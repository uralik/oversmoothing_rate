from fairseq.tasks import register_task
from fairseq import metrics, utils
from fairseq.data import data_utils
from fairseq.tasks.translation import TranslationTask, TranslationConfig
from fairseq.models.transformer import transformer_iwslt_de_en
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerModel,
    base_architecture,
    transformer_wmt_en_de_big,
)
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
from dataclasses import dataclass, field
import logging
import numpy as np

import collections
import itertools

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)

@dataclass
class TranslationOversmoothingConfig(TranslationConfig):
    stat_save_path: str = field(
        default='extra_stat.pkl',
        metadata={
            "help": "where to save extra stuff"
        },
    )

@register_task("translation_oversmoothing", dataclass=TranslationOversmoothingConfig)
class OversmoothingTranslationTask(TranslationTask):
    cfg: TranslationOversmoothingConfig

    def __init__(self, cfg: TranslationOversmoothingConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.extra_statistics = None
        self.initialize_extra_statistics()
        self.validation_epoch = False

    def initialize_extra_statistics(self):
        """
        extra_statistics only handles lists, i.e. add any list into logging_outputs and it will be dumped in the extra_statistics
        """
        self.extra_statistics = collections.defaultdict(list)

    @torch.no_grad()
    def compute_eos_log_probabilities(self, model_logit, non_pad_mask):
        model_lprobs = model_logit.log_softmax(dim=-1)
        eos_model_lprobs = model_lprobs[:,:,self.tgt_dict.eos()]
        real_eos_ids = non_pad_mask.sum(dim=1).long() - 1  # convert lengths to eos ids
        probs_before_last_ts = []
        probs_last_ts = []

        for probs_seq, real_eos_id in zip(eos_model_lprobs, real_eos_ids):
            probs_before_last_ts.append(probs_seq[:real_eos_id].tolist())
            probs_last_ts.append(probs_seq[real_eos_id].item())

        return probs_before_last_ts, probs_last_ts

    @torch.no_grad()
    def compute_eos_ranks(self, model_logit, non_pad_mask):
        eos_ranks = (model_logit[:,:,self.tgt_dict.eos()].unsqueeze(dim=-1) <= model_logit[:,:]).float().sum(dim=-1)
        real_eos_ids = non_pad_mask.sum(dim=1).long() - 1  # convert lengths to eos ids
        ranks_before_last_ts = []
        rank_last_ts = []
        for rank_seq, real_eos_id in zip(eos_ranks, real_eos_ids):
            ranks_before_last_ts.append(rank_seq[:real_eos_id].tolist())
            rank_last_ts.append(rank_seq[real_eos_id].item())

        return ranks_before_last_ts, rank_last_ts

    def _inference_with_bleu(self, generator, sample, model):
        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)

        # we keep all generated hyps such that we can compute pool-level quality later on
        hyps, scores, refs = [], [], []
        for i in range(len(gen_out)):
            generated_hyps = [decode(gen_out[i][hypid]["tokens"]) for hypid in range(len(gen_out[i]))]
            generated_scores = [gen_out[i][hypid]["score"].item() for hypid in range(len(gen_out[i]))]
            hyps.append(generated_hyps)
            scores.append(generated_scores)
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        return {
            'refs_text': refs,
            'hyps_text': hyps,
            'hyps_scores': scores,
            'gen_out': gen_out }
  
    

    @torch.no_grad()
    def valid_step(self, sample, model, criterion):
        # original valid step
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = criterion(model, sample)

        # processing target sequences
        non_pad_mask = (sample['target'] != 1).float()[:,:,None]
        target_net_output = model(**sample['net_input'])
        target_model_lprobs = F.log_softmax(target_net_output[0], dim=-1)
        target_model_true_lprobs = torch.gather(target_model_lprobs, dim=-1, index=sample['target'].unsqueeze(-1))
        target_eos_log_probs_nt, target_eos_log_probs_t = self.compute_eos_log_probabilities(target_net_output[0], non_pad_mask.squeeze(-1))
        target_false_eos_ranks, target_true_eos_ranks = self.compute_eos_ranks(target_net_output[0], non_pad_mask.squeeze(-1))
        target_seq_lengths = non_pad_mask.squeeze(-1).sum(-1).tolist()
        logging_output['target_eos_log_probs_nt'] = target_eos_log_probs_nt
        logging_output['target_eos_log_probs_t'] = target_eos_log_probs_t
        logging_output['target_false_eos_ranks'] = target_false_eos_ranks
        logging_output['target_true_eos_ranks'] = target_true_eos_ranks
        logging_output['target_seq_lengths'] = target_seq_lengths
        logging_output['target_model_log_probs'] = target_model_true_lprobs.detach().cpu().tolist()
        logging_output['target_src_seq_lengths'] = sample['net_input']['src_lengths'].tolist()

        # now we generate only once!
        if self.cfg.eval_bleu:
            result_dict = self._inference_with_bleu(self.sequence_generator, sample, model)
            generated = result_dict['gen_out']
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync

            beam_hyps = []
            for i in range(len(generated)):
                beam_hyps.append(generated[i][0]["tokens"])
            generated_seq_lengths = [hyp.numel() for hyp in beam_hyps]

            beam_hyps_target = data_utils.collate_tokens(beam_hyps, pad_idx=self.tgt_dict.pad(), move_eos_to_beginning=False)
            beam_hyps_input = data_utils.collate_tokens(beam_hyps, pad_idx=self.tgt_dict.pad(), move_eos_to_beginning=True)

            generated_net_output = model(sample['net_input']['src_tokens'], sample['net_input']['src_lengths'], beam_hyps_input)
            generated_non_pad_mask = (beam_hyps_target != 1).float()[:,:,None]  # 1 is PAD token
            generated_model_lprobs = F.log_softmax(generated_net_output[0], dim=-1)
            generated_model_true_lprobs = torch.gather(generated_model_lprobs, dim=-1, index=beam_hyps_target.unsqueeze(-1))
            generated_eos_log_probs_nt, generated_eos_log_probs_t = self.compute_eos_log_probabilities(generated_net_output[0], generated_non_pad_mask.squeeze(-1))
            generated_false_eos_ranks, generated_true_eos_ranks = self.compute_eos_ranks(generated_net_output[0], generated_non_pad_mask.squeeze(-1))

            logging_output['generated_seq_lengths'] = generated_seq_lengths
            logging_output['generated_eos_log_probs_nt'] = generated_eos_log_probs_nt
            logging_output['generated_eos_log_probs_t'] = generated_eos_log_probs_t
            logging_output['generated_false_eos_ranks'] = generated_false_eos_ranks
            logging_output['generated_true_eos_ranks'] = generated_true_eos_ranks
            logging_output['generated_hyps_text'] = result_dict['hyps_text']
            logging_output['generated_refs_text'] = result_dict['refs_text']
            logging_output['generated_hyps_scores'] = result_dict['hyps_scores']
            logging_output['generated_model_log_probs'] = generated_model_true_lprobs.detach().cpu().tolist()
        
        return loss, sample_size, logging_output


    def reduce_metrics(self, logging_outputs, criterion):

        super().reduce_metrics(logging_outputs, criterion)

        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)  # this one does not include padding

        nll_loss_sum = sum(log.get('nll_loss', torch.tensor(0)) for log in logging_outputs)

        oversmoothing_loss = sum(log.get('oversmoothing_loss', torch.tensor(0)) for log in logging_outputs)

        oversmoothing_rate = sum(log.get('oversmoothing_rate', torch.tensor(0)) for log in logging_outputs)
        oversmoothing_rate = oversmoothing_rate / nsentences  # see criterion for details

        with metrics.aggregate('train'):
            metrics.log_scalar('target/nll_loss_sum', nll_loss_sum)
            metrics.log_scalar('target/oversmoothing_loss', oversmoothing_loss)
            metrics.log_scalar('target/oversmoothing_rate', oversmoothing_rate)

        # extra stats to pickle
        if self.validation_epoch:
            
            keys_with_list_type_values = []
            for k,v in logging_outputs[0].items():
                if isinstance(v, list):
                    keys_with_list_type_values.append(k)

            for k in keys_with_list_type_values:
                reduced_list = [log[k] for log in logging_outputs]
                reduced_list = sum(reduced_list, [])
                self.extra_statistics[k] += reduced_list

            with metrics.aggregate('valid'):
                metrics.log_scalar('target/oversmoothing_rate', oversmoothing_rate)

    def log_tensorboard(self):
            """
            Logging entire valid epoch metrics 
            """
            
            def list_avg(l):
                if sum(l) == 0:
                    return 0
                else:
                    return float(sum(l)) / len(l)

            with metrics.aggregate('valid'):
                seq_types = ['target', 'generated']
                for seq_type in seq_types:
                    # ranks
                    seq_avg_false_eos_rank = [list_avg(l) for l in self.extra_statistics[f'{seq_type}_false_eos_ranks']]
                    metrics.log_scalar(f'{seq_type}/false_eos_rank', list_avg(seq_avg_false_eos_rank))
                    metrics.log_scalar(f'{seq_type}/true_eos_rank', list_avg(self.extra_statistics[f'{seq_type}_true_eos_ranks']))

                # log length ratio in tb
                if self.extra_statistics[f'generated_seq_lengths']:
                    metrics.log_scalar(f'target_generated_lenratio', (torch.tensor(self.extra_statistics[f'target_seq_lengths']) / torch.tensor(self.extra_statistics[f'generated_seq_lengths'])).mean().item())
