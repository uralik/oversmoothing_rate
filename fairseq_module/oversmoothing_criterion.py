from dataclasses import dataclass, field
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterionConfig
from fairseq.criterions.label_smoothed_cross_entropy import register_criterion, LabelSmoothedCrossEntropyCriterion
import logging
import torch
import math

@dataclass
class OversmoothingCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    oversmoothing_weight: float = field(
        default=0.5,
        metadata={'help': '(1-os_weight)*NLL + os_weight*OSL'}
    )
    oversmoothing_margin: float = field(
        default=0,
        metadata={'help': 'max(log p(eos|prefix) - log p(suffix|prefix) + M, 0)'}
    )

logger = logging.getLogger(__name__)

def compute_oversmoothing_logratio(logits, target, non_pad_mask, eos_idx, margin=1e-5):
    full_lprobs = torch.log_softmax(logits, dim=-1)
    target_lprobs = torch.gather(full_lprobs, dim=-1, index=target.unsqueeze(-1))

    # reverse cumsum fast workaround, this makes approximation error for suffix_lprob[:,-1]
    # in other words, after this operation the smallest suffix of one token does not equal eaxctly to that
    # true eos_probability. So it is better to exlcude those positions from OSL since theoretically loss there is 0.
    target_lprobs_withoutpad = (target_lprobs * non_pad_mask).squeeze(-1)
    suffix_lprob = target_lprobs_withoutpad + torch.sum(target_lprobs_withoutpad, dim=-1, keepdims=True) - torch.cumsum(target_lprobs_withoutpad, dim=-1)
    
    eos_lprobs = full_lprobs[:,:,eos_idx] * non_pad_mask.squeeze(-1)

    oversmoothing_loss = torch.maximum(eos_lprobs - suffix_lprob + margin, torch.zeros_like(suffix_lprob))
    oversmoothing_loss = (oversmoothing_loss.sum(dim=1) / non_pad_mask.squeeze(dim=-1).sum(dim=1)).mean()

    # computing the oversmoothing rate here for free
    with torch.no_grad():
        oversmoothed = eos_lprobs > suffix_lprob
        oversmoothed = oversmoothed * non_pad_mask.squeeze(-1)  # exclude pad cases from oversmoothing rate
        oversmoothed = oversmoothed * (target != eos_idx).float() # exclude t=true_eos from oversmoothing counts

        num_osr_per_seq = non_pad_mask.squeeze(-1).sum(-1) - 1  # exclude the <eos> from each seq count
        osr = oversmoothed.sum(-1) / num_osr_per_seq # compute oversmoothing per sequence

    return oversmoothing_loss, osr

@register_criterion("oversmoothing_loss",
                    dataclass=OversmoothingCriterionConfig)
class OversmoothingCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg,
                 label_smoothing,
                 ignore_prefix_size=0,
                 report_accuracy=False,
                 oversmoothing_weight=0.5,
                 oversmoothing_margin=0):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.pad_idx = task.tgt_dict.pad()
        self.eos_idx = task.tgt_dict.eos()
        self.label_smoothing_eps = label_smoothing
        self.oversmoothing_margin = oversmoothing_margin
        self.oversmoothing_weight = oversmoothing_weight

        logger.info(f'Oversmoothing loss, margin={oversmoothing_margin}, weight={oversmoothing_weight}')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        updated_features, extra_stats = model(**sample["net_input"])

        loss, nll_loss = self.compute_loss(model, (updated_features, extra_stats), sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "nll_loss": nll_loss.item(),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        non_pad_mask = (sample["target"] != 1).float()[:, :, None]

        target = model.get_targets(sample, updated_features)
        oversmoothing_loss, osr_per_sequence = compute_oversmoothing_logratio(updated_features, target, non_pad_mask, self.eos_idx, self.oversmoothing_margin)

        loss = (1-self.oversmoothing_weight)*loss + self.oversmoothing_weight*oversmoothing_loss

        logging_output['oversmoothing_loss'] = oversmoothing_loss.item()
        logging_output['oversmoothing_rate'] = osr_per_sequence.sum().item()  # this will need to be reduced with nsentences!

        logging_output['loss'] = loss.item()

        return loss, sample_size, logging_output
