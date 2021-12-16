import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import scipy
import scipy.stats
from typing import Dict, List
from matplotlib import pyplot as plt
import collections
from collections import defaultdict
import math
from tqdm.auto import tqdm
from glob import glob
import os
import json
import gc
import itertools
from sacrebleu.metrics import BLEU
from bleurt import score


class StatsParserOversmoothing:
    def __init__(self, experiments_directory, params, lengths, pkl_prefix='best', tqdm=False, reduce_ram=True):
        self.pkl_prefix = pkl_prefix
        self.model_configs = []
        self.models = []
        self.model_full_configs = []  # useful to reinstantiate fairseq tasks later to access dicts etc.
        self.model_decoding_settings = []

        self.bleu = BLEU()
        self.scorer = score.BleurtScorer('/home/mae9785/BLEURT-20-D12')

        self.aggregations = {
            'max': np.max,
            'min': np.min,
            'avg': np.mean
        }

        self.bleurt_scores = {}
        self.length_ratios = {}
        self.nt_eos_log_probs = {}

        self.experiments_directory = experiments_directory
        self.params = params
        self.lengths = lengths
        self.tqdm = tqdm
        self.reduce_ram = reduce_ram
        self.load_results()
        self.preprocess_metrics()
        self.compute_mean_eos_rank_globally()
        self.compute_length_ratio()
        
    def load_results(self):
        # Pathlib doesnt follow symlinks due to ** in the rglob
        #pkl_fnames = list(Path(self.experiments_directory).rglob('*.pkl'))
        if isinstance(self.experiments_directory, list):
            pkl_fnames = []
            for exp_dir in self.experiments_directory:
                pkl_fnames.extend(glob(os.path.join(exp_dir, '*', f'{self.pkl_prefix}*.pkl')))
        else:
            pkl_fnames = glob(os.path.join(self.experiments_directory, '*', f'{self.pkl_prefix}*.pkl'))
        progress_bar = tqdm(pkl_fnames, desc="Parsing pkls", disable=(not self.tqdm), total=len(pkl_fnames))

        for pkl_file in progress_bar:
            if 'args.pkl' in pkl_file:
                continue
            args = {}
            args_path = glob(os.path.join('/', *pkl_file.split('/')[:-1], '*_args.pkl'))
            if len(args_path) > 0:
                args = pickle.load(open(args_path[0], 'rb'))
                
            model = pickle.load(open(pkl_file, 'rb'))
            gc.collect()
            #import pdb; pdb.set_trace()
            cfg_to_load = 'saved_cfg' if 'saved_cfg' in model[0] else 'cfg'
            full_cfg = vars(model[0][cfg_to_load]['model'])
            cfg_criterion = model[0][cfg_to_load]['criterion']
            decoding_settings = json.loads(model[0][cfg_to_load]['task']['eval_bleu_args'])
            seed = model[0][cfg_to_load].common.seed
            current_config = {
                'seed': seed
            }
            for parameter in self.params:
                current_config[parameter] = None
                if parameter in cfg_criterion:
                    current_config[parameter] = cfg_criterion[parameter]
                if parameter in decoding_settings:
                    current_config[parameter] = decoding_settings[parameter]
                if parameter in full_cfg:
                    current_config[parameter] = full_cfg[parameter]
                if f'--{parameter}' in args:
                    args_parameter = f'--{parameter}'
                    current_config[parameter] = args[args_parameter]
            self.model_configs += [current_config]
            self.models += [model]
            self.model_full_configs += [model[0]['cfg']]  # we keep it in order to check overridden args e.g. beam size
            self.model_decoding_settings.append(decoding_settings)

    @staticmethod
    def __calculate_terminal_ll(update, seq_type, reduce='sum'):
        if reduce == 'sum':
            result = sum(update[f'{seq_type}_eos_log_probs_t'])
        elif reduce == 'mean':
            result = float(sum(update[f'{seq_type}_eos_log_probs_t']))/len(update[f'{seq_type}_eos_log_probs_t'])
        
        return result

    def __calculate_nonterminal_ll(self, update, seq_type, reduce='sum', model_id=None):
        flat_list = list(itertools.chain.from_iterable(update[f'{seq_type}_eos_log_probs_nt']))
        if reduce == 'sum':
            result = sum(flat_list)
        elif reduce == 'mean':
            result = float(sum(flat_list))/len(flat_list)
        if seq_type == 'target' and model_id is not None:
            nt_eos_log_probs = []
            for seq in update[f'{seq_type}_eos_log_probs_nt']:
                nt_eos_log_probs += [np.mean(seq)]
            self.nt_eos_log_probs = nt_eos_log_probs
        return result

    def __recalculate_len_ratio(self, update, model_id): # update stands for self.models[model_id][-1]
        tl = np.array(update['target_seq_lengths'])
        gl = np.array(update['generated_seq_lengths'])
        lr_seq = []
        for tl_seq, gl_seq in zip(tl, gl):
            lr_seq += [tl_seq / gl_seq]
        self.length_ratios[model_id] = lr_seq
        return tl.sum() / gl.sum()

    def __recalculate_len_ratio_seq(self, update, model_id): # update stands for self.models[model_id][-1]
        tl = np.array(update['target_seq_lengths'])
        gl = np.array(update['generated_seq_lengths'])
        lr_seq = []
        for tl_seq, gl_seq in zip(tl, gl):
            lr_seq += [tl_seq / gl_seq]
        return np.mean(lr_seq)
           
    def compute_bleu(self, update):
        beam_all_hyps = update['generated_hyps_text']
        beam_top_hyps = [l[0] for l in beam_all_hyps]
        refs = update['generated_refs_text']
        return self.bleu.corpus_score(beam_top_hyps, [refs]).score

    def compute_bleurt(self, update, model_id):
        return 0
        nbest1_hyps = [hyps[0] for hyps in update['generated_hyps_text']]
        refs = update['generated_refs_text']
        scores = self.scorer.score(references=refs, candidates=nbest1_hyps, batch_size=100)
        self.bleurt_scores[model_id] = scores
        return np.mean(np.array(scores) * 100)

    def calc_number_of_abrupted_seqs(self, update):
        ok = 0
        total = 0
        for hyps in update['generated_hyps_text']:
            total += 1
            top_hyp = hyps[0]
            ending = top_hyp.split()[-1]
            if ending in ['.', '?', '!', '...', '&quot;'] or ending[-1] == '.':
                ok += 1
        return (total - ok) / total * 100

    def calc_empty_winrate(self, update, mode='target'):
        winrate = 0
        for seq, eos_seq in zip(update[f'{mode}_model_log_probs'], update[f'{mode}_eos_log_probs_nt']):
            seq_log_prob = sum([s[0] for s in seq[:len(eos_seq) + 1]])
            eos_log_prob = eos_seq[0]
            if seq_log_prob > eos_log_prob:
                winrate += 1
        return winrate / len(update['generated_eos_log_probs_nt'])

    def preprocess_metrics(self):
        for i, model in enumerate(self.models):
            for update in model:
                update['bleu'] = self.compute_bleu(update)
                update['abrupted'] = self.calc_number_of_abrupted_seqs(update)
                update['bleurt'] = self.compute_bleurt(update, i)
                update['target_generated_lenratio_mean'] = self.__recalculate_len_ratio(update, i)
                update['target_generated_lenratio_seq_mean'] = self.__recalculate_len_ratio_seq(update, i)
                update['target_empty_winrate'] = self.calc_empty_winrate(update, 'target')
                update['generated_empty_winrate'] = self.calc_empty_winrate(update, 'generated')
                for seq_type in ['target']:
                    update[f'{seq_type}_nonterminal_ll'] = self.__calculate_nonterminal_ll(update, seq_type, model_id=i)
                    update[f'{seq_type}_terminal_ll'] = self.__calculate_terminal_ll(update, seq_type)
                    update[f'{seq_type}_nonterminal_ll_mean'] = self.__calculate_nonterminal_ll(update, seq_type, reduce='mean')
                    update[f'{seq_type}_terminal_ll_mean'] = self.__calculate_terminal_ll(update, seq_type, reduce='mean')
                   
                if 'stats' in update:
                    for metric in update['stats']:
                        update[f'tb_{metric}'] = update['stats'][metric]

                if 'stat' in update:
                    for metric in update['stat']:
                        update[f'tb_{metric}'] = update['stat'][metric]

    @staticmethod
    def generate_cfg_description(cfg, i):
        description = f'model #{i}, '
        for parameter in cfg:
            description += f'{parameter}: {cfg[parameter]}, '
        return description[:-2]

    def check_parameters(self, parameters: List[str]):
        parameter_info = collections.OrderedDict()
        for i, model in enumerate(self.models):
            description = self.generate_cfg_description(self.model_configs[i], i)
            parameter_info[description] = {}
            for update in model:
                update_str = f'updates: {update["num_updates"]} epoch: {update["epoch"]}'
                parameters_desc = ''
                for parameter in parameters:
                    parameters_desc += f'{parameter}: {update[parameter]};  '
                parameter_info[description][update_str] = parameters_desc
        return self.__generate_dataframe(parameter_info)

    def compute_mean_eos_rank_globally(self):
        for model, decoding_settings in zip(self.models, self.model_decoding_settings):
            # using the last update
            for seq_type in ['target', 'generated']:
                false_eos_ranks = sum(model[-1][f'{seq_type}_false_eos_ranks'], [])
                false_eos_ranks_stats = scipy.stats.describe(false_eos_ranks)
                false_eos_ranks_median = np.median(false_eos_ranks)
                # using r suffix to deal with html table logic 
                model[-1][f'median_{seq_type}_false_eos_r'] = false_eos_ranks_median
                model[-1][f'skew_{seq_type}_false_eos_r'] = false_eos_ranks_stats.skewness
                model[-1][f'mean_{seq_type}_false_eos_r'] = false_eos_ranks_stats.mean
                model[-1][f'std_{seq_type}_false_eos_r'] = np.sqrt(false_eos_ranks_stats.variance)
                # how many false eos ranks are in beam in the worst case
                #false_eos_ranks = np.array(false_eos_ranks)
                #in_beam_false_eos_count = np.sum(false_eos_ranks <= decoding_settings['beam']) / float(len(false_eos_ranks))
                #in_beam_false_eos_med = np.median(false_eos_ranks[np.array(false_eos_ranks) <= decoding_settings['beam']])
                #model[-1][f'{seq_type}_false_eos_r_in_beam'] = in_beam_false_eos_count
                #model[-1][f'{seq_type}_false_eos_r_in_beam_med'] = in_beam_false_eos_med

                true_eos_ranks_stats = scipy.stats.describe(model[-1][f'{seq_type}_true_eos_ranks'])
                true_eos_ranks_median = np.median(model[-1][f'{seq_type}_true_eos_ranks'])
                model[-1][f'skew_{seq_type}_true_eos_r'] = true_eos_ranks_stats.skewness
                model[-1][f'median_{seq_type}_true_eos_r'] = true_eos_ranks_median
                model[-1][f'mean_{seq_type}_true_eos_r'] = true_eos_ranks_stats.mean
                model[-1][f'std_{seq_type}_true_eos_r'] = np.sqrt(true_eos_ranks_stats.variance)
                #in_beam_true_eos_count = np.sum(np.array(model[-1][f'{seq_type}_true_eos_ranks']) <= decoding_settings['beam']) / float(len(model[-1][f'{seq_type}_true_eos_ranks']))
                #model[-1][f'{seq_type}_true_eos_r_in_beam'] = in_beam_true_eos_count


    def compute_length_ratio(self):
        for model in self.models:
            # using the last update
            target_lengths = np.array(model[-1][f'target_seq_lengths'])
            generated_lengths = np.array(model[-1][f'generated_seq_lengths'])
            target_generated_ratio = target_lengths / generated_lengths
            model[-1]['target_generated_lenratio_mean'] = np.mean(target_generated_ratio)
            model[-1]['target_generated_lenratio_median'] = np.median(target_generated_ratio)


    def check_model(self, model_id, metrics):
        model_info = {}
        for update in self.models[model_id]:
            update_str = f'updates: {update["num_updates"]} epoch: {update["epoch"]}'
            for parameter in self.params:
                if parameter not in model_info:
                    model_info[parameter] = {}
                model_info[parameter][update_str] = self.model_configs[model_id][parameter]
            for metric in metrics:
                if metric not in model_info:
                    model_info[metric] = {}
                model_info[metric][update_str] = update[metric]
        return self.__generate_dataframe(model_info)

    def draw_distributions(self, model_id, distribution_name, update_step=-1,
                           xlim=200, ylim=250):  # either 'target_eos_len_distribution' or 'generated_eos_len_distribution'
        model = self.models[model_id]
        distribution = model[update_step][distribution_name]
        num_eos_tokens = len(distribution.keys())
        grid_height = int(np.ceil(num_eos_tokens / 2))
        fig, axes = plt.subplots(grid_height, 2, figsize=(30, grid_height * 10))
        axes = axes.ravel()
        for eos_id, axis in zip(sorted(distribution), axes):
            length_distribution = {}
            for length in distribution[eos_id]:
                if length not in length_distribution:
                    length_distribution[length] = 0
                length_distribution[length] += 1
            axis.bar(length_distribution.keys(), length_distribution.values(), width=1)
            axis.set_title(f'EOS #{eos_id}', fontsize=14)
            axis.set_xlim([0, xlim])
            axis.set_ylim([0, ylim])
            axis.grid()

    def draw_single_distribution_for_several_eos(self, model_id, distribution_name, update_step=-1, eos_ids=[],
                                                 xlim=200, ylim=250):
        model = self.models[model_id]
        distribution = model[update_step][distribution_name]
        plt.figure(figsize=(10, 8))
        for eos_id in eos_ids:
            length_distribution = {}
            if eos_id not in distribution:
                continue
            for length in distribution[eos_id]:
                if length not in length_distribution:
                    length_distribution[length] = 0
                length_distribution[length] += 1
            plt.bar(length_distribution.keys(), length_distribution.values(), width=1, label=eos_id)
        plt.xlim([0, xlim])
        plt.ylim([0, ylim])
        plt.legend()
        plt.grid()

    def get_aggregated_bin_eosprobs(self, model, seq_agg, dataset_agg, mode, min_bin_samples, min_bin_size, max_length):
        rank_before_last_distr = {}
        rank_last_distr = {}

        lengths_bounds = [(i, i + min_bin_size) for i in range(0, max_length, min_bin_size)]
        length_to_bounds_mapping = {}
        for length in range(max_length):
            for len_bound in lengths_bounds:
                if len_bound[0] <= length < len_bound[1]:
                    length_to_bounds_mapping[length] = len_bound

        count_lengths = {}
        #import ipdb; ipdb.set_trace()

        for prev_ranks, last_rank, seq_len in zip(model[-1][f'{mode}_false_eos_probs'],
                                                     model[-1][f'{mode}_true_eos_probs'],
                                                     model[-1][f'{mode}_seq_lengths']):
            len_bound = length_to_bounds_mapping[int(seq_len)]
            if len_bound not in count_lengths:
                count_lengths[len_bound] = 0
                rank_before_last_distr[len_bound] = []
                rank_last_distr[len_bound] = []

            count_lengths[len_bound] += 1
            rank_before_last_distr[len_bound].append(self.aggregations[seq_agg](prev_ranks))
            rank_last_distr[len_bound].append(last_rank)

        merged_rank_before_last_distr = [[[], [], 0]]
        merged_rank_last_distr = [[[], [], 0]]
        for len_bound in lengths_bounds:
            if len_bound in rank_before_last_distr:
                if merged_rank_before_last_distr[-1][2] >= min_bin_samples:
                    merged_rank_before_last_distr += [
                        [list(len_bound),
                         rank_before_last_distr[len_bound],
                         count_lengths[len_bound]]
                    ]
                    merged_rank_last_distr += [
                        [list(len_bound),
                         rank_last_distr[len_bound],
                         count_lengths[len_bound]]
                    ]
                else:
                    merged_rank_before_last_distr[-1][0] += list(len_bound)
                    merged_rank_before_last_distr[-1][1] += rank_before_last_distr[len_bound]
                    merged_rank_before_last_distr[-1][2] += count_lengths[len_bound]
                    merged_rank_last_distr[-1][0] += list(len_bound)
                    merged_rank_last_distr[-1][1] += rank_last_distr[len_bound]
                    merged_rank_last_distr[-1][2] += count_lengths[len_bound]

        if len(merged_rank_before_last_distr) > 1:
            if merged_rank_before_last_distr[-1][2] < min_bin_samples:
                merged_rank_before_last_distr[-2][0] += merged_rank_before_last_distr[-1][0]
                merged_rank_before_last_distr[-2][1] += merged_rank_before_last_distr[-1][1]
                merged_rank_before_last_distr[-2][2] += merged_rank_before_last_distr[-1][2]
                del merged_rank_before_last_distr[-1]
                merged_rank_last_distr[-2][0] += merged_rank_last_distr[-1][0]
                merged_rank_last_distr[-2][1] += merged_rank_last_distr[-1][1]
                merged_rank_last_distr[-2][2] += merged_rank_last_distr[-1][2]
                del merged_rank_last_distr[-1]

        mean_bin_lengths_before_last, aggregated_bin_values_before_last = [], []
        mean_bin_lengths_last, aggregated_bin_values_last = [], []
        bin_bounds = []
        for value_before_last, value_last in zip(merged_rank_before_last_distr, merged_rank_last_distr):
            bin_bounds += [(np.min(value_last[0]), np.max(value_last[0]))]
            mean_bin_lengths_before_last += [(np.min(value_before_last[0]) + np.max(value_before_last[0])) / 2]
            aggregated_bin_values_before_last += [self.aggregations[dataset_agg](value_before_last[1])]
            mean_bin_lengths_last += [(np.min(value_last[0]) + np.max(value_last[0])) / 2]
            aggregated_bin_values_last += [self.aggregations[dataset_agg](value_last[1])]
        return (mean_bin_lengths_before_last, aggregated_bin_values_before_last), \
               (mean_bin_lengths_last, aggregated_bin_values_last), bin_bounds

    def get_aggregated_bin_ranks(self, model, seq_agg, dataset_agg, mode, min_bin_samples, min_bin_size, max_length):
        rank_before_last_distr = {}
        rank_last_distr = {}

        lengths_bounds = [(i, i + min_bin_size) for i in range(0, max_length, min_bin_size)]
        length_to_bounds_mapping = {}
        for length in range(max_length):
            for len_bound in lengths_bounds:
                if len_bound[0] <= length < len_bound[1]:
                    length_to_bounds_mapping[length] = len_bound

        count_lengths = {}
        #import ipdb; ipdb.set_trace()

        for prev_ranks, last_rank, seq_len in zip(model[-1][f'{mode}_false_eos_ranks'],
                                                     model[-1][f'{mode}_true_eos_ranks'],
                                                     model[-1][f'{mode}_seq_lengths']):
            len_bound = length_to_bounds_mapping[int(seq_len)]
            if len_bound not in count_lengths:
                count_lengths[len_bound] = 0
                rank_before_last_distr[len_bound] = []
                rank_last_distr[len_bound] = []

            count_lengths[len_bound] += 1
            rank_before_last_distr[len_bound].append(self.aggregations[seq_agg](prev_ranks))
            rank_last_distr[len_bound].append(last_rank)

        merged_rank_before_last_distr = [[[], [], 0]]
        merged_rank_last_distr = [[[], [], 0]]
        for len_bound in lengths_bounds:
            if len_bound in rank_before_last_distr:
                if merged_rank_before_last_distr[-1][2] >= min_bin_samples:
                    merged_rank_before_last_distr += [
                        [list(len_bound),
                         rank_before_last_distr[len_bound],
                         count_lengths[len_bound]]
                    ]
                    merged_rank_last_distr += [
                        [list(len_bound),
                         rank_last_distr[len_bound],
                         count_lengths[len_bound]]
                    ]
                else:
                    merged_rank_before_last_distr[-1][0] += list(len_bound)
                    merged_rank_before_last_distr[-1][1] += rank_before_last_distr[len_bound]
                    merged_rank_before_last_distr[-1][2] += count_lengths[len_bound]
                    merged_rank_last_distr[-1][0] += list(len_bound)
                    merged_rank_last_distr[-1][1] += rank_last_distr[len_bound]
                    merged_rank_last_distr[-1][2] += count_lengths[len_bound]

        if len(merged_rank_before_last_distr) > 1:
            if merged_rank_before_last_distr[-1][2] < min_bin_samples:
                merged_rank_before_last_distr[-2][0] += merged_rank_before_last_distr[-1][0]
                merged_rank_before_last_distr[-2][1] += merged_rank_before_last_distr[-1][1]
                merged_rank_before_last_distr[-2][2] += merged_rank_before_last_distr[-1][2]
                del merged_rank_before_last_distr[-1]
                merged_rank_last_distr[-2][0] += merged_rank_last_distr[-1][0]
                merged_rank_last_distr[-2][1] += merged_rank_last_distr[-1][1]
                merged_rank_last_distr[-2][2] += merged_rank_last_distr[-1][2]
                del merged_rank_last_distr[-1]

        mean_bin_lengths_before_last, aggregated_bin_values_before_last = [], []
        mean_bin_lengths_last, aggregated_bin_values_last = [], []
        bin_bounds = []
        for value_before_last, value_last in zip(merged_rank_before_last_distr, merged_rank_last_distr):
            bin_bounds += [(np.min(value_last[0]), np.max(value_last[0]))]
            mean_bin_lengths_before_last += [(np.min(value_before_last[0]) + np.max(value_before_last[0])) / 2]
            aggregated_bin_values_before_last += [self.aggregations[dataset_agg](value_before_last[1])]
            mean_bin_lengths_last += [(np.min(value_last[0]) + np.max(value_last[0])) / 2]
            aggregated_bin_values_last += [self.aggregations[dataset_agg](value_last[1])]
        return (mean_bin_lengths_before_last, aggregated_bin_values_before_last), \
               (mean_bin_lengths_last, aggregated_bin_values_last), bin_bounds

    def draw_ranks(self, seq_agg='max', dataset_agg='avg', mode='target', min_bin_samples=200, min_bin_size=5, max_length=250,
                   metrics=(('#EOS', 'number_eos_tokens'), ('ME', 'marginal_entropy_weight'), ('CE', 'conditional_entropy_weight')),
                   model_ids=None, prob_or_rank='rank'):
        if model_ids is None:
            model_ids = [(i, False) for i in range(len(self.models))]

        fig, axes = plt.subplots(2, 1, figsize=(20, 15))

        for model_id, is_bold in model_ids:
            line_width = 6.0 if is_bold else 1
            model = self.models[model_id]
            cfg = self.model_configs[model_id]
            description = f'{model_id}, '
            for metric_name, metric_id in metrics:
                description += f'{metric_name}: {cfg[metric_id]}, '
            description = description[:-2]
            if prob_or_rank == 'rank':
                (mean_bin_lengths_before_last, aggregated_bin_values_before_last), \
                    (mean_bin_lengths_last, aggregated_bin_values_last), _ = self.get_aggregated_bin_ranks(model, seq_agg, dataset_agg, 
                                                                                                        mode, min_bin_samples,
                                                                                                        min_bin_size, max_length)
            elif prob_or_rank == 'prob':
                (mean_bin_lengths_before_last, aggregated_bin_values_before_last), \
                    (mean_bin_lengths_last, aggregated_bin_values_last), _ = self.get_aggregated_bin_eosprobs(model, seq_agg, dataset_agg, 
                                                                                                        mode, min_bin_samples,
                                                                                                        min_bin_size, max_length)
            else:
                raise NotImplementedError
                
            axes[0].plot(mean_bin_lengths_before_last, aggregated_bin_values_before_last, label=description, marker="o",
                         linewidth=line_width)
            axes[1].plot(mean_bin_lengths_last, aggregated_bin_values_last, label=description, marker="o")
        axes[0].grid()
        axes[1].grid()
        axes[0].set_xlabel(f'{mode} length', fontsize=14)
        axes[1].set_xlabel(f'{mode} length', fontsize=14)
        axes[0].set_ylabel(f'seq-agg={seq_agg} False EOS {prob_or_rank}', fontsize=14)
        axes[1].set_ylabel(f'True EOS {prob_or_rank}', fontsize=14)
        axes[0].legend(fontsize=12)
        axes[1].legend(fontsize=12)

    def get_criteria_table(self, metrics, params, model_ids, seq_agg='avg', dataset_agg='avg', min_bin_samples=400, min_bin_size=10, max_length=250):
        style_border_right_solid = 'border-right: 1px solid;'
        style_border_right_solid_bold = 'border-right: 3px solid;'
        style_border_right_dotted = 'border-right: 1px dotted;'
        style_align_center = 'text-align: center;"'

#        header_desc = [f'<td style="{style_border_right_solid}"></td>']
#        subheader_desc = [f'<td style="{style_border_right_solid}"></td>']

        header_desc = []
        subheader_desc = []

        for i, (param_name, param_id) in enumerate(params):
            if i == len(params) - 1:
                header_desc += [f'<td style="{style_border_right_solid_bold}{style_align_center}"><strong>{param_name}</strong></td>']
                subheader_desc += [f'<td style="{style_border_right_solid_bold}"></td>']
            else:
                header_desc += [f'<td style="{style_border_right_solid}{style_align_center}"><strong>{param_name}</strong></td>']
                subheader_desc += [f'<td style="{style_border_right_solid}"></td>']

        for metric, metric_name in metrics:
            if 'eos_rank' not in metric and 'eos_prob' not in metric:
                header_desc += [
                    f'<td style="{style_border_right_solid}{style_align_center}"><strong>{metric_name}</strong></td>']
                subheader_desc += [f'<td style="{style_border_right_solid}"></td>']
            else:
                mode = metric.split('_')[0]
                (a, b), (c, d), bin_bounds = self.get_aggregated_bin_ranks(self.models[0], seq_agg, dataset_agg, mode,
                                                                           min_bin_samples,
                                                                           min_bin_size, 250)
                subheader_desc += [
                    f'<td style="{style_border_right_dotted}{style_align_center}">{item[0]}-{item[1]}</td>'
                    if i != len(bin_bounds) - 1 else f'<td style="{style_border_right_solid}{style_align_center}">{item[0]}-{item[1]}</td>'
                    for i, item in enumerate(bin_bounds)
                ]
                header_desc += [
                    f'<td colspan="{len(bin_bounds)}" style="{style_border_right_solid}{style_align_center}"><strong>{metric_name}</strong></td>'
                ]

        header = f'<tr>{"".join(header_desc)}</tr><tr>{"".join(subheader_desc)}</tr>'

        content = []
        for model_id in model_ids:
            content += ['<tr style="border: 1px solid;">']
            for i, (param_name, param_id) in enumerate(params):
                if i == len(params) - 1:
                    content += [f'<td style="{style_border_right_solid_bold}">{self.model_configs[model_id][param_id]}</td>']
                else:
                    content += [f'<td style="{style_border_right_solid}">{self.model_configs[model_id][param_id]}</td>']
            for metric, metric_name in metrics:
                if 'eos_rank' not in metric and 'eos_prob' not in metric:
                    content += [
                        f'<td style="{style_border_right_solid}{style_align_center}">{round(self.models[model_id][-1][metric], 3)}</td>'
                    ]
                else:
                    if 'rank' in metric:
                        mode = metric.split('_')[0]
                        (_1, aggregated_bin_values_before_last), \
                        (_2, aggregated_bin_values_last), _ = self.get_aggregated_bin_ranks(self.models[model_id],
                                                                                            seq_agg, dataset_agg, mode,
                                                                                            min_bin_samples,
                                                                                            min_bin_size, max_length=max_length)
                    elif 'prob' in metric:
                        mode = metric.split('_')[0]
                        (_1, aggregated_bin_values_before_last), \
                        (_2, aggregated_bin_values_last), _ = self.get_aggregated_bin_eosprobs(self.models[model_id],
                                                                                            seq_agg, dataset_agg, mode,
                                                                                            min_bin_samples,
                                                                                            min_bin_size, max_length=max_length)
                    else:
                        raise NotImplementedError

                    if 'false' in metric:
                        for i, item in enumerate(aggregated_bin_values_before_last):
                            if i == len(aggregated_bin_values_before_last) - 1:
                                content += [
                                    f'<td style="{style_border_right_solid}{style_align_center}">{round(item, 3)}</td>']
                            else:
                                content += [
                                    f'<td style="{style_border_right_dotted}{style_align_center}">{round(item, 3)}</td>']
                    else:
                        for i, item in enumerate(aggregated_bin_values_last):
                            if i == len(aggregated_bin_values_last) - 1:
                                content += [f'<td style="{style_border_right_solid}">{round(item, 3)}</td>']
                            else:
                                content += [f'<td style="{style_border_right_dotted}">{round(item, 3)}</td>']
            content += ['</tr>']

        table = f'<table style="{style_border_right_solid} font-size: 10pt">{header}{"".join(content)}</table>'
        return table
    
    def get_ranks_distribution(self, params, min_bin_size=10, draw_plot=False):
        vocab_size = self.models[0][-1]['vocab_size']
       
        mapping = {}
        start = 0
        for i in range(vocab_size + 1):
            mapping[i] = start + min_bin_size / 2 
            if i == start:
                start += min_bin_size

        selected_ids = self.filter_models({'self': self}, params)
        bins = collections.OrderedDict()
        for selected_id in selected_ids['self']:
            model = self.models[selected_id]
            config = self.model_configs[selected_id]
            for i, ranks in enumerate(model[-1]['target_false_eos_ranks']):
                for j, rank in enumerate(ranks):
                    bounds_mean = mapping[rank]
                    if bounds_mean not in bins:
                        bins[bounds_mean] = 0
                    bins[bounds_mean] += 1
        
        bins = collections.OrderedDict(sorted(bins.items()))

        if draw_plot:
            plt.figure(figsize=(15, 10))
            plt.bar(list(bins.keys()), list(bins.values()), width=4)
            plt.yscale('symlog')
            plt.grid()
            plt.show()
        return bins

    @staticmethod
    def filter_models(parsers, params):
        selected_ids = {}
        for experiment, parser in parsers.items():
            selected_ids[experiment] = []
            for model_id, config in enumerate(parser.model_configs):
                eq = 0
                for param in params:
                    if config[param] in params[param]:
                        eq += 1
                if eq == len(params):
                    selected_ids[experiment] += [model_id]
        return selected_ids

    def get_soft_oversmoothing(self, minl=0, maxl=200, seq_type='target', beam=1000):
        model_ids = np.arange(len(self.models))
        suffix_soft_dicts = {}
        for model_id in model_ids:
            model_beam = self.model_configs[model_id]['beam']
            if model_beam != beam:
                continue
            seed = self.model_configs[model_id]['seed']
            model_neos = self.model_configs[model_id]['number_eos_tokens']
            if model_neos not in suffix_soft_dicts:
                suffix_soft_dicts[model_neos] = {}
            suffix_soft_dicts[model_neos][seed] = self.get_suffix_dict(self.models[model_id], minl=minl, maxl=maxl, seq_type=seq_type)
        return suffix_soft_dicts

    def get_hard_oversmoothing(self, minl=0, maxl=200, seq_type='target', beam=1000):
        model_ids = np.arange(len(self.models))
        suffix_hard_dicts = {}
        for model_id in model_ids:
            model_beam = self.model_configs[model_id]['beam']
            if model_beam != beam:
                continue
            seed = self.model_configs[model_id]['seed']
            model_neos = self.model_configs[model_id]['number_eos_tokens']
            if model_neos not in suffix_hard_dicts:
                suffix_hard_dicts[model_neos] = {}
            suffix_hard_dicts[model_neos][seed] = self.get_suffix_hard_dict(self.models[model_id], minl=minl, maxl=maxl, seq_type=seq_type)
        return suffix_hard_dicts

    def get_suffix_dict(self, pickle_dict, minl=0, maxl=1000, seq_type='target'):
        suffix_length_to_list = defaultdict(list)
        for seq_lprobs, seq_eos_probs in zip(pickle_dict[0][f'{seq_type}_model_lprobs'], pickle_dict[0][f'{seq_type}_false_eos_probs']):
            if len(seq_lprobs) > maxl or len(seq_lprobs) < minl:
                continue
            for prefix_len in range(len(seq_lprobs)-1):
                suffix_lprob = sum(seq_lprobs[prefix_len:])
                full_seq_lprob = sum(seq_lprobs)
                prefix_lprob = sum(seq_lprobs[:prefix_len])+math.log(seq_eos_probs[prefix_len])
                suffix_length = len(seq_lprobs[prefix_len:])
                suffix_length_to_list[suffix_length].append(seq_eos_probs[prefix_len] - math.exp(suffix_lprob))
        return suffix_length_to_list

    def get_suffix_hard_dict(self, pickle_dict, minl=0, maxl=1000, seq_type='target'):
        suffix_length_to_list = defaultdict(list)
        for seq_lprobs, seq_eos_probs in zip(pickle_dict[0][f'{seq_type}_model_lprobs'], pickle_dict[0][f'{seq_type}_false_eos_probs']):
            if len(seq_lprobs) > maxl or len(seq_lprobs) < minl:
                continue
            for prefix_len in range(len(seq_lprobs)-1):
                suffix_lprob = sum(seq_lprobs[prefix_len:])
                full_seq_lprob = sum(seq_lprobs)
                prefix_lprob = sum(seq_lprobs[:prefix_len])+math.log(seq_eos_probs[prefix_len])
                suffix_length = len(seq_lprobs[prefix_len:])
                suffix_length_to_list[suffix_length].append(float(seq_eos_probs[prefix_len] > math.exp(suffix_lprob)))
        return suffix_length_to_list


