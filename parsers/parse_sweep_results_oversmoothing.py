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

        self.aggregations = {
            'max': np.max,
            'min': np.min,
            'avg': np.mean
        }

        self.experiments_directory = experiments_directory
        self.params = params
        self.lengths = lengths
        self.tqdm = tqdm
        self.reduce_ram = reduce_ram
        self.load_results()
        self.preprocess_metrics()
        self.compute_mean_eos_rank_globally()
        
    def load_results(self):
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
        return tl.sum() / gl.sum()

    def __recalculate_len_ratio_seq(self, update, model_id): # update stands for self.models[model_id][-1]
        tl = np.array(update['target_seq_lengths'])
        gl = np.array(update['generated_seq_lengths'])
        lr_seq = []
        for tl_seq, gl_seq in zip(tl, gl):
            lr_seq += [tl_seq / gl_seq]
        return np.mean(lr_seq), np.median(lr_seq)
           
    def compute_bleu(self, update):
        beam_all_hyps = update['generated_hyps_text']
        beam_top_hyps = [l[0] for l in beam_all_hyps]
        refs = update['generated_refs_text']
        return self.bleu.corpus_score(beam_top_hyps, [refs]).score

    def preprocess_metrics(self):
        for i, model in enumerate(self.models):
            for update in model:
                update['bleu'] = self.compute_bleu(update)
                update['target_generated_lenratio_mean'] = self.__recalculate_len_ratio(update, i)
                lr_seq_mean, lr_seq_median = self.__recalculate_len_ratio_seq(update, i)
                update['target_generated_lenratio_seq_mean'] = lr_seq_mean
                update['target_generated_lenratio_median'] = lr_seq_median
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

                true_eos_ranks_stats = scipy.stats.describe(model[-1][f'{seq_type}_true_eos_ranks'])
                true_eos_ranks_median = np.median(model[-1][f'{seq_type}_true_eos_ranks'])
                model[-1][f'skew_{seq_type}_true_eos_r'] = true_eos_ranks_stats.skewness
                model[-1][f'median_{seq_type}_true_eos_r'] = true_eos_ranks_median
                model[-1][f'mean_{seq_type}_true_eos_r'] = true_eos_ranks_stats.mean
                model[-1][f'std_{seq_type}_true_eos_r'] = np.sqrt(true_eos_ranks_stats.variance)


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

    def get_criteria_table(self, metrics, params, model_ids, seq_agg='avg', dataset_agg='avg', min_bin_samples=400, min_bin_size=10, max_length=250):
        style_border_right_solid = 'border-right: 1px solid;'
        style_border_right_solid_bold = 'border-right: 3px solid;'
        style_border_right_dotted = 'border-right: 1px dotted;'
        style_align_center = 'text-align: center;"'

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

