from compare_mt import compare_mt_main, reporters
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from glob import glob
import os
from tqdm.auto import tqdm
import json


class TranslationParser:
    """
    params: dict {'param_name_with_underscore': 'shortname'}
    """
    def __init__(self, experiments_directory, params, pkl_prefix='generat', use_tqdm=False):
        self.models = {}
        self.experiments_directory = experiments_directory
        self.pkl_prefix = pkl_prefix
        self.tqdm = use_tqdm
        self.params_names = params
        self.model_ids = []
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
            args_path = glob(os.path.join('/', *pkl_file.split('/')[:-1], '*_args.pkl'))
            
            args = {}
            generated_results = pickle.load(open(pkl_file, 'rb'))
            if len(args_path) > 0:
                args = pickle.load(open(args_path[0], 'rb'))
        
            model_cfg = vars(generated_results['saved_cfg']['model'])
            cfg_criterion = generated_results['saved_cfg']['criterion']
            seed = generated_results['saved_cfg'].common.seed
            current_config = {
                'seed': seed
            }
            
            name = []
            for param_id in params:
                param_name = params[param_id]
                name += ['']
                if param_id in cfg_criterion:
                    name[-1] = f'{param_name}:{cfg_criterion[param_id]}'
                elif param_id in model_cfg:
                    name[-1] = f'{param_name}:{model_cfg[param_id]}'
                elif f'--{param_id.replace("-","_")}' in args:
                    args_param_id = f'--{param_id}'
                    name[-1] = f'{param_name}:{args[args_param_id]}'
                else:
                    name[-1] = f'{param_name}:None'
            name = ','.join(name)
            self.models[name] = {}
            self.models[name]['params'] = {
                param.split(':')[0]: param.split(':')[1]
                for param in name.split(',')
            }
            self.models[name]['generated'] = {}
            for sample_id in generated_results['stats']:
                self.models[name]['generated'][sample_id] = generated_results['stats'][sample_id]
            self.model_ids = list(self.models.keys())

    def __get_targets(self, model_id):
        targets = []
        for sample_id in self.models[model_id]['generated']:
            targets += [self.models[model_id]['generated'][sample_id]['target'].split()]
        return targets

    def __get_sources(self, model_id):
        sources = []
        for sample_id in self.models[model_id]['generated']:
            sources += [self.models[model_id]['generated'][sample_id]['source'].split()]
        return sources

    def __get_hypotheses(self, model_ids, hypotheses_rank):
        hypotheses = []
        for model_id in model_ids:
            hypotheses += [[]]
            for sample_id in self.models[model_id]['generated']:
                hypotheses[-1] += [
                    self.models[model_id]['generated'][sample_id]['hypotheses'][hypotheses_rank][0].split()]
        return hypotheses

    def compare_bleu(self, model_ids, hypotheses_rank=0, case_insensitive=False, prob_thresh=0.05,
                     significance_test=False, bootstrap=1000, score_type='bleu'):
        reporters.sys_names = model_ids
        if significance_test and len(model_ids) > 2:
            raise Exception('Cannot run significance test for more than two models')
        targets = self.__get_targets(model_ids[0])
        sources = self.__get_sources(model_ids[0])
        hypotheses = self.__get_hypotheses(model_ids, hypotheses_rank)
        reporter = compare_mt_main.generate_score_report(targets, hypotheses, src=sources, bootstrap=bootstrap,
                                                         prob_thresh=prob_thresh, case_insensitive=case_insensitive)
        return reporter.html_content()

    def generate_sentences(self, model_ids, hypotheses_rank=0, score_type='bleu',
                           report_length=20, case_insensitive=False):
        reporters.sys_names = model_ids
        targets = self.__get_targets(model_ids[0])
        sources = self.__get_sources(model_ids[0])
        hypotheses = self.__get_hypotheses(model_ids, hypotheses_rank)
        reporter = compare_mt_main.generate_sentence_examples(targets, hypotheses, src=sources,
                                                              report_length=report_length,
                                                              case_insensitive=case_insensitive)
        return reporter.html_content()

    def lengths_diff_stats(self, model_ids, bucket_type='lengthdiff', hypotheses_rank=0):
        reporters.sys_names = model_ids
        targets = self.__get_targets(model_ids[0])
        sources = self.__get_sources(model_ids[0])
        hypotheses = self.__get_hypotheses(model_ids, hypotheses_rank)
        reporter = compare_mt_main.generate_sentence_bucketed_report(targets, hypotheses, src=sources,
                                                                     bucket_type=bucket_type)
        return reporter.html_content()

    def tokens_accuracy_stats(self, model_ids, bucket_type='freq', hypotheses_rank=0):
        reporters.sys_names = model_ids
        targets = self.__get_targets(model_ids[0])
        sources = self.__get_sources(model_ids[0])
        hypotheses = self.__get_hypotheses(model_ids, hypotheses_rank)
        reporter = compare_mt_main.generate_word_accuracy_report(targets, hypotheses, src=sources,
                                                                 bucket_type=bucket_type)
        return reporter.html_content()

    def ngram_stats(self, model_ids, min_ngram_len=1, max_ngram_len=4,
                    compare_type='match', report_len=50, hypotheses_rank=0):
        reporters.sys_names = model_ids
        targets = self.__get_targets(model_ids[0])
        hypotheses = self.__get_hypotheses(model_ids, hypotheses_rank)
        reporter = compare_mt_main.generate_ngram_report(targets, hypotheses,
                                                         min_ngram_length=min_ngram_len, max_ngram_length=max_ngram_len,
                                                         report_length=report_len, compare_type=compare_type)
        return reporter.html_content()

    def __aggregate_by_param(self, model_ids, param):
        aggregated = {}
        for model_id in model_ids:
            param_value = self.models[model_id]['params'][param]
            if param_value not in aggregated:
                aggregated[param_value] = []
            aggregated[param_value] += [self.models[model_id]]
        return aggregated

    def get_ends_stats(self, model_ids=None, hypotheses_rank=0):
        if model_ids is None:
            model_ids = list(self.models.keys())
        endings = {}
        for model_id in model_ids:
            endings[model_id] = {'E': 0, 'not E': 0}
        for model_id in model_ids:
            model = self.models[model_id]
            for generated in model['generated'].values():
                ending = generated['hypotheses'][hypotheses_rank][0].split(' ')[-1]
                if ending in ['.', '?', '!', '...', '&quot;'] or ending[-1] == '.':
                    endings[model_id]['E'] += 1
                else:
                    endings[model_id]['not E'] += 1
        return pd.DataFrame(endings).T

    @staticmethod
    def __calculate_length(model, hypotheses_rank=0):
        lengths = {}
        for sample in model['generated'].values():
            length = len(sample['hypotheses'][hypotheses_rank][0].split(' '))
            if length not in lengths:
                lengths[length] = 0
            lengths[length] += 1
        return lengths

    def __get_target_lengths(self, model_id):
        lengths = {}
        for sample in self.models[model_id]['generated'].values():
            length = len(sample['target'].split(' '))
            if length not in lengths:
                lengths[length] = 0
            lengths[length] += 1
        return lengths

    def get_lengths_plot(self, model_ids=None, xlim=60, hypotheses_rank=0):
        if model_ids is None:
            model_ids = list(self.models.keys())
        target_lengths = self.__get_target_lengths(model_ids[0])
        height = len(model_ids) // 2 + len(model_ids) % 2
        width = 2
        
        fig = plt.figure(figsize=(12, height * 7))

        for i, model_id in enumerate(model_ids):
            model = self.models[model_id]
            fig.add_subplot(height, width, i + 1)
            lengths = self.__calculate_length(model, hypotheses_rank)
            plt.bar(lengths.keys(), lengths.values(), alpha=0.5, ls='dotted', lw=3, label='generated')
            plt.bar(target_lengths.keys(), target_lengths.values(), alpha=0.8,
                    ls='dotted', lw=3, label='target')
            plt.xlim(0, xlim)
            plt.title(model_id)
            plt.legend()
            plt.xlabel('Sequence length')
            plt.grid()
        plt.show()
