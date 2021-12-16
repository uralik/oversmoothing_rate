from parse_sweep_results_oversmoothing import StatsParserOversmoothing
from translations_parser import TranslationParser
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import itertools
import os
import pandas as pd
import argparse
import torch
from glob import glob

params = {
    ('beam', 'beam'),
    ('lr', 'lr'),
    ('unnormalized', 'unnormalized'),
    ('seed', 'seed'),
    ('OSL weight', 'oversmoothing_weight'),
    ('OSL Margin', 'oversmoothing_margin'),
}

table_eos_quality_target = [
    ('bleu', 'BLEU'),
    ('bleurt', 'BLEURT'),
    ('tb_ppl', 'PPL'),
    ('tb_nll_loss', 'NLL'),
    ('abrupted', 'Abrupted'),
    ('tb_target/oversmoothing_rate', 'OS rate'),
    ('tb_target/oversmoothing_loss', 'OS loss'),
    ('target_empty_winrate', 'Target Empty WR'),
    ('generated_empty_winrate', 'Generated Empty WR'),
    ('target_generated_lenratio_mean', '|T|/|G| mean'),
    ('target_generated_lenratio_seq_mean', 'LR seq'),
    ('target_generated_lenratio_median', '|T|/|G| med'),
    ('mean_target_false_eos_r', 'T False EOS R avg'),
    ('std_target_false_eos_r', 'T False EOS R std'),
    ('mean_target_true_eos_r', 'T True EOS R avg'),
    ('std_target_true_eos_r', 'T True EOS R std'),
    ('mean_generated_false_eos_r', 'G False EOS R avg'),
    ('std_generated_false_eos_r', 'G False EOS R std'),
    ('mean_generated_true_eos_r', 'G True EOS R avg'),
    ('std_generated_true_eos_r', 'G True EOS R std'),
    ('target_terminal_ll', 'T Term LL'),
    ('target_nonterminal_ll', 'T NTerm LL'),
    ('generated_terminal_ll', 'G Term LL'),
    ('generated_nonterminal_ll', 'G Nterm LL'),
    ('target_terminal_ll_mean', 'T Term LL Avg'),
    ('target_nonterminal_ll_mean', 'T NTerm LL Avg'),
    ('generated_terminal_ll_mean', 'G Term LL Avg'),
    ('generated_nonterminal_ll_mean', 'G Nterm LL Avg'),
]

parser_params = {
    'label_smoothing',
    'beam',
    'lr',
    'unnormalized',
    'oversmoothing_weight', 
    'oversmoothing_margin',
}

class ParserWrapper:
    def __init__(self, path, beam, output):
        self.path = path
        self.beam = beam
        self.output = output
        self.exp_name = path.split('/')[-1]

        if not os.path.exists(f'{output}/{self.exp_name}'):
            os.makedirs(f'{output}/{self.exp_name}', exist_ok=True)

        pkls = {
            'exp': path
        }

        self.parsers = {}
        for experiment in pkls:
            self.parsers[experiment] = StatsParserOversmoothing(pkls[experiment], parser_params, lengths=None, tqdm=True, pkl_prefix=f'validate_beam{beam}_testset/best')

    def filter_models(self, params):
        selected_ids = {}
        for experiment, parser in self.parsers.items():
            selected_ids[experiment] = []
            for model_id, config in enumerate(parser.model_configs):
                eq = 0
                for param in params:
                    if config[param] in params[param]:
                        eq += 1
                if eq == len(params):
                    selected_ids[experiment] += [model_id]
        return selected_ids

    def generate_csv_results(self, target=False):
        selected_ids = self.filter_models({'oversmoothing_margin': [0.0001]})
        html = self.parsers['exp'].get_criteria_table(table_eos_quality_target, params, selected_ids['exp'])
    
        results = pd.read_html(html, header=0)
        results = results[0].drop(0)
        results = pd.DataFrame(results)
        results['beam'] = results['beam'].astype(int)
        results['seed'] = results['seed'].astype(int)
        if target:
            results.to_csv(f'{self.output}/{self.exp_name}/metrics_{self.exp_name}.csv')
        else:
            results.to_csv(f'{self.output}/{self.exp_name}/metrics_{self.exp_name}_beam{self.beam}.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--beam', nargs='?', type=int, help='beam size')
    parser.add_argument('-p', '--path', nargs='?', help='path to experiment sweep')
    parser.add_argument('-o', '--output', nargs='?', help='output path')
    args = parser.parse_args()

    pw = ParserWrapper(path=args.path, beam=args.beam, output=args.output)
    pw.generate_csv_results(target=args.target)

