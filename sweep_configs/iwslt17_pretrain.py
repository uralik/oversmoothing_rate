import itertools
import fire
import collections
import getpass
import datetime
import os
import pickle
from glob import glob

from sweep_utils import add_train_iwslt17_de_fr_zh_oversmoothing, all_vs_all_grid, compose_cmd_args, add_common_validation

def pretrain_iwslt17_pure_baseline_nllstop(sweep_step, language):
    experiment_name = f'pretrain_iwslt17_pure_baseline_nllstop_{language}'

    kv_opts = collections.OrderedDict()
    kv_opts = add_train_iwslt17_de_fr_zh_oversmoothing(kv_opts, language)

    kv_opts['--criterion'] = 'label_smoothed_cross_entropy'
    kv_opts['--arch'] = 'transformer_iwslt_de_en'
    kv_opts['--task'] = 'translation'

    kv_opts['--validate-interval-updates'] = '2000'
    kv_opts['--best-checkpoint-metric'] = 'nll_loss'
    kv_opts['--patience'] = 5
    del kv_opts['--validate-interval']

    kv_opts['--max-tokens'] = '4096'
    kv_opts['--lr'] = '5e-4'  # we train for real now

    kv_opts['--label-smoothing'] = 0.1

    del kv_opts['--eos-choice']
    del kv_opts['--marginal-entropy-weight']
    del kv_opts['--conditional-entropy-weight']
    del kv_opts['--user-dir']
    del kv_opts['--eval-bleu']
    del kv_opts['--maximize-best-checkpoint-metric']
    del kv_opts['--eval-bleu-args']

    # grid is defined here
    grid = collections.OrderedDict()
    grid['--seed'] = ['2421', '2804', '9361', '4872', '6765']

    sweep_step_dict = all_vs_all_grid(grid)[sweep_step-1]

    for k,v in sweep_step_dict.items():
        kv_opts[k] = v

    save_dir = os.environ.get('EXPERIMENTS_DIRECTORY_IWSLT')
    save_dir = os.path.join(save_dir, experiment_name, f'sweep_step_{sweep_step}')
    save_dir_tb = os.path.join(save_dir, 'tb')

    kv_opts['--save-dir'] = save_dir
    kv_opts['--tensorboard-logdir'] = save_dir_tb

    # saving the args dict in save-dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cmd_args_filename = os.path.join(save_dir, experiment_name+f'_{sweep_step}'+'_args.pkl')
    pickle.dump(kv_opts, open(cmd_args_filename, 'wb'))

    return kv_opts

def validate_trained_sweep_ontest(sweep_step, experiment_name_to_validate, beam, language):
    experiment_name = f'validate_beam{beam}_testset'

    pretrain_args_pkl_filename = os.path.join(os.environ.get('EXPERIMENTS_DIRECTORY_IWSLT'), f'{experiment_name_to_validate}_{language}', f'sweep_step_{sweep_step}', f'{experiment_name_to_validate}_{language}_{sweep_step}'+'_args.pkl')
    args_from_trained_model = pickle.load(open(pretrain_args_pkl_filename, 'rb'))

    kv_opts = collections.OrderedDict()
    kv_opts = add_common_validation(kv_opts, args_from_trained_model)

    kv_opts['data'] = os.path.join(os.environ.get('DATA_IWSLT'), f'iwslt17.tokenized.{language}-en')

    kv_opts['--max-tokens'] = '256'
    kv_opts['--valid-subset'] = 'test'

    kv_opts['--eval-bleu'] = True
    kv_opts['--eval-bleu-args'] = '\'{"beam": %d, "max_len_a": 1.2, "max_len_b": 10, "min_length": 0, "unnormalized": true}\'' % beam
    kv_opts['--eval-bleu-detok'] = 'moses'
    kv_opts['--eval-bleu-remove-bpe'] = 'sentencepiece'
    kv_opts['--scoring'] = 'sacrebleu'
    kv_opts['--stat-save-path'] = os.path.join(args_from_trained_model['--save-dir'], experiment_name, 'best_extra_state.pkl')

    if not os.path.exists(kv_opts['--stat-save-path']):
        os.makedirs(os.path.dirname(kv_opts['--stat-save-path']), exist_ok=True)

    cmd_args_filename = os.path.join(args_from_trained_model['--save-dir'], experiment_name, f'validate_{sweep_step}'+'_args.pkl')
    pickle.dump(kv_opts, open(cmd_args_filename, 'wb'))

    return kv_opts

def main(call_fn, sweep_step, **kwargs):
    call_fn = globals()[call_fn]
    dict_args = call_fn(sweep_step, **kwargs)
    return compose_cmd_args(dict_args)


if __name__ == "__main__":
    fire.Fire(main)
