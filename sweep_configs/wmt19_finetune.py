import itertools
import fire
import collections
import getpass
import datetime
import os
import pickle
from glob import glob

from sweep_utils import add_train_wmt19_ende_oversmoothing_finetunebig, add_train_wmt19_deen_oversmoothing_finetunebig, add_train_wmt19_oversmoothing_finetunebig, validate_trained_sweep, generate_trained_sweep, get_static_paths, add_max_logit_policy_wentreg, all_vs_all_grid, compose_cmd_args, add_common_validation

def finetune_wmt19_ruen_osl(sweep_step):
    experiment_name = finetune_wmt19_ruen_osl.__name__

    kv_opts = collections.OrderedDict()
    kv_opts['--user-dir'] = get_static_paths('--user-dir', getpass.getuser())
    kv_opts = add_train_wmt19_oversmoothing_finetunebig(kv_opts)

    save_dir = get_static_paths('savedir_absolute path', 'ik1147')
    save_dir = os.path.join(save_dir, experiment_name, f'sweep_step_{sweep_step}')
    save_dir_tb = os.path.join(save_dir, 'tb')

    kv_opts['--save-dir'] = save_dir
    kv_opts['--tensorboard-logdir'] = save_dir_tb

    kv_opts['--validate-interval-updates'] = '2000'

    kv_opts['--max-update'] = '500000'
    kv_opts['--patience'] = '5'

    kv_opts['--best-checkpoint-metric'] = 'loss'

    grid = collections.OrderedDict()
    grid['--seed'] = ['2421', '2804', '9361', '4872', '6765']
    grid['--oversmoothing-margin'] = ['0.0001']
    grid['--oversmoothing-weight'] = ['0.00', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95', '1.00']

    sweep_step_dict = all_vs_all_grid(grid)[sweep_step-1]

    for k,v in sweep_step_dict.items():
        kv_opts[k] = v

    # saving the args dict in save-dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cmd_args_filename = os.path.join(save_dir, experiment_name+f'_{sweep_step}'+'_args.pkl')
    pickle.dump(kv_opts, open(cmd_args_filename, 'wb'))

    return kv_opts

def finetune_wmt19_deen_osl(sweep_step):
    experiment_name = finetune_wmt19_deen_osl.__name__

    kv_opts = collections.OrderedDict()
    kv_opts['--user-dir'] = get_static_paths('--user-dir', getpass.getuser())
    kv_opts = add_train_wmt19_deen_oversmoothing_finetunebig(kv_opts)

    save_dir = get_static_paths('savedir_absolute path', 'ik1147')
    save_dir = os.path.join(save_dir, experiment_name, f'sweep_step_{sweep_step}')
    save_dir_tb = os.path.join(save_dir, 'tb')

    kv_opts['--save-dir'] = save_dir
    kv_opts['--tensorboard-logdir'] = save_dir_tb

    kv_opts['--validate-interval-updates'] = '2000'

    kv_opts['--max-update'] = '500000'
    kv_opts['--patience'] = '5'

    kv_opts['--best-checkpoint-metric'] = 'loss'

    grid = collections.OrderedDict()
    grid['--seed'] = ['2421', '2804', '9361', '4872', '6765']
    grid['--oversmoothing-margin'] = ['0.0001']
    grid['--oversmoothing-weight'] = ['0.00', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95', '1.00']

    sweep_step_dict = all_vs_all_grid(grid)[sweep_step-1]

    for k,v in sweep_step_dict.items():
        kv_opts[k] = v

    # saving the args dict in save-dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cmd_args_filename = os.path.join(save_dir, experiment_name+f'_{sweep_step}'+'_args.pkl')
    pickle.dump(kv_opts, open(cmd_args_filename, 'wb'))

    return kv_opts

def finetune_wmt19_ende_osl(sweep_step):
    experiment_name = finetune_wmt19_ende_osl.__name__

    kv_opts = collections.OrderedDict()
    kv_opts['--user-dir'] = get_static_paths('--user-dir', getpass.getuser())
    kv_opts = add_train_wmt19_ende_oversmoothing_finetunebig(kv_opts)

    save_dir = get_static_paths('savedir_absolute path', getpass.getuser())
    save_dir = os.path.join(save_dir, experiment_name, f'sweep_step_{sweep_step}')
    save_dir_tb = os.path.join(save_dir, 'tb')

    kv_opts['--save-dir'] = save_dir
    kv_opts['--tensorboard-logdir'] = save_dir_tb

    kv_opts['--validate-interval-updates'] = '2000'

    kv_opts['--max-update'] = '500000'
    kv_opts['--patience'] = '5'

    kv_opts['--best-checkpoint-metric'] = 'loss'

    grid = collections.OrderedDict()
    grid['--seed'] = ['2421']
    grid['--oversmoothing-margin'] = ['0.0001']
    grid['--oversmoothing-weight'] = ['0.00', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95', '1.00']

    sweep_step_dict = all_vs_all_grid(grid)[sweep_step-1]

    for k,v in sweep_step_dict.items():
        kv_opts[k] = v

    # saving the args dict in save-dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cmd_args_filename = os.path.join(save_dir, experiment_name+f'_{sweep_step}'+'_args.pkl')
    pickle.dump(kv_opts, open(cmd_args_filename, 'wb'))

    return kv_opts

def validate_trained_sweep_ontest(sweep_step, experiment_name_to_validate, beam):
    experiment_name = f'validate_beam{beam}_testset'

    pretrain_args_pkl_filename = os.path.join(get_static_paths('savedir_absolute path', 'ik1147'), experiment_name_to_validate, f'sweep_step_{sweep_step}', experiment_name_to_validate+f'_{sweep_step}'+'_args.pkl')
    args_from_trained_model = pickle.load(open(pretrain_args_pkl_filename, 'rb'))

    kv_opts = collections.OrderedDict()
    kv_opts = add_common_validation(kv_opts, args_from_trained_model)

    kv_opts['--batch-size'] = 1
    kv_opts['--valid-subset'] = 'test'

    kv_opts['--eval-bleu'] = True
    kv_opts['--scoring'] = 'sacrebleu'
    kv_opts['--eval-bleu-args'] = '\'{"beam": %d, "max_len_a": 1.2, "max_len_b": 10, "min_length": 0, "unnormalized": true}\'' % beam
    kv_opts['--eval-bleu-detok'] = 'moses'
    kv_opts['--eval-bleu-remove-bpe'] = True
    kv_opts['--scoring'] = 'sacrebleu'

    kv_opts['--stat-save-path'] = os.path.join(get_static_paths('savedir_absolute path', 'ik1147'), experiment_name_to_validate, f'sweep_step_{sweep_step}', experiment_name, 'best_extra_state.pkl')

    kv_opts['--path'] = os.path.join(get_static_paths('savedir_absolute path', 'ik1147'), experiment_name_to_validate, f'sweep_step_{sweep_step}', 'checkpoint_best.pt')

    del kv_opts['--max-tokens']

    if not os.path.exists(kv_opts['--stat-save-path']):
        os.makedirs(os.path.dirname(kv_opts['--stat-save-path']), exist_ok=True)

    cmd_args_filename = os.path.join(get_static_paths('savedir_absolute path', 'ik1147'), experiment_name_to_validate, f'sweep_step_{sweep_step}', experiment_name, f'validate_{sweep_step}'+'_args.pkl')
    pickle.dump(kv_opts, open(cmd_args_filename, 'wb'))

    return kv_opts

def main(call_fn, sweep_step, **kwargs):
    call_fn = globals()[call_fn]
    dict_args = call_fn(sweep_step, **kwargs)
    return compose_cmd_args(dict_args)


if __name__ == "__main__":
    fire.Fire(main)
