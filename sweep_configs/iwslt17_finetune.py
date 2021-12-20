import itertools
import fire
import collections
import getpass
import datetime
import os
import pickle
from glob import glob

from sweep_utils import validate_trained_sweep,  get_static_paths, add_train_iwslt17_de_fr_zh_oversmoothing, all_vs_all_grid, compose_cmd_args, add_common_validation

def finetune_iwslt17_oversmoothing_grid_label_smoothing(sweep_step, language):
    experiment_name = f'finetune_iwslt17_oversmoothing_grid_label_smoothing_{language}'

    kv_opts = collections.OrderedDict()
    kv_opts['--user-dir'] = get_static_paths('--user-dir', getpass.getuser())
    kv_opts = add_train_iwslt17_de_fr_zh_oversmoothing(kv_opts, language)
    
    seed_mapping = {
        '2421': '1',
        '2804': '2',
        '9361': '3',
        '4872': '4',
        '6765': '5'
    }

    kv_opts['data'] = f'/scratch/mae9785/iwslt17-data-bin/iwslt17.tokenized.{language}-en'

    # altering pretrain args to adjust for new experiment
    save_dir = get_static_paths('savedir_absolute path', getpass.getuser())
    save_dir = os.path.join(save_dir, experiment_name, f'sweep_step_{sweep_step}')
    save_dir_tb = os.path.join(save_dir, 'tb')
    kv_opts['--save-dir'] = save_dir
    kv_opts['--tensorboard-logdir'] = save_dir_tb

    kv_opts['--warmup-updates'] = '1'
    kv_opts['--warmup-init-lr'] = '5e-4'
    kv_opts['--max-tokens'] = '4096'
    kv_opts['--lr'] = '5e-4'  # we train for real now
    kv_opts['--label-smoothing'] = 0.0

    grid = collections.OrderedDict()
    grid['--seed'] = ['2421', '2804', '9361', '4872', '6765']
    grid['--oversmoothing-margin'] = ['0.0001']
    grid['--oversmoothing-weight'] = [str(0.05 * weight) for weight in range(21)]

    sweep_step_dict = all_vs_all_grid(grid)[sweep_step-1]

    for k,v in sweep_step_dict.items():
        kv_opts[k] = v

    kv_opts['--finetune-from-model'] = f'/scratch/mae9785/nmt/pretrain_iwslt17_{language}_pure_baseline/sweep_step_{seed_mapping[kv_opts["--seed"]]}/checkpoint_best.pt'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cmd_args_filename = os.path.join(save_dir, experiment_name+f'_{sweep_step}'+'_args.pkl')
    pickle.dump(kv_opts, open(cmd_args_filename, 'wb'))

    return kv_opts

def validate_trained_sweep_ontest(sweep_step, experiment_name_to_validate, beam, language):
    experiment_name = f'validate_beam{beam}_testset'

    pretrain_args_pkl_filename = os.path.join(get_static_paths('savedir_absolute path', getpass.getuser()), f'{experiment_name_to_validate}_{language}', f'sweep_step_{sweep_step}', f'{experiment_name_to_validate}_{language}_{sweep_step}'+'_args.pkl')
    args_from_trained_model = pickle.load(open(pretrain_args_pkl_filename, 'rb'))

    kv_opts = collections.OrderedDict()
    kv_opts = add_common_validation(kv_opts, args_from_trained_model)

    kv_opts['data'] = f'/scratch/mae9785/iwslt17-data-bin/iwslt17.tokenized.{language}-en'

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
