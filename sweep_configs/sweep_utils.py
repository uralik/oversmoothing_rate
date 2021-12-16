#!/usr/bin/env python

import pickle
import itertools
import fire
import collections
import getpass
import datetime
import os

def validate_trained_sweep(sweep_step, experiment_name_to_validate, beam, max_tokens=None):
    experiment_name = f'validate_beam{beam}'

    pretrain_args_pkl_filename = os.path.join(get_static_paths('savedir_absolute path', getpass.getuser()), experiment_name_to_validate, f'sweep_step_{sweep_step}', experiment_name_to_validate+f'_{sweep_step}'+'_args.pkl')
    args_from_trained_model = pickle.load(open(pretrain_args_pkl_filename, 'rb'))

    kv_opts = collections.OrderedDict()
    kv_opts = add_common_validation(kv_opts, args_from_trained_model)

    kv_opts['--eval-bleu-args'] = '\'{"beam": %d, "max_len_a": 1.2, "max_len_b": 10, "min_length": 0, "unnormalized": true}\'' % beam
    kv_opts['--stat-save-path'] = os.path.join(args_from_trained_model['--save-dir'], experiment_name, 'best_extra_state.pkl')

    if beam > 100:
        kv_opts['--max-tokens'] = '512'

    if max_tokens is not None:
        kv_opts['--max-tokens'] = max_tokens

    if not os.path.exists(kv_opts['--stat-save-path']):
        os.makedirs(os.path.dirname(kv_opts['--stat-save-path']), exist_ok=True)

    cmd_args_filename = os.path.join(args_from_trained_model['--save-dir'], experiment_name, f'validate_{sweep_step}'+'_args.pkl')
    pickle.dump(kv_opts, open(cmd_args_filename, 'wb'))

    return kv_opts


def generate_trained_sweep(sweep_step, experiment_name_to_generate, beam, nbest, max_tokens=None):
    experiment_name = f'generate_beam{beam}_nbest{nbest}'
    pretrain_args_pkl_filename = os.path.join(get_static_paths('savedir_absolute path', getpass.getuser()), experiment_name_to_generate, f'sweep_step_{sweep_step}', experiment_name_to_generate+f'_{sweep_step}'+'_args.pkl')
    args_from_trained_model = pickle.load(open(pretrain_args_pkl_filename, 'rb'))

    kv_opts = collections.OrderedDict()
    kv_opts = add_common_generation(kv_opts, args_from_trained_model)

    kv_opts['--beam'] = beam
    kv_opts['--nbest'] = nbest
    kv_opts['--results-path'] = os.path.join(args_from_trained_model['--save-dir'], experiment_name)

    if beam > 100:
        kv_opts['--max-tokens'] = '512'

    if max_tokens is not None:
        kv_opts['--max-tokens'] = max_tokens

    # saving the args dict in save-dir
    if not os.path.exists(kv_opts['--results-path']):
        os.makedirs(kv_opts['--results-path'], exist_ok=True)

    cmd_args_filename = os.path.join(kv_opts['--results-path'], f'generate_{sweep_step}'+'_args.pkl')
    pickle.dump(kv_opts, open(cmd_args_filename, 'wb'))

    return kv_opts

# amending opts for downstream task

def add_common_validation(kv_opts: collections.OrderedDict, args_from_trained_model: collections.OrderedDict) -> collections.OrderedDict:
    kv_opts['data'] = args_from_trained_model['data']
    kv_opts['--user-dir'] = get_static_paths('--user-dir', getpass.getuser())
    kv_opts['--task'] = args_from_trained_model['--task']
    kv_opts['--path'] = os.path.join(args_from_trained_model['--save-dir'], 'checkpoint_best.pt')
    if '--eval-bleu-args' in args_from_trained_model:
        kv_opts['train_sweep_eval_bleu_args'] = args_from_trained_model['--eval-bleu-args']  # keep the eval bleu args which were used during the actual training to know what beam size was used at training in case of eos penalty
    kv_opts['--max-tokens'] = '2048'
    kv_opts['--valid-subset'] = 'valid'

    return kv_opts

def add_common_generation(kv_opts: collections.OrderedDict, args_from_trained_model: collections.OrderedDict) -> collections.OrderedDict:
    kv_opts = add_common_validation(kv_opts, args_from_trained_model)
    kv_opts['--gen-subset'] = 'valid'
    kv_opts['--unnormalized'] = True
    kv_opts['--remove-bpe'] = True
    kv_opts['--post-process'] = True
    kv_opts['--max-len-a'] = 1.2
    kv_opts['--max-len-b'] = 10
    return kv_opts

# user specific opts

def get_static_paths(key: str, username: str) -> str:
    kv_opts = {
        '--user-dir': {
            'ik1147': '/scratch/ik1147/nmt_multiple_eos/nmt_eos/fairseq_module',
            'mae9785': '/home/mae9785/ml2/eos/nmt_eos/fairseq_module'
        },
        'data': {
            'ik1147': '/scratch/ik1147/public/fairseq_data_bin',
            'mae9785': '/scratch/mae9785/iwslt17-data-bin',
            'all': '/scratch/ik1147/public/wmt16'
        },
        'savedir_absolute path': {
            'ik1147': '/scratch/ik1147/nmt_multiple_eos/nmt_eos',
            'mae9785': '/scratch/mae9785/nmt'
        }
    }

    return kv_opts[key][username]

# task specific opts

def add_train_iwslt14_deen(kv_opts: collections.OrderedDict) -> collections.OrderedDict:
    kv_opts['data'] = os.path.join(get_static_paths('data', getpass.getuser()),'iwslt14.tokenized.de-en')
    kv_opts['--task'] = 'translation_eos'
    kv_opts['--user-dir'] = './fairseq_module/'
    kv_opts['--optimizer'] = 'adam'
    kv_opts['--adam-betas'] = '\'(0.9, 0.98)\''
    kv_opts['--lr-scheduler'] = 'inverse_sqrt'
    kv_opts['--warmup-updates'] = '4000'
    kv_opts['--weight-decay'] = '0.0001'
    kv_opts['--criterion'] = 'label_smoothed_cross_entropy_meos'
    kv_opts['--max-tokens'] = '4096'
    kv_opts['--arch'] = 'transformer_iwslt14_deen_meos'
    kv_opts['--clip-norm'] = '0.0'
    kv_opts['--lr'] = '5e-4'
    kv_opts['--dropout'] = '0.3'
    kv_opts['--eval-bleu'] = True
    kv_opts['--eval-bleu-args'] = '\'{"beam": 10, "max_len_a": 1.2, "max_len_b": 10, "min_length": 0, "unnormalized": true}\''
    kv_opts['--eval-bleu-detok'] = 'moses'
    kv_opts['--eval-bleu-remove-bpe'] = True
    kv_opts['--no-epoch-checkpoints'] = True
    kv_opts['--best-checkpoint-metric'] = 'bleu'
    kv_opts['--maximize-best-checkpoint-metric'] = True
    kv_opts['--patience'] = '10'
    kv_opts['--validate-interval'] = '5'  # more sparse 500 # fine grained 50

    # label smoothing is turned on here
    kv_opts['--label-smoothing'] = '0.1'

    return kv_opts

def add_train_iwslt17_de_fr_zh_en(kv_opts: collections.OrderedDict, language: str) -> collections.OrderedDict:
    kv_opts['data'] = os.path.join(get_static_paths('data', getpass.getuser()), f'iwslt17.tokenized.{language}-en')
    kv_opts['--task'] = 'translation_eos'
    kv_opts['--user-dir'] = '../fairseq_module/'
    kv_opts['--optimizer'] = 'adam'
    kv_opts['--adam-betas'] = '\'(0.9, 0.98)\''
    kv_opts['--lr-scheduler'] = 'inverse_sqrt'
    kv_opts['--warmup-updates'] = '4000'
    kv_opts['--weight-decay'] = '0.0001'
    kv_opts['--criterion'] = 'label_smoothed_cross_entropy_meos'
    kv_opts['--max-tokens'] = '4096'
    kv_opts['--arch'] = 'transformer_iwslt14_deen_meos'
    kv_opts['--clip-norm'] = '0.0'
    kv_opts['--lr'] = '5e-4'
    kv_opts['--dropout'] = '0.3'
    kv_opts['--eval-bleu'] = True
    kv_opts['--eval-bleu-args'] = '\'{"beam": 10, "max_len_a": 1.2, "max_len_b": 10, "min_length": 0, "unnormalized": true}\''
    kv_opts['--eval-bleu-detok'] = 'moses'
    kv_opts['--eval-bleu-remove-bpe'] = True
    kv_opts['--no-epoch-checkpoints'] = True
    kv_opts['--best-checkpoint-metric'] = 'bleu'
    kv_opts['--maximize-best-checkpoint-metric'] = True
    kv_opts['--patience'] = '10'
    kv_opts['--validate-interval'] = '5'  # more sparse 500 # fine grained 50

    # label smoothing is turned on here
    kv_opts['--label-smoothing'] = '0.1'

    return kv_opts


# def add_train_wmt16_deen(kv_opts: collections.OrderedDict) -> collections.OrderedDict:
#     kv_opts['data'] = os.path.join(get_static_paths('data', 'ik1147'),'wmt16_en_de_bpe32k')
#     kv_opts['--task'] = 'translation_eos'
#     kv_opts['--user-dir'] = './fairseq_module/'
#     kv_opts['--optimizer'] = 'adam'
#     kv_opts['--adam-betas'] = '\'(0.9, 0.98)\''
#     kv_opts['--lr-scheduler'] = 'inverse_sqrt'
#     kv_opts['--warmup-updates'] = '4000'
#     kv_opts['--weight-decay'] = '0.0'
#     kv_opts['--criterion'] = 'label_smoothed_cross_entropy'
#     kv_opts['--max-tokens'] = '4096'
#     kv_opts['--arch'] = 'transformer_vaswani_wmt_en_de_big_meos'
#     kv_opts['--clip-norm'] = '0.0'
#     kv_opts['--lr'] = '5e-4'
#     kv_opts['--dropout'] = '0.3'
#     kv_opts['--eval-bleu'] = True
#     kv_opts['--eval-bleu-args'] = '\'{"beam": 10, "max_len_a": 1.2, "max_len_b": 10, "min_length": 0}\''
#     kv_opts['--eval-bleu-detok'] = 'moses'
#     kv_opts['--eval-bleu-remove-bpe'] = True
#     kv_opts['--no-epoch-checkpoints'] = True
#     kv_opts['--best-checkpoint-metric'] = 'bleu'
#     kv_opts['--maximize-best-checkpoint-metric'] = True
#     kv_opts['--patience'] = '10'
#     kv_opts['--validate-interval'] = '5'  # more sparse 500 # fine grained 50
#     kv_opts['--share-all-embeddings'] = True

#     # label smoothing is turned on here
#     kv_opts['--label-smoothing'] = '0.1'

#     return kv_opts

# def add_train_wmt16_scratch(kv_opts: collections.OrderedDict) -> collections.OrderedDict:
#     kv_opts['data'] = os.path.join(get_static_paths('data', 'ik1147'),'wmt16_en_de_bpe32k')
#     kv_opts['--task'] = 'translation_eos'
#     kv_opts['--user-dir'] = './fairseq_module/'
#     kv_opts['--optimizer'] = 'adam'
#     kv_opts['--adam-betas'] = '\'(0.9, 0.98)\''
#     kv_opts['--lr-scheduler'] = 'inverse_sqrt'
#     kv_opts['--warmup-updates'] = '4000'
#     kv_opts['--weight-decay'] = '0.0'
#     kv_opts['--criterion'] = 'label_smoothed_cross_entropy'
#     kv_opts['--max-tokens-valid'] = '4096'
#     kv_opts['--max-tokens'] = '32768'
#     kv_opts['--arch'] = 'transformer_wmt_meos'
#     kv_opts['--clip-norm'] = '0.0'
#     kv_opts['--lr'] = '5e-4'
#     kv_opts['--dropout'] = '0.3'
#     kv_opts['--no-epoch-checkpoints'] = True
#     kv_opts['--best-checkpoint-metric'] = 'nll_loss'

#     # label smoothing is turned on here
#     kv_opts['--label-smoothing'] = '0.1'

#     return kv_opts

# def add_train_wmt19_scratch(kv_opts: collections.OrderedDict) -> collections.OrderedDict:
#     kv_opts['data'] = '/scratch/ik1147/nmt_multiple_eos/wmt19_data/tokenized.ru-en_preprocessed'
#     kv_opts['--task'] = 'translation_eos'
#     kv_opts['--user-dir'] = '../fairseq_module/'
#     kv_opts['--optimizer'] = 'adam'
#     kv_opts['--adam-betas'] = '\'(0.9, 0.98)\''
#     kv_opts['--lr-scheduler'] = 'inverse_sqrt'
#     kv_opts['--warmup-updates'] = '4000'
#     kv_opts['--weight-decay'] = '0.0'
#     kv_opts['--criterion'] = 'label_smoothed_cross_entropy'
#     kv_opts['--max-tokens-valid'] = '4096'
#     kv_opts['--max-tokens'] = '4096'
#     kv_opts['--arch'] = 'transformer_wmt_meos'
#     kv_opts['--clip-norm'] = '0.0'
#     kv_opts['--lr'] = '5e-4'
#     kv_opts['--dropout'] = '0.3'
#     kv_opts['--no-epoch-checkpoints'] = True
#     kv_opts['--best-checkpoint-metric'] = 'nll_loss'

#     # label smoothing is turned on here
#     kv_opts['--label-smoothing'] = '0.1'

#     return kv_opts

# def add_finetune_wmt19_ruen(kv_opts: collections.OrderedDict) -> collections.OrderedDict:
#     kv_opts['data'] = '/scratch/ik1147/nmt_multiple_eos/wmt19_data/tokenized.ru-en_preprocessed'
#     kv_opts['--task'] = 'translation_eos'
#     kv_opts['--optimizer'] = 'adam'
#     kv_opts['--adam-betas'] = '\'(0.9, 0.98)\''
#     kv_opts['--lr-scheduler'] = 'inverse_sqrt'
#     kv_opts['--warmup-updates'] = '4000'
#     kv_opts['--weight-decay'] = '0.0'
#     kv_opts['--criterion'] = 'label_smoothed_cross_entropy'
#     kv_opts['--label-smoothing'] = '0.1'
#     kv_opts['--max-tokens'] = '3584'
#     kv_opts['--arch'] = 'transformer_vaswani_wmt_en_de_big_meos'
#     kv_opts['--encoder-ffn-embed-dim'] = '8192'
#     kv_opts['--share-decoder-input-output-embed'] = True
#     kv_opts['--warmup-updates'] = '1'
#     kv_opts['--patience'] = '5'
#     kv_opts['--validate-interval-updates'] = '5000'
#     kv_opts['--clip-norm'] = '0.0'
#     kv_opts['--lr'] = '5e-4'
#     kv_opts['--dropout'] = '0.1'
#     kv_opts['--eval-bleu'] = True
#     kv_opts['--eval-bleu-args'] = '\'{"beam": 10, "max_len_a": 1.2, "max_len_b": 10, "min_length": 0}\''
#     kv_opts['--eval-bleu-detok'] = 'moses'
#     kv_opts['--eval-bleu-remove-bpe'] = True
#     kv_opts['--no-epoch-checkpoints'] = True
#     kv_opts['--best-checkpoint-metric'] = 'bleu'
#     kv_opts['--maximize-best-checkpoint-metric'] = True
#     kv_opts['--max-update'] = 200000

#     return kv_opts

def add_train_wmt19_oversmoothing_small(kv_opts: collections.OrderedDict) -> collections.OrderedDict:
    kv_opts['data'] = '/scratch/ik1147/nmt_multiple_eos/wmt19_data/tokenized.ru-en_preprocessed'
    kv_opts['--task'] = 'translation_oversmoothing'
    kv_opts['--optimizer'] = 'adam'
    kv_opts['--adam-betas'] = '\'(0.9, 0.98)\''
    kv_opts['--lr-scheduler'] = 'inverse_sqrt'
    kv_opts['--warmup-updates'] = '4000'
    kv_opts['--weight-decay'] = '0.0'
    kv_opts['--criterion'] = 'oversmoothing_loss'
    kv_opts['--max-tokens-valid'] = '4096'
    kv_opts['--max-tokens'] = '4096'
    kv_opts['--arch'] = 'transformer_iwslt_de_en'
    kv_opts['--clip-norm'] = '0.0'
    kv_opts['--lr'] = '5e-4'
    kv_opts['--dropout'] = '0.3'
    kv_opts['--no-epoch-checkpoints'] = True
    kv_opts['--best-checkpoint-metric'] = 'loss'

    # label smoothing is turned on here
    kv_opts['--label-smoothing'] = '0.1'

    return kv_opts

def add_train_wmt19_oversmoothing_finetunebig(kv_opts: collections.OrderedDict) -> collections.OrderedDict:
    kv_opts['data'] = '/scratch/ik1147/nmt_multiple_eos/wmt19_data/tokenized.ru-en_preprocessed'
    kv_opts['--finetune-from-model'] = '/scratch/ik1147/nmt_multiple_eos/wmt19_pretrained_models/wmt19.ru-en.ffn8192.pt'
    kv_opts['--task'] = 'translation_oversmoothing'
    kv_opts['--optimizer'] = 'adam'
    kv_opts['--adam-betas'] = '\'(0.9, 0.98)\''
    kv_opts['--lr-scheduler'] = 'inverse_sqrt'
    kv_opts['--weight-decay'] = '0.0'
    kv_opts['--criterion'] = 'oversmoothing_loss'
    kv_opts['--label-smoothing'] = '0.0'  # no label smoothing for finetuning
    kv_opts['--max-tokens'] = '3584'
    kv_opts['--arch'] = 'transformer_vaswani_wmt_en_de_big'
    kv_opts['--encoder-ffn-embed-dim'] = '8192'
    kv_opts['--share-decoder-input-output-embed'] = True
    kv_opts['--warmup-updates'] = '1'
    kv_opts['--patience'] = '5'
    kv_opts['--validate-interval-updates'] = '5000'
    kv_opts['--clip-norm'] = '0.0'
    kv_opts['--lr'] = '5e-4'
    kv_opts['--dropout'] = '0.1'
    kv_opts['--no-epoch-checkpoints'] = True
    kv_opts['--best-checkpoint-metric'] = 'loss'

    return kv_opts

def add_train_wmt16_oversmoothing_finetunebig(kv_opts: collections.OrderedDict) -> collections.OrderedDict:
    kv_opts['data'] = os.path.join(get_static_paths('data', 'ik1147'),'wmt16_en_de_bpe32k')
    kv_opts['--finetune-from-model'] = '/scratch/ik1147/public/wmt16/wmt16.en-de.joined-dict.transformer/model.pt'
    kv_opts['--task'] = 'translation_oversmoothing'
    kv_opts['--optimizer'] = 'adam'
    kv_opts['--adam-betas'] = '\'(0.9, 0.98)\''
    kv_opts['--lr-scheduler'] = 'inverse_sqrt'
    kv_opts['--warmup-updates'] = '4000'
    kv_opts['--weight-decay'] = '0.0'
    kv_opts['--criterion'] = 'oversmoothing_loss'
    kv_opts['--max-tokens'] = '4096'
    kv_opts['--arch'] = 'transformer_vaswani_wmt_en_de_big'
    kv_opts['--clip-norm'] = '0.0'
    kv_opts['--lr'] = '5e-4'
    kv_opts['--dropout'] = '0.3'
    kv_opts['--no-epoch-checkpoints'] = True
    kv_opts['--patience'] = '5'
    kv_opts['--validate-interval-updates'] = '5000'
    kv_opts['--share-all-embeddings'] = True
    kv_opts['--best-checkpoint-metric'] = 'loss'

    # no label smoothing for finetuning
    kv_opts['--label-smoothing'] = '0.0'

    return kv_opts

def add_train_wmt19_deen_oversmoothing_finetunebig(kv_opts: collections.OrderedDict) -> collections.OrderedDict:
    kv_opts['data'] = '/scratch/ik1147/nmt_multiple_eos/wmt19_deen_data/tokenized.de-en_preprocessed'
    kv_opts['--finetune-from-model'] = '/scratch/ik1147/nmt_multiple_eos/wmt19_deen_pretrained_model/wmt19.de-en.ffn8192.pt'
    kv_opts['--task'] = 'translation_oversmoothing'
    kv_opts['--optimizer'] = 'adam'
    kv_opts['--adam-betas'] = '\'(0.9, 0.98)\''
    kv_opts['--lr-scheduler'] = 'inverse_sqrt'
    kv_opts['--weight-decay'] = '0.0'
    kv_opts['--criterion'] = 'oversmoothing_loss'
    kv_opts['--label-smoothing'] = '0.0'  # no label smoothing for finetuning
    kv_opts['--max-tokens'] = '3584'
    kv_opts['--arch'] = 'transformer_vaswani_wmt_en_de_big'
    kv_opts['--encoder-ffn-embed-dim'] = '8192'
    kv_opts['--share-decoder-input-output-embed'] = True
    kv_opts['--warmup-updates'] = '1'
    kv_opts['--patience'] = '5'
    kv_opts['--validate-interval-updates'] = '5000'
    kv_opts['--clip-norm'] = '0.0'
    kv_opts['--lr'] = '5e-4'
    kv_opts['--dropout'] = '0.1'
    kv_opts['--no-epoch-checkpoints'] = True
    kv_opts['--best-checkpoint-metric'] = 'loss'

    return kv_opts

def add_train_wmt19_ende_oversmoothing_finetunebig(kv_opts: collections.OrderedDict) -> collections.OrderedDict:
    kv_opts['data'] = '/scratch/ik1147/nmt_multiple_eos/wmt19_ende_data/tokenized.en-de_preprocessed'
    kv_opts['--finetune-from-model'] = '/scratch/ik1147/nmt_multiple_eos/wmt19_ende_pretrained_model/wmt19.en-de.ffn8192.pt'
    kv_opts['--task'] = 'translation_oversmoothing'
    kv_opts['--optimizer'] = 'adam'
    kv_opts['--adam-betas'] = '\'(0.9, 0.98)\''
    kv_opts['--lr-scheduler'] = 'inverse_sqrt'
    kv_opts['--weight-decay'] = '0.0'
    kv_opts['--criterion'] = 'oversmoothing_loss'
    kv_opts['--label-smoothing'] = '0.0'  # no label smoothing for finetuning
    kv_opts['--max-tokens'] = '3584'
    kv_opts['--arch'] = 'transformer_vaswani_wmt_en_de_big'
    kv_opts['--encoder-ffn-embed-dim'] = '8192'
    kv_opts['--share-decoder-input-output-embed'] = True
    kv_opts['--warmup-updates'] = '1'
    kv_opts['--patience'] = '5'
    kv_opts['--validate-interval-updates'] = '5000'
    kv_opts['--clip-norm'] = '0.0'
    kv_opts['--lr'] = '5e-4'
    kv_opts['--dropout'] = '0.1'
    kv_opts['--no-epoch-checkpoints'] = True
    kv_opts['--best-checkpoint-metric'] = 'loss'

    return kv_opts

def add_train_iwslt17_de_fr_zh_oversmoothing(kv_opts: collections.OrderedDict, language: str) -> collections.OrderedDict:
    kv_opts['data'] = os.path.join(get_static_paths('data', getpass.getuser()), f'iwslt17.tokenized.{language}-en')
    kv_opts['--task'] = 'translation_oversmoothing'
    kv_opts['--optimizer'] = 'adam'
    kv_opts['--adam-betas'] = '\'(0.9, 0.98)\''
    kv_opts['--lr-scheduler'] = 'inverse_sqrt'
    kv_opts['--warmup-updates'] = '4000'
    kv_opts['--weight-decay'] = '0.0'
    kv_opts['--criterion'] = 'oversmoothing_loss'
    kv_opts['--label-smoothing'] = '0.1'
    kv_opts['--max-tokens'] = '4096'
    kv_opts['--arch'] = 'transformer_iwslt_de_en'
    kv_opts['--warmup-updates'] = '1'
    kv_opts['--patience'] = '5'
    kv_opts['--validate-interval-updates'] = '2000'
    kv_opts['--clip-norm'] = '0.0'
    kv_opts['--lr'] = '5e-4'
    kv_opts['--dropout'] = '0.3'
    kv_opts['--no-epoch-checkpoints'] = True
    kv_opts['--best-checkpoint-metric'] = 'loss'

    return kv_opts

# method specific opts

def add_max_logit_policy_wentreg(kv_opts: collections.OrderedDict) -> collections.OrderedDict:
    kv_opts['--eos-choice'] = 'max'
    kv_opts['--marginal-entropy-weight'] = '1.0'
    kv_opts['--conditional-entropy-weight'] = '1.0'

    return kv_opts

# utilities

def all_vs_all_grid(grid_dict: collections.OrderedDict) -> list:
    assert isinstance(grid_dict, collections.OrderedDict), "grid dictionary is supposed to be the ordered dict to avoid "
    sweeps = list(collections.OrderedDict(zip(grid_dict.keys(), values)) for values in itertools.product(*grid_dict.values()))
    return sweeps

def compose_cmd_args(dict_args, newlines=False):
    # first we put data mandatory field

    delimeter = '\n' if newlines else ' '
    trail_char = '\\' if newlines else ''
    cmd_args = [f"{dict_args['data']} {trail_char}"]
    for arg, val in dict_args.items():
        if '--' not in arg:
            continue
        if isinstance(val, bool):
            if val == False:
                # do not include bool arg if it is False
                continue
            else:
                cmd_args.append(f"{arg} {trail_char}")
        else:
            cmd_args.append(f"{arg} {val} {trail_char}")

    cmd_args[-1] = cmd_args[-1].rstrip("{trail_char}")

    return f'{delimeter}'.join(cmd_args)

def load_pkl_args(pkl_args_filename):
    dict_args = pickle.load(open(pkl_args_filename, 'rb'))
    print(compose_cmd_args(dict_args, newlines=True))

def main(call_fn, **kwargs):
    call_fn = globals()[call_fn]
    call_fn(**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
