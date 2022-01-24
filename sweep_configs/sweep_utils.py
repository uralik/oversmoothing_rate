#!/usr/bin/env python

import pickle
import itertools
import fire
import collections
import getpass
import datetime
import os

# amending opts for downstream task

def add_common_validation(kv_opts: collections.OrderedDict, args_from_trained_model: collections.OrderedDict) -> collections.OrderedDict:
    kv_opts['data'] = args_from_trained_model['data']
    kv_opts['--user-dir'] = os.environ.get('FAIRSEQ_MODULE')
    kv_opts['--task'] = args_from_trained_model['--task']
    kv_opts['--path'] = os.path.join(args_from_trained_model['--save-dir'], 'checkpoint_best.pt')
    if '--eval-bleu-args' in args_from_trained_model:
        kv_opts['train_sweep_eval_bleu_args'] = args_from_trained_model['--eval-bleu-args']  # keep the eval bleu args which were used during the actual training to know what beam size was used at training in case of eos penalty
    kv_opts['--max-tokens'] = '2048'
    kv_opts['--valid-subset'] = 'valid'

    return kv_opts

# task specific opts

def add_train_wmt19_oversmoothing_finetunebig(kv_opts: collections.OrderedDict) -> collections.OrderedDict:
    kv_opts['data'] = os.path.join(os.environ.get('DATA_WMT19_RUEN'), 'tokenized.ru-en_preprocessed')
    kv_opts['--finetune-from-model'] = os.environ.get('PRETRAINED_MODEL_WMT19_RUEN')
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
    kv_opts['data'] = os.path.join(os.environ.get('DATA_WMT16_ENDE'),'wmt16_en_de_bpe32k')
    kv_opts['--finetune-from-model'] = os.environ.get('PRETRAINED_MODEL_WMT16_ENDE')
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
    kv_opts['data'] = os.path.join(os.environ.get('DATA_WMT19_DEEN'), 'tokenized.de-en_preprocessed')
    kv_opts['--finetune-from-model'] = os.environ.get('PRETRAINED_MODEL_WMT19_DEEN')
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
    kv_opts['data'] = os.path.join(os.environ.get('DATA_WMT19_ENDE'), 'tokenized.en-de_preprocessed')
    kv_opts['--finetune-from-model'] = os.environ.get('PRETRAINED_MODEL_WMT19_ENDE')
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
    kv_opts['data'] = os.path.join(os.environ.get('DATA_IWSLT'), f'iwslt17.tokenized.{language}-en')
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
