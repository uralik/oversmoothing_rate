from pathlib import Path
import argparse


def merge_test_validation_files(data_folder, source_lang, target_lang, prefix):
    total_files = len(list(Path(data_folder).rglob(f'{prefix}?.{source_lang}-{target_lang}.{source_lang}')))
    valid_source = ''
    valid_target = ''

    valid_bpe_source = ''
    valid_bpe_target = ''
    for i in range(int(total_files)):
        with open(f'{data_folder}/{prefix}{i}.{source_lang}-{target_lang}.{source_lang}', 'r') as f:
            valid_source += f.read()
        
        with open(f'{data_folder}/{prefix}{i}.{source_lang}-{target_lang}.{target_lang}', 'r') as f:
            valid_target += f.read()

        with open(f'{data_folder}/{prefix}{i}.bpe.{source_lang}-{target_lang}.{source_lang}', 'r') as f:
            valid_bpe_source += f.read()
        
        with open(f'{data_folder}/{prefix}{i}.bpe.{source_lang}-{target_lang}.{target_lang}', 'r') as f:
            valid_bpe_target += f.read()

    with open(f'{data_folder}/{prefix}.{source_lang}-{target_lang}.{source_lang}', 'w') as f:
        f.write(valid_source)

    with open(f'{data_folder}/{prefix}.{source_lang}-{target_lang}.{target_lang}', 'w') as f:
        f.write(valid_target)

    with open(f'{data_folder}/{prefix}.bpe.{source_lang}-{target_lang}.{source_lang}', 'w') as f:
        f.write(valid_bpe_source)

    with open(f'{data_folder}/{prefix}.bpe.{source_lang}-{target_lang}.{target_lang}', 'w') as f:
        f.write(valid_bpe_target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', nargs='?', help='path to data folder')
    parser.add_argument('-s', '--source_lang', nargs='?', help='source language')
    parser.add_argument('-t', '--target_lang', nargs='?', help='target language')
    parser.add_argument('-p', '--prefix', nargs='?', help='valid or test')
    args = parser.parse_args()

    merge_test_validation_files(args.data, args.source_lang, args.target_lang, args.prefix)
