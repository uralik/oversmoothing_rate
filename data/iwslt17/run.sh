#!/bin/bash

pip install sentencepiece

bash prepare-iwslt17-multilingual.sh

python merge_validation_files.py -d iwslt17.de_fr_zh.en.bpe16k -s $1 -t en -p valid
python merge_validation_files.py -d iwslt17.de_fr_zh.en.bpe16k -s $1 -t en -p test

TEXT=iwslt17.de_fr_zh.en.bpe16k
fairseq-preprocess --source-lang $1 --target-lang en --trainpref $TEXT/train.bpe.$1-en --validpref $TEXT/valid.bpe.$1-en --testpref $TEXT/test.bpe.$1-en --destdir data-bin/iwslt17.$1.en.bpe16k
