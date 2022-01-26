#!/bin/bash

fairseq-preprocess -s ru -t en --trainpref ./tokenized.ru-en/train --validpref ./tokenized.ru-en/valid --testpref ./tokenized.ru-en/test --tgtdict ../${PRETRAINED_MODEL_WMT19_RUEN}/dict.en.txt --srcdict ../${PRETRAINED_MODEL_WMT19_RUEN}/dict.ru.txt  --destdir ./tokenized.ru-en_preprocessed --workers 8
