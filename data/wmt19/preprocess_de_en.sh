#!/bin/bash
# change -s and -t to preprocess en->de pair 

fairseq-preprocess -s de -t en --trainpref ./tokenized.en-de/train --testpref ./tokenized.en-de/test --validpref ./tokenized.en-de/valid --tgtdict ../${PRETRAINED_MODEL_WMT19_ENDE}/dict.de.txt --srcdict ../${PRETRAINED_MODEL_WMT19_ENDE}/dict.de.txt  --destdir ./tokenized.de-en_preprocessed --workers 32
