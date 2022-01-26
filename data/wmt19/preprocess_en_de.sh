#!/bin/bash
# change -s and -t to preprocess de->en pair 

fairseq-preprocess -s en -t de --trainpref ./tokenized.en-de/train --testpref ./tokenized.en-de/test --validpref ./tokenized.en-de/valid --tgtdict ../${PRETRAINED_MODEL_WMT19_ENDE}/dict.de.txt --srcdict ../${PRETRAINED_MODEL_WMT19_ENDE}/dict.de.txt  --destdir ./tokenized.en-de_preprocessed --workers 32
