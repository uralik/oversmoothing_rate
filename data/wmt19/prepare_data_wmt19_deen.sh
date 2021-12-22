#!/bin/bash
# Adapted from https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-wmt14en2de.sh#L5

# THIS ONE IS USING BPE CODES FROM FAIRSEQ PRETRAINED CHECKPOINTS

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning fastBPE...'
git clone https://github.com/glample/fastBPE.git

cd fastBPE/
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
cd ..

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl
BPEROOT=fastBPE/fast
BPE_TOKENS=30000
BPE_CODES=fastbpe.code  # comes with pretrained fairseq checkpoint

URLS=(
    "http://statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz"
    "http://data.statmt.org/news-commentary/v14/training/news-commentary-v14.de-en.tsv.gz"
    "http://data.statmt.org/wmt19/translation-task/dev.tgz"
    "http://data.statmt.org/wmt19/translation-task/test.tgz"
)
FILES=(
    "training-parallel-commoncrawl.tgz"
    "training-parallel-nc-v12.tgz"
    "news-commentary-v14.de-en.tsv.gz"
    "dev.tgz"
    "test.tgz"
)
CORPORA=(
    "commoncrawl.de-en"
    "training/news-commentary-v12.de-en"
    "news-commentary-v14.de-en"  # this needs a tsv split first! use split_tsv.sh
)

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
lang=de-en
prep=tokenized.de-en
tmp=$prep/tmp
orig=orig
dev=dev/newstest2016

mkdir -p $orig $tmp $prep

# comment to skip downloading

cd $orig

for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit -1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        fi
    fi
done
cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    rm $tmp/train.tags.$lang.tok.$l
    for f in "${CORPORA[@]}"; do
        cat $orig/$f.$l | \
            perl $NORM_PUNC $l | \
            perl $REM_NON_PRINT_CHAR | \
            perl $TOKENIZER -threads 8 -a -l $l >> $tmp/train.tags.$lang.tok.$l
    done
done

echo "pre-processing test data..."
for l in $src $tgt; do
    if [ "$l" == "$src" ]; then
        t="src"
    else
        t="ref"
    fi
    grep '<seg id' $orig/sgm/newstest2019-deen-$t.$l.sgm | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -a -l $l > $tmp/test.$l
    echo ""
done

echo "splitting train and valid..."
for l in $src $tgt; do
    awk '{if (NR%100 == 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/valid.$l
    awk '{if (NR%100 != 0)  print $0; }' $tmp/train.tags.$lang.tok.$l > $tmp/train.$l
done

# TRAIN=$tmp/train.ru-en
# BPE_CODE=$BPE_CODES
# rm -f $TRAIN
# for l in $src $tgt; do
#     cat $tmp/train.$l >> $TRAIN
# done

# DO NOT LEARN BPE HERE
# echo "learn_bpe.py on ${TRAIN}..."
# python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        $BPEROOT applybpe $tmp/bpe.$f $tmp/$f ende30k.$BPE_CODES
    done
done

perl $CLEAN -ratio 1.5 $tmp/bpe.train $src $tgt $prep/train 1 250
perl $CLEAN -ratio 1.5 $tmp/bpe.valid $src $tgt $prep/valid 1 250

for L in $src $tgt; do
    cp $tmp/bpe.test.$L $prep/test.$L
done