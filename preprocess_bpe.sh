#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh


SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt
BPE_TOKENS=31000

#####load the scripts to call later easier######
. $SCRIPTS/generic.sh

src=en
tgt=de
lang=en-de
INPUT_DIR=data-bin/Europarl
OUTPUT_DIR=data-bin/Europarl/output
tmp=$OUTPUT_DIR

mkdir $OUTPUT_DIR
# Process each file
PREFIX=train.tags.de-en.tok.tc.clean.tagged



##############BPE##############################################

TRAIN=$OUTPUT_DIR/train.$lang
BPE_CODE=$OUTPUT_DIR/code


for LANG in $src $tgt; do
    echo $INPUT_DIR/$PREFIX.$LANG
    cat $INPUT_DIR/$PREFIX.$LANG >> $TRAIN
done
echo "learn_bpe.py on ${TRAIN}..."
####prepare code from the whole trainng of both src and trg######
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE
echo $BPE_CODE

####learning byte code######
for L in $src $tgt; do
    for f in $PREFIX.$L; do
        echo "apply_bpe.py to ${f}..."


        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $INPUT_DIR/$f > $OUTPUT_DIR/$f
    done
done

