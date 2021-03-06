#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

#####load the scripts to call later easier######
. $SCRIPTS/generic.sh
# Process each file
src=en
tgt=de
lang=en-de
INPUT_DIR=data-bin/Europarl
OUTPUT_DIR=data-bin/Europarl/output
tmp=$OUTPUT_DIR

mkdir $OUTPUT_DIR
PREFIX=$INPUT_DIR/train.tags.de-en



#############take care of the inputs formats ########################################
####we just need the train files to be valid.en-de.en and train.es for example######
####we just need the train files to be test_.en and test_.es for example######
############## tokenize, truecase, clean##############################################
echo  $PREFIX.$LANG
for LANG in $src $tgt
do
  log "Processing training [$LANG] data..."
  tokenize $LANG < $PREFIX.$LANG > $PREFIX.tok.$LANG
  #rm $PREFIX.$LANG   #not to remove the original now
  train_truecaser $PREFIX.tok.$LANG $OUTPUT_DIR/truecasing.$LANG
  truecase $OUTPUT_DIR/truecasing.$LANG < $PREFIX.tok.$LANG > $PREFIX.tok.tc.$LANG
  rm $PREFIX.tok.$LANG
done

# Remove too long and too short sentences
##############clean##############################################
log "Cleaning corpus..."
clean_corpus $PREFIX.tok.tc $src $tgt clean
rm $PREFIX.tok.tc.{$src,$trg}

log "*** Training data is prepared at $PREFIX.tok.tc.clean.{$src,$tgt}"






