#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

SCRIPTS=/home/christine/Phd/Cristina_cooperation/joint/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=/home/christine/Phd/Cristina_cooperation/joint/subword-nmt
BPE_TOKENS=31000

#####load the scripts to call later easier######
. $SCRIPTS/generic.sh

src=en
tgt=es
lang=en-es
INPUT_DIR=Europarl_talks_Eva/EN_ES_extended_gendered
OUTPUT_DIR=Europarl_talks_Eva/EN_ES_extended_gendered/output
type=extended_gendered
tmp=$OUTPUT_DIR
split_no=2000

mkdir $OUTPUT_DIR
# Process each file
PREFIX=$INPUT_DIR/train



#############take care of the inputs formats ########################################
####we just need the train files to be train.en and train.es for example######
####we just need the train files to be test_.en and test_.es for example######
############## tokenize, truecase, clean##############################################
echo  $PREFIX.$LANG
for LANG in $src $tgt
do
  log "Processing training [$LANG] data..."
  tokenize $LANG < $PREFIX.$LANG > $PREFIX.tok.$LANG
  rm $PREFIX.$LANG
  train_truecaser $PREFIX.tok.$LANG $OUTPUT_DIR/truecasing.$LANG
  truecase $OUTPUT_DIR/truecasing.$LANG < $PREFIX.tok.$LANG > $PREFIX.tok.tc.$LANG
  rm $PREFIX.tok.$LANG
done

# Remove too long and too short sentences
##############clean##############################################
log "Cleaning corpus..."
clean_corpus $PREFIX.tok.tc $src $tgt clean
rm $PREFIX.tok.tc.{$src,$trg}

log "*** Training data is prepared at $PREFIX.tok.tc.clean.{en,kk}"


########tokenize, truecase test, valid##############
#test file before splitting
test_file=test_
for LANG in $src $tgt
do
  log "Processing dev/test [$LANG] data..."
  tokenize $LANG < $INPUT_DIR/$test_file.$LANG > $INPUT_DIR/$test_file.tok.$LANG
  rm $INPUT_DIR/$test_file.$LANG
  ####using the same truecasing we used in tokenization######
  truecase $OUTPUT_DIR/truecasing.$LANG < $INPUT_DIR/$test_file.tok.$LANG > $INPUT_DIR/$test_file.tok.tc.$LANG
  rm $INPUT_DIR/$test_file.tok.$LANG
done

#############create test and valid#############

for LANG in $src $tgt
do
  log "creating dev/test [$LANG] data..."

  head -n $split_no $INPUT_DIR/$test_file.tok.tc.$LANG > $INPUT_DIR/test.tok.tc.$LANG
  tail -n $split_no $INPUT_DIR/$test_file.tok.tc.$LANG > $INPUT_DIR/valid.tok.tc.$LANG

  rm $INPUT_DIR/$test_file.tok.tc.$LANG

done



#############BPE#############


##############convert train.tags.en-es.en to train.en##############################################
echo "creating train.en, train.es"
for l in $src $tgt; do
    awk '{if (NR%23 != 0)  print $0; }' $PREFIX.tok.tc.$LANG > $OUTPUT_DIR/train.$l

done




##############BPE##############################################
TRAIN=$OUTPUT_DIR/train.$lang
BPE_CODE=$OUTPUT_DIR/code

echo ${TRAIN}

rm -f $TRAIN
for l in $src $tgt; do
    echo $PREFIX.tok.tc.$LANG
    cat $PREFIX.tok.tc.$LANG >> $TRAIN
done
echo "learn_bpe.py on ${TRAIN}..."
####prepare code from the whole trainng of both src and trg######
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE
echo $BPE_CODE

####learning byte code######
for L in $src $tgt; do
    for f in train.tok.tc.clean.$L valid.tok.tc.$L test.tok.tc.$L; do
        echo "apply_bpe.py to ${f}..."


        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $INPUT_DIR/$f > $OUTPUT_DIR/$f
    done
done

rm -f $train
rm -f $OUTPUT_DIR/train.$src
$OUTPUT_DIR/train.$trg

########if extended  replace tokenized sep with <sep> ##################
if [ type='extended' ]
then
sed -i -e 's/< sep >/<sep>/g' *.es
sed -i -e 's/< sep >/<sep>/g' *.en
fi


########if gendered training : replace male or any o f its versions with MALE...... and so for female ##################
if [ type='gendered' ]
then
sed -i -e 's/female/FEMALE/g' *.es
sed -i -e 's/male/MALE/g' *.es
fi
