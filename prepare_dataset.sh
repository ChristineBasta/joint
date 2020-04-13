#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh
# Process each file
src=en
tgt=es
lang=en-es
INPUT_DIR=data-bin/iwslt14
OUTPUT_DIR=data-bin/iwslt14/output
tmp=$OUTPUT_DIR

mkdir $OUTPUT_DIR
PREFIX=$INPUT_DIR/train.tags.$lang



SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl

#####load the scripts to call later easier######
. $SCRIPTS/generic.sh
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


done

#Adding src and trg tokens in beginning of sentences
src_file=$PREFIX.tok.tc.$src
src_file_prepared=$PREFIX.tok.tc.tagged.$src
trg_file=$PREFIX.tok.tc.$tgt
trg_file_prepared=$PREFIX.tok.tc.tagged.$tgt
python data_loading/utitilies.py --src_file  $src_file --src_file_prepared $src_file_prepared --trg_file $trg_file --trg_file_prepared $trg_file_prepared
rm $PREFIX.tok.$src
rm $PREFIX.tok.$tgt
rm $PREFIX.tok.tc.$src
rm $PREFIX.tok.tc.$tgt

#rm $PREFIX.tok.tc.clean.$src
#rm $PREFIX.tok.tc.clean.$tgt
log "*** Tagging is done and data is prepared at $src_file_prepared and $trg_file_prepared "

# Remove too long and too short sentences
##############clean##############################################
log "Cleaning corpus..."
clean_corpus $PREFIX.tok.tc.tagged $src $tgt clean
rm $PREFIX.tok.tc.tagged.$src
rm $PREFIX.tok.tc.tagged.$tgt
log "*** Training data is prepared at $PREFIX.tok.tc.clean.{$src,$tgt}"



#BPE step for getting the BPE version of dataset
BPEROOT=subword-nmt
BPE_TOKENS=31000
PREFIX=$PREFIX.tok.tc.tagged.clean
TRAIN=$OUTPUT_DIR/train.$lang
BPE_CODE=$OUTPUT_DIR/code
echo $PREFIX.tok.tc.tagged.clean

for LANG in $src $tgt; do
    echo $PREFIX.$LANG
    cat $PREFIX.$LANG >> $TRAIN
done
echo "learn_bpe.py on ${TRAIN}..."
####prepare code from the whole trainng of both src and trg######
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE
echo $BPE_CODE
file=train
####learning byte code######
for L in $src $tgt; do
    for f in $PREFIX.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $f > $f.bpe

    done
done

rm $PREFIX.$src
rm $PREFIX.$tgt
rm $TRAIN
log "*** BPE files  prepared at $PREFIX.$src.bpe and $PREFIX.$tgt.bpe "
############### making valid and train#################################33

### Prepare test and development data #########################################

test_folder=$INPUT_DIR/test
test_file=IWSLT14.TED.tst2010.$lang
log "Extracting text from SGM files..."
#test
cat $test_folder/$test_file.$src.xml \
   | LC_ALL=C $MOSES_SCRIPTS/ems/support/input-from-sgm-documents.perl \
   > $test_folder/test.$lang.$src

cat $test_folder/$test_file.$tgt.xml\
   | LC_ALL=C $MOSES_SCRIPTS/ems/support/input-from-sgm-documents.perl \
   > $test_folder/test.$lang.$tgt


#valid
dev_file=IWSLT14.TED.dev2010.$lang
log "Extracting text from SGM files..."
#test
cat $test_folder/$dev_file.$src.xml \
   | LC_ALL=C $MOSES_SCRIPTS/ems/support/input-from-sgm-documents.perl \
   > $test_folder/valid.$lang.$src

cat $test_folder/$dev_file.$tgt.xml\
   | LC_ALL=C $MOSES_SCRIPTS/ems/support/input-from-sgm-documents.perl \
   > $test_folder/valid.$lang.$tgt

###########tokenize and truecase the test and dev###################3
DEVTEST_PREFIX=$test_folder/valid.$lang

for LANG in $src $tgt
do
  log "Processing dev [$LANG] data..."
  tokenize $LANG < $DEVTEST_PREFIX.$LANG > $DEVTEST_PREFIX.tok.$LANG
  rm $DEVTEST_PREFIX.$LANG
  truecase $OUTPUT_DIR/truecasing.$LANG < $DEVTEST_PREFIX.tok.$LANG > $DEVTEST_PREFIX.tok.tc.$LANG
  rm $DEVTEST_PREFIX.tok.$LANG
done
python data_loading/utitilies.py --src_file  $DEVTEST_PREFIX.tok.tc.$src --src_file_prepared $DEVTEST_PREFIX.tok.tc.tagged.$src --trg_file $DEVTEST_PREFIX.tok.tc.$tgt --trg_file_prepared $DEVTEST_PREFIX.tok.tc.tagged.$tgt
for L in $src $tgt; do
    for f in $DEVTEST_PREFIX.tok.tc.tagged.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $f > $f.bpe

    done
done

rm $DEVTEST_PREFIX.tok.tc.tagged.$src
rm $DEVTEST_PREFIX.tok.tc.tagged.$tgt
rm $DEVTEST_PREFIX.tok.tc.$src
rm $DEVTEST_PREFIX.tok.tc.$tgt
DEVTEST_PREFIX=$test_folder/test.$lang
for LANG in $src $tgt
do
  log "Processing test [$LANG] data..."
  tokenize $LANG < $DEVTEST_PREFIX.$LANG > $DEVTEST_PREFIX.tok.$LANG
  rm $DEVTEST_PREFIX.$LANG
  truecase $OUTPUT_DIR/truecasing.$LANG < $DEVTEST_PREFIX.tok.$LANG > $DEVTEST_PREFIX.tok.tc.$LANG
  rm $DEVTEST_PREFIX.tok.$LANG
done


python data_loading/utitilies.py --src_file  $DEVTEST_PREFIX.tok.tc.$src --src_file_prepared $DEVTEST_PREFIX.tok.tc.tagged.$src --trg_file $DEVTEST_PREFIX.tok.tc.$tgt --trg_file_prepared $DEVTEST_PREFIX.tok.tc.tagged.$tgt
####learning byte code######

for L in $src $tgt; do
    for f in $DEVTEST_PREFIX.tok.tc.tagged.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $f > $f.bpe

    done
done

rm $DEVTEST_PREFIX.tok.tc.tagged.$src
rm $DEVTEST_PREFIX.tok.tc.tagged.$tgt
rm $DEVTEST_PREFIX.tok.tc.$src
rm $DEVTEST_PREFIX.tok.tc.$tgt


#Renaming all files
mv $INPUT_DIR/test/test.$lang.tok.tc.tagged.$src.bpe $INPUT_DIR/test.$lang.$src
mv $INPUT_DIR/test/test.$lang.tok.tc.tagged.$tgt.bpe $INPUT_DIR/test.$lang.$tgt

mv $INPUT_DIR/test/valid.$lang.tok.tc.tagged.$src.bpe $INPUT_DIR/valid.$lang.$src
mv $INPUT_DIR/test/valid.$lang.tok.tc.tagged.$tgt.bpe $INPUT_DIR/valid.$lang.$tgt

mv $INPUT_DIR/train.tags.$lang.tok.tc.tagged.clean.$src.bpe $INPUT_DIR/train.$lang.$src
mv $INPUT_DIR/train.tags.$lang.tok.tc.tagged.clean.$tgt.bpe $INPUT_DIR/train.$lang.$tgt


#################preparing dictionaries####################################
########### edit################33
TEXT=$INPUT_DIR

 /usr/bin/python3.6 get_dictionary.py --joined-dictionary --source-lang en --target-lang es \
  --trainpref $TEXT/train.$lang --validpref $TEXT/valid.$lang --testpref $TEXT/test.$lang \
  --destdir $TEXT/dictionaries