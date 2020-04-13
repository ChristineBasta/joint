# Dataset download and preparation
#cd examples
#./prepare-iwslt14-31K.sh
#cd ..
#Dataset adding token and src tags

# Dataset binarization:
TEXT=data-bin/Europarl
 /usr/bin/python3.6 get_dictionary.py --joined-dictionary --source-lang de --target-lang en \
  --trainpref $TEXT/train --validpref $TEXT/train --testpref $TEXT/train \
  --destdir data-bin/Europarl/dictionaries