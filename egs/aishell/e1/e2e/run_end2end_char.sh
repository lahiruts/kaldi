#!/bin/bash
# Copyright 2019   Lahiru Fano Labs

# This top-level script demonstrates character-based end-to-end LF-MMI training
# (specifically single-stage flat-start LF-MMI models) on Aishell2. It is exactly
# like "run_end2end_phone.sh" excpet it uses a trivial grapheme-based
# (i.e. character-based) lexicon and a stronger neural net (i.e. TDNN-LSTM)

#This script assumes the data is avaialble in the Kaldi format.
set -euo pipefail
stage=3
trainset=train
#trainset=train_spe2e_hires
devset=dev
testset=test
pythoncmd=python3.7
numChars=3000

. ./cmd.sh ## You'll want to change cmd.sh to something that will work
           ## on your system. This relates to the queue.
. ./utils/parse_options.sh
. ./path.sh

# We use the suffix _nosp for the phoneme-based dictionary and
# lang directories (for consistency with run.sh) and the suffix
# _char for character-based dictionary and lang directories.
dict_dir=data/local/dict_char
if [ $stage -le 0 ]; then
    for x in train dev test; do
        mv data/${x}/text data/${x}/text.org
        $pythoncmd e2e/reformat_text_file.py data/${x}/text.org data/${x}/text
    done
    mkdir -p $dict_dir
    #create the silence_phones.txt file. Make sure special tags in the transcrips (<SPK/> ..etc)  
    #matches to the proper silence phone in the lexicon
    printf "SIL\nSPN\nNSN\n" > ${dict_dir}/silence_phones.txt
    printf "SIL\n" > ${dict_dir}/optional_silence.txt  #create manually if diallowed silences are present. 
    $pythoncmd e2e/cal_coverage.py data/${trainset}/text.org $numChars ${dict_dir}/nonsilence_phones.txt
    $pythoncmd e2e/create_e2e_lexicon.py ${dict_dir}/nonsilence_phones.txt ${dict_dir}/lexicon.txt
fi


if [ $stage -le 1 ]; then
  utils/prepare_lang.sh --position-dependent-phones false \
    $dict_dir "<UNK>" data/local/lang data/lang || exit 1;
fi

# arpa LM
if [ $stage -le 2 ]; then
  e2e/train_lms.sh \
      data/local/dict_char/lexicon.txt \
      data/${trainset}/text \
      data/local/lm || exit 1;
fi


# G compilation, check LG composition
if [ $stage -le 3 ]; then
  utils/format_lm.sh data/lang data/local/lm/4gram/lm_unpruned.gz \
    data/local/dict_char/lexicon.txt data/lang_test || exit 1;
fi


if [ $stage -le 4 ]; then
  # make MFCC features for the test data. Only hires since it's flat-start.
  if [ -f data/test_hires/feats.scp ]; then
    echo "$0: It seems that features for the test sets already exist."
    echo "skipping this stage..."
  else
    echo "$0: extracting MFCC features for the test sets"
    for x in test dev; do
      mv data/$x data/${x}_hires
      steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj 20 \
                         --mfcc-config conf/mfcc_hires.conf data/${x}_hires
      steps/compute_cmvn_stats.sh data/${x}_hires
    done
  fi
fi


if [ -f data/${trainset}_spe2e_hires/feats.scp ]; then
  echo "$0: It seems that features for the perturbed training data already exist."
  echo "If you want to extract them anyway, remove them first and run this"
  echo "stage again. Skipping this stage..."
else
  if [ $stage -le 5 ]; then
    echo "$0: perturbing the training data to allowed lengths..."
    #utils/data/get_utt2dur.sh data/$trainset  # necessary for the next command

    # 12 in the following command means the allowed lengths are spaced
    # by 12% change in length.
    # replace the tab with space charater in the wav.scp for this file to work properly. 
    $pythoncmd utils/data/perturb_speed_to_allowed_lengths.py 12 data/${trainset} \
                                                   data/${trainset}_spe2e_hires
   
    #Note we may have to remove tabs sed -e 's/\t/ /g' data/train/bk_text >  data/train/text
    cat data/${trainset}_spe2e_hires/utt2dur | \
      awk '{print $1 " " substr($1,5)}' >data/${trainset}_spe2e_hires/utt2uniq
    utils/fix_data_dir.sh data/${trainset}_spe2e_hires
  fi

  if [ $stage -le 6 ]; then
    echo "$0: extracting MFCC features for the training data..."
    steps/make_mfcc_pitch.sh --nj 68 --mfcc-config conf/mfcc_hires.conf \
                       --cmd "$train_cmd" data/${trainset}_spe2e_hires
    steps/compute_cmvn_stats.sh data/${trainset}_spe2e_hires
  fi
fi

exit 0

if [ $stage -le 7 ]; then
  echo "$0: calling the flat-start chain recipe..."
  e2e/run_tdnnf_flatstart_char.sh
fi
