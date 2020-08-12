#!/usr/bin/env bash
# Copyright 2020   Lahiru Samarakoon
# Apache 2.0.

. ./cmd.sh

set -e
stage=7
aug_list="reverb music noise babble clean"  #clean refers to the original train dir
num_reverb_copies=1

# Alignment and train directories
clean_ali=exp/chain/e2e_nnet3_ali
train_set=train


include_original=true

. ./path.sh
. ./utils/parse_options.sh


if [ $stage -le 0 ]; then
    echo "$0: speed perturbation  for the training data..."
    #speed perturb for the clean training set
    utils/data/get_utt2dur.sh data/${train_set}
    utils/data/get_reco2dur.sh data/${train_set}

    utils/data/perturb_data_dir_speed.sh 0.9 data/${train_set} data/${train_set}_sp_0.9
    utils/data/perturb_data_dir_speed.sh 1.1 data/${train_set} data/${train_set}_sp_1.1
fi


if [ $stage -le 1 ]; then
  # Adding simulated RIRs to the original data directory
  echo "$0: Preparing data/${train_set}_reverb directory"

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  if [ ! -f data/$train_set/reco2dur ]; then
    utils/data/get_reco2dur.sh --nj 6 --cmd "$train_cmd" data/$train_set || exit 1;
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD train_nodup.
  # Note that we don't add any additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --prefix "reverb" \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications $num_reverb_copies \
    --source-sampling-rate 16000 \
    data/$train_set data/${train_set}_reverb
fi

if [ $stage -le 2 ]; then
  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # We will use them as additive noises for data augmentation.
  steps/data/make_musan.sh --sampling-rate 16000 --use-vocals "true" \
        musan data

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-prefix "noise" --modify-spk-id "true" \
    --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" \
    data/${train_set} data/${train_set}_noise

  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-prefix "music" --modify-spk-id "true" \
    --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" \
    data/${train_set} data/${train_set}_music

  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-prefix "babble" --modify-spk-id "true" \
    --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" \
    --bg-noise-dir "data/musan_speech" \
    data/${train_set} data/${train_set}_babble

fi


src_dir=exp/chain/e2e_tdnnf_char1b
ali_dir=exp/chain/e2e_nnet3_ali

if [ $stage -le 3 ]; then
    #Extract High Resolution MFCC features for the clean training set
    echo "$0: extracting High resolution MFCC features for the training data..."

    utils/data/perturb_data_dir_volume.sh data/${train_set}

    steps/make_mfcc_pitch.sh --nj 60 --mfcc-config conf/mfcc_hires.conf \
                       --cmd "$train_cmd" data/${train_set}
    steps/compute_cmvn_stats.sh data/${train_set}


    echo "$0: forced alignments of training data..."
    mv $src_dir/frame_subsampling_factor $src_dir/frame_subsampling_factor.bk
    nj=$(cat $src_dir/num_jobs)
    steps/nnet3/align_lats.sh --nj $nj --cmd "$train_cmd" --generate_ali_from_lats true data/$train_set \
    data/lang $src_dir $ali_dir 
    mv $src_dir/frame_subsampling_factor.bk $src_dir/frame_subsampling_factor
    rm $ali_dir/fsts.*.gz
fi


if [ $stage -le 4 ]; then
    #Extract High Resolution MFCC features for the clean training set
    echo "$0: extracting High resolution MFCC features for the training data 0.9 sp..."
    #utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_0.9
    
    #steps/make_mfcc_pitch.sh --nj 60 --mfcc-config conf/mfcc_hires.conf \
    #                   --cmd "$train_cmd" data/${train_set}_sp_0.9
    #steps/compute_cmvn_stats.sh data/${train_set}_sp_0.9

    echo "$0: forced alignments of training data sp 0.9 ..."
    mv $src_dir/frame_subsampling_factor $src_dir/frame_subsampling_factor.bk
    nj=$(cat $src_dir/num_jobs)
    steps/nnet3/align_lats.sh --nj $nj --cmd "$train_cmd" --generate_ali_from_lats true data/${train_set}_sp_0.9 \
    data/lang $src_dir ${ali_dir}_0.9
    mv $src_dir/frame_subsampling_factor.bk $src_dir/frame_subsampling_factor
    rm ${ali_dir}_0.9/fsts.*.gz
fi


if [ $stage -le 5 ]; then
    #Extract High Resolution MFCC features for the clean training set
    echo "$0: extracting High resolution MFCC features for the training data 1.1 sp ..."
    #utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_1.1

    #steps/make_mfcc_pitch.sh --nj 60 --mfcc-config conf/mfcc_hires.conf \
    #                   --cmd "$train_cmd" data/${train_set}_sp_1.1
    #steps/compute_cmvn_stats.sh data/${train_set}_sp_1.1

    echo "$0: forced alignments of training data sp 1.1 ..."
    mv $src_dir/frame_subsampling_factor $src_dir/frame_subsampling_factor.bk
    nj=$(cat $src_dir/num_jobs)
    steps/nnet3/align_lats.sh --nj $nj --cmd "$train_cmd" --generate_ali_from_lats true data/${train_set}_sp_1.1 \
    data/lang $src_dir ${ali_dir}_1.1
    mv $src_dir/frame_subsampling_factor.bk $src_dir/frame_subsampling_factor
    rm ${ali_dir}_1.1/fsts.*.gz
fi
data_sets="train_babble train_music train_noise train_reverb"

if [ $stage -le 6 ]; then
    #Extract High Resolution MFCC features for the augmented training set
    echo "$0: extracting High resolution MFCC features for the augmeted training data  ..."
    for dataset in $data_sets; do
	utils/data/perturb_data_dir_volume.sh data/${dataset}
        steps/make_mfcc_pitch.sh --nj 60 --mfcc-config conf/mfcc_hires.conf \
                       --cmd "$train_cmd" data/${dataset}
        steps/compute_cmvn_stats.sh data/${dataset}
     done
fi


if [ $stage -le 7 ]; then
    #combine clean data with the augmented data
    #utils/combine_data.sh data/${train_set}_aug data/train data/train_reverb data/train_babble data/train_music data/train_noise
    
    echo "$0: Creating alignments of aug data by copying alignments of clean data"
    #steps/copy_ali_dir.sh --nj 60 --cmd "$train_cmd" \
    #--include-original "$include_original" \
    #data/${train_set}_aug ${clean_ali} ${clean_ali}_aug 


    echo "$0: Creating lattices of aug data by copying alignments of clean data"
    steps/copy_lat_dir.sh --nj 60 --cmd "$train_cmd" \
    --include-original "$include_original" \
    data/${train_set}_aug ${clean_ali} ${clean_ali}_aug


fi



if [ $stage -le 8 ]; then
    #combine augmeted data with speed perturbated data.
    #utils/combine_data.sh data/${train_set}_aug_sp data/${train_set}_aug data/${train_set}_sp_0.9 data/${train_set}_sp_1.1
    
    #combine augmented alignments with speed perturbated alignments
    steps/combine_ali_dirs.sh --nj 60 data/${train_set}_aug_sp ${clean_ali}_aug_sp ${clean_ali}_aug ${clean_ali}_0.9 ${clean_ali}_1.1
fi
exit 0

if [ $stage -le 2 ]; then
  # Extract low-resolution MFCCs for the augmented data
  # To be used later to generate alignments for augmented data
  echo "$0: Extracting low-resolution MFCCs for the augmented data. Useful for generating alignments"
  mfccdir=mfcc_aug
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/mfcc/swbd-$date/s5c/$mfccdir/storage $mfccdir/storage
  fi
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
                     data/${train_set}_aug exp/make_mfcc/${train_set}_aug $mfccdir
  steps/compute_cmvn_stats.sh data/${train_set}_aug exp/make_mfcc/${train_set}_aug $mfccdir
  utils/fix_data_dir.sh data/${train_set}_aug || exit 1;
fi

if [ $stage -le 3 ] && $generate_alignments; then
  # obtain the alignment of augmented data from clean data
  include_original=false
  prefixes=""
  for n in $aug_list; do
    if [ "$n" == "reverb" ]; then
      for i in `seq 1 $num_reverb_copies`; do
        prefixes="$prefixes "reverb$i
      done
    elif [ "$n" != "clean" ]; then
      prefixes="$prefixes "$n
    else
      # The original train directory will not have any prefix
      # include_original flag will take care of copying the original alignments
      include_original=true
    fi
  done
  echo "$0: Creating alignments of aug data by copying alignments of clean data"
  steps/copy_ali_dir.sh --nj 40 --cmd "$train_cmd" \
    --include-original "$include_original" --prefixes "$prefixes" \
    data/${train_set}_aug exp/${clean_ali} exp/${clean_ali}_aug
fi

if [ $stage -le 4 ]; then
  mfccdir=mfcc_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    date=$(date +'%m_%d_%H_%M')
    utils/create_split_dir.pl /export/b0{1,2,3,4}/$USER/kaldi-data/mfcc/swbd-$date/s5c/$mfccdir/storage $mfccdir/storage
  fi

  for dataset in ${train_set}_aug; do
    echo "$0: Creating hi resolution MFCCs for dir data/$dataset"
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    utils/data/perturb_data_dir_volume.sh data/${dataset}_hires

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires;
  done
fi

if [ $stage -le 5 ]; then
  mfccdir=mfcc_hires
  for dataset in eval2000 train_dev $maybe_rt03; do
    echo "$0: Creating hi resolution MFCCs for data/$dataset"
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf \
        data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
  done
fi

