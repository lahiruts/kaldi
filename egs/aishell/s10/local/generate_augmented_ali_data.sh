#!/usr/bin/env bash
# Copyright 2020   Lahiru Samarakoon
# Apache 2.0.
. ./cmd.sh

set -e
stage=3
aug_list="reverb music noise babble clean"  #clean refers to the original train dir
num_reverb_copies=1
nj=45

name=king
location=data/${name}
pythoncmd=python3.7


# Alignment and train directories
src_dir=exp/init_model
ali_dir=exp/${name}/nnet3_ali

#clean_ali=exp/chain/e2e_nnet3_ali
#train_set=train


include_original=true

. ./path.sh
. ./utils/parse_options.sh


if [ $stage -le 0 ]; then
    echo "$0: speed perturbation  for the training data..."
    #speed perturb for the clean training set
    utils/data/get_utt2dur.sh $location/train
    utils/data/get_reco2dur.sh $location/train
    ./utils/fix_data_dir.sh $location/train

    utils/data/perturb_data_dir_speed.sh 0.9 $location/train $location/train_sp_0.9
    utils/data/perturb_data_dir_speed.sh 1.1 $location/train $location/train_sp_1.1
fi


if [ $stage -le 1 ]; then
  # Adding simulated RIRs to the original data directory
  echo "$0: Preparing ${name} reverb directory"

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  if [ ! -f $location/train/reco2dur ]; then
    utils/data/get_reco2dur.sh --nj 6 --cmd "$train_cmd" $location/train || exit 1;
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
    $location/train $location/train_reverb
fi

if [ $stage -le 2 ]; then
  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # We will use them as additive noises for data augmentation.
  #steps/data/make_musan.sh --sampling-rate 16000 --use-vocals "true" \
  #      musan data

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-prefix "noise" --modify-spk-id "true" \
    --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" \
    $location/train  $location/train_noise

  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-prefix "music" --modify-spk-id "true" \
    --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" \
    $location/train $location/train_music

  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-prefix "babble" --modify-spk-id "true" \
    --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" \
    --bg-noise-dir "data/musan_speech" \
    $location/train $location/train_babble

fi


if [ $stage -le 2 ]; then
    echo "reformat the text file"
    for dir_name in train test val train_babble train_music train_noise train_reverb train_sp_0.9 train_sp_1.1 train_aug train_aug_sp;do
	   mv $location/${dir_name}/text $location/${dir_name}/text.org
	   $pythoncmd e2e/reformat_text_file_with_tags.py $location/${dir_name}/text.org $location/${dir_name}/text
    done 

fi


if [ $stage -le 3 ]; then
    #Extract High Resolution MFCC features for the clean training set
    echo "$0: extracting High resolution MFCC features for the training data..."

    #utils/data/perturb_data_dir_volume.sh $location/train

    #steps/make_mfcc_pitch.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    #                   --cmd "$train_cmd" $location/train
    #steps/compute_cmvn_stats.sh $location/train


    echo "$0: forced alignments of training data..."
    mv $src_dir/frame_subsampling_factor $src_dir/frame_subsampling_factor.bk
    #nj=$(cat $src_dir/num_jobs)
    steps/nnet3/align_lats.sh --nj $nj --cmd "$train_cmd" --generate_ali_from_lats true $location/train \
    data/lang $src_dir $ali_dir 
    mv $src_dir/frame_subsampling_factor.bk $src_dir/frame_subsampling_factor
    rm $ali_dir/fsts.*.gz
fi


if [ $stage -le 4 ]; then
    #Extract High Resolution MFCC features for the clean training set
    echo "$0: extracting High resolution MFCC features for the training data 0.9 sp..."
    #utils/data/perturb_data_dir_volume.sh $location/train_sp_0.9
    
    #steps/make_mfcc_pitch.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    #                   --cmd "$train_cmd" $location/train_sp_0.9
    #steps/compute_cmvn_stats.sh $location/train_sp_0.9

    echo "$0: forced alignments of training data sp 0.9 ..."
    mv $src_dir/frame_subsampling_factor $src_dir/frame_subsampling_factor.bk
    #nj=$(cat $src_dir/num_jobs)
    steps/nnet3/align_lats.sh --nj $nj --cmd "$train_cmd" --generate_ali_from_lats true $location/train_sp_0.9 \
    data/lang $src_dir ${ali_dir}_0.9
    mv $src_dir/frame_subsampling_factor.bk $src_dir/frame_subsampling_factor
    rm ${ali_dir}_0.9/fsts.*.gz
fi


if [ $stage -le 5 ]; then
    #Extract High Resolution MFCC features for the clean training set
    echo "$0: extracting High resolution MFCC features for the training data 1.1 sp ..."
    #utils/data/perturb_data_dir_volume.sh $location/train_sp_1.1

    #steps/make_mfcc_pitch.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    #                   --cmd "$train_cmd" $location/train_sp_1.1
    #steps/compute_cmvn_stats.sh $location/train_sp_1.1

    echo "$0: forced alignments of training data sp 1.1 ..."
    mv $src_dir/frame_subsampling_factor $src_dir/frame_subsampling_factor.bk
    #nj=$(cat $src_dir/num_jobs)
    steps/nnet3/align_lats.sh --nj $nj --cmd "$train_cmd" --generate_ali_from_lats true $location/train_sp_1.1 \
    data/lang $src_dir ${ali_dir}_1.1
    mv $src_dir/frame_subsampling_factor.bk $src_dir/frame_subsampling_factor
    rm ${ali_dir}_1.1/fsts.*.gz
fi

data_sets="train_babble train_music train_noise train_reverb"

if [ $stage -le 6 ]; then
    #Extract High Resolution MFCC features for the augmented training set
    echo "$0: extracting High resolution MFCC features for the augmeted training data  ..."
    #for dataset in $data_sets; do
    #	utils/data/perturb_data_dir_volume.sh ${location}/${dataset}
    #    steps/make_mfcc_pitch.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
    #                   --cmd "$train_cmd" ${location}/${dataset}
    #    steps/compute_cmvn_stats.sh ${location}/${dataset}
    # done
fi


if [ $stage -le 7 ]; then
    #combine clean data with the augmented data
    #utils/combine_data.sh ${location}/train_aug ${location}/train ${location}/train_reverb ${location}/train_babble ${location}/train_music ${location}/train_noise
    
    echo "$0: Creating alignments of aug data by copying alignments of clean data"
    steps/copy_ali_dir.sh --nj $nj --cmd "$train_cmd" \
    --include-original "$include_original" \
    ${location}/train_aug ${ali_dir} ${ali_dir}_aug 


    echo "$0: Creating lattices of aug data by copying alignments of clean data"
    steps/copy_lat_dir.sh --nj $nj --cmd "$train_cmd" \
    --include-original "$include_original" \
    ${location}/train_aug ${ali_dir} ${ali_dir}_aug

fi


if [ $stage -le 8 ]; then
    #combine augmeted data with speed perturbated data.
    #utils/combine_data.sh ${location}/train_aug_sp ${location}/train_aug ${location}/train_sp_0.9 ${location}/train_sp_1.1
    
    #combine augmented alignments with speed perturbated alignments
    steps/combine_ali_dirs.sh --nj $nj ${location}/train_aug_sp ${ali_dir}_aug_sp ${ali_dir}_aug ${ali_dir}_0.9 ${ali_dir}_1.1
fi
exit 0

