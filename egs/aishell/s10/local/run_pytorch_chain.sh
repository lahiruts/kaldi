#!/bin/bash

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

set -e

stage=1

nj=20

#train_set=train
#gmm=tri3
#nnet3_affix=_pybind
#tree_affix= 
#tdnn_affix=
#train_affix=
online_cmvn=true
train_ivector=false
num_epochs=6
dropout_schedule=0,0@0.20,0.5@0.50,0      # you might set this to 0,0 or 0.5,0.5 to train.
frame_subsampling_factor=3
feat_type="delta"
lang=default

. ./path.sh
. ./cmd.sh

. parse_options.sh

ali_dir=exp/chain/nnet3_ali
lat_dir=exp/chain/nnet3_ali
lang_dir=data/lang_chain_e2e_char
tree_dir=exp/chain/e2e_realign_char_tree_tied1a
train_data_dir=data/train_spe2e_hires
dir=exp/chain_pybind/tdnn_sp

#echo $train_ivector_dir

#tree_dir=exp/chain${nnet3_affix}/tree_bi${tree_affix}
#lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
#dir=exp/chain${nnet3_affix}/tdnn${tdnn_affix}_sp
#train_data_dir=data/${train_set}_sp_hires
#lores_train_data_dir=data/${train_set}_sp

if  [[ $stage -le 0 ]]; then
  echo "$0: Making Phone LM and denominator and normalization FST"
  mkdir -p $dir/den_fsts/log

  # We may later reorganize this.
  cp $tree_dir/tree $dir/${lang}.tree

  echo "$0: creating phone language-model"
  $train_cmd $dir/den_fsts/log/make_phone_lm_${lang}.log \
    chain-est-phone-lm --num-extra-lm-states=2000 \
       "ark:gunzip -c $tree_dir/ali.*.gz | ali-to-phones $tree_dir/final.mdl ark:- ark:- |" \
       $dir/den_fsts/${lang}.phone_lm.fst
  mkdir -p $dir/init
  copy-transition-model $tree_dir/final.mdl $dir/init/${lang}_trans.mdl
  echo "$0: creating denominator FST"
  $train_cmd $dir/den_fsts/log/make_den_fst.log \
     chain-make-den-fst $dir/${lang}.tree $dir/init/${lang}_trans.mdl $dir/den_fsts/${lang}.phone_lm.fst \
     $dir/den_fsts/${lang}.den.fst $dir/den_fsts/${lang}.normalization.fst || exit 1;
fi


# You should know how to calculate your model's left/right context **manually**
model_left_context=28
model_right_context=28
egs_left_context=$(($model_left_context + 1))
egs_right_context=$(($model_right_context + 1))
frames_per_eg=150,110,90
frames_per_iter=1500000
minibatch_size=128

hidden_dim=1024
bottleneck_dim=128
prefinal_bottleneck_dim=256
kernel_size_list="3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3" # comma separated list
subsampling_factor_list="1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1" # comma separated list

log_level=info # valid values: debug, info, warning

# true to save network output as kaldi::CompressedMatrix
# false to save it as kaldi::Matrix<float>
save_nn_output_as_compressed=false
if [ $stage -le 1 ]; then
  echo "$0: about to dump raw egs."
  # Dump raw egs.
  steps/chain2/get_raw_egs.sh --cmd "$train_cmd" \
    --lang "${lang}" \
    --online-cmvn $online_cmvn \
    --left-context $egs_left_context \
    --right-context $egs_right_context \
    --frame-subsampling-factor $frame_subsampling_factor \
    --alignment-subsampling-factor $frame_subsampling_factor \
    --frames-per-chunk $frames_per_eg \
    --feat-type $feat_type \
    ${train_data_dir} ${dir} ${lat_dir} ${dir}/raw_egs
fi


if [ $stage -le 2 ]; then
  echo "$0: about to process egs"
  steps/chain2/process_egs.sh  --cmd "$train_cmd" \
      --num-repeats 1 \
    ${dir}/raw_egs ${dir}/processed_egs
fi

num_workers=4
if [ $stage -le 3 ]; then
  echo "$0: about to randomize egs"
  steps/chain2/randomize_egs.sh --frames-per-job 3000000 --num-workers $num_workers \
    ${dir}/processed_egs ${dir}/egs
fi

info_file=$dir/raw_egs/info.txt
feat_dim=$(grep 'feat_dim' $info_file | awk '{print $NF}')
ivector_dim=0
ivector_period=0
if $train_ivector; then
  ivector_dim=$(grep 'ivector_dim' $info_file | awk '{print $NF}')
  ivector_period=$(cat $train_ivector_dir/ivector_period)
fi
echo "ivector_dim: $ivector_dim", "ivector_period, $ivector_period"

merged_egs_dir=merged_egs_chain2
if [[ $stage -le 4 ]]; then
  echo "$0: merging egs"

  mkdir -p $dir/$merged_egs_dir
  num_egs=$(ls -1 $dir/egs/train.*.scp | wc -l)

  $train_cmd --max-jobs-run $nj JOB=1:$num_egs $dir/$merged_egs_dir/log/merge_egs.JOB.log \
    nnet3-chain-shuffle-egs scp:$dir/egs/train.JOB.scp ark:- \| \
    nnet3-chain-merge-egs --minibatch-size=$minibatch_size ark:- \
      ark,scp:$dir/$merged_egs_dir/cegs.JOB.ark,$dir/$merged_egs_dir/cegs.JOB.scp || exit 1

  rm -f $dir/raw_egs/cegs.*.ark
fi

training_eg_dir=egs_chain2_for_training
# we have to make sure each scp file holding the same number of lines,
# as we will load them with multiple workers in PyTorch and there is an
# assumption in DDP training that num-mininbatches should be equal
# across workers.
if [[ $stage -le 5 ]]; then
  echo "$0: align eg numbers in each scp file"

  mkdir -p $dir/$training_eg_dir/tmp_scp_dir
  steps/chain2/align_eg_numbers.sh $dir/$merged_egs_dir $dir/$training_eg_dir/tmp_scp_dir

  # TODO: make this more efficient as for each ark file there are only few arks
  #       we really need to copy from other ark files.
  num_egs=$(ls -1 $dir/$training_eg_dir/tmp_scp_dir/*.scp | wc -l)
  $train_cmd --max-jobs-run $nj JOB=1:$num_egs $dir/$training_eg_dir/log/copy_egs.JOB.log \
    nnet3-chain-copy-egs scp:$dir/$training_eg_dir/tmp_scp_dir/cegs.JOB.scp \
      ark,scp:$dir/$training_eg_dir/cegs.JOB.ark,$dir/$training_eg_dir/cegs.JOB.scp || exit 1

  rm -r $dir/$training_eg_dir/tmp_scp_dir
  rm -f $dir/$merged_egs_dir/cegs.*.ark
fi

#cuda_train_cmd="$cuda_train_cmd --gpu 4 exp/temp/logs/temp.log"
#$cuda_train_cmd python3 ./chain/temp.py


#export KALDI_ROOT=/mnt/ssd/lahiru/Devinstalls/kaldi_april_20/kaldi
#export PYTHONPATH=$KALDI_ROOT/src/pybind:$PYTHONPATH
export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_sequential.so

output_dim=$(grep 'num_leaves' $info_file | awk '{print $NF}')
train_dir=train
if [[ $stage -le 6 ]]; then
  echo "$0: training..."

  mkdir -p $dir/$train_dir/tensorboard
  train_checkpoint=
  if [[ -f $dir/$train_dir/best_model.pt ]]; then
    train_checkpoint=$dir/$train_dir/best_model.pt
  fi

  INIT_FILE=$dir/$train_dir/ddp_init
  rm -f $INIT_FILE # delete old one before starting
  init_method=file://$(readlink -f $INIT_FILE)
  # use '127.0.0.1' for training on a single machine
  init_method=tcp://127.0.0.1:7275
  #init_method=tcp://127.0.0.1
  echo "$0: init method is $init_method"

  num_epochs=$num_epochs
  lr=1e-3
  
  # use_ddp = false & world_size = 1: training model with one GPU
  # use_ddp = true & use_multiple_machine = false: training model with multiple GPUs on a single machine
  # use_ddp = true & use_multiple_machine = true:  training model with GPU on multiple machines

  use_ddp=true
  world_size=$num_workers
  use_multiple_machine=false
  # you can assign GPUs with --device-ids "$device_ids"
  # device_ids="4, 5, 6, 7"
  if $use_multiple_machine ; then
    # suppose you are using Sun GridEngine
    cuda_train_cmd="$cuda_train_cmd --gpu 1 JOB=1:$world_size $dir/$train_dir/logs/job.JOB.log"
  else
    cuda_train_cmd="$cuda_train_cmd --gpu $world_size $dir/$train_dir/logs/train.log"
    echo "multi-gpu single machine ddp"
  fi
  
  $cuda_train_cmd python3 ./chain/train.py \
        --bottleneck-dim $bottleneck_dim \
        --checkpoint=${train_checkpoint:-} \
        --dir $dir/$train_dir \
        --feat-dim $feat_dim \
        --hidden-dim $hidden_dim \
        --is-training true \
        --ivector-dim $ivector_dim \
        --kernel-size-list "$kernel_size_list" \
        --log-level $log_level \
        --output-dim $output_dim \
        --prefinal-bottleneck-dim $prefinal_bottleneck_dim \
        --subsampling-factor-list "$subsampling_factor_list" \
        --train.cegs-dir $dir/$training_eg_dir \
        --train.ddp.init-method $init_method \
        --train.ddp.multiple-machine $use_multiple_machine \
        --train.ddp.world-size $world_size \
        --train.den-fst $dir/den_fsts/${lang}.den.fst \
        --train.dropout-schedule "$dropout_schedule" \
        --train.egs-left-context $egs_left_context \
        --train.egs-right-context $egs_right_context \
        --train.l2-regularize 5e-5 \
        --train.leaky-hmm-coefficient 0.1 \
        --train.lr $lr \
        --train.num-epochs $num_epochs \
        --train.use-ddp $use_ddp \
        --train.valid-cegs-scp $dir/processed_egs/train_subset.scp \
        --train.xent-regularize 0.1 || exit 1;
fi

if [[ $stage -le 7 ]]; then
  echo "inference: computing likelihood"
  for x in test dev; do
    mkdir -p $dir/$train_dir/inference/$x
    if [[ -f $dir/$train_dir/inference/$x/nnet_output.scp ]]; then
      echo "$dir/$train_dir/inference/$x/nnet_output.scp already exists! Skip"
    else
      if $train_ivector; then
        ivector_scp="exp/nnet3${nnet3_affix}/ivectors_${x}_hires/ivector_online.scp"
      fi
      feat_scp="data/${x}_hires/feats.scp"
      if $online_cmvn; then
        if [[ "$feat_type" == "delta" ]]; then
          apply-cmvn-online --spk2utt=ark:data/${x}_hires/spk2utt $dir/raw_egs/global_cmvn.stats \
              scp:data/${x}_hires/feats.scp ark:- | add-deltas --print-args=false --delta-order=2 --delta-window=2 \
              ark:- ark,scp:data/${x}_hires/data/online_cmvn_feats.ark,data/${x}_hires/online_cmvn_feats.scp
        fi
        feat_scp="data/${x}_hires/online_cmvn_feats.scp"
      fi
      best_epoch=$(grep 'best epoch' $dir/$train_dir/best-epoch-info | awk '{print $NF}')
      inference_checkpoint=$dir/$train_dir/epoch-${best_epoch}.pt
      $cuda_inference_cmd --gpu 1 $dir/$train_dir/inference/logs/${x}.log \
        python3 ./chain/inference.py \
        --bottleneck-dim $bottleneck_dim \
        --checkpoint $inference_checkpoint \
        --dir $dir/$train_dir/inference/$x \
        --feat-dim $feat_dim \
        --feats-scp "$feat_scp" \
        --hidden-dim $hidden_dim \
        --is-training false \
        --ivector-dim $ivector_dim \
        --ivector-period $ivector_period \
        --ivector-scp "$ivector_scp" \
        --log-level $log_level \
        --kernel-size-list "$kernel_size_list" \
        --prefinal-bottleneck-dim $prefinal_bottleneck_dim \
        --model-left-context $model_left_context \
        --model-right-context $model_right_context \
        --output-dim $output_dim \
        --save-as-compressed $save_nn_output_as_compressed \
        --subsampling-factor-list "$subsampling_factor_list" || exit 1;
    fi
  done
fi

if [[ $stage -le 8 ]]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  cp $tree_dir/final.mdl $dir/final.mdl
  cp $tree_dir/tree $dir/tree
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi


if [[ $stage -le 9 ]]; then
  echo "decoding"
  for x in test dev; do
    if [[ ! -f $dir/$train_dir/inference/$x/nnet_output.scp ]]; then
      echo "exp/chain/inference/$x/nnet_output.scp does not exist!"
      echo "Please run inference.py first"
      exit 1
    fi
    echo "decoding $x"

    ./local/decode.sh \
      --nj $nj \
      $dir/graph \
      $dir/init/${lang}_trans.mdl \
      $dir/$train_dir/inference/$x/nnet_output.scp \
      $dir/$train_dir/decode_res/$x
  done
fi

if [[ $stage -le 10 ]]; then
  echo "scoring"

  for x in test dev; do
    ./local/score.sh --cmd "$decode_cmd" \
      data/${x}_hires \
      $dir/graph \
      $dir/$train_dir/decode_res/$x || exit 1
  done

  for x in test dev; do
    head $dir/$train_dir/decode_res/$x/scoring_kaldi/best_*
  done
fi
