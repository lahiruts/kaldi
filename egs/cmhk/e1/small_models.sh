#!/bin/bash
set -e

stage=3
cmd=run.pl

train_stage=-10
get_egs_stage=-10
speed_perturb=true


decode_iter=
decode_nj=40

# training options
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

lang=data/lang_chain_e2e_char
treedir=exp/chain/e2e_char_tree_tied1a
train_set=train_aug_sp
ali_dir=exp/chain/e2e_nnet3_ali_aug_sp
src_dir=exp/chain/e2e_tdnnf_char1b
#dir=exp/chain/tdnnf_char1b
hdim=768
bdim=128
dir=exp/chain/tdnnf_char1b_layer_9_${hdim}_${bdim}

if [ $stage -le 1 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi
 

treedir=exp/chain/e2e_realign_char_tree_tied1a

if [ $stage -le 2 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  #steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 1 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 5000 data/$train_set $lang $ali_dir $treedir
fi


if [ $stage -le 3 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  affine_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=43 name=input

  relu-batchnorm-dropout-layer name=tdnn1 input=Append(-1,0,1) $tdnn_opts dim=$hdim
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=$hdim bottleneck-dim=$bdim time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=$hdim bottleneck-dim=$bdim time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=$hdim bottleneck-dim=$bdim time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=$hdim bottleneck-dim=$bdim time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=$hdim bottleneck-dim=$bdim time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=$hdim bottleneck-dim=$bdim time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=$hdim bottleneck-dim=$bdim time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=$hdim bottleneck-dim=$bdim time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 4 ]; then

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$train_cmd" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 6 \
    --trainer.optimization.num-jobs-initial 9 \
    --trainer.optimization.num-jobs-final 9 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --lat-dir $ali_dir \
    --dir $dir  || exit 1;

fi


graph=$dir/graph_4g


if [ $stage -le 5 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.

  utils/lang/check_phones_compatible.sh \
    data/lang_test/phones.txt $lang/phones.txt


  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $graph
fi

if [ $stage -le 6 ]; then
  #for test_set in test_01_hires test_02_hires test_03_hires test_04_hires val_hires; do
  for test_set in test_01_hires; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj 20 --cmd "$decode_cmd" --max-active 500 \
      $graph data/${test_set} $dir/decode_${test_set}_maxactive_500 || exit 1;
  done
fi


if [ $stage -le 7 ]; then
    steps/online/nnet3/decode.sh --nj 20 --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 --max-active 500 --online_config conf/online.conf \
         $graph data/test_01_hires \
         ${dir}/online_decode_test_01_hires || exit 1;

fi

exit 0

#Realign the data using LFMMI model

src_dir=exp/chain/tdnnf_char1b
ali_dir=exp/chain/nnet3_ali
if [ $stage -le 7 ]; then
#Generate alignment lattices from the final model
  mv $src_dir/frame_subsampling_factor $src_dir/frame_subsampling_factor.bk
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $src_dir/num_jobs) || exit 1;
  steps/nnet3/align_lats.sh --nj $nj --cmd "$train_cmd" --generate_ali_from_lats true data/$train_set \
    data/lang $src_dir $ali_dir
  rm $ali_dir/fsts.*.gz # save space
  mv $src_dir/frame_subsampling_factor.bk $src_dir/frame_subsampling_factor

fi


exit 0



for val in 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400
do
echo $val

if [ $stage -le 4 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/lang/check_phones_compatible.sh \
    data/lang_test/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 \
    --iter ${val} \
    data/lang_test $dir $treedir/graph_4g_${val} || exit 1;

fi

if [ $stage -le 5 ]; then
  frames_per_chunk=150
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      #data_affix=$(echo $data | sed s/test_//)
      nspk=$(wc -l <data/${data}/spk2utt)
      for lmtype in 4g; do
        steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --iter ${val} \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          $treedir/graph_${lmtype}_${val}  data/${data} ${dir}/decode_${val}_${lmtype}_${data} || exit 1
      done
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

echo "Done. Date: $(date). Results:"
done

train_set=train
test_sets="test"
ali_dir=exp/chain/e2e_nnet3_ali
treedir=exp/chain/tri4_cd_tree_sp
lang=data/lang_chain
src_dir=exp/chain/e2e_tdnnf_char1b

iter=200
#ali_dir=exp/chain/e2e_nnet3_ali_${iter}
#if [ $stage -le 6 ]; then
#    nj=$(cat $src_dir/num_jobs) || exit 1;
#    steps/nnet3/align.sh --nj $nj --cmd "$train_cmd" --iter $iter data/$train_set \
#    data/lang $src_dir $ali_dir
#fi


for iter in 200 400 600 800 1000 1200 1400
do
echo $iter
ali_dir=exp/chain/e2e_nnet3_ali_${iter}

if [ $stage -le 6 ]; then
  mv $src_dir/frame_subsampling_factor $src_dir/frame_subsampling_factor.bk 
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $src_dir/num_jobs) || exit 1;
  steps/nnet3/align_lats.sh --nj $nj --cmd "$train_cmd" --iter $iter --generate_ali_from_lats true data/$train_set \
    data/lang $src_dir $ali_dir 
  rm $ali_dir/fsts.*.gz # save space
  mv $src_dir/frame_subsampling_factor.bk $src_dir/frame_subsampling_factor
fi

done

exit 0

ali_dir=exp/chain/e2e_nnet3_ali_200
if [ $stage -le 7 ]; then
  nj=$(cat $src_dir/num_jobs) 
  # If generate_alignments is true, ali.*.gz is generated in lats dir
  $cmd JOB=1:$nj $ali_dir/log/generate_alignments.JOB.log \
    lattice-best-path --acoustic-scale=$acoustic_scale "ark:gunzip -c $ali_dir/lat.JOB.gz |" \
    ark:/dev/null "ark:|gzip -c >$ali_dir/ali.JOB.gz" || exit 1;
fi

#local/chain/compare_wer.sh $dir
