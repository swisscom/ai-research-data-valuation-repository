#!/bin/bash
set -e

# configs for 'chain'
stage=0
decode_nj=10
train_set=train_clean_100
unlabelled_set=
gmm=tri3b
nnet3_affix=_cleaned_he
njobs=


echo " $unlabelled_set "

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1d
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# TDNN options
frames_per_eg=150,110,100
remove_egs=true
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

test_online_decoding=true  # if true, it will run the last decoding stage.

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

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.

if [ $stage -le 1 ]; then
local/nnet3/run_ivector_common.sh --stage $stage \
                                --train-set $train_set \
                                --gmm $gmm \
                                --num-threads-ubm 6 --num-processes 3 \
                                --nnet3-affix "$nnet3_affix" || exit 1;
fi

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix:+_$affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

# if we are using the speed-perturbed data we need to generate
# alignments for it.

if [ $stage -le 2 ]; then
local/chain/run_chain_common.sh --stage $stage \
                              --gmm-dir $gmm_dir \
                              --ali-dir $ali_dir \
                              --lores-train-data-dir ${lores_train_data_dir} \
                              --lang $lang \
                              --lat-dir $lat_dir \
                              --num-leaves 7000 \
                              --tree-dir $tree_dir || exit 1;
fi

if [ $stage -le 14 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  affine_opts="l2-regularize=0.008 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.008 dropout-proportion=0.0 bypass-scale=0.75"
  linear_opts="l2-regularize=0.008 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.008"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=1536
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf16 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf17 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1536 small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 15 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{09,10,11,12}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 2500000 \
    --trainer.num-epochs 2 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 2 \
    --trainer.optimization.initial-effective-lrate 0.00015 \
    --trainer.optimization.final-effective-lrate 0.000015 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;

fi

graph_dir=$dir/graph_tgsmall

if [ $stage -le 16 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov data/lang_test_tgsmall $dir $graph_dir
  # remove <UNK> from the graph, and convert back to const-FST.
  fstrmsymbols --apply-to-output=true --remove-arcs=true "echo 3|" $graph_dir/HCLG.fst - | \
    fstconvert --fst_type=const > $graph_dir/temp.fst
  mv $graph_dir/temp.fst $graph_dir/HCLG.fst
fi


iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi

# First evaluation of the  model

if [ $stage -le 17 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in test_clean test_other dev_clean dev_other; do
      (
       steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
       data/${decode_set}_hires exp/nnet3${nnet3_affix}/extractor \
       exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires || exit 1
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts --use_gpu true --num_threads 20 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_tgsmall || exit 1
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


if [ $stage -le 18 ]; then
  for decode_set in $unlabelled_set; do
      (
       steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
       data/${decode_set}_hires exp/nnet3${nnet3_affix}/extractor \
       exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires || exit 1
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts --use_gpu true --num_threads 20 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_tgsmall || exit 1
      ) || touch $dir/.error &
    done
    wait
    if [ -f $dir/.error ]; then
      echo "$0: something went wrong in decoding"
      exit 1
    fi
fi

if [ $stage -le 19 ]; then

for x in $dir/decode_${unlabelled_set}_fglarge; do (
    # Make sure to empty the files first
    rm $x/prob_avg.txt || true
    rm $x/mergedfile.txt || true
    rm $x/decoded_text.txt || true
    rm $x/mergedlm.txt || true
    rm $x/mergedac.txt || true
    rm $x/nbest_text.txt || true
    rm $x/mergedctm.txt || true

    decode_nj=10

    for (( i=1; i<=$decode_nj; i++ )) do (
     lattice-to-ctm-conf --acoustic-scale=0.1 --decode-mbr=true ark:'gunzip -c '$x'/lat.'$i'.gz |' $x/$i.ctm 2>> $x/prob_avg.txt
     lattice-mbr-decode --acoustic-scale=0.1 ark:'gunzip -c '$x'/lat.'$i'.gz |' ark:$x/1.tra ark:/dev/null ark,t:$x/$i.sau
     lattice-best-path --acoustic-scale=0.1 ark:'gunzip -c '$x'/lat.'$i'.gz |' 'ark,t:|utils/int2sym.pl -f 2- '$graph_dir'/words.txt >> '$x'/decoded_text.txt' > /dev/null 2>&1 )
     lattice-to-nbest --acoustic-scale=0.1 --n=5 ark:'gunzip -c '$x'/lat.'$i'.gz |' ark:$x/nbest.lats
     nbest-to-linear ark:$x/nbest.lats ark,t:$x/1.ali 'ark,t:|utils/int2sym.pl -f 2- '$graph_dir'/words.txt >> '$x'/nbest_text.txt'  ark,t:$x/lmcost.$i  ark,t:$x/accost.$i
    done

    cat $x/*.sau > $x/mergedfile.txt
    cat $x/lmcost.* > $x/mergedlm.txt
    cat $x/accost.* > $x/mergedac.txt
    cat $x/*.ctm > $x/mergedctm.txt

    python3 local/chain/prepare_filtering.py --path $x --unlabelled no

    cp $x/decoded_text.txt data/${unlabelled_set}_hires/text

    )
   done
fi



if [ $stage -eq 20 ]; then
   #decode_${unlabelled_set}_fglarge
   for x in $dir/decode_test_other_fglarge; do (

   # Make sure that files are empty.
   rm $x/prob_avg.txt || true
   rm $x/mergedfile.txt || true
   rm $x/decoded_text.txt || true

    decode_nj=50
    # Extract useful infos about each decoded utterance.
    #for i in {1..$decode_nj};  do (
    for (( i=1; i<=$decode_nj; i++ )) do (
      lattice-to-ctm-conf --acoustic-scale=0.1 --decode-mbr=true ark:'gunzip -c '$x'/lat.'$i'.gz |' $x/1.ctm 2>> $x/prob_avg.txt
      lattice-mbr-decode --acoustic-scale=0.1 ark:'gunzip -c '$x'/lat.'$i'.gz |' ark:$x/1.tra ark:/dev/null ark,t:$x/$i.sau
      lattice-best-path ark:'gunzip -c '$x'/lat.'$i'.gz |' 'ark,t:|utils/int2sym.pl -f 2- '$graph_dir'/words.txt >> '$x'/decoded_text.txt' > /dev/null 2>&1 )
    done

    cat $x/*.sau > $x/mergedfile.txt

    # Steps to classify the datapoints and output the trusted ones.
    python3 local/chain/prepare_filtering.py --path $x 
    #python3 filtering/score_computation.py
    #python3 filtering/output_true_ids.py

    # Move the decoded text in data folder for the next retraining iteration.
    cp $x/decoded_text.txt data/${unlabelled_set}_hires/text

    )
   done
fi


exit 0;
