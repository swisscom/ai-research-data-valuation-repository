#!/bin/bash

# This script uses weight transfer as a transfer learning method to transfer
# already trained neural net model on wsj to rm.
#
# Model preparation: The last layer (prefinal and output layer) from
# already-trained wsj model is removed and 3 randomly initialized layer
# (new tdnn layer, prefinal, and output) are added to the model.
#
# Training: The transferred layers are retrained with smaller learning-rate,
# while new added layers are trained with larger learning rate

set -e


# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
iteration=

decode_nj=10
train_set=
unlabelled_set=
gmm=tri3b
affix=1d
tree_affix=

frames_per_eg=150,110,100
remove_egs=true
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

decode_iter=
# configs for transfer learning
src_mdl=exp/chain_cleaned/tdnn_1d_sp/final.mdl # Input chain model
                                                   # trained on source dataset (wsj).
                                                   # This model is transfered to the target domain.

src_mfcc_config=conf/mfcc_hires.conf # mfcc config used to extract higher dim
                                                  # mfcc features for ivector and DNN training
                                                  # in the source domain.
src_ivec_extractor_dir=exp/nnet3_cleaned/extractor  # Source ivector extractor dir used to extract ivector for
                         # source data. The ivector for target data is extracted using this extractor.
                         # It should be nonempty, if ivector is used in the source model training.

common_egs_dir=
primary_lr_factor=0.25 # The learning-rate factor for transferred layers from source
                       # model. e.g. if 0, the paramters transferred from source model
                       # are fixed.
                       # The learning-rate factor for new added layers is 1.0.



# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


nnet_affix=_new_wsj_${iteration}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

required_files="$src_mfcc_config $src_mdl"
use_ivector=false
ivector_dim=$(nnet3-am-info --print-args=false $src_mdl | grep "ivector-dim" | cut -d" " -f2)
if [ -z $ivector_dim ]; then ivector_dim=0 ; fi

if [ ! -z $src_ivec_extractor_dir ]; then
  if [ $ivector_dim -eq 0 ]; then
    echo "$0: Source ivector extractor dir '$src_ivec_extractor_dir' is specified "
    echo "but ivector is not used in training the source model '$src_mdl'."
  else
    required_files="$required_files $src_ivec_extractor_dir/final.dubm $src_ivec_extractor_dir/final.mat $src_ivec_extractor_dir/final.ie"
    use_ivector=true
  fi
else
  if [ $ivector_dim -gt 0 ]; then
    echo "$0: ivector is used in training the source model '$src_mdl' but no "
    echo " --src-ivec-extractor-dir option as ivector dir for source model is specified." && exit 1;
  fi
fi

for f in $required_files; do
  if [ ! -f $f ]; then
    echo "$0: no such file $f." && exit 1;
  fi
done

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 4" if you have already
# run those things.

local/nnet3/run_ivector_common.sh --stage $stage \
                                --train-set $train_set \
                                --gmm $gmm \
                                --num-threads-ubm 6 --num-processes 3 \
                                --nnet3-affix "$nnet_affix" || exit 1;

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain${nnet_affix}
lat_dir=exp/chain${nnet_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet_affix}/tdnn${affix:+_$affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet_affix}/ivectors_${train_set}_sp_hires


local/chain/run_chain_common.sh --stage $stage \
                                --gmm-dir $gmm_dir \
                                --ali-dir $ali_dir \
                                --lores-train-data-dir ${lores_train_data_dir} \
                                --lang $lang \
                                --lat-dir $lat_dir \
                                --num-leaves 7000 \
                                --tree-dir $tree_dir || exit 1;


if [ $stage -le 7 ]; then
  echo "$0: Create neural net configs using the xconfig parser for";
  echo " generating new layers, that are specific to rm. These layers ";
  echo " are added to the transferred part of the wsj network.";
  num_targets=$(tree-info --print-args=false $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  mkdir -p $dir
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  relu-renorm-layer name=tdnn-target input=Append(tdnnf17.batchnorm@-3,tdnnf17.batchnorm) dim=1536
  ## adding the layers for chain branch
  relu-renorm-layer name=prefinal-chain input=tdnn-target dim=1536 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5
  relu-renorm-layer name=prefinal-xent input=tdnn-target dim=1536 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --existing-model $src_mdl \
    --xconfig-file  $dir/configs/network.xconfig  \
    --config-dir $dir/configs/

  # Set the learning-rate-factor to be primary_lr_factor for transferred layers "
  # and adding new layers to them.
  $train_cmd $dir/log/generate_input_mdl.log \
    nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor" $src_mdl - \| \
      nnet3-init --srand=1 - $dir/configs/final.config $dir/input.raw  || exit 1;
fi

if [ $stage -le 8 ]; then
  echo "$0: generate egs for chain to train new model on rm dataset."
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/rm-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi
  ivector_dir=
  if $use_ivector; then ivector_dir="exp/nnet2${nnet_affix}/ivectors" ; fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir $train_ivector_dir \
    --chain.xent-regularize $xent_regularize \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=200" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --num_utts_subset 100" \
    --egs.chunk-width 150 \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 2500000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 1 \
    --trainer.optimization.num-jobs-final 1 \
    --trainer.optimization.initial-effective-lrate 0.005 \
    --trainer.optimization.final-effective-lrate 0.0005 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs true \
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir || exit 1;
fi

if [ $stage -le 9 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  ivec_opt=""
  if $use_ivector;then
    ivec_opt="--online-ivector-dir exp/nnet3/ivectors_test_clean_hires"
  fi
  echo "$dir"
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  #use_ivector=false
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test_tgsmall $dir $dir/graph

  for decode_set in test_clean dev_clean; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts --use_gpu true --num_threads 10 \
          --online-ivector-dir exp/nnet3${nnet_affix}/ivectors_${decode_set}_hires \
          $dir/graph data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_tgsmall || exit 1
     steps/lmrescore_const_arpa.sh \
          --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
          data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_{tgsmall,fglarge} || exit 1
      ) || touch $dir/.error &
  done

fi

if [ $stage -eq 10 ]; then
  for decode_set in $unlabelled_set; do
      (
      steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 20 \
      data/${decode_set}_hires exp/nnet3${nnet_affix}/extractor \
      exp/nnet3${nnet_affix}/ivectors_${decode_set}_hires || exit 1;

      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts --use_gpu true --num_threads 10 \
          --online-ivector-dir exp/nnet3${nnet_affix}/ivectors_${decode_set}_hires \
          $dir/graph data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_tgsmall || exit 1
     steps/lmrescore_const_arpa.sh \
          --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
          data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_{tgsmall,fglarge} || exit 1
      ) || touch $dir/.error &
    done
    wait
    if [ -f $dir/.error ]; then
      echo "$0: something went wrong in decoding"
      exit 1
    fi
fi

if [ $stage -eq 11 ]; then

   for x in $dir/decode_${unlabelled_set}_fglarge; do (

   # Make sure to empty the files first
   rm $x/prob_avg.txt || true
   rm $x/mergedfile.txt || true
   rm $x/decoded_text.txt || true
   rm $x/mergedlm.txt || true
   rm $x/mergedac.txt || true
   rm $x/nbest_text.txt || true
   rm $x/mergedctm.txt || true

   #decode_nj=2

    for (( i=1; i<=$decode_nj; i++ )) do (
      lattice-to-ctm-conf --acoustic-scale=0.1 --decode-mbr=true ark:'gunzip -c '$x'/lat.'$i'.gz |' $x/$i.ctm 2>> $x/prob_avg.txt
      lattice-mbr-decode --acoustic-scale=0.1 ark:'gunzip -c '$x'/lat.'$i'.gz |' ark:$x/1.tra ark:/dev/null ark,t:$x/$i.sau
      lattice-best-path ark:'gunzip -c '$x'/lat.'$i'.gz |' 'ark,t:|utils/int2sym.pl -f 2- '$dir'/graph/words.txt >> '$x'/decoded_text.txt' > /dev/null 2>&1 )
      lattice-to-nbest --acoustic-scale=0.1 --n=5 ark:'gunzip -c '$x'/lat.'$i'.gz |' ark:$x/nbest.lats
      nbest-to-linear ark:$x/nbest.lats ark,t:$x/1.ali 'ark,t:|utils/int2sym.pl -f 2- '$dir'/graph/words.txt >> '$x'/nbest_text.txt'  ark,t:$x/lmcost.$i  ark,t:$x/accost.$i
    done

    cat $x/*.sau > $x/mergedfile.txt
    cat $x/lmcost.* > $x/mergedlm.txt
    cat $x/accost.* > $x/mergedac.txt    
    cat $x/*.ctm > $x/mergedctm.txt

    python3 local/chain/prepare_filtering.py --path $x
    #python3 filtering/score_computation.py
    python3 filtering/output_true_ids.py

    cp $x/decoded_text.txt data/${unlabelled_set}_hires/text

    )
   done
fi

if [ $stage -eq 12 ]; then
  cp $dir/final.mdl $src_mdl
fi

wait;
exit 0;
