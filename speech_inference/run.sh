#!/bin/bash


# Set this to somewhere where you want to put your data, or where
# someone else has already put it.  You'll want to change this
# if you're not on the CLSP grid.
data=/workspace/jupyter/data/

# base url for downloads.
data_url=www.openslr.org/resources/12
lm_url=www.openslr.org/resources/11

stage=22
i=
decoded=


. ./cmd.sh
. ./path.sh
. parse_options.sh

# you might not want to do this for interactive shells.
set -e


if [ $stage -le 1 ]; then
  # download the data.  Note: we're using the 100 hour setup for
  # now; later in the script we'll download more and use it to train neural
  # nets.
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
    local/download_and_untar.sh $data $data_url $part
  done


  # download the LM resources
  local/download_lm.sh $lm_url data/local/lm
fi

if [ $stage -le 2 ]; then
  # format the data as Kaldi data directories
  for part in dev-clean test-clean dev-other test-other train-clean-100; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done
fi

## Optional text corpus normalization and LM training
## These scripts are here primarily as a documentation of the process that has been
## used to build the LM. Most users of this recipe will NOT need/want to run
## this step. The pre-built language models and the pronunciation lexicon, as
## well as some intermediate data(e.g. the normalized text used for LM training),
## are available for download at http://www.openslr.org/11/
#local/lm/train_lm.sh $LM_CORPUS_ROOT \
#  data/local/lm/norm/tmp data/local/lm/norm/norm_texts data/local/lm

## Optional G2P training scripts.
## As the LM training scripts above, this script is intended primarily to
## document our G2P model creation process
#local/g2p/train_g2p.sh data/local/dict/cmudict data/local/lm

if [ $stage -le 3 ]; then
  # when the "--stage 3" option is used below we skip the G2P steps, and use the
  # lexicon we have already downloaded from openslr.org/11/
  local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
   data/local/lm data/local/lm data/local/dict_nosp

  utils/prepare_lang.sh data/local/dict_nosp \
   "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

  local/format_lms.sh --src-dir data/lang_nosp data/local/lm
fi

if [ $stage -le 4 ]; then
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_tglarge
  utils/build_const_arpa_lm.sh data/local/lm/lm_fglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_fglarge
fi

if [ $stage -le 5 ]; then
  mfccdir=mfcc
  # spread the mfccs over various machines, as this data-set is quite large.
  if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
    mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
    utils/create_split_dir.pl /export/b{02,11,12,13}/$USER/kaldi-data/egs/librispeech/s5/$mfcc/storage \
     $mfccdir/storage
  fi
fi


if [ $stage -le 6 ]; then
  for part in dev_clean test_clean dev_other test_other train_clean_100; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done
fi

if [ $stage -le 7 ]; then
  # Make some small data subsets for early system-build stages.  Note, there are 29k
  # utterances in the train_clean_100 directory which has 100 hours of data.
  # For the monophone stages we select the shortest utterances, which should make it
  # easier to align the data from a flat start.

  utils/subset_data_dir.sh --shortest data/train_clean_100 2000 data/train_2kshort
  utils/subset_data_dir.sh data/train_clean_100 5000 data/train_5k
  utils/subset_data_dir.sh data/train_clean_100 10000 data/train_10k
fi

if [ $stage -le 8 ]; then
  # train a monophone system
  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
                      data/train_2kshort data/lang_nosp exp/mono

  # decode using the monophone model
  (
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
                     exp/mono exp/mono/graph_nosp_tgsmall
  )&
fi

if [ $stage -le 9 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
                    data/train_5k data/lang_nosp exp/mono exp/mono_ali_5k

  # train a first delta + delta-delta triphone system on a subset of 5000 utterances
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
                        2000 10000 data/train_5k data/lang_nosp exp/mono_ali_5k exp/tri1

  # decode using the tri1 model
  (
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
                     exp/tri1 exp/tri1/graph_nosp_tgsmall
  )&
fi

if [ $stage -le 10 ]; then
  steps/align_si.sh --nj 10 --cmd "$train_cmd" \
                    data/train_10k data/lang_nosp exp/tri1 exp/tri1_ali_10k


  # train an LDA+MLLT system.
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
                          --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
                          data/train_10k data/lang_nosp exp/tri1_ali_10k exp/tri2b

  # decode using the LDA+MLLT model
  (
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
                     exp/tri2b exp/tri2b/graph_nosp_tgsmall
  )&
fi

if [ $stage -le 11 ]; then
  # Align a 10k utts subset using the tri2b model
  steps/align_si.sh  --nj 10 --cmd "$train_cmd" --use-graphs true \
                     data/train_10k data/lang_nosp exp/tri2b exp/tri2b_ali_10k

  # Train tri3b, which is LDA+MLLT+SAT on 10k utts
  steps/train_sat.sh --cmd "$train_cmd" 2500 15000 \
                     data/train_10k data/lang_nosp exp/tri2b_ali_10k exp/tri3b

  # decode using the tri3b model
  (
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
                     exp/tri3b exp/tri3b/graph_nosp_tgsmall
  )&
fi

if [ $stage -le 12 ]; then
  # align the entire train_clean_100 subset using the tri3b model
  steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
    data/train_clean_100 data/lang_nosp \
    exp/tri3b exp/tri3b_ali_clean_100

  # train another LDA+MLLT+SAT system on the entire 100 hour subset
  steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
                      data/train_clean_100 data/lang_nosp \
                      exp/tri3b_ali_clean_100 exp/tri4b

  # decode using the tri4b model
  (
    utils/mkgraph.sh data/lang_nosp_test_tgsmall \
                     exp/tri4b exp/tri4b/graph_nosp_tgsmall
  )&
fi

if [ $stage -le 13 ]; then
  # Now we compute the pronunciation and silence probabilities from training data,
  # and re-create the lang directory.
  steps/get_prons.sh --cmd "$train_cmd" \
                     data/train_clean_100 data/lang_nosp exp/tri4b
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
                                  data/local/dict_nosp \
                                  exp/tri4b/pron_counts_nowb.txt exp/tri4b/sil_counts_nowb.txt \
                                  exp/tri4b/pron_bigram_counts_nowb.txt data/local/dict

  utils/prepare_lang.sh data/local/dict \
                        "<UNK>" data/local/lang_tmp data/lang
  local/format_lms.sh --src-dir data/lang data/local/lm

  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_tglarge.arpa.gz data/lang data/lang_test_tglarge
  utils/build_const_arpa_lm.sh \
    data/local/lm/lm_fglarge.arpa.gz data/lang data/lang_test_fglarge

  # decode using the tri4b model with pronunciation and silence probabilities
  (
    utils/mkgraph.sh \
      data/lang_test_tgsmall exp/tri4b exp/tri4b/graph_tgsmall
  )&
fi

# Prepare 360 hours (will be unlabelled pool)
if [ $stage -le 20 ]; then
  local/download_and_untar.sh $data $data_url train-clean-360

  for part in train-clean-360; do
   #  use underscore-separated names in data directories.
   local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
 done

 for part in train_clean_360; do
   steps/make_mfcc.sh --cmd "$train_cmd" --nj 40 data/$part exp/make_mfcc/$part $mfccdir
   steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
 done

fi

# To run faster, as first try, sample n=10000 utterances
if [ $stage -eq 21 ]; then
  n=10000

  for datadir in train_clean_360; do
   utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done

 for datadir in train_clean_360; do
    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires || exit 1;
    steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
    utils/fix_data_dir.sh data/${datadir}_hires
 done
 utils/subset_data_dir.sh data/train_clean_360_hires $n data/train_clean_360_sub
fi


# Precise which unlabelled dataset you want to use
unlabelled_set=train_clean_360

# Initial Training
if [ $stage -eq 22 ]; then
  # Split labelled set in two: first one will be used to train nnet3, the other
  # will be decoded and used to train the codi.
  rm -r data/train_clean_100/split2 || true
  rm -r data/train_clean_100_split_1_hires || true
  rm -r data/train_clean_100_split_2_hires || true

  utils/split_data.sh data/train_clean_100 2
  mv data/train_clean_100/split2/1/ data/train_clean_100_split_1_hires
  mv data/train_clean_100/split2/2/ data/train_clean_100_split_2_hires

  for datadir in train_clean_100_split_2 train_clean_100_split_1; do
      steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
       --cmd "$train_cmd" data/${datadir}_hires || exit 1;
      steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
      utils/fix_data_dir.sh data/${datadir}_hires
  done

  local/chain/run_tdnn.sh --unlabelled_set $unlabelled_set --njobs 70 --unlabelled yes --train_set train_clean_100_split_1 
  echo "Training and decoding unlabelled set done, now decoding second labelled set"
  local/chain/run_tdnn.sh --unlabelled_set train_clean_100_split_2 --stage 18 --unlabelled no
fi

# Iterative Retraining
if [ $stage -eq 23 ]; then

    echo "Retraining: Iteration $i"
    
    rm -r data/retrain || true
    rm -r data/retrain_sp || true
    rm -r data/retrain_sp_hires || true
    rm -r exp/tri3b_ali_retrain_sp || true

     # Prepare folders
     rm -r exp/tri3b_ali_${unlabelled_set}_trusted_hires_sp || true
     rm -r data/${unlabelled_set}_trusted_hires_sp || true
     rm -r data/${unlabelled_set}_trusted_hires || true
     cp -r data/${unlabelled_set}_hires data/${unlabelled_set}_hires_temp || true
     cp -r data/${unlabelled_set}_untrusted_hires data/${unlabelled_set}_hires_temp || true
     rm -r data/${unlabelled_set}_untrusted_hires || true

     rm data/${unlabelled_set}_hires_temp/text
     if [ $decoded -eq 0 ]; then
        cp data/${unlabelled_set}/text data/${unlabelled_set}_hires_temp/text
        echo "Succefully copied true text"
     else
        cp data/${unlabelled_set}_hires/text data/${unlabelled_set}_hires_temp/text
        echo "Succefully copied decoded text"
     fi

     # Split initial dataset in two: trusted and untrusted using IDS outputted from CoDi
     python3 local/chain/prepare_datasets.py --path_data data/${unlabelled_set}_hires_temp

     rm -r data/${unlabelled_set}_hires_temp
     
     utils/utt2spk_to_spk2utt.pl data/${unlabelled_set}_trusted_hires/utt2spk > data/${unlabelled_set}_trusted_hires/spk2utt
     utils/utt2spk_to_spk2utt.pl data/${unlabelled_set}_untrusted_hires/utt2spk > data/${unlabelled_set}_untrusted_hires/spk2utt

     for datadir in ${unlabelled_set}_untrusted ${unlabelled_set}_trusted; do
        utils/fix_data_dir.sh data/${datadir}_hires
     done

    #local/chain/tuning/run_tdnn_wsj_rm_1a.sh --train_set ${unlabelled_set}_trusted_hires --unlabelled_set ${unlabelled_set}_untrusted --iteration $i
    
    utils/combine_data.sh \
    data/retrain data/train_all data/${unlabelled_set}_trusted_hires
    
    local/chain/run_tdnn.sh --unlabelled_set $unlabelled_set --njobs 70 --train_set retrain --iteration $i --decoded $decoded
    
    # utils/copy_data_dir.sh data/retrain data/train_all
    # rm -r data/retrain || true

fi

if [ $stage -eq 24 ]; then

for datadir in train_clean_360_sub; do

wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
unzip rirs_noises.zip

foreground_snrs="20:10:15:5:0"
background_snrs="20:10:15:5:0"
num_reps=1
rvb_opts=()
# This is the config for the system using simulated RIRs and point-source noises
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
    rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
    rvb_opts+=(--noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list)
python3 steps/data/reverberate_data_dir.py \
      "${rvb_opts[@]}" \
      --prefix "rev" \
      --foreground-snrs $foreground_snrs \
      --background-snrs $background_snrs \
      --speech-rvb-probability 1 \
      --pointsource-noise-addition-probability 1 \
      --isotropic-noise-addition-probability 1 \
      --num-replications $num_reps \
      --max-noises-per-minute 3 \
      --source-sampling-rate 16000 \
      data/${datadir}_hires data/${datadir}_rvb
done

fi


if [ $stage -eq 25 ]; then

for i in 25
do
 let n="1000*i"
 echo " Training with $n utterances "
 rm -r data/train_clean_360_sub_hires || true
 rm -r exp/tri3b_ali_train_clean_360_sub_hires_sp || true
 rm -r data/train_clean_360_sub_hires_sp || true
 rm -r data/train_clean_360_sub_hires_sp_hires || true

 utils/subset_data_dir.sh data/train_clean_360_hires $n data/train_clean_360_sub_hires

 for datadir in train_clean_360_sub; do
        steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
         --cmd "$train_cmd" data/${datadir}_hires || exit 1;
        steps/compute_cmvn_stats.sh data/${datadir}_hires || exit 1;
        utils/fix_data_dir.sh data/${datadir}_hires
  done


  local/chain/tuning/run_tdnn_wsj_rm_1a.sh --train_set train_clean_360_sub_hires --unlabelled_set ${unlabelled_set} --iteration $i
done
fi

# Wait for decodings in the background
wait
