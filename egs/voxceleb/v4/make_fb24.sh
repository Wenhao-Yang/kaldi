#!/bin/bash
# Yangwenhao 2019-07-15 20:27
#
#
# In this recipe, I want to implement DeepSpeaker using Kaldi nnet3. Training the Resnet-34 with voxceleb1 dataset.
#
# There should be following steps:
#
#   Making fbanks ,VAD and augment the dataset with/without noisy.
#   Train a Resnet-34 model to classification
#   Extract x-vector from trained Resnet-34.
#   Scoring with PLDA and Cosine.
#
# Results (mostly equal error-rates) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e

fbankdir=`pwd`/data/vox1_fb24/feats
vaddir=`pwd`/data/vox1_fb24/vads


# The trials file is downloaded by local/make_voxceleb1.pl.
train_dir=data/vox1_fb24/train
test_dir=data/vox1_fb24/test
stage=2

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  echo "==========================Making Fbank features and VAD============================"
  for name in $train_dir $test_dir; do

#    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
#      $test_dir exp/make_mfcc $mfccdir
#    utils/fix_data_dir.sh $test_dir
#    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
#      $test_dir exp/make_vad $vaddir
#    utils/fix_data_dir.sh $test_dir
    # Making Fbank features
    steps/make_fbank.sh --write-utt2num-frames true --fbank_config conf/fbank.conf --nj 4 --cmd "$train_cmd" \
        ${name} exp/make_vox1_fbank24 $fbankdir
    utils/fix_data_dir.sh ${name}

    # Todo: Making spectrograms if?

    # Todo: Is there any better VAD solutioin?
    sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" ${name} exp/make_vox1_fbank24 $vaddir
    utils/fix_data_dir.sh ${name}
  done
fi

if [ $stage -le 2 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 4 --cmd "$train_cmd" $train_dir data/vox1_fb24/train_no_sli data/vox1_fb24/feats_no_sil
  utils/fix_data_dir.sh data/vox1_fb24/train_no_sli
fi
if [ $stage -le 3 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 4 --cmd "$train_cmd" $test_dir data/vox1_fb24/test_no_sli data/vox1_fb24/feats_no_sil
  utils/fix_data_dir.sh data/vox1_fb24/test_no_sli
fi






