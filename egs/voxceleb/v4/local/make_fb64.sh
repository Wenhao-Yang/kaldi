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

fbankdir=`pwd`/data/vox1_fb64/feats
vaddir=`pwd`/data/vox1_fb64/vads

# The trials file is downloaded by local/make_voxceleb1.pl.
train_dir=data/vox1_fb64/train
test_dir=data/vox1_fb64/test
stage=0

if [ $stage -le 0 ]; then
  echo "===================================Data preparing=================================="
  # This script creates data/voxceleb1_test and data/voxceleb1_train.
  # Our evaluation set is the test portion of VoxCeleb1.
  local/make_voxceleb1_trials.pl data
  local/make_voxceleb1.py $train_dir $test_dir
  utils/utt2spk_to_spk2utt.pl $test_dir/utt2spk >$test_dir/spk2utt
  utils/validate_data_dir.sh --no-text --no-feats $test_dir
  utils/utt2spk_to_spk2utt.pl $train_dir/utt2spk >$train_dir/spk2utt
  utils/validate_data_dir.sh --no-text --no-feats $train_dir
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  echo "==========================Making Fbank features and VAD============================"
  for name in $train_dir $test_dir; do
    steps/make_fbank.sh --write-utt2num-frames true --fbank_config conf/fbank.conf --nj 4 --cmd "$train_cmd" \
        ${name} exp/make_vox1_fbank64 $fbankdir
    utils/fix_data_dir.sh ${name}

    # Todo: Making spectrograms if?
    # Todo: Is there any better VAD solutioin?
    sid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" ${name} exp/make_vox1_fbank64 $vaddir
    utils/fix_data_dir.sh ${name}
  done
fi

if [ $stage -le 2 ]; then
  echo "=====================================CMVN========================================"
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.

  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 4 --cmd "$train_cmd" $train_dir data/vox1_fb64/train_no_sli data/vox1_fb64/feats_no_sil
  utils/fix_data_dir.sh data/vox1_fb64/train_no_sli

  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 4 --cmd "$train_cmd" $test_dir data/vox1_fb64/test_no_sli data/vox1_fb64/feats_no_sil
  utils/fix_data_dir.sh data/vox1_fb64/test_no_sli
fi






