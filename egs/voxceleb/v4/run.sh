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
mfccdir=`pwd`/mfcc
fbankdir=`pwd`/fbank64
vaddir=`pwd`/fbank64


# The trials file is downloaded by local/make_voxceleb1.pl.
voxceleb1_trials=data/voxceleb1_test/trials
voxceleb1_root=/export/corpora/VoxCeleb1
voxceleb2_root=/export/corpora/VoxCeleb2
nnet_dir=exp/xvector_nnet_1a

res_dir=exp/resnt
tdnn_dir=exp/tdnn

musan_root=/export/corpora/JHU/musan
test_dir=data/voxceleb1_test
train_dir=data/voxceleb1_train

stage=12

if [ $stage -le 0 ]; then
  echo "===================================Data preparing=================================="
  # This script creates data/voxceleb1_test and data/voxceleb1_train.
  # Our evaluation set is the test portion of VoxCeleb1.
  local/make_voxceleb1_trials.pl data
  local/make_voxceleb1.py
  utils/utt2spk_to_spk2utt.pl $test_dir/utt2spk >$test_dir/spk2utt
  utils/validate_data_dir.sh --no-text --no-feats $test_dir
  utils/utt2spk_to_spk2utt.pl $train_dir/utt2spk >$train_dir/spk2utt
  utils/validate_data_dir.sh --no-text --no-feats $train_dir
  exit
fi
#stage = 100
if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  echo "==========================Making Fbank features and VAD============================"
  for name in voxceleb1_train voxceleb1_test; do

#    steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
#      $test_dir exp/make_mfcc $mfccdir
#    utils/fix_data_dir.sh $test_dir
#    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
#      $test_dir exp/make_vad $vaddir
#    utils/fix_data_dir.sh $test_dir
    # Making Fbank features
    steps/make_fbank.sh --write-utt2num-frames true --fbank_config conf/fbank.conf --nj 40 --cmd "$train_cmd" \
        data/${name} exp/make_fbank $fbankdir
    utils/fix_data_dir.sh data/${name}

    # Todo: Making spectrograms if?


    # Todo: Is there any better VAD solutioin?
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
fi


# In this section, we augment the VoxCeleb1 data with reverberation,
# noise, music, and babble, and combine it with the clean data.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the VoxCeleb2 list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/train data/train_reverb
  cp data/train/vad.scp data/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
  rm -rf data/train_reverb
  mv data/train_reverb.new data/train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 16000 $musan_root data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble
fi

if [ $stage -le 3 ]; then
  # Take a random subset of the augmentations
  utils/subset_data_dir.sh data/train_aug 1000000 data/train_aug_1m
  utils/fix_data_dir.sh data/train_aug_1m

  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/train_aug_1m exp/make_mfcc $mfccdir

  # Combine the clean and augmented VoxCeleb2 list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/train_combined data/train_aug_1m data/train
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    data/voxceleb1_train data/voxceleb1_train_no_sil exp/voxceleb1_train_no_sil
  utils/fix_data_dir.sh data/voxceleb1_train_no_sil
fi
# exit

if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 3s (300 frames) per utterance.
  echo "====================Remove utterance that are too short==========================="
  min_len=300
  mv data/voxceleb1_train_no_sil/utt2num_frames data/voxceleb1_train_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/voxceleb1_train_no_sil/utt2num_frames.bak > data/voxceleb1_train_no_sil/utt2num_frames
  utils/filter_scp.pl data/voxceleb1_train_no_sil/utt2num_frames data/voxceleb1_train_no_sil/utt2spk > data/voxceleb1_train_no_sil/utt2spk.new
  mv data/voxceleb1_train_no_sil/utt2spk.new data/voxceleb1_train_no_sil/utt2spk
  utils/fix_data_dir.sh data/voxceleb1_train_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/voxceleb1_train_no_sil/spk2utt > data/voxceleb1_train_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/voxceleb1_train_no_sil/spk2num | utils/filter_scp.pl - data/voxceleb1_train_no_sil/spk2utt > data/voxceleb1_train_no_sil/spk2utt.new
  mv data/voxceleb1_train_no_sil/spk2utt.new data/voxceleb1_train_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/voxceleb1_train_no_sil/spk2utt > data/voxceleb1_train_no_sil/utt2spk

  utils/filter_scp.pl data/voxceleb1_train_no_sil/utt2spk data/voxceleb1_train_no_sil/utt2num_frames > data/voxceleb1_train_no_sil/utt2num_frames.new
  mv data/voxceleb1_train_no_sil/utt2num_frames.new data/voxceleb1_train_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/voxceleb1_train_no_sil
fi


# Stages 6 through 8 are handled in run_xvector.sh
# local/nnet3/xvector/run_resnet.sh --stage $stage --train-stage -1 \
# --data data/voxceleb1_train_no_sil --nnet-dir $res_dir \
# --egs-dir $res_dir/egs


#exit
# Stages 6 through 8 are handled in run_xvector.sh
#local/nnet3/xvector/run_tdnn.sh --stage $stage --train-stage -1 \
#  --data data/voxceleb1_train_no_sil --nnet-dir $tdnn_dir \
#  --egs-dir $tdnn_dir/egs

# if [ $stage -le 9 ]; then
#   # Extract x-vectors for centering, LDA, and PLDA training.
#   sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 12G"  --nj 20 \
#     $res_dir data/voxceleb1_train \
#     $res_dir/xvectors_train

#   # Extract x-vectors used in the evaluation.
#   sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 12G"  --nj 20 \
#     $res_dir data/voxceleb1_test \
#     $res_dir/xvectors_voxceleb1_test
# fi

#if [ $stage -le 9 ]; then
#  # Extract x-vectors used in the evaluation.
#  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj 40 \
#    $tdnn_dir data/voxceleb1_test \
#    $tdnn_dir/xvectors_voxceleb1_test
#fi

if [ $stage -le 10 ]; then
  # Compute the mean vector for centering the evaluation xvectors.
  $train_cmd $res_dir/xvectors_train/log/compute_mean.log \
    ivector-mean scp:$res_dir/xvectors_train/xvector.scp \
    $res_dir/xvectors_train/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd $res_dir/xvectors_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$res_dir/xvectors_train/xvector.scp ark:- |" \
    ark:data/voxceleb1_train/utt2spk $res_dir/xvectors_train/transform.mat || exit 1;

  # Train the PLDA model.
  # subtract global mean and do lda transform before PLDA classification
  $train_cmd $res_dir/xvectors_train/log/plda.log \
    ivector-compute-plda ark:data/voxceleb1_train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$res_dir/xvectors_train/xvector.scp ark:- | transform-vec $res_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $res_dir/xvectors_train/plda || exit 1;
fi


if [ $stage -le 11 ]; then
  $train_cmd exp/scores/log/voxceleb1_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $res_dir/xvectors_train/plda - |" \
    "ark:ivector-subtract-global-mean $res_dir/xvectors_train/mean.vec scp:$res_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $res_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $res_dir/xvectors_train/mean.vec scp:$res_dir/xvectors_voxceleb1_test/xvector.scp ark:- | transform-vec $res_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_test || exit 1;
fi

if [ $stage -le 12 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_voxceleb1_test $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # EER: 3.128%
  # minDCF(p-target=0.01): 0.3258
  # minDCF(p-target=0.001): 0.5003
  #
  # For reference, here's the ivector system from ../v1:
  # EER: 5.329%
  # minDCF(p-target=0.01): 0.4933
  # minDCF(p-target=0.001): 0.6168
fi

# ResNet10 fbank64
# EER: 5.493%
# minDCF(p-target=0.01): 0.4672
# minDCF(p-target=0.001): 0.6370








