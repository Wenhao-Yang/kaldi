#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

# sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe


# The trials file is downloaded by local/make_timit_v2.pl.
# timit_trials=data/timit_test/trials
# timit_root=/export/corpora/VoxCeleb1
# voxceleb2_root=/export/corpora/VoxCeleb2
timit_root=/data/timit


# train=data/train
# test=data/test

# train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_fb24_dnn_20
# test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_fb24_dnn_20
# datafrom=py24_dnn

train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_fb40_dnn_20
test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_fb40_dnn_20
datafrom=py40_dnn

timit_trials=${test}/trials

stage=1

if [ $stage -le 0 ]; then
  # if [ ! -d ${train} ]; then
  #   mkdir -p ${train}
  #   cp ${timit_root}/train/* ${train}
  # fi

  # if [ ! -d ${test} ]; then
  #   mkdir -p ${test}
  #   cp ${timit_root}/test/* ${test}
  # fi
  local/timit_data_prep.sh $timit_root || exit 1
  local/timit_format_data.sh

  for name in ${train} ${test} ; do
    utils/fix_data_dir.sh ${name}
    utils/validate_data_dir.sh --no-text --no-feats ${name}
  done

fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in ${train} ${test}; do
    # steps/make_mfcc.sh --write-utt2num-frames true \
    #   --mfcc-config conf/mfcc.conf --nj 12 --cmd "$train_cmd" \
    #   data/${name} exp/make_mfcc $mfccdir
    # utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
      ${name} exp/make_vad_${datafrom} $vaddir
    utils/fix_data_dir.sh ${name}
  done
fi

if [ $stage -le 2 ]; then
  # Train the UBM.
  # 训练2048的diag GMM
  #
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 8G" \
    --nj 12 --num-threads 8 \
    ${train} 512 \
    exp/diag_ubm_${datafrom}
  # 训练2048的full GMM
  sid/train_full_ubm.sh --cmd "$train_cmd --mem 8G" \
    --nj 12 --remove-low-count-gaussians true \
    ${train} \
    exp/diag_ubm_${datafrom} exp/full_ubm_${datafrom}
fi

if [ $stage -le 3 ]; then
  # In this stage, we train the i-vector extractor.
  #
  # Note that there are well over 1 million utterances in our training set,
  # and it takes an extremely long time to train the extractor on all of this.
  # Also, most of those utterances are very short.  Short utterances are
  # harmful for training the i-vector extractor.  Therefore, to reduce the
  # training time and improve performance, we will only train on the 100k
  # longest utterances.
  # utils/subset_data_dir.sh \
  #   --utt-list <(sort -n -k 2 data/train/utt2num_frames | tail -n 100000) \
  #   data/train data/train_100k
  # # Train the i-vector extractor.
  sid/train_ivector_extractor.sh --cmd "$train_cmd" --nj 4 --num-processes 2 --num-threads 2\
    --ivector-dim 128 --num-iters 5 \
    exp/full_ubm_${datafrom}/final.ubm ${train} \
    exp/extractor_${datafrom}
fi

if [ $stage -le 4 ]; then
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 12 \
    exp/extractor_${datafrom} ${train} \
    exp/ivectors_train_${datafrom}

  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj 8 \
    exp/extractor_${datafrom} ${test} \
    exp/ivectors_timit_test_${datafrom}
fi

if [ $stage -le 5 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd exp/ivectors_train_${datafrom}/log/compute_mean.log \
    ivector-mean scp:exp/ivectors_train_${datafrom}/ivector.scp \
    exp/ivectors_train_${datafrom}/mean.vec || exit 1;

  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=128
  $train_cmd exp/ivectors_train_${datafrom}/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train_${datafrom}/ivector.scp ark:- |" \
    ark:${train}/utt2spk exp/ivectors_train_${datafrom}/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd exp/ivectors_train_${datafrom}/log/plda.log \
    ivector-compute-plda ark:${train}/spk2utt \
    "ark:ivector-subtract-global-mean scp:exp/ivectors_train_${datafrom}/ivector.scp ark:- | transform-vec exp/ivectors_train_${datafrom}/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    exp/ivectors_train_${datafrom}/plda || exit 1;
fi

if [ $stage -le 6 ]; then
  $train_cmd exp/scores/log/timit_test_${datafrom}_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_train_${datafrom}/plda - |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train_${datafrom}/mean.vec scp:exp/ivectors_timit_test_${datafrom}/ivector.scp ark:- | transform-vec exp/ivectors_train_${datafrom}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train_${datafrom}/mean.vec scp:exp/ivectors_timit_test_${datafrom}/ivector.scp ark:- | transform-vec exp/ivectors_train_${datafrom}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$timit_trials' | cut -d\  --fields=1,2 |" exp/scores_timit_test_${datafrom} || exit 1;
fi

if [ $stage -le 7 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $timit_trials exp/scores_timit_test_${datafrom}) 2> /dev/null`
  # sid/compute_min_dcf.py --p-target 0.01 exp/scores_timit_test_${datafrom} $timit_trials
  # sid/compute_min_dcf.py --p-target 0.001 exp/scores_timit_test_${datafrom} $timit_trials
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_timit_test_${datafrom} $timit_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_timit_test_${datafrom} $timit_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"

# 2048 UBM 400-200
# EER: 31.1%
# minDCF(p-target=0.01): 0.9926
# minDCF(p-target=0.001): 0.9926

# 1024 UBM 256-196
# EER: 13.69%
# minDCF(p-target=0.01): 0.9360
# minDCF(p-target=0.001): 0.9360

# 1024 UBM 256-128
# EER: 14.14%
# minDCF(p-target=0.01): 0.9375
# minDCF(p-target=0.001): 0.9375

# 1024 UBM 128-120
# EER: 8.78%
# minDCF(p-target=0.01): 0.9288
# minDCF(p-target=0.001): 0.9330

# 1024 UBM remove<1e-5 128-120
# EER: 7.738%
# minDCF(p-target=0.01): 0.8899
# minDCF(p-target=0.001): 0.8899

# 512 UBM remove<1e-4 128-120
# EER: 2.232%
# minDCF(p-target=0.01): 0.4688
# minDCF(p-target=0.001): 0.4688

# 512 UBM remove<1e-4 128-128
# EER: 2.381%
# minDCF(p-target=0.01): 0.4375
# minDCF(p-target=0.001): 0.4375

# fb24 512 UBM remove<1e-4 128-128
# EER: 3.849%
# minDCF(p-target=0.01): 0.5722
# minDCF(p-target=0.001): 0.9368

# dnn.fb24 512 UBM remove<1e-4 128-128
# EER: 3.783%
# minDCF(p-target=0.01): 0.5188
# minDCF(p-target=0.001): 0.8144

# fb40 512 UBM remove<1e-4 128-128
# EER: 4.947%
# minDCF(p-target=0.01): 0.7026
# minDCF(p-target=0.001): 0.9821

# dnn.fb40 512 UBM remove<1e-4 128-128
# EER: 5.013%
# minDCF(p-target=0.01): 0.7062
# minDCF(p-target=0.001): 0.9493

fi
