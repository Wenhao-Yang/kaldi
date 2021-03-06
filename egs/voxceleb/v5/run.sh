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
vaddir=`pwd`/data/vad

# sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe


# The trials file is downloaded by local/make_vox1_v2.pl.
# vox1_trials=data/vox1_test/trials
# vox1_root=/export/corpora/VoxCeleb1
# voxceleb2_root=/export/corpora/VoxCeleb2
#vox1_root=/data/vox1


# train=data/train
# test=data/test

#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_fb24
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_fb24
#datafrom=py24_512

#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_dfb24
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_dfb24
#datafrom=py24_dnn

train=/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/dev_dfb24_soft
test=/home/yangwenhao/local/project/lstm_speaker_verification/data/Vox1_pyfb/test_dfb24_soft
datafrom=dpy24_soft

# train=/home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/train_mfcc_20
# test=/home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/test_mfcc_20
# datafrom=mfcc

# train=/home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/train_mfcc_dnn_20
# test=/home/yangwenhao/local/project/lstm_speaker_verification/data/vox1/test_mfcc_dnn_20
# datafrom=mfcc_dnn


vox1_trials=${test}/trials_h
stage=6

if [ $stage -le 0 ]; then
  # if [ ! -d ${train} ]; then
  #   mkdir -p ${train}
  #   cp ${vox1_root}/train/* ${train}
  # fi

  # if [ ! -d ${test} ]; then
  #   mkdir -p ${test}
  #   cp ${vox1_root}/test/* ${test}
  # fi
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
    sid/compute_vad_decision.sh --nj 12 --cmd "$train_cmd" \
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
    ${train} 1024 \
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
  sid/train_ivector_extractor.sh --cmd "$train_cmd" --nj 3 --num-processes 2 --num-threads 2 \
    --ivector-dim 128 --num-iters 5 \
    exp/full_ubm_${datafrom}/final.ubm ${train} \
    exp/extractor_${datafrom}
fi

if [ $stage -le 4 ]; then
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 8G" --nj 12 \
    exp/extractor_${datafrom} ${train} \
    exp/ivectors_train_${datafrom}

  sid/extract_ivectors.sh --cmd "$train_cmd --mem 8G" --nj 8 \
    exp/extractor_${datafrom} ${test} \
    exp/ivectors_vox1_test_${datafrom}
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
  $train_cmd exp/scores/log/vox1_test_${datafrom}_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_train_${datafrom}/plda - |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train_${datafrom}/mean.vec scp:exp/ivectors_vox1_test_${datafrom}/ivector.scp ark:- | transform-vec exp/ivectors_train_${datafrom}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train_${datafrom}/mean.vec scp:exp/ivectors_vox1_test_${datafrom}/ivector.scp ark:- | transform-vec exp/ivectors_train_${datafrom}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$vox1_trials' | cut -d\  --fields=1,2 |" exp/scores_vox1_test_${datafrom} || exit 1;
fi

if [ $stage -le 7 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $vox1_trials exp/scores_vox1_test_${datafrom}) 2> /dev/null`
  # sid/compute_min_dcf.py --p-target 0.01 exp/scores_vox1_test_${datafrom} $vox1_trials
  # sid/compute_min_dcf.py --p-target 0.001 exp/scores_vox1_test_${datafrom} $vox1_trials
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_vox1_test_${datafrom} $vox1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_vox1_test_${datafrom} $vox1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"

fi

# fb24 GMM 2048

#EER: 6.66%
#minDCF(p-target=0.01): 0.5668
#minDCF(p-target=0.001): 0.6931

# fb24 GMM 2048
#EER: 6.707%
#minDCF(p-target=0.01): 0.5776
#minDCF(p-target=0.001): 0.7098

# fb24 GMM 1024
#EER: 6.845%
#minDCF(p-target=0.01): 0.5863
#minDCF(p-target=0.001): 0.6670

# fb24.dnn GMM 1024
#EER: 7.253%
#minDCF(p-target=0.01): 0.6088
#minDCF(p-target=0.001): 0.7399

