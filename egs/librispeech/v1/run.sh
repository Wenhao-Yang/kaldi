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


# The trials file is downloaded by local/make_libri_v2.pl.
# libri_trials=data/libri_test/trials
# libri_root=/export/corpora/VoxCeleb1
# voxceleb2_root=/export/corpora/VoxCeleb2
libri_root=/data/libri


#/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/test_fb24_dnn_new

train=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb/dev_fb24
test=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb/test_fb24
datafrom=py24

#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb/dev_lfb24
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb/test_lfb24
#datafrom=lpy24

#
#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb/dev_dfb24_64
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb/test_dfb24_64
#datafrom=dpy24_64
#
#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb/dev_dfb24_fix
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb/test_dfb24_fix
#datafrom=dpy24_fix

#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb/dev_dfb24_fix_f1
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/pyfb/test_dfb24_fix_f1
#datafrom=dpy24_fix_f1

# train=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/train_mfcc_20
# test=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/test_mfcc_20
# datafrom=mfcc

# train=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/train_mfcc_dnn_20
# test=/home/yangwenhao/local/project/lstm_speaker_verification/data/libri/test_mfcc_dnn_20
# datafrom=mfcc_dnn


libri_trials=${test}/trials.2

stage=6

if [ $stage -le 0 ]; then
  # if [ ! -d ${train} ]; then
  #   mkdir -p ${train}
  #   cp ${libri_root}/train/* ${train}
  # fi

  # if [ ! -d ${test} ]; then
  #   mkdir -p ${test}
  #   cp ${libri_root}/test/* ${test}
  # fi
  local/libri_data_prep.sh $libri_root || exit 1
  local/libri_format_data.sh

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
      ${name} \
       exp/make_vad_${datafrom} \
       $vaddir
    utils/fix_data_dir.sh ${name}
  done
fi
#stage=100
if [ $stage -le 2 ]; then
  # Train the UBM.
  # 训练2048的diag GMM
  #
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 8G" \
    --nj 12 --num-threads 8 \
    ${train} 256 \
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
    exp/ivectors_libri_test_${datafrom}
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
  $train_cmd exp/scores/log/libri_test_${datafrom}_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 exp/ivectors_train_${datafrom}/plda - |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train_${datafrom}/mean.vec scp:exp/ivectors_libri_test_${datafrom}/ivector.scp ark:- | transform-vec exp/ivectors_train_${datafrom}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean exp/ivectors_train_${datafrom}/mean.vec scp:exp/ivectors_libri_test_${datafrom}/ivector.scp ark:- | transform-vec exp/ivectors_train_${datafrom}/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$libri_trials' | cut -d\  --fields=1,2 |" exp/scores_libri_test_${datafrom} || exit 1;
fi

if [ $stage -le 7 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $libri_trials exp/scores_libri_test_${datafrom}) 2> /dev/null`
  # sid/compute_min_dcf.py --p-target 0.01 exp/scores_libri_test_${datafrom} $libri_trials
  # sid/compute_min_dcf.py --p-target 0.001 exp/scores_libri_test_${datafrom} $libri_trials
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_libri_test_${datafrom} $libri_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_libri_test_${datafrom} $libri_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"

# VAD 5.5 0.5
# fb24 640 UBM 128
#EER: 2.89%
#minDCF(p-target=0.01): 0.7239
#minDCF(p-target=0.001): 0.9970

# VAD 5.5 0.5
# fb24 512 UBM 128
#EER: 2.67%
#minDCF(p-target=0.01): 0.7224
#minDCF(p-target=0.001): 0.9962

# VAD 5.5 0.5
# fb24 256 UBM 128
#EER: 2.506%
#minDCF(p-target=0.01): 0.7093
#minDCF(p-target=0.001): 0.9983


# dfb24 640 UBM 128
#EER: 2.945%
#minDCF(p-target=0.01): 0.7514
#minDCF(p-target=0.001): 0.9981

# dfb24 512 UBM 128
#EER: 2.833%
#minDCF(p-target=0.01): 0.7368
#minDCF(p-target=0.001): 0.9997

# dfb24 256 UBM 128
#EER: 2.702%
#minDCF(p-target=0.01): 0.7519
#minDCF(p-target=0.001): 0.9980

fi
