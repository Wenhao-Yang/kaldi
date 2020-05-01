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

#/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_fb24_dnn_new

#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/train_fb24
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/test_fb24
#datafrom=py24
#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/train_lfb24
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/test_lfb24
#datafrom=lpy24
#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/train_afb24
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/test_afb24
#datafrom=apy24

#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/train_dfb24_fix
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/test_dfb24_fix
#datafrom=dpy24_fix
#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/train_dfb24_wei
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/test_dfb24_wei
#datafrom=dpy24_wei
#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/train_dfb24_mdv
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/test_dfb24_mdv
#datafrom=dpy24_mdv

train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/train_fb30
test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/test_fb30
datafrom=py30
#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/train_dfb30_var
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/test_dfb30_var
#datafrom=dpy30_var
#train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/train_dfb30_fix
#test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/pyfb/test_dfb30_fix
#datafrom=dpy30_fix

# train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_fb40_dnn_20
# test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_fb40_dnn_20
# datafrom=py40_dnn

# train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_mfcc_20
# test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_mfcc_20
# datafrom=mfcc

# train=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_mfcc_dnn_20
# test=/home/yangwenhao/local/project/lstm_speaker_verification/data/timit/test_mfcc_dnn_20
# datafrom=mfcc_dnn


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
#    sid/compute_vad_decision.sh --nj 8  /home/yangwenhao/local/project/lstm_speaker_verification/data/timit/train_fb24_dnn_m exp/make_vad_py24_dnn_m data/vad
    sid/compute_vad_decision.sh --nj 8 --cmd "$train_cmd" \
      ${name} exp/make_vad_${datafrom} $vaddir
    utils/fix_data_dir.sh ${name}
  done
fi
#stage=4

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
  sid/train_ivector_extractor.sh --cmd "$train_cmd" --nj 4 --num-processes 2 --num-threads 3 \
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


fi

# finally results
# py24
# 640GMMs 128
#EER: 4.603%
#minDCF(p-target=0.01): 0.6626
#minDCF(p-target=0.001): 0.9634

# 512GMMs 128
#EER: 3.796%
#minDCF(p-target=0.01): 0.5419
#minDCF(p-target=0.001): 0.9092

# 256 GMMs 128
#EER: 2.976%
#minDCF(p-target=0.01): 0.4211
#minDCF(p-target=0.001): 0.8286


#20200425 20:40
#linear py24
# 640 GMMs 128
#EER: 4.061%
#minDCF(p-target=0.01): 0.5816
#minDCF(p-target=0.001): 0.8654

# 512GMMs 128
#EER: 3.558%
#minDCF(p-target=0.01): 0.5142
#minDCF(p-target=0.001): 0.8065

# 256 GMMs 128
#EER: 2.804%
#minDCF(p-target=0.01): 0.4038
#minDCF(p-target=0.001): 0.6626



#20200425 21:00
#amel py24 512GMMs 128
#EER: 4.656%
#minDCF(p-target=0.01): 0.6148
#minDCF(p-target=0.001): 0.8327

#dnnpy24 with variance weight, 512GMMs 128
#EER: 6.538%
#minDCF(p-target=0.01): 0.7332
#minDCF(p-target=0.001): 0.9351

#20200425 19:42
#py24 with mean weight from fix 512GMMs 128
#EER: 3.73%
#minDCF(p-target=0.01): 0.5288
#minDCF(p-target=0.001): 0.7901

#py24 with mean weight from fix 256 GMMs 128
#EER: 2.791%
#minDCF(p-target=0.01): 0.4039
#minDCF(p-target=0.001): 0.6152



#20200425 21:19
#py24 with mean/std-min weight from fix 512GMMs 128
#EER: 3.783%
#minDCF(p-target=0.01): 0.5330
#minDCF(p-target=0.001): 0.8437

#20200425 21:19
#py24 with mean/std weight from fix 512GMMs 128
#EER: 3.677%
#minDCF(p-target=0.01): 0.5225
#minDCF(p-target=0.001): 0.8214

#20200425 19:42
#dpy24 with mean weight from var

# 640 GMMs 128
#train - EER: 4.206%
#minDCF(p-target=0.01): 0.5938
#minDCF(p-target=0.001): 0.8959

#traintest EER: 4.312%
#minDCF(p-target=0.01): 0.5934
#minDCF(p-target=0.001): 0.8400

#test - EER: 4.153%
#minDCF(p-target=0.01): 0.6024
#minDCF(p-target=0.001): 0.8483

# 512 GMMs 128
#EER: 3.638%
#minDCF(p-target=0.01): 0.5046
#minDCF(p-target=0.001): 0.8105

# 256 GMMs 128
#EER: 2.791%
#minDCF(p-target=0.01): 0.4039
#minDCF(p-target=0.001): 0.6152



#20200425 20:17
#py30 512GMMs 128
#EER: 4.127%
#minDCF(p-target=0.01): 0.6105
#minDCF(p-target=0.001): 0.9454

#20200425 20:40
##py30 with mean weight from fix 512GMMs 128
#EER: 4.153%
#minDCF(p-target=0.01): 0.5808
#minDCF(p-target=0.001): 0.8818

#20200425 20:28
##py30 with mean weight from var 512GMMs 128
#EER: 4.246%
#minDCF(p-target=0.01): 0.5843
#minDCF(p-target=0.001): 0.9068

#py24 with mean weight from var 640 GMMs 128
#train classifier weight
#EER: 4.22%
#minDCF(p-target=0.01): 0.5967
#minDCF(p-target=0.001): 0.8987
#train verification weight
#EER: 4.471%
#minDCF(p-target=0.01): 0.5947
#minDCF(p-target=0.001): 0.9065

#py24 with mean weight from var 512 GMMs 128
#train classifier weight
#EER: 3.836%
#minDCF(p-target=0.01): 0.5219
#minDCF(p-target=0.001): 0.7617
#train verification weight
#EER: 3.717%
#minDCF(p-target=0.01): 0.5108
#minDCF(p-target=0.001): 0.8170

#py24 with mean weight from var 256 GMMs 128
# train classifier weight
# EER: 2.963%
#minDCF(p-target=0.01): 0.4018
#minDCF(p-target=0.001): 0.6544
#train verification weight
# EER: 2.778%
#minDCF(p-target=0.01): 0.3823
#minDCF(p-target=0.001): 0.6537