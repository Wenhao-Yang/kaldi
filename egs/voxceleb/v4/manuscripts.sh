local/nnet3/xvector/prepare_feats_for_egs.sh --nj 10 data/voxceleb1_test data/voxceleb1_test_no_sil exp/voxceleb1_test_no_sil


steps/make_fbank.sh --write-utt2num-frames true --fbank_config data/make_test/fb_conf data/make_test exp/make_test data/make_test

compute-fbank-feats --config=  scp:make_test.scp ark:-

spk2utt
A.J._Buckley-1zcIwhmdeo4-0001.wav A.J._Buckley