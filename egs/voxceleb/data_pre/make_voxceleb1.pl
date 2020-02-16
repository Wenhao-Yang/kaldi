#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#           2018  David Snyder
#
# Usage: make_voxceleb1.pl /export/voxceleb1 data/

if (! -e "voxceleb1_test.txt") {
  system("wget -O voxceleb1_test.txt http://www.openslr.org/resources/49/voxceleb1_test.txt");
}

if (! -e "vox1_meta.csv") {
  system("wget -O vox1_meta.csv http://www.openslr.org/resources/49/vox1_meta.csv");
}

open(TRIAL_IN, "<", "voxceleb1_test.txt") or die "Could not open the verification trials file $data_base/voxceleb1_test.txt";
open(META_IN, "<", "vox1_meta.csv") or die "Could not open the meta data file $data_base/vox1_meta.csv";
open(TRIAL_OUT, ">", "trials") or die "Could not open the output file $out_test_dir/trials";

my %id2spkr = ();
while (<META_IN>) {
  chomp;
  my ($vox_id, $spkr_id, $gender, $nation, $set) = split;
  $id2spkr{$vox_id} = $spkr_id;
}

my $test_spkrs = ();
while (<TRIAL_IN>) {
  chomp;
  my ($tar_or_non, $path1, $path2) = split;

  # Create entry for left-hand side of trial
  my ($spkr_id, $filename) = split('/', $path1);
  my $rec_id = substr($filename, 0, 11);
  my $segment = substr($filename, 15, 8);
  my $utt_id1 = "$spkr_id-$rec_id-$segment";
  $test_spkrs{$spkr_id} = ();

  # Create entry for right-hand side of trial
  my ($spkr_id, $filename) = split('/', $path2);
  my $rec_id = substr($filename, 0, 11);
  my $segment = substr($filename, 15, 8);
  my $utt_id2 = "$spkr_id-$rec_id-$segment";
  $test_spkrs{$spkr_id} = ();

  my $target = "nontarget";
  if ($tar_or_non eq "1") {
    $target = "target";
  }
  print TRIAL_OUT "$utt_id1 $utt_id2 $target\n";
}
close(TRIAL_OUT) or die;
close(TRIAL_IN) or die;
close(META_IN) or die;


