#!/usr/bin/perl
#
# Copyright 2018  Ewald Enzinger
#           2018  David Snyder
#
# Usage: make_voxceleb1.pl /export/voxceleb1 data/

if (@ARGV != 1) {
  print STDERR "Usage: $0 <path-to-data-dir>\n";
  print STDERR "e.g. $0 data/\n";
  exit(1);
}

($out_dir) = @ARGV;
my $out_test_dir = "$out_dir/voxceleb1_test";

if (system("mkdir -p $out_test_dir") != 0) {
  die "Error making directory $out_test_dir";
}

if (! -e "$out_dir/voxceleb1_test.txt") {
  system("wget -O $out_dir/voxceleb1_test.txt http://www.openslr.org/resources/49/voxceleb1_test.txt");
}

if (! -e "$out_dir/vox1_meta.csv") {
  system("wget -O $our_dir/vox1_meta.csv http://www.openslr.org/resources/49/vox1_meta.csv");
}

open(TRIAL_IN, "<", "$out_dir/voxceleb1_test.txt") or die "Could not open the verification trials file $out_dir/voxceleb1_test.txt";
open(META_IN, "<", "$out_dir/vox1_meta.csv") or die "Could not open the meta data file $out_dir/vox1_meta.csv";
open(TRIAL_OUT, ">", "$out_test_dir/trials") or die "Could not open the output file $out_test_dir/trials";

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
