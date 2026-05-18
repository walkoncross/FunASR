#!/usr/bin/env bash
input=$1

./transcribe_by_paraformer.sh "$input"
./transcribe_by_paraformer_no_vad.sh "$input"

./transcribe_by_nano.sh "$input"
./transcribe_by_nano_no_vad.sh "$input"

./transcribe_by_sensevoice.sh "$input"
./transcribe_by_sensevoice_no_vad.sh "$input"