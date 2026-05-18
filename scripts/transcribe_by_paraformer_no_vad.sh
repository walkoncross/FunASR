#!/usr/bin/env bash
# paraformer-zh: without VAD, suitable for short audio (< 30s), character-level timestamps
# Usage: ./transcribe_by_paraformer_no_vad.sh <audio_file_or_dir>
input=$1

python transcribe.py \
  -i "$input" \
  -d mps \
  -m paraformer-zh \
  --vad-model "" \
  --punc-model ct-punc \
  --separate-channel
