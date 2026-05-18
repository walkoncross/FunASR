#!/usr/bin/env bash
# paraformer-zh: with VAD, suitable for long audio, character-level timestamps, punc model adds punctuation
# Usage: ./transcribe_by_paraformer.sh <audio_file_or_dir>
input=$1

python transcribe.py \
  -i "$input" \
  -d mps \
  -m paraformer-zh \
  --vad-model fsmn-vad \
  --punc-model ct-punc \
  --silence-gap 0.5 \
  --separate-channel
