#!/usr/bin/env bash
# SenseVoiceSmall: without VAD, suitable for short audio (< 30s), multilingual + emotion recognition, use_itn enables numeric normalization
# Usage: ./transcribe_by_sensevoice_no_vad.sh <audio_file_or_dir>
input=$1

python transcribe.py \
  -i "$input" \
  -d mps \
  -m iic/SenseVoiceSmall \
  --vad-model "" \
  --punc-model "" \
  --language auto \
  --use-itn \
  --separate-channel
