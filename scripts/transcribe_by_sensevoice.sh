#!/usr/bin/env bash
# SenseVoiceSmall: with VAD, suitable for long audio, multilingual + emotion recognition, use_itn enables numeric normalization
# Usage: ./transcribe_by_sensevoice.sh <audio_file_or_dir>
input=$1

python transcribe.py \
  -i "$input" \
  -d mps \
  -m iic/SenseVoiceSmall \
  --vad-model fsmn-vad \
  --punc-model "" \
  --language auto \
  --use-itn \
  --separate-channel
