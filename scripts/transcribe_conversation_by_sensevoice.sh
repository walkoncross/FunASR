#!/usr/bin/env bash
# SenseVoiceSmall: two-channel conversation transcription, multilingual + emotion recognition
# Note: --merge-vad merges short segments and may reduce splitting granularity; enable as needed
# Usage: ./transcribe_conversation_by_sensevoice.sh <stereo_audio>
input=$1

python transcribe_conversation.py \
  -i "$input" \
  -d mps \
  -m iic/SenseVoiceSmall \
  --vad-model fsmn-vad \
  --punc-model "" \
  --language auto \
  --use-itn
