#!/usr/bin/env bash
# paraformer-zh: two-channel conversation transcription, character-level timestamps, highest multi-turn splitting accuracy
# Usage: ./transcribe_conversation_by_paraformer.sh <stereo_audio>
input=$1

python transcribe_conversation.py \
  -i "$input" \
  -d mps \
  -m paraformer-zh \
  --vad-model fsmn-vad \
  --punc-model ct-punc
