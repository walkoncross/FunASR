#!/usr/bin/env bash
# Fun-ASR-Nano-2512: two-channel conversation transcription, end-to-end LLM-ASR, fsmn-vad segmentation
# Usage: ./transcribe_conversation_by_nano.sh <stereo_audio>
input=$1

python transcribe_conversation.py \
  -i "$input" \
  -d mps \
  -m FunAudioLLM/Fun-ASR-Nano-2512 \
  --vad-model fsmn-vad \
  --punc-model ""
