#!/usr/bin/env bash
# Fun-ASR-Nano-2512: without VAD, suitable for short audio (< 30s), end-to-end LLM-ASR, no punc model needed
# Usage: ./transcribe_by_nano_no_vad.sh <audio_file_or_dir>
input=$1

python transcribe.py \
  -i "$input" \
  -d mps \
  -m FunAudioLLM/Fun-ASR-Nano-2512 \
  --vad-model "" \
  --punc-model "" \
  --separate-channel
