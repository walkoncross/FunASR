#!/usr/bin/env bash
# Fun-ASR-Nano-2512: with VAD, suitable for long audio, end-to-end LLM-ASR, no punc model needed
# Usage: ./transcribe_by_nano.sh <audio_file_or_dir>
input=$1

python transcribe.py \
  -i "$input" \
  -d mps \
  -m FunAudioLLM/Fun-ASR-Nano-2512 \
  --vad-model fsmn-vad \
  --punc-model "" \
  --silence-gap 0.5 \
  --separate-channel
