#!/usr/bin/env bash
# paraformer-zh-streaming: streaming transcription, 600ms chunk / 300ms lookahead
# Usage: ./transcribe_streaming_by_paraformer.sh <audio_file_or_dir>
input=$1

python transcribe_streaming.py \
  -i "$input" \
  -d mps \
  -m paraformer-zh-streaming \
  --punc-model ct-punc \
  --chunk-size 0 10 5 \
  --encoder-look-back 4 \
  --decoder-look-back 1
