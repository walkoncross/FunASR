#!/usr/bin/env bash
# paraformer-zh-streaming: streaming transcription, real-time profile (AI outbound / AI customer service)
#   chunk-size 0 8 4  => 480ms chunk / 240ms look-ahead (lower first-token latency)
#   encoder-look-back 4 / decoder-look-back 1 => minimal history to reduce per-chunk latency
# Usage: ./transcribe_streaming_by_paraformer.sh <audio_file_or_dir>
input=$1

python transcribe_streaming.py \
  -i "$input" \
  -d mps \
  -m paraformer-zh-streaming \
  --punc-model ct-punc \
  --chunk-size 0 8 4 \
  --encoder-look-back 4 \
  --decoder-look-back 1 \
  --separate-channels \
  --latency-mode realtime
