#!/usr/bin/env bash
# paraformer-zh-streaming：流式转写，600ms chunk / 300ms lookahead
# 用法：./transcribe_streaming_by_paraformer.sh <音频文件或目录>
input=$1

python transcribe_streaming.py \
  -i "$input" \
  -d mps \
  -m paraformer-zh-streaming \
  --punc-model ct-punc \
  --chunk-size 0 10 5 \
  --encoder-look-back 4 \
  --decoder-look-back 1
