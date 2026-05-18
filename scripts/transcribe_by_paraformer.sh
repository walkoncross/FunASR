#!/usr/bin/env bash
# paraformer-zh：字符级时间戳，支持热词，punc 模型加标点
# 用法：./transcribe_by_paraformer.sh <音频文件或目录>
input=$1

python transcribe.py \
  -i "$input" \
  -d mps \
  -m paraformer-zh \
  --vad-model fsmn-vad \
  --punc-model ct-punc \
  --silence-gap 0.5
