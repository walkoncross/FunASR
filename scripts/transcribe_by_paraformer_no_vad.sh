#!/usr/bin/env bash
# paraformer-zh：不带 VAD，适合短音频（< 30s），字符级时间戳
# 用法：./transcribe_by_paraformer_no_vad.sh <音频文件或目录>
input=$1

python transcribe.py \
  -i "$input" \
  -d mps \
  -m paraformer-zh \
  --vad-model "" \
  --punc-model ct-punc
