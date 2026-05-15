#!/usr/bin/env bash
# SenseVoiceSmall：多语言 + 情感识别，use_itn 开启数字规范化
# 用法：./transcribe_by_sensevoice.sh <音频文件或目录>
input=$1

python transcribe.py \
  -i "$input" \
  -d mps \
  -m iic/SenseVoiceSmall \
  --vad-model fsmn-vad \
  --punc-model "" \
  --language auto \
  --use-itn \
  --merge-vad \
  --merge-length-s 15
