#!/usr/bin/env bash
# SenseVoiceSmall：不带 VAD，适合短音频（< 30s），多语言 + 情感识别，use_itn 开启数字规范化
# 用法：./transcribe_by_sensevoice_no_vad.sh <音频文件或目录>
input=$1

python transcribe.py \
  -i "$input" \
  -d mps \
  -m iic/SenseVoiceSmall \
  --vad-model "" \
  --punc-model "" \
  --language auto \
  --use-itn \
  --separate-channel

