#!/usr/bin/env bash
# SenseVoiceSmall：双声道对话转写，多语言 + 情感识别
# 注意：--merge-vad 会合并短分段，可能降低对话切分粒度，按需开启
# 用法：./transcribe_conversation_by_sensevoice.sh <立体声音频>
input=$1

python transcribe_conversation.py \
  -i "$input" \
  -d mps \
  -m iic/SenseVoiceSmall \
  --vad-model fsmn-vad \
  --punc-model "" \
  --language auto \
  --use-itn
