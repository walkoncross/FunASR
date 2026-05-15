#!/usr/bin/env bash
# paraformer-zh：双声道对话转写，字符级时间戳，多轮切分精度最高
# 用法：./transcribe_conversation_by_paraformer.sh <立体声音频>
input=$1

python transcribe_conversation.py \
  -i "$input" \
  -d mps \
  -m paraformer-zh \
  --vad-model fsmn-vad \
  --punc-model ct-punc
