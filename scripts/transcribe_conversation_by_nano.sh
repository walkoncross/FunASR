#!/usr/bin/env bash
# Fun-ASR-Nano-2512：双声道对话转写，端到端 LLM-ASR，fsmn-vad 分段
# 用法：./transcribe_conversation_by_nano.sh <立体声音频>
input=$1

python transcribe_conversation.py \
  -i "$input" \
  -d mps \
  -m FunAudioLLM/Fun-ASR-Nano-2512 \
  --vad-model fsmn-vad \
  --punc-model ""
