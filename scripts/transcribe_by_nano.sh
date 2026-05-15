#!/usr/bin/env bash
# Fun-ASR-Nano-2512：端到端 LLM-ASR，无需 punc 模型，fsmn-vad 分段
# 用法：./transcribe_by_nano.sh <音频文件或目录>
input=$1

python transcribe.py \
  -i "$input" \
  -d mps \
  -m FunAudioLLM/Fun-ASR-Nano-2512 \
  --vad-model fsmn-vad \
  --punc-model ""
