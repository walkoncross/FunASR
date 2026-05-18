#!/usr/bin/env bash
# Fun-ASR-Nano-2512：不带 VAD，适合短音频（< 30s），端到端 LLM-ASR，无需 punc 模型
# 用法：./transcribe_by_nano_no_vad.sh <音频文件或目录>
input=$1

python transcribe.py \
  -i "$input" \
  -d mps \
  -m FunAudioLLM/Fun-ASR-Nano-2512 \
  --vad-model "" \
  --punc-model "" \
  --separate-channel
