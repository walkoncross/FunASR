input=$1

python transcribe_conversation.py \
  -i $input \
  -d mps \
  -m ~/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512 \
  --punc-model ""