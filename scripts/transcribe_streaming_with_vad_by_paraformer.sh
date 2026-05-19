#!/usr/bin/env bash
# paraformer-zh-streaming + fsmn-vad: simulated-online streaming with VAD
#   channel 0 = user  (客户 / 用户)
#   channel 1 = agent (人工坐席 / AI坐席)
#
#   chunk-size 0 8 4  => 480ms chunk / 240ms look-ahead (real-time profile)
#   vad-chunk-ms 200  => VAD processes 200ms windows
#   --channels 0 1    => transcribe both channels (default)
#   --channels 0      => transcribe user channel only
#   --latency-mode fast      => no sleep, max throughput (default); final_chunk_ms measured
#   --latency-mode realtime  => sleep per wall-step; true end-to-end ttft_ms measured
#
# Usage: ./transcribe_streaming_with_vad_by_paraformer.sh <audio_file_or_dir>
input=$1

python transcribe_streaming_with_vad.py \
  -i "$input" \
  -d mps \
  -m paraformer-zh-streaming \
  --vad-model fsmn-vad \
  --punc-model ct-punc \
  --chunk-size 0 8 4 \
  --encoder-look-back 4 \
  --decoder-look-back 1 \
  --vad-chunk-ms 200 \
  --channels 0 1 \
  --latency-mode fast
