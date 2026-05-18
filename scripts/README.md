# FunASR Scripts

Speech transcription scripts based on [FunASR](https://github.com/modelscope/FunASR), supporting both ModelScope and HuggingFace backends.

## Scripts

| Script | Purpose |
|--------|---------|
| `transcribe.py` | Single-file or batch non-streaming transcription with multi-channel support |
| `transcribe_conversation.py` | Two-channel conversation transcription; outputs timestamped multi-turn dialogue JSON |
| `transcribe_streaming.py` | Streaming transcription; outputs recognition results in real time, chunk by chunk |

### Shortcut Scripts (model-preset parameters)

| Script | Model | VAD | Use case |
|--------|-------|-----|----------|
| `transcribe_by_nano.sh` | Fun-ASR-Nano-2512 | yes | Single file / directory (long audio) |
| `transcribe_by_nano_no_vad.sh` | Fun-ASR-Nano-2512 | no | Same (short audio < 30s, faster) |
| `transcribe_by_sensevoice.sh` | SenseVoiceSmall | yes | Multilingual + ITN (long audio) |
| `transcribe_by_sensevoice_no_vad.sh` | SenseVoiceSmall | no | Multilingual + ITN (short audio < 30s) |
| `transcribe_by_paraformer.sh` | paraformer-zh | yes | Character-level timestamps (long audio) |
| `transcribe_by_paraformer_no_vad.sh` | paraformer-zh | no | Character-level timestamps (short audio < 30s) |
| `transcribe_conversation_by_nano.sh` | Fun-ASR-Nano-2512 | yes | Two-channel conversation transcription |
| `transcribe_conversation_by_sensevoice.sh` | SenseVoiceSmall | yes | Two-channel conversation, multilingual |
| `transcribe_conversation_by_paraformer.sh` | paraformer-zh | yes | Two-channel conversation, highest accuracy |
| `transcribe_streaming_by_paraformer.sh` | paraformer-zh-streaming | — | Streaming transcription |

---

## transcribe.py

Single-file or batch non-streaming transcription. Supports paraformer-zh, SenseVoice, Fun-ASR-Nano, and other models.

### Usage

```bash
python scripts/transcribe.py -i <audio_file_or_dir> [OPTIONS]
# or use a shortcut script
./scripts/transcribe_by_paraformer.sh <audio_file_or_dir>
```

### Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--input` | `-i` | required | Audio file or directory |
| `--output` | `-o` | `./results/` | Output directory |
| `--model` | `-m` | `paraformer-zh` | ASR model name or local path |
| `--vad-model` | `-vm` | `fsmn-vad` | VAD model name or path; leave empty to disable |
| `--punc-model` | `-pm` | `ct-punc` | Punctuation model name or path; leave empty to disable |
| `--hub` | | `modelscope` | Model source: `modelscope` / `hf` |
| `--device` | `-d` | `cpu` | Inference device: `cpu` / `cuda:0` / `mps` |
| `--batch-size` | `-bs` | `1` | Inference batch size |
| `--separate-channel` | `-sc` | `False` | Transcribe each channel independently |
| `--hotwords` | | `None` | Hotwords string, space-separated |
| `--enable-update` | | `False` | Enable FunASR PyPI version check |
| `--language` | | `None` | Language code (SenseVoice): `auto` / `zh` / `en` / `yue` / `ja` / `ko` |
| `--use-itn` | | `False` | Enable inverse text normalization / ITN (SenseVoice) |
| `--merge-vad` | | `False` | Merge short VAD segments (SenseVoice) |
| `--merge-length-s` | | `15.0` | Max duration in seconds when merging VAD segments (requires `--merge-vad`) |
| `--silence-gap` | `-sg` | `0.5` | Silence gap threshold in seconds for splitting utterances; `0` disables splitting |

### Examples

```bash
# Transcribe a single file (outputs JSON to ./results/)
python scripts/transcribe.py -i audio.wav

# Batch transcribe a directory
python scripts/transcribe.py -i ./audio_dir/

# Separate stereo channels, use MPS acceleration
python scripts/transcribe.py -i stereo.wav -sc -d mps

# SenseVoice: multilingual + ITN
python scripts/transcribe.py -i audio.wav -m iic/SenseVoiceSmall \
  --punc-model "" --language auto --use-itn --merge-vad

# Fun-ASR-Nano: no punc model needed
python scripts/transcribe.py -i audio.wav \
  -m FunAudioLLM/Fun-ASR-Nano-2512 --punc-model ""

# Use a local model (skip network check)
python scripts/transcribe.py -i audio.wav \
  -m ~/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  -vm ~/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  -pm ~/.cache/modelscope/hub/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large
```

### Output Format

Output JSON filename: `<stem>.<model>.<vad>.<punc>.json`

Examples:
- `recording.paraformer-zh.fsmn-vad.ct-punc.json`
- `recording.SenseVoiceSmall.fsmn-vad.no-punc.json`
- `recording.Fun-ASR-Nano-2512.fsmn-vad.no-punc.json`
- `recording.paraformer-zh.no-vad.ct-punc.json`

```json
{
  "source": "/path/to/audio.wav",
  "filename": "audio.wav",
  "audio_dur_s": 12.345,
  "transcribe_s": 1.234,
  "rtf": 0.1,
  "rtfx": 10.0,
  "vad_s": null,
  "vad_rtf": null,
  "vad_rtfx": null,
  "punct_s": null,
  "punct_rtf": null,
  "punct_rtfx": null,
  "model_name": "paraformer-zh",
  "vad_model": "fsmn-vad",
  "punc_model": "ct-punc",
  "text": "first segment result second segment result",
  "segments": [
    {"text": "first segment result", "start": 0.0, "end": 5.0},
    {"text": "second segment result", "start": 5.2, "end": 10.1}
  ]
}
```

- `start` / `end` are in seconds; for SenseVoice, they correspond to VAD segment boundaries.
- `rtf`: real-time factor (lower is faster); `rtfx`: inverse RTF = `1/rtf` (higher is faster).
- With channel separation (`-sc`), filenames get a `.channel0` / `.channel1` segment and the JSON includes `"channel": 0`.

---

## transcribe_conversation.py

Two-channel conversation transcription. Processes each channel separately, splits utterances by silence gaps after VAD segmentation, then interleaves both channels by timestamp into a multi-turn dialogue JSON.

Supports paraformer-zh (character-level timestamps, highest turn-splitting accuracy) and Fun-ASR-Nano / SenseVoice (token-level timestamps).

### Usage

```bash
python scripts/transcribe_conversation.py -i <stereo_audio> [OPTIONS]
# or use a shortcut script
./scripts/transcribe_conversation_by_paraformer.sh <stereo_audio>
```

### Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--input` | `-i` | required | Stereo audio file |
| `--output` | `-o` | `results/<basename>.conversation.<model>.<vad>.<punc>.json` | JSON output path |
| `--model` | `-m` | `paraformer-zh` | ASR model name or local path |
| `--vad-model` | `-vm` | `fsmn-vad` | VAD model name or path |
| `--punc-model` | `-pm` | `ct-punc` | Punctuation model name or path; leave empty to disable |
| `--hub` | | `modelscope` | Model source: `modelscope` / `hf` |
| `--device` | `-d` | `cpu` | Inference device: `cpu` / `cuda:0` / `mps` |
| `--batch-size` | `-bs` | `1` | Inference batch size |
| `--channels` | `-c` | `2` | Number of channels to process |
| `--silence-gap` | `-sg` | `0.5` | Silence gap threshold in seconds; gaps longer than this split a new utterance |
| `--hotwords` | | `None` | Hotwords string, space-separated |
| `--enable-update` | | `False` | Enable FunASR PyPI version check |
| `--language` | | `None` | Language code (SenseVoice): `auto` / `zh` / `en` / `yue` / `ja` / `ko` |
| `--use-itn` | | `False` | Enable inverse text normalization / ITN (SenseVoice) |
| `--merge-vad` | | `False` | Merge short VAD segments (SenseVoice — reduces turn-split granularity, use with care) |
| `--merge-length-s` | | `15.0` | Max duration in seconds when merging VAD segments (requires `--merge-vad`) |

### Examples

```bash
# Default parameters (paraformer-zh)
python scripts/transcribe_conversation.py -i stereo.wav

# Fun-ASR-Nano with MPS acceleration
python scripts/transcribe_conversation.py -i stereo.wav -d mps \
  -m FunAudioLLM/Fun-ASR-Nano-2512 --punc-model ""

# Increase silence gap for longer utterances
python scripts/transcribe_conversation.py -i stereo.wav --silence-gap 1.0
```

### Output Format

Filename format: `<stem>.conversation.<model>.<vad>.<punc>.json`

```json
{
  "source": "/path/to/stereo.wav",
  "filename": "stereo.wav",
  "channels": 2,
  "audio_dur_s": 306.68,
  "transcribe_s": 52.32,
  "rtf": 0.1707,
  "rtfx": 5.86,
  "vad_s": 0.0,
  "vad_rtf": null,
  "vad_rtfx": null,
  "punct_s": null,
  "punct_rtf": null,
  "punct_rtfx": null,
  "model_name": "paraformer-zh",
  "vad_model": "fsmn-vad",
  "punc_model": "ct-punc",
  "conversations": [
    {"role": "channel_0", "text": "Hello, is this Ms. Yuan Xuezhen?", "start": 17.17, "end": 19.3},
    {"role": "channel_1", "text": "I know, you must be the one.", "start": 20.7, "end": 22.5}
  ]
}
```

`role` corresponds to the channel index (`channel_0` = first channel); `start` / `end` are in seconds.

---

## transcribe_streaming.py

Streaming speech transcription using paraformer-zh-streaming. Feeds audio in fixed-size chunks and outputs recognition results in real time, simulating a low-latency ASR scenario.

### Usage

```bash
python scripts/transcribe_streaming.py -i <audio_file_or_dir> [OPTIONS]
# or use a shortcut script
./scripts/transcribe_streaming_by_paraformer.sh <audio_file_or_dir>
```

### Parameters

| Parameter | Short | Default | Description |
|-----------|-------|---------|-------------|
| `--input` | `-i` | required | Audio file or directory |
| `--output` | `-o` | `./results/` | Output directory |
| `--model` | `-m` | `paraformer-zh-streaming` | Streaming ASR model name or local path |
| `--punc-model` | `-pm` | `ct-punc` | Punctuation model name or path; leave empty to disable |
| `--hub` | | `modelscope` | Model source: `modelscope` / `hf` |
| `--device` | `-d` | `cpu` | Inference device: `cpu` / `cuda:0` / `mps` |
| `--chunk-size` | | `0 10 5` | Streaming chunk config `[lookahead chunk shift]` in frames (60ms each). `0 10 5` = 600ms chunk / 300ms lookahead |
| `--encoder-look-back` | | `4` | Number of encoder self-attention look-back chunks |
| `--decoder-look-back` | | `1` | Number of decoder cross-attention look-back encoder chunks |
| `--hotwords` | | `None` | Hotwords string, space-separated |
| `--enable-update` | | `False` | Enable FunASR PyPI version check |
| `--separate-channel` | `-sc` | `False` | Split channels and transcribe each separately |

**chunk-size explained**: `[lookahead, chunk, shift]`, where 1 frame = 60ms.
- `0 10 5`: 600ms chunk, 300ms lookahead (default — balanced latency and accuracy)
- `0 8 4`: 480ms chunk, 240ms lookahead (lower latency)

### Examples

```bash
# Default 600ms chunks
python scripts/transcribe_streaming.py -i audio.wav -d mps

# 480ms low-latency mode
python scripts/transcribe_streaming.py -i audio.wav -d mps \
  --chunk-size 0 8 4
```

### Output Format

Filename format: `<stem>.streaming.<model>.no-vad.<punc>.json`

With `--separate-channel`: `<stem>.channel0.streaming.<model>.no-vad.<punc>.json`

```json
{
  "source": "/path/to/audio.wav",
  "filename": "audio.wav",
  "audio_dur_s": 5.12,
  "transcribe_s": 1.23,
  "rtf": 0.24,
  "rtfx": 4.16,
  "vad_s": null,
  "vad_rtf": null,
  "vad_rtfx": null,
  "punct_s": null,
  "punct_rtf": null,
  "punct_rtfx": null,
  "model_name": "paraformer-zh-streaming",
  "vad_model": null,
  "punc_model": "ct-punc",
  "text": "full transcription text",
  "chunks": [
    {"chunk": 0, "is_final": false, "text": "partial result"},
    {"chunk": 1, "is_final": false, "text": "more results"},
    {"chunk": 42, "is_final": true, "text": "final complete text"}
  ]
}
```

The final `text` field is taken from the `is_final=true` chunk. `chunks` records the real-time output of each chunk and can be used to analyze streaming latency.

---

## Model Comparison

| Model | Timestamps | Multilingual | Punctuation | Streaming | Best for |
|-------|-----------|--------------|-------------|-----------|----------|
| paraformer-zh | Character-level (ms) | No | Requires ct-punc | No | Chinese transcription; highest accuracy for multi-turn dialogue |
| paraformer-zh-streaming | None | No | Requires ct-punc | Yes | Real-time low-latency scenarios |
| SenseVoiceSmall | Token-level (s) | Yes | Built-in ITN | No | Multilingual, emotion recognition |
| Fun-ASR-Nano-2512 | Token-level (s) | No | Built-in | No | End-to-end, no extra models needed |

---

## VAD and Long Audio

All three models (paraformer-zh, SenseVoiceSmall, Fun-ASR-Nano) have no hard truncation limit in their encoders, but **using long audio without VAD has two problems**:

1. **OOM**: Encoder attention complexity is O(T²); memory grows quadratically with audio length.
2. **Accuracy drop**: Models are trained on segments up to ~30s; recognition accuracy degrades on longer inputs.

**Recommendations**:
- Long audio (> 30s): use `--vad-model fsmn-vad`. VAD segments the audio first, then each segment is recognized independently — supports any length.
- Short audio (< 30s): VAD can be omitted (`--vad-model ""`), saving one model load and reducing latency.

**`--silence-gap` vs. VAD**: `--silence-gap` performs a second-pass local split on timestamps after VAD segmentation and model inference. It has no effect when VAD is disabled (only one result segment). For SenseVoice, splits align with VAD segment boundaries; `--silence-gap` does not change split granularity (one result per VAD segment).

---

## Model Download and Cache

On first run, models are downloaded automatically from ModelScope to `~/.cache/modelscope/hub/models/`.

**Skip network check**: pass the local cache absolute path to `-m` / `-vm` / `-pm`. FunASR loads the model directly when it detects an existing path, bypassing any network request.

Common local model paths:

| Model | Local path |
|-------|-----------|
| paraformer-zh | `~/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| paraformer-zh-streaming | `~/.cache/modelscope/hub/models/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404` |
| fsmn-vad | `~/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| ct-punc | `~/.cache/modelscope/hub/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large` |
| SenseVoiceSmall | `~/.cache/modelscope/hub/models/iic/SenseVoiceSmall` |
| Fun-ASR-Nano-2512 | `~/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512` |
