# FunASR Scripts

基于 [FunASR](https://github.com/modelscope/FunASR) 的语音转写脚本，支持 ModelScope / HuggingFace 双后端。

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `transcribe.py` | 单文件或批量非流式转写，支持多声道分离 |
| `transcribe_conversation.py` | 双声道对话转写，输出带时间戳的多轮对话 JSON |
| `transcribe_streaming.py` | 流式转写，逐块实时输出识别结果 |

### 快捷脚本（按模型预设参数）

| 脚本 | 模型 | 用途 |
|------|------|------|
| `transcribe_by_nano.sh` | Fun-ASR-Nano-2512 | 单文件/目录转写 |
| `transcribe_by_sensevoice.sh` | SenseVoiceSmall | 单文件/目录转写，多语言 + ITN |
| `transcribe_by_paraformer.sh` | paraformer-zh | 单文件/目录转写，字符级时间戳 |
| `transcribe_conversation_by_nano.sh` | Fun-ASR-Nano-2512 | 双声道对话转写 |
| `transcribe_conversation_by_sensevoice.sh` | SenseVoiceSmall | 双声道对话转写，多语言 |
| `transcribe_conversation_by_paraformer.sh` | paraformer-zh | 双声道对话转写，多轮精度最高 |
| `transcribe_streaming_by_paraformer.sh` | paraformer-zh-streaming | 流式转写 |

---

## transcribe.py

单文件或批量非流式语音转写，支持 paraformer-zh / SenseVoice / Fun-ASR-Nano 等模型。

### 用法

```bash
python scripts/transcribe.py -i <音频文件或目录> [OPTIONS]
# 或使用快捷脚本
./scripts/transcribe_by_paraformer.sh <音频文件或目录>
```

### 参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | 必填 | 音频文件或目录 |
| `--output` | `-o` | `./results/` | 输出目录 |
| `--model` | `-m` | `paraformer-zh` | ASR 模型名称或本地路径 |
| `--vad-model` | `-vm` | `fsmn-vad` | VAD 模型名称或路径，留空则禁用 |
| `--punc-model` | `-pm` | `ct-punc` | 标点模型名称或路径，留空则禁用 |
| `--hub` | | `modelscope` | 模型来源：`modelscope` / `hf` |
| `--device` | `-d` | `cpu` | 推理设备：`cpu` / `cuda:0` / `mps` |
| `--batch-size` | `-bs` | `1` | 推理 batch size |
| `--separate-channel` | `-sc` | `False` | 分离声道分别转录，每声道独立输出 |
| `--hotwords` | | `None` | 热词字符串，空格分隔 |
| `--enable-update` | | `False` | 启用 FunASR PyPI 版本检查 |
| `--language` | | `None` | 语言代码（SenseVoice）：`auto` / `zh` / `en` / `yue` / `ja` / `ko` |
| `--use-itn` | | `False` | 启用标点与数字规范化 ITN（SenseVoice） |
| `--merge-vad` | | `False` | 合并短 VAD 分段（SenseVoice） |
| `--merge-length-s` | | `15.0` | 合并 VAD 分段的最大时长，秒（需配合 `--merge-vad`） |

### 示例

```bash
# 转写单个文件（输出 JSON 到 ./results/）
python scripts/transcribe.py -i audio.wav

# 批量转写目录
python scripts/transcribe.py -i ./audio_dir/

# 分离双声道，使用 MPS 加速
python scripts/transcribe.py -i stereo.wav -sc -d mps

# SenseVoice：多语言 + ITN
python scripts/transcribe.py -i audio.wav -m iic/SenseVoiceSmall \
  --punc-model "" --language auto --use-itn --merge-vad

# Fun-ASR-Nano：无需 punc 模型
python scripts/transcribe.py -i audio.wav \
  -m FunAudioLLM/Fun-ASR-Nano-2512 --punc-model ""

# 使用本地模型（跳过联网检查）
python scripts/transcribe.py -i audio.wav \
  -m ~/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  -vm ~/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  -pm ~/.cache/modelscope/hub/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large
```

### 输出格式

统一输出 JSON，文件名为 `<stem>.funasr.json`。

**默认（无 `--timestamp`）**：
```json
{
  "source": "/path/to/audio.wav",
  "filename": "audio.wav",
  "text": [
    "第一段识别结果",
    "第二段识别结果"
  ],
  "audio_dur_s": 12.345,
  "transcribe_s": 1.234,
  "rtf": 0.1
}
```

**启用 `--timestamp`**：
```json
{
  "source": "/path/to/audio.wav",
  "filename": "audio.wav",
  "text": [
    {"text": "第一段识别结果", "start": 0.0, "end": 5.0},
    {"text": "第二段识别结果", "start": 5.2, "end": 10.1}
  ],
  "audio_dur_s": 12.345,
  "transcribe_s": 1.234,
  "rtf": 0.1
}
```

声道分离模式（`-sc`）时，文件名附加 `_channel0` / `_channel1` 后缀，JSON 中额外包含 `"channel": 0`。

---

## transcribe_conversation.py

双声道对话转写。逐声道处理，利用 fsmn-vad 分段后按静音间隔切分话语，两声道结果按时间戳交错排序输出多轮对话 JSON。

支持 paraformer-zh（字符级时间戳，切分精度最高）和 Fun-ASR-Nano / SenseVoice（token 级时间戳）。

### 用法

```bash
python scripts/transcribe_conversation.py -i <立体声音频> [OPTIONS]
# 或使用快捷脚本
./scripts/transcribe_conversation_by_paraformer.sh <立体声音频>
```

### 参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | 必填 | 立体声音频文件 |
| `--output` | `-o` | `results/<basename>-conversation.json` | JSON 输出路径 |
| `--model` | `-m` | `paraformer-zh` | ASR 模型名称或本地路径 |
| `--vad-model` | `-vm` | `fsmn-vad` | VAD 模型名称或路径 |
| `--punc-model` | `-pm` | `ct-punc` | 标点模型名称或路径，留空则禁用 |
| `--hub` | | `modelscope` | 模型来源：`modelscope` / `hf` |
| `--device` | `-d` | `cpu` | 推理设备：`cpu` / `cuda:0` / `mps` |
| `--batch-size` | `-bs` | `1` | 推理 batch size |
| `--channels` | `-c` | `2` | 处理声道数 |
| `--silence-gap` | `-sg` | `0.5` | 句间静音间隔阈值（秒），超过则切断为新话语 |
| `--hotwords` | | `None` | 热词字符串，空格分隔 |
| `--enable-update` | | `False` | 启用 FunASR PyPI 版本检查 |
| `--language` | | `None` | 语言代码（SenseVoice）：`auto` / `zh` / `en` / `yue` / `ja` / `ko` |
| `--use-itn` | | `False` | 启用标点与数字规范化 ITN（SenseVoice） |
| `--merge-vad` | | `False` | 合并短 VAD 分段（SenseVoice，会降低对话切分粒度，慎用） |
| `--merge-length-s` | | `15.0` | 合并 VAD 分段的最大时长，秒（需配合 `--merge-vad`） |

### 示例

```bash
# 默认参数（paraformer-zh）
python scripts/transcribe_conversation.py -i stereo.wav

# Fun-ASR-Nano，MPS 加速
python scripts/transcribe_conversation.py -i stereo.wav -d mps \
  -m FunAudioLLM/Fun-ASR-Nano-2512 --punc-model ""

# 调大切分间隔，得到更长的句子
python scripts/transcribe_conversation.py -i stereo.wav --silence-gap 1.0
```

### 输出格式

```json
{
  "source": "/path/to/stereo.wav",
  "filename": "stereo.wav",
  "channels": 2,
  "conversations": [
    {"role": "channel_0", "text": "喂，你好，是袁雪珍女士吗？", "start": 17.17, "end": 19.3},
    {"role": "channel_1", "text": "我知道你应该是那个啊。", "start": 20.7, "end": 22.5}
  ]
}
```

`role` 对应声道编号（`channel_0` = 第 1 声道），`start` / `end` 单位为秒。

---

## transcribe_streaming.py

流式语音转写，使用 paraformer-zh-streaming 逐块实时输出识别结果，模拟低延迟 ASR 场景。

### 用法

```bash
python scripts/transcribe_streaming.py -i <音频文件或目录> [OPTIONS]
# 或使用快捷脚本
./scripts/transcribe_streaming_by_paraformer.sh <音频文件或目录>
```

### 参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | 必填 | 音频文件或目录 |
| `--output` | `-o` | `./results/` | 输出目录 |
| `--model` | `-m` | `paraformer-zh-streaming` | 流式 ASR 模型名称或本地路径 |
| `--punc-model` | `-pm` | `ct-punc` | 标点模型名称或路径，留空则禁用 |
| `--hub` | | `modelscope` | 模型来源：`modelscope` / `hf` |
| `--device` | `-d` | `cpu` | 推理设备：`cpu` / `cuda:0` / `mps` |
| `--chunk-size` | | `0 10 5` | 流式分块配置 `[lookahead chunk shift]`，单位帧（60ms）。`0 10 5` = 600ms chunk / 300ms lookahead |
| `--encoder-look-back` | | `4` | Encoder self-attention 回看 chunk 数 |
| `--decoder-look-back` | | `1` | Decoder cross-attention 回看 encoder chunk 数 |
| `--hotwords` | | `None` | 热词字符串，空格分隔 |
| `--enable-update` | | `False` | 启用 FunASR PyPI 版本检查 |

**chunk-size 说明**：`[lookahead, chunk, shift]`，单位为帧（1 帧 = 60ms）。
- `0 10 5`：600ms chunk，300ms lookahead（默认，平衡延迟与精度）
- `0 8 4`：480ms chunk，240ms lookahead（更低延迟）

### 示例

```bash
# 默认 600ms 分块
python scripts/transcribe_streaming.py -i audio.wav -d mps

# 480ms 低延迟模式
python scripts/transcribe_streaming.py -i audio.wav -d mps \
  --chunk-size 0 8 4
```

### 输出格式

**json**：
```json
{
  "source": "/path/to/audio.wav",
  "filename": "audio.wav",
  "text": "完整转写文本",
  "chunks": [
    {"chunk": 0, "is_final": false, "text": "部分识别结果"},
    {"chunk": 1, "is_final": false, "text": "更多识别结果"},
    {"chunk": 42, "is_final": true, "text": "最终完整文本"}
  ],
  "audio_dur_s": 5.12,
  "transcribe_s": 1.23,
  "rtf": 0.24
}
```

最终文本（`text`）取 `is_final=true` 块的输出；`chunks` 记录每块的实时输出，可用于分析流式延迟。

---

## 模型对比

| 模型 | 时间戳 | 多语言 | 标点 | 流式 | 适用场景 |
|------|--------|--------|------|------|---------|
| paraformer-zh | 字符级（ms） | 否 | 需 ct-punc | 否 | 中文转写，对话多轮精度最高 |
| paraformer-zh-streaming | 无 | 否 | 需 ct-punc | 是 | 实时低延迟场景 |
| SenseVoiceSmall | token 级（s） | 是 | 内置 ITN | 否 | 多语言、情感识别 |
| Fun-ASR-Nano-2512 | token 级（s） | 否 | 内置 | 否 | 端到端，无需额外模型 |

---

## 关于模型下载与缓存

首次运行会自动从 ModelScope 下载模型到 `~/.cache/modelscope/hub/models/`。

**跳过联网检查**：将 `-m / -vm / -pm` 指定为本地缓存的绝对路径，FunASR 检测到路径存在时直接加载，跳过网络请求。

常用模型本地路径：

| 模型 | 本地路径 |
|------|---------|
| paraformer-zh | `~/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| paraformer-zh-streaming | `~/.cache/modelscope/hub/models/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404` |
| fsmn-vad | `~/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| ct-punc | `~/.cache/modelscope/hub/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large` |
| SenseVoiceSmall | `~/.cache/modelscope/hub/models/iic/SenseVoiceSmall` |
| Fun-ASR-Nano-2512 | `~/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512` |
