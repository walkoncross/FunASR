# FunASR Scripts

基于 [FunASR](https://github.com/modelscope/FunASR) 的语音转写脚本，支持 ModelScope / HuggingFace 双后端。

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `transcribe.py` | 单文件或批量转写，支持多声道分离 |
| `transcribe_conversation.py` | 双声道对话转写，输出带时间戳的对话 JSON |

---

## transcribe.py

单文件或批量语音转写。

### 用法

```bash
python scripts/transcribe.py -i <音频文件或目录> [OPTIONS]
```

### 参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | 必填 | 音频文件或目录 |
| `--output` | `-o` | `./results/` | 输出目录 |
| `--output-format` | `-f` | `json` | 输出格式：`txt` / `json` |
| `--model` | `-m` | `paraformer-zh` | ASR 模型名称或本地路径 |
| `--vad-model` | `-vm` | `fsmn-vad` | VAD 模型名称或路径 |
| `--punc-model` | `-pm` | `ct-punc` | 标点模型名称或路径 |
| `--hub` | | `modelscope` | 模型来源：`modelscope` / `hf` |
| `--device` | `-d` | `cpu` | 推理设备：`cpu` / `cuda:0` / `mps` |
| `--batch-size` | `-bs` | `1` | 推理 batch size |
| `--separate-channel` | `-sc` | `False` | 分离声道分别转录，每声道独立输出 |
| `--hotwords` | | `None` | 热词字符串，空格分隔 |
| `--enable-update` | | `False` | 启用 FunASR PyPI 版本检查 |

### 示例

```bash
# 转写单个文件（输出 JSON 到 ./results/）
python scripts/transcribe.py -i audio.wav

# 批量转写目录
python scripts/transcribe.py -i ./audio_dir/ -f txt

# 分离双声道，使用 MPS 加速
python scripts/transcribe.py -i stereo.wav -sc -d mps

# 使用本地模型（跳过联网检查）
python scripts/transcribe.py -i audio.wav \
  -m ~/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  -vm ~/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  -pm ~/.cache/modelscope/hub/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large
```

### 输出格式

**txt**：纯文本，每个文件对应一个 `<stem>.funasr.txt`。

**json**：
```json
{
  "source": "/path/to/audio.wav",
  "filename": "audio.wav",
  "text": "识别结果",
  "audio_dur_s": 12.345,
  "transcribe_s": 1.234,
  "rtf": 0.1
}
```

声道分离模式（`-sc`）时，文件名附加 `_channel0` / `_channel1` 后缀，JSON 中额外包含 `"channel": 0`。

---

## transcribe_conversation.py

双声道对话转写。逐声道处理，利用 FunASR 内置 fsmn-vad 的字符级时间戳按静音间隔切分话语，最终按时间排序输出对话 JSON。

### 用法

```bash
python scripts/transcribe_conversation.py -i <立体声音频> [OPTIONS]
```

### 参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | 必填 | 立体声音频文件 |
| `--output` | `-o` | `results/<basename>-conversation.json` | JSON 输出路径 |
| `--model` | `-m` | `paraformer-zh` | ASR 模型名称或本地路径 |
| `--vad-model` | `-vm` | `fsmn-vad` | VAD 模型名称或路径 |
| `--punc-model` | `-pm` | `ct-punc` | 标点模型名称或路径 |
| `--hub` | | `modelscope` | 模型来源：`modelscope` / `hf` |
| `--device` | `-d` | `cpu` | 推理设备：`cpu` / `cuda:0` / `mps` |
| `--batch-size` | `-bs` | `1` | 推理 batch size |
| `--channels` | `-c` | `2` | 处理声道数 |
| `--silence-gap` | `-sg` | `0.5` | 句间静音间隔阈值（秒），超过则切断为新话语 |
| `--hotwords` | | `None` | 热词字符串，空格分隔 |
| `--enable-update` | | `False` | 启用 FunASR PyPI 版本检查 |

### 示例

```bash
# 默认参数转写双声道录音
python scripts/transcribe_conversation.py -i stereo.wav

# 调大切分间隔，得到更长的句子
python scripts/transcribe_conversation.py -i stereo.wav --silence-gap 1.0

# 使用 MPS 加速 + 本地模型
python scripts/transcribe_conversation.py -i stereo.wav -d mps \
  -m ~/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch \
  -vm ~/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch \
  -pm ~/.cache/modelscope/hub/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large
```

### 输出格式

```json
{
  "source": "/path/to/stereo.wav",
  "filename": "stereo.wav",
  "channels": 2,
  "conversations": [
    {"role": "channel_0", "text": "喂，你好，是袁雪珍女士吗？", "start": 17.17, "end": 19.3},
    {"role": "channel_1", "text": "我知道你应该是那个啊。", "start": 20.7, "end": 22.5},
    ...
  ]
}
```

`role` 字段对应声道编号（`channel_0` = 第 1 声道，`channel_1` = 第 2 声道），`start` / `end` 单位为秒。

---

## 关于模型下载

首次运行时会自动从 ModelScope 下载模型到 `~/.cache/modelscope/hub/models/iic/`，下载完成后会缓存在本地。

**跳过联网检查**：直接将 `-m / -vm / -pm` 指定为本地缓存路径，FunASR 检测到路径存在时会跳过网络请求，直接加载本地文件。

默认缓存路径：

| 模型 | 本地路径 |
|------|---------|
| paraformer-zh | `~/.cache/modelscope/hub/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` |
| fsmn-vad | `~/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch` |
| ct-punc | `~/.cache/modelscope/hub/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large` |
