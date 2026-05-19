# FunASR Scripts

基于 [FunASR](https://github.com/modelscope/FunASR) 的语音转写脚本，支持 ModelScope / HuggingFace 双后端。

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `transcribe.py` | 单文件或批量非流式转写，支持多声道分离 |
| `transcribe_conversation.py` | 双声道对话转写，输出带时间戳的多轮对话 JSON |
| `transcribe_streaming.py` | 流式转写，逐块实时输出识别结果 |
| `transcribe_streaming_with_vad.py` | 模拟在线双声道流式转写（VAD + 流式 ASR），适用于 AI 外呼 / AI 客服 |

### 快捷脚本（按模型预设参数）

| 脚本 | 模型 | VAD | 用途 |
|------|------|-----|------|
| `transcribe_by_nano.sh` | Fun-ASR-Nano-2512 | 带 | 单文件/目录转写（长音频） |
| `transcribe_by_nano_no_vad.sh` | Fun-ASR-Nano-2512 | 不带 | 同上（短音频 < 30s，速度更快） |
| `transcribe_by_sensevoice.sh` | SenseVoiceSmall | 带 | 多语言 + ITN（长音频） |
| `transcribe_by_sensevoice_no_vad.sh` | SenseVoiceSmall | 不带 | 多语言 + ITN（短音频 < 30s） |
| `transcribe_by_paraformer.sh` | paraformer-zh | 带 | 字符级时间戳（长音频） |
| `transcribe_by_paraformer_no_vad.sh` | paraformer-zh | 不带 | 字符级时间戳（短音频 < 30s） |
| `transcribe_conversation_by_nano.sh` | Fun-ASR-Nano-2512 | 带 | 双声道对话转写 |
| `transcribe_conversation_by_sensevoice.sh` | SenseVoiceSmall | 带 | 双声道对话转写，多语言 |
| `transcribe_conversation_by_paraformer.sh` | paraformer-zh | 带 | 双声道对话转写，精度最高 |
| `transcribe_streaming_by_paraformer.sh` | paraformer-zh-streaming | — | 流式转写（文件模拟） |
| `transcribe_streaming_with_vad_by_paraformer.sh` | paraformer-zh-streaming + fsmn-vad | 带 | 模拟在线双声道流式转写 |

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
| `--silence-gap` | `-sg` | `0.5` | 按静音间隔切分的阈值（秒），`0` 表示不切分 |

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

输出 JSON，文件名格式为 `<stem>.<model>.<vad>.<punc>.json`。

示例：
- `录音1.paraformer-zh.fsmn-vad.ct-punc.json`
- `录音1.SenseVoiceSmall.fsmn-vad.no-punc.json`
- `录音1.Fun-ASR-Nano-2512.fsmn-vad.no-punc.json`
- `录音1.paraformer-zh.no-vad.ct-punc.json`

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
  "text": "第一段识别结果 第二段识别结果",
  "segments": [
    {"text": "第一段识别结果", "start": 0.0, "end": 5.0},
    {"text": "第二段识别结果", "start": 5.2, "end": 10.1}
  ]
}
```

- `start` / `end` 单位为秒；SenseVoice 各段 start/end 对应 VAD 段边界
- `rtf`：实时率（越小越快）；`rtfx`：逆实时率 = `1/rtf`（越大越快）
- 声道分离模式（`-sc`）时，文件名添加 `.channel0` / `.channel1` 段，JSON 中额外包含 `"channel": 0`

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
| `--output` | `-o` | `results/<basename>.conversation.<model>.<vad>.<punc>.json` | JSON 输出路径 |
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

文件名格式：`<stem>.conversation.<model>.<vad>.<punc>.json`

```json
{
  "source": "/path/to/stereo.wav",
  "filename": "stereo.wav",
  "channels": 2,
  "audio_dur_s": 306.68,
  "transcribe_s": 52.32,
  "rtf": 0.1707,
  "rtfx": 5.86,
  "vad_s": null,
  "vad_rtf": null,
  "vad_rtfx": null,
  "punct_s": null,
  "punct_rtf": null,
  "punct_rtfx": null,
  "model_name": "paraformer-zh",
  "vad_model": "fsmn-vad",
  "punc_model": "ct-punc",
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

> **关于 VAD**：本脚本处理的是已录制的音频文件，直接按固定 chunk 切块推理，无需 VAD。
> 若对接真实在线场景（麦克风流、电话 RTP 流），需在上游加 VAD（如 fsmn-vad）检测说话起止，
> 典型链路为：`音频流 → fsmn-vad（端点检测）→ 有效语音段 → paraformer-zh-streaming → 文本`，
> VAD 同时承担"用户是否说完"的判断，用于触发 AI 回复。

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
| `--chunk-size` | | `0 8 4` | 流式分块配置 `[lookahead chunk shift]`，单位帧（60ms）。默认 `0 8 4` = 480ms chunk / 240ms lookahead（实时场景） |
| `--encoder-look-back` | | `4` | Encoder self-attention 回看 chunk 数。实时场景用 `4`，离线批量用 `8` |
| `--decoder-look-back` | | `1` | Decoder cross-attention 回看 encoder chunk 数。实时场景用 `1`，离线批量用 `2` |
| `--hotwords` | | `None` | 热词字符串，空格分隔 |
| `--enable-update` | | `False` | 启用 FunASR PyPI 版本检查 |
| `--separate-channel` | `-sc` | `False` | 分声道独立转写 |

**chunk-size 配置档位**：`[lookahead, chunk, shift]`，单位为帧（1 帧 = 60ms）。

| 配置 | chunk 窗口 | look-ahead | encoder-look-back | decoder-look-back | 适用场景 |
|------|-----------|-----------|-------------------|-------------------|---------|
| `0 8 4` | 480ms | 240ms | 4 | 1 | **在线 AI 外呼 / AI 客服**（默认，低延迟优先） |
| `0 10 5` | 600ms | 300ms | 4 | 1 | 平衡延迟与精度 |
| `0 16 8` | 960ms | 480ms | 8 | 2 | 离线批量处理（精度优先） |

> 延迟越低，每次推理可用上下文越短，识别准确率相应略降；离线批量处理时推荐使用 `0 16 8 + encoder-look-back 8 + decoder-look-back 2`。

### 示例

```bash
# 在线 AI 外呼 / AI 客服（默认，低延迟）
python scripts/transcribe_streaming.py -i audio.wav -d mps

# 离线批量处理（精度优先）
python scripts/transcribe_streaming.py -i audio.wav -d mps \
  --chunk-size 0 16 8 \
  --encoder-look-back 8 \
  --decoder-look-back 2
```

### 输出格式

文件名格式：`<stem>.streaming.<model>.no-vad.<punc>.json`（流式无 VAD）

使用 `--separate-channel` 时：`<stem>.channel0.streaming.<model>.no-vad.<punc>.json`

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
  "text": "完整转写文本",
  "chunks": [
    {"chunk": 0, "is_final": false, "text": "部分识别结果"},
    {"chunk": 1, "is_final": false, "text": "更多识别结果"},
    {"chunk": 42, "is_final": true, "text": "最终完整文本"}
  ]
}
```

最终文本（`text`）为所有 chunk 输出拼接；`chunks` 记录每块的实时输出，可用于分析流式延迟。

---

## transcribe_streaming_with_vad.py

模拟在线双声道流式转写，使用 **fsmn-vad + paraformer-zh-streaming** 构建完整的端点检测→流式 ASR 链路，面向 AI 外呼和 AI 客服场景。

### 与 transcribe_streaming.py 的区别

| 维度 | transcribe_streaming.py | transcribe_streaming_with_vad.py |
|------|------------------------|----------------------------------|
| VAD | 无，固定 chunk 切块 | fsmn-vad 端点检测，仅语音段送 ASR |
| 场景 | 已录制文件的批量模拟 | 模拟在线实时双路电话流 |
| 声道处理 | 各 channel 独立跑完整文件 | 两路共享时间轴，独立 VAD/ASR 状态 |
| 输出结构 | 平铺 chunks 列表 | 带 `role/start_s/end_s` 的对话轮次 |
| flush 触发 | 文件末尾 is_final | VAD 检测到段尾即 flush（模拟"用户说完"信号） |

### 架构

```
wall-clock 推进（每步 max(asr_chunk_stride, vad_chunk_samples)）
    │
    ├── channel 0 (user / 客户)
    │       └── fsmn-vad  ──→ 检测到说话开始：开始累积 speech_buf
    │                    ──→ 检测到说话结束：ASR flush → utterance 输出
    │
    └── channel 1 (agent / 坐席)
            └── fsmn-vad  ──→ 同上（独立 vad_cache / asr_cache）
                                            │
                                两路 utterances 按 start_s 排序合并
```

VAD 检测到"说话结束"是 AI 外呼/客服中**决定何时回复**的关键信号，本脚本通过 `is_final=True` 触发 ASR flush 来模拟这一行为。

### 用法

```bash
python scripts/transcribe_streaming_with_vad.py -i <音频文件或目录> [OPTIONS]
# 或使用快捷脚本
./scripts/transcribe_streaming_with_vad_by_paraformer.sh <音频文件或目录>
```

### 参数

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | 必填 | 音频文件或目录（需双声道） |
| `--output` | `-o` | `./results/` | 输出目录 |
| `--model` | `-m` | `paraformer-zh-streaming` | 流式 ASR 模型 |
| `--vad-model` | `-vm` | `fsmn-vad` | VAD 模型 |
| `--punc-model` | `-pm` | `ct-punc` | 标点模型，留空则禁用 |
| `--hub` | | `modelscope` | 模型来源：`modelscope` / `hf` |
| `--device` | `-d` | `cpu` | 推理设备：`cpu` / `cuda:0` / `mps` |
| `--chunk-size` | | `0 8 4` | ASR 分块配置，同 transcribe_streaming.py |
| `--encoder-look-back` | | `4` | Encoder self-attention 回看 chunk 数 |
| `--decoder-look-back` | | `1` | Decoder cross-attention 回看 encoder chunk 数 |
| `--vad-chunk-ms` | | `200` | VAD 每次处理的窗口大小（ms） |
| `--hotwords` | | `None` | 热词字符串，空格分隔 |
| `--enable-update` | | `False` | 启用 FunASR PyPI 版本检查 |

### 输出格式

文件名格式：`<stem>.streaming-vad.<asr>.<vad>.<punc>.json`

```json
{
  "source": "/path/to/audio.wav",
  "filename": "audio.wav",
  "audio_dur_s": 306.68,
  "transcribe_s": 12.34,
  "rtf": 0.040,
  "rtfx": 24.87,
  "asr_model": "paraformer-zh-streaming",
  "vad_model": "fsmn-vad",
  "punc_model": "ct-punc",
  "conversations": [
    {"role": "user",  "text": "你好，我想取消订单", "start_s": 17.17, "end_s": 19.50},
    {"role": "agent", "text": "好的，请稍等",       "start_s": 20.10, "end_s": 21.30}
  ]
}
```

`conversations` 结构与 `transcribe_conversation.py` 的输出格式对齐（均含 `role/text/start_s/end_s`），下游分析代码可复用。

---

## 模型对比

| 模型 | 时间戳 | 多语言 | 标点 | 流式 | 适用场景 |
|------|--------|--------|------|------|---------|
| paraformer-zh | 字符级（ms） | 否 | 需 ct-punc | 否 | 中文转写，对话多轮精度最高 |
| paraformer-zh-streaming | 无 | 否 | 需 ct-punc | 是 | 实时低延迟场景 |
| SenseVoiceSmall | token 级（s） | 是 | 内置 ITN | 否 | 多语言、情感识别 |
| Fun-ASR-Nano-2512 | token 级（s） | 否 | 内置 | 否 | 端到端，无需额外模型 |

---

## 关于 VAD 与长音频

三种模型（paraformer-zh、SenseVoiceSmall、Fun-ASR-Nano）的 encoder 均无硬截断限制，但**长音频不带 VAD 存在两个问题**：

1. **OOM**：encoder 的 attention 复杂度为 O(T²)，长音频显存消耗随时长平方增长
2. **精度下降**：模型训练时单段最长约 30s，超长输入识别准确率下降

**建议**：
- 长音频（> 30s）：带 `--vad-model fsmn-vad`，VAD 先切段再分别识别，支持任意时长
- 短音频（< 30s）：可省略 VAD（`--vad-model ""`），减少一次模型加载，延迟更低

**`--silence-gap` 与 VAD 的关系**：`--silence-gap` 是在 VAD 分段完成、模型推理返回后，对时间戳在本地再做二次切分；不带 VAD 时该参数无效（整段只有 1 条结果）。SenseVoice 按 VAD 段边界切分，`--silence-gap` 不影响切分粒度（每段对应一个 VAD 段）。

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
