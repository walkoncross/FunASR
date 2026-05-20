# 流式 VAD + 流式 ASR 搭配设计说明

## 背景

FunASR 官方 README 分别给出了流式 VAD 和流式 ASR 的独立示例，但没有说明两者如何组合使用。`transcribe_streaming_with_vad.py` 是本项目对这一链路的完整实现，本文说明设计思路与关键细节。

---

## 两个模型的差异

| | 流式 VAD (fsmn-vad) | 流式 ASR (paraformer-zh-streaming) |
|---|---|---|
| `chunk_size` 含义 | `int`，单位 ms（如 `200`） | `[lookahead, chunk, shift]`，单位帧（1 帧 = 60ms） |
| `cache` 内容 | FSMN 隐状态、滑动窗口缓冲 | Encoder/Decoder KV cache |
| 输出 | `[[beg_ms, end_ms]]` 时间戳 | 文字 token |
| flush 触发 | `is_final=True` | `is_final=True` |

流式 VAD 的 `cache` 跨 chunk 保留 FSMN 内部状态，是真正的有状态流式推理。官方示例中两者各自独立循环，直接组合会面临一个核心问题：

> **VAD 的 chunk 粒度（200ms）和 ASR 的 chunk 粒度（480ms）不同，且 ASR 只应收到 VAD 判断为"语音"的片段。**

---

## 搭配方案

### wall_step：统一推进粒度

取两者 chunk 大小的较大值作为每步推进量：

```python
wall_step = max(chunk_stride_asr, vad_chunk_samples)
          = max(7680, 3200)   # 480ms vs 200ms
          = 7680 samples      # 480ms
```

每步同时喂给 VAD 和 ASR，避免时间轴不对齐。

### 每步的处理逻辑

```
┌─ 每个 wall_step (480ms) ────────────────────────────────────┐
│  1. VAD.generate(pcm, cache, is_final)                      │
│     → []              静音，不动                            │
│     → [[beg, -1]]     语音开始，记录 seg_start_sample       │
│     → [[-1, end]]     语音结束，触发 ASR flush              │
│     → [[beg, end]]    同一块内开始+结束，直接 flush         │
│                                                             │
│  2. 根据 VAD 结果决定 ASR 动作：                            │
│     - 静音中           不调用 ASR                           │
│     - 语音段中间       ASR.generate(pcm, is_final=False)    │
│     - 语音段结束       ASR.generate(pcm, is_final=True)     │
│                        → 输出一条 utterance，清空 asr_cache │
└─────────────────────────────────────────────────────────────┘
```

### cache 生命周期

| cache | 生命周期 |
|---|---|
| `vad_cache` | 全程保持，贯穿整个音频文件 |
| `asr_cache` | 每条 utterance 结束后清空（`state.asr_cache = {}`） |

VAD cache 持续检测端点；ASR cache 只在当前语音段内有效，跨句清空避免上文污染下文。

---

## VAD 输出格式详解

流式 VAD 每个 chunk 可能返回以下四种结果（时间为相对音频起点的绝对毫秒数）：

| 输出 | 含义 |
|---|---|
| `[[beg, end]]` | 语音开始和结束都在本 chunk 内 |
| `[[beg, -1]]` | 检测到语音开始，尚未结束 |
| `[[-1, end]]` | 检测到语音结束（开始在前一 chunk） |
| `[]` | 本 chunk 为静音 |

`is_final=True` 时 VAD flush 残余缓冲，可能额外输出一个结束点。

---

## 与 transcribe_streaming.py 的本质区别

`transcribe_streaming.py` 的流式 ASR 示例是将整段音频无脑按 chunk 切分送入 ASR，`is_final` 只在最后一个 chunk 触发。这等价于"已知语音边界"的场景，适合实验室基准测试。

加入流式 VAD 后的关键变化：

1. **ASR 只处理有语音的片段**，静音段跳过，降低无效计算
2. **VAD 的"说话结束"信号是 AI 回复的触发点**——在 AI 外呼/客服中，系统需要知道"用户说完了"才能开始回复，这正是 `is_final=True` 语义所对应的时刻
3. **每句话独立的 ASR cache**，不同说话人、不同轮次之间不相互干扰

---

## 延迟指标说明

脚本输出以下延迟字段（每条 utterance）：

| 字段 | 含义 |
|---|---|
| `seg_asr_ms` | 本句所有 ASR chunk 推理时间之和，即**端到端延迟**（用户说完 → 系统有文字） |
| `final_chunk_ms` | 最后一个 `is_final=True` chunk 的单次推理时间 |
| `vad_s` | 本句 VAD 推理累计时间（秒） |

### 为何没有 TTFT

paraformer-zh-streaming 在 `generate()` API 层面**不支持 token 级流式输出**。其 `inference()` 内部循环逐 chunk 累积 token，只在最后一个 `is_final=True` chunk 结束后调用 `sentence_postprocess()` 一次性返回完整文本。因此"第一个 token 返回时刻"恒等于"最后一个 chunk 推理结束时刻"，TTFT 无法独立测量，`seg_asr_ms` 即是端到端延迟的完整描述。

这是 CIF（Continuous Integrate-and-Fire）架构的设计决策：模型在 `is_final=False` 时刻意屏蔽 chunk 末尾的 alpha 累积，防止不完整词边界提前输出。

---

## 双声道处理

两个声道（`user` / `agent`）各自维护独立的 `ChannelState`，包含独立的：

- `vad_cache`：互不干扰的端点检测状态
- `asr_cache`：互不干扰的解码上下文
- `seg_asr_elapsed_ms` / `seg_vad_elapsed_ms`：独立计时

两路共享同一条 wall-clock 时间轴推进，最终所有 utterances 按 `start_s` 排序合并为统一的对话序列。

`--channels` 参数可选择只处理单声道（如 `--channels 0` 只处理用户侧），便于单路场景测试。
