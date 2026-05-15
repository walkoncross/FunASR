# coding=utf-8
"""
FunASR 双声道对话转写脚本（ModelScope / HuggingFace 后端）

对立体声音频逐声道处理，利用 FunASR 内置 fsmn-vad 完成分段，
将各声道的话语按时间排序，输出对话 JSON。

Usage:
  python scripts/transcribe_conversation.py -i <stereo_audio> [OPTIONS]

  --model/-m          ASR 模型名称或本地路径 (default: paraformer-zh)
  --vad-model/-vm     VAD 模型名称或路径 (default: fsmn-vad)
  --punc-model/-pm    标点模型名称或路径 (default: ct-punc)
  --input/-i          立体声音频文件（必填）
  --output/-o         JSON 输出路径 (default: results/<basename>-conversation.json)
  --hub               模型来源: modelscope / hf (default: modelscope)
  --device/-d         推理设备: cpu / cuda:0 / mps (default: cpu)
  --batch-size/-bs    推理 batch size (default: 1)
  --channels/-c       处理声道数 (default: 2)
  --silence-gap/-sg   句间静音间隔阈值(s)，超过则切断为新话语 (default: 0.5)
  --hotwords          热词字符串，空格分隔
  --enable-update     启用 FunASR 版本检查（默认禁用）

Output format:
  {
    "source": "...",
    "filename": "...",
    "channels": 2,
    "conversations": [
      {"role": "channel_0", "text": "...", "start": 0.0, "end": 1.2},
      {"role": "channel_1", "text": "...", "start": 0.9, "end": 2.3},
      ...
    ]
  }
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FunASR 双声道对话转写工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "-m", default="paraformer-zh", help="ASR 模型名称或本地路径")
    parser.add_argument("--vad-model", "-vm", default="fsmn-vad", help="VAD 模型名称或路径")
    parser.add_argument("--punc-model", "-pm", default="ct-punc", help="标点模型名称或路径")
    parser.add_argument("--input", "-i", required=True, help="立体声音频文件路径")
    parser.add_argument("--output", "-o", default=None,
                        help="JSON 输出路径（默认: results/<basename>-conversation.json）")
    parser.add_argument("--hub", default="modelscope", choices=["modelscope", "hf"],
                        help="模型来源：modelscope 或 hf（HuggingFace）")
    parser.add_argument("--device", "-d", default="cpu", help="推理设备：cpu / cuda:0 / mps")
    parser.add_argument("--batch-size", "-bs", type=int, default=1, help="推理 batch size")
    parser.add_argument("--channels", "-c", type=int, default=2, help="处理声道数")
    parser.add_argument("--silence-gap", "-sg", type=float, default=0.5, dest="silence_gap",
                        help="句间静音间隔阈值(s)，token 间隔超过此值则切断为新话语")
    parser.add_argument("--hotwords", default=None, help="热词字符串，空格分隔")
    parser.add_argument("--enable-update", action="store_true", default=False,
                        help="启用 FunASR 版本检查（默认禁用）")
    return parser.parse_args()


def load_model(args):
    from funasr import AutoModel

    kwargs = dict(
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        disable_update=not args.enable_update,
    )
    # 空字符串视为不传，避免 AutoModel 尝试构建空模型名导致报错
    if args.vad_model:
        kwargs["vad_model"] = args.vad_model
        if args.vad_model == "fsmn-vad":
            logger.info("使用内置 fsmn-vad 模型")
            kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
    if args.punc_model:
        kwargs["punc_model"] = args.punc_model

    if args.hub == "hf":
        kwargs["hub"] = "hf"
        logger.info("使用 HuggingFace Hub 加载模型")
    else:
        logger.info("使用 ModelScope 加载模型")
    if args.hotwords:
        kwargs["hotword"] = args.hotwords

    return AutoModel(**kwargs)


def _norm_timestamps(timestamps: list) -> list[tuple[float, float]]:
    """
    统一时间戳格式为 (start_ms, end_ms) 的列表。
    支持两种输入格式：
      - paraformer:     [[start_ms, end_ms], ...]          单位 ms
      - Fun-ASR-Nano:   [{"start_time": s, "end_time": e}, ...]  单位 s（经 vad 偏移后）
    """
    if not timestamps:
        return []
    first = timestamps[0]
    if isinstance(first, dict):
        # Fun-ASR-Nano timestamps：单位秒，转为 ms
        return [(t["start_time"] * 1000, t["end_time"] * 1000) for t in timestamps]
    else:
        # paraformer timestamp：已是 ms
        return [(t[0], t[1]) for t in timestamps]


def _split_utterances_by_gap(text: str, timestamps: list, silence_gap_s: float) -> list[dict]:
    """
    将 token 级时间戳按静音间隔切分成多条话语。
    支持 paraformer ([[start_ms, end_ms]]) 和 Fun-ASR-Nano ([{"start_time":s,"end_time":e}]) 格式。
    text 中每个 token 对应 timestamps 中一个元素。
    """
    if not timestamps:
        return [{"text": text, "start": 0.0, "end": 0.0}] if text else []

    norm_ts = _norm_timestamps(timestamps)
    silence_gap_ms = silence_gap_s * 1000.0
    utterances = []
    seg_tokens: list[str] = []
    seg_start_ms: float = norm_ts[0][0]
    prev_end_ms: float = norm_ts[0][1]

    # Fun-ASR-Nano timestamps 的 token 是子词/字，text 是最终文本，长度可能不等。
    # 此时只用时间戳划出时间边界，不做字符对齐——按 gap 切出若干时间段，
    # 再把完整 text 按段数均分（退化方案）。
    # paraformer 保证字符与时间戳一一对齐，可以精确切分。
    chars = list(text)
    char_aligned = (len(chars) == len(norm_ts))

    if char_aligned:
        # 精确模式：逐字符切分（paraformer）
        n = len(norm_ts)
        for i in range(n):
            start_ms, end_ms = norm_ts[i]
            gap = start_ms - prev_end_ms
            if seg_tokens and gap >= silence_gap_ms:
                utterances.append({
                    "text": "".join(seg_tokens).strip(),
                    "start": round(seg_start_ms / 1000.0, 3),
                    "end": round(prev_end_ms / 1000.0, 3),
                })
                seg_tokens = []
                seg_start_ms = start_ms
            seg_tokens.append(chars[i])
            prev_end_ms = end_ms
        if seg_tokens:
            utterances.append({
                "text": "".join(seg_tokens).strip(),
                "start": round(seg_start_ms / 1000.0, 3),
                "end": round(prev_end_ms / 1000.0, 3),
            })
    else:
        # 时间边界模式：按 gap 切出时间段列表，text 不拆分（Fun-ASR-Nano）
        # 先找出所有切断点对应的时间范围
        seg_ranges: list[tuple[float, float]] = []  # (start_ms, end_ms)
        seg_start_ms = norm_ts[0][0]
        prev_end_ms = norm_ts[0][1]
        for start_ms, end_ms in norm_ts[1:]:
            gap = start_ms - prev_end_ms
            if gap >= silence_gap_ms:
                seg_ranges.append((seg_start_ms, prev_end_ms))
                seg_start_ms = start_ms
            prev_end_ms = end_ms
        seg_ranges.append((seg_start_ms, prev_end_ms))

        if len(seg_ranges) == 1:
            # 无切断点，整段作为一条话语
            utterances.append({
                "text": text.strip(),
                "start": round(seg_ranges[0][0] / 1000.0, 3),
                "end": round(seg_ranges[0][1] / 1000.0, 3),
            })
        else:
            # inference_with_vad 将各 VAD 分段文本以空格拼接（auto_model.py L591）
            # 因此按空格拆分后与 seg_ranges 一一对应
            parts = text.split(" ")
            for idx, (s_ms, e_ms) in enumerate(seg_ranges):
                seg_text = parts[idx].strip() if idx < len(parts) else ""
                utterances.append({
                    "text": seg_text,
                    "start": round(s_ms / 1000.0, 3),
                    "end": round(e_ms / 1000.0, 3),
                })

    return [u for u in utterances if u["text"]]


def transcribe_channel(model, wav_path: str, args) -> list[dict]:
    """
    转写单声道 wav，返回话语列表。
    支持两种时间戳格式：
      - paraformer: item["timestamp"] = [[start_ms, end_ms], ...]  字符级
      - Fun-ASR-Nano: item["timestamps"] = [{"start_time":s, "end_time":e}, ...]  token 级（秒）
    用 --silence-gap 阈值按静音间隔切分为多条话语。
    """
    generate_kwargs = {}
    if args.hotwords:
        generate_kwargs["hotword"] = args.hotwords
    # 强制每个 VAD segment 单独推理：
    # - batch_size_threshold_s=0：禁止多 segment 合并成 batch
    # - batch_size_s=0：使 inference_with_vad 内部 batch_size=1（毫秒粒度→样本数=1）
    # 避免 Fun-ASR-Nano 等不支持 batch decoding 的模型报错
    if args.vad_model:
        generate_kwargs["batch_size_threshold_s"] = 0
        generate_kwargs["batch_size_s"] = 0

    results = model.generate(input=wav_path, **generate_kwargs)

    utterances = []
    if not results:
        return utterances

    for item in results:
        text = item.get("text", "").strip()
        if not text:
            continue
        # paraformer: "timestamp"（字符级 ms 列表）；Fun-ASR-Nano: "timestamps"（token 级秒字典）
        ts = item.get("timestamp") or item.get("timestamps") or []
        segs = _split_utterances_by_gap(text, ts, args.silence_gap)
        utterances.extend(segs)

    return utterances


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        logger.error("--input 必须是文件: %r", args.input)
        sys.exit(1)

    audio_data, sample_rate = sf.read(args.input, always_2d=True)
    total_dur_s = audio_data.shape[0] / sample_rate
    num_channels = audio_data.shape[1]
    channels_to_process = min(args.channels, num_channels)

    if num_channels < args.channels:
        logger.warning("音频仅有 %d 声道，实际处理 %d 声道", num_channels, channels_to_process)

    basename = os.path.splitext(os.path.basename(args.input))[0]
    output_path = args.output or f"results/{basename}-conversation.json"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    logger.info("[config] model=%s  vad=%s  punc=%s", args.model, args.vad_model, args.punc_model)
    logger.info("[config] hub=%s  device=%s  batch_size=%d", args.hub, args.device, args.batch_size)
    logger.info("[config] silence_gap=%.2fs  channels=%d", args.silence_gap, args.channels)
    logger.info("[input]  %s  (%d ch, %.1fs)", args.input, num_channels, total_dur_s)

    t0 = time.perf_counter()
    model = load_model(args)
    logger.info("[timing] 模型加载: %.3fs", time.perf_counter() - t0)

    all_utterances = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for ch in range(channels_to_process):
            channel_audio = audio_data[:, ch]
            tmp_wav = os.path.join(tmpdir, f"ch{ch}.wav")
            sf.write(tmp_wav, channel_audio, sample_rate)

            logger.info("\n[channel %d] 开始转写...", ch)
            t1 = time.perf_counter()
            utterances = transcribe_channel(model, tmp_wav, args)
            elapsed = time.perf_counter() - t1

            ch_dur = len(channel_audio) / sample_rate
            rtf = elapsed / ch_dur if ch_dur > 0 else 0.0
            logger.info("[channel %d] %d 段话语  耗时=%.3fs  RTF=%.4f",
                        ch, len(utterances), elapsed, rtf)

            for u in utterances:
                logger.info("  [%.2f-%.2fs] %r", u["start"], u["end"], u["text"])
                all_utterances.append({
                    "role": f"channel_{ch}",
                    "text": u["text"],
                    "start": u["start"],
                    "end": u["end"],
                })

    # 按开始时间排序，同时刻按声道顺序
    all_utterances.sort(key=lambda u: (u["start"], u["role"]))

    output = {
        "source": args.input,
        "filename": os.path.basename(args.input),
        "channels": channels_to_process,
        "conversations": all_utterances,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("\n[result] 共 %d 条话语", len(all_utterances))
    for u in all_utterances[:8]:
        logger.info("  [%.2f-%.2f] %s: %r", u["start"], u["end"], u["role"], u["text"])
    if len(all_utterances) > 8:
        logger.info("  ... (%d 条更多)", len(all_utterances) - 8)
    logger.info("[output] JSON 已写入: %s", output_path)


if __name__ == "__main__":
    main()
