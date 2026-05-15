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


def _split_utterances_by_gap(text: str, timestamps: list, silence_gap_s: float) -> list[dict]:
    """
    将字符级 timestamp [[start_ms, end_ms], ...] 按静音间隔切分成多条话语。
    text 中每个字符对应 timestamps 中一个元素（FunASR paraformer 输出保证对齐）。
    """
    if not timestamps:
        return [{"text": text, "start": 0.0, "end": 0.0}] if text else []

    silence_gap_ms = silence_gap_s * 1000.0
    utterances = []
    seg_chars: list[str] = []
    seg_start_ms: float = timestamps[0][0]
    prev_end_ms: float = timestamps[0][1]

    chars = list(text)  # 字符列表，与 timestamps 对齐
    # 若字符数与 timestamp 数不匹配，按最短截断
    n = min(len(chars), len(timestamps))

    for i in range(n):
        start_ms, end_ms = timestamps[i]
        gap = start_ms - prev_end_ms

        if seg_chars and gap >= silence_gap_ms:
            # 切断：保存当前句
            utterances.append({
                "text": "".join(seg_chars).strip(),
                "start": round(seg_start_ms / 1000.0, 3),
                "end": round(prev_end_ms / 1000.0, 3),
            })
            seg_chars = []
            seg_start_ms = start_ms

        seg_chars.append(chars[i])
        prev_end_ms = end_ms

    # 最后一段
    if seg_chars:
        utterances.append({
            "text": "".join(seg_chars).strip(),
            "start": round(seg_start_ms / 1000.0, 3),
            "end": round(prev_end_ms / 1000.0, 3),
        })

    return [u for u in utterances if u["text"]]


def transcribe_channel(model, wav_path: str, args) -> list[dict]:
    """
    转写单声道 wav，返回话语列表。
    FunASR paraformer + fsmn-vad 返回 1 条结果，timestamp 为字符级 [[start_ms, end_ms], ...]。
    用 --silence-gap 阈值将字符级时间戳重新切分为多条话语。
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
        ts = item.get("timestamp")
        if not text:
            continue
        segs = _split_utterances_by_gap(text, ts or [], args.silence_gap)
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
