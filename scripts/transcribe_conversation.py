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
    parser.add_argument("--hotwords", default=None, help="热词字符串，空格分隔")
    parser.add_argument("--enable-update", action="store_true", default=False,
                        help="启用 FunASR 版本检查（默认禁用）")
    return parser.parse_args()


def load_model(args):
    from funasr import AutoModel

    kwargs = dict(
        model=args.model,
        vad_model=args.vad_model,
        punc_model=args.punc_model,
        device=args.device,
        batch_size=args.batch_size,
        disable_update=not args.enable_update,
    )
    if args.hub == "hf":
        kwargs["hub"] = "hf"
        logger.info("使用 HuggingFace Hub 加载模型")
    else:
        logger.info("使用 ModelScope 加载模型")
    if args.hotwords:
        kwargs["hotword"] = args.hotwords

    return AutoModel(**kwargs)


def transcribe_channel(model, wav_path: str, args) -> list[dict]:
    """
    转写单声道 wav，返回话语列表。
    FunASR + fsmn-vad 会在结果中携带 timestamp 字段（句级时间戳）。
    每条结果格式示例：
      {"text": "你好", "timestamp": [[0, 500], [500, 1200]], ...}
    timestamp 单位为毫秒。
    """
    generate_kwargs = {}
    if args.hotwords:
        generate_kwargs["hotword"] = args.hotwords

    results = model.generate(input=wav_path, **generate_kwargs)

    utterances = []
    if not results:
        return utterances

    for item in results:
        text = item.get("text", "").strip()
        if not text:
            continue

        # fsmn-vad 输出的 timestamp 是 [[start_ms, end_ms], ...] 字符级/词级时间戳
        # 取整段的首尾作为话语时间
        ts = item.get("timestamp")
        if ts and len(ts) > 0:
            start_s = ts[0][0] / 1000.0
            end_s = ts[-1][1] / 1000.0
        else:
            start_s = 0.0
            end_s = 0.0

        utterances.append({"text": text, "start": start_s, "end": end_s})

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
