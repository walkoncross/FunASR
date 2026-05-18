# coding=utf-8
"""
FunASR 语音转写脚本（支持 ModelScope / HuggingFace 后端）

Usage:
  python scripts/transcribe.py -i <音频文件或目录> [OPTIONS]

  --model/-m          ASR 模型名称或本地路径 (default: paraformer-zh)
  --vad-model/-vm     VAD 模型名称或路径 (default: fsmn-vad)
  --punc-model/-pm    标点模型名称或路径 (default: ct-punc)
  --input/-i          音频文件或目录（必填）
  --output/-o         输出目录 (default: ./results/)
  --hub               模型来源: modelscope / hf (default: modelscope)
  --device/-d         推理设备: cpu / cuda:0 / mps (default: cpu)
  --batch-size/-bs    推理 batch size (default: 1)
  --separate-channel/-sc  分离声道分别转录
  --hotwords          热词字符串，空格分隔
  --enable-update     启用 FunASR 版本检查（默认禁用）

Output format (json):
  {
    "source": "...",
    "filename": "...",
    "text": [
      {"text": "识别结果", "start": 0.0, "end": 5.0},
      ...
    ],
    "audio_dur_s": 12.34,
    "transcribe_s": 1.23,
    "rtf": 0.1
  }
  start / end 单位为秒；SenseVoice 无时间戳时两者均为 null。
  启用 --silence-gap 时，每个 VAD 段内按静音间隔再切分为多条；
  默认 silence_gap=0.5s（按静音间隔切分），设为 0 则每个 VAD 段整体作为一条。
"""

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".aac"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FunASR 语音转写工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "-m", default="paraformer-zh", help="ASR 模型名称或本地路径")
    parser.add_argument("--vad-model", "-vm", default="fsmn-vad", help="VAD 模型名称或路径")
    parser.add_argument("--punc-model", "-pm", default="ct-punc", help="标点模型名称或路径")
    parser.add_argument("--input", "-i", required=True, help="音频文件或目录")
    parser.add_argument("--output", "-o", default="./results/", help="输出目录")
    parser.add_argument("--hub", default="modelscope", choices=["modelscope", "hf"],
                        help="模型来源：modelscope 或 hf（HuggingFace）")
    parser.add_argument("--device", "-d", default="cpu", help="推理设备：cpu / cuda:0 / mps")
    parser.add_argument("--batch-size", "-bs", type=int, default=1, help="推理 batch size")
    parser.add_argument("--separate-channel", "-sc", action="store_true", default=False,
                        help="分离声道分别转录，每声道独立输出")
    parser.add_argument("--hotwords", default=None, help="热词字符串，空格分隔")
    parser.add_argument("--enable-update", action="store_true", default=False,
                        help="启用 FunASR 版本检查（默认禁用）")
    # SenseVoice 专用参数
    parser.add_argument("--language", default=None,
                        help="语言代码（SenseVoice）：auto / zh / en / yue / ja / ko / nospeech")
    parser.add_argument("--use-itn", action="store_true", default=False,
                        help="启用标���与数字规范化 ITN（SenseVoice）")
    parser.add_argument("--merge-vad", action="store_true", default=False,
                        help="合并短 VAD 分段（SenseVoice）")
    parser.add_argument("--merge-length-s", type=float, default=15.0,
                        help="合并 VAD 分段的最大时长，秒（SenseVoice，需配合 --merge-vad）")
    parser.add_argument("--silence-gap", "-sg", type=float, default=0.5, dest="silence_gap",
                        help="按静音间隔切分的阈值（秒），0 表示不切分（整段一条）")
    return parser.parse_args()


def _timed(label: str, fn, *args, audio_dur_s: float = 0.0, **kwargs):
    """执行 fn 并返回 (result, elapsed_s)，打印耗时和 RTF。"""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    logger.info("[timing] %s: %.3fs", label, elapsed)
    if audio_dur_s > 0:
        rtf = elapsed / audio_dur_s
        logger.info("[RTF]    RTF=%.4f  RTFx=%.2f", rtf, 1 / rtf)
    return result, elapsed


def _audio_duration(path: str) -> float:
    """返回音频时长（秒），依赖 soundfile；失败时返回 0。"""
    try:
        import soundfile as sf
        return sf.info(path).duration
    except Exception:
        return 0.0


def load_model(args) -> "AutoModel":
    """根据参数加载 FunASR AutoModel，支持 modelscope / hf 后端。"""
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


def _clean_text(text: str) -> str:
    """清除 SenseVoice 原始标签，返回干净文本。"""
    if text and re.search(r"<\|(?:zh|en|yue|ja|ko|nospeech)\|>", text):
        try:
            from funasr.utils.postprocess_utils import rich_transcription_postprocess
            return rich_transcription_postprocess(text)
        except ImportError:
            return re.sub(r"<\|[^|]+\|>", "", text).strip()
    return text


def _ts_bounds(ts_list: list) -> tuple[float | None, float | None]:
    """
    从时间戳列表中取首尾，返回 (start_s, end_s)。
    - paraformer timestamp item: [start_ms, end_ms]
    - Fun-ASR-Nano timestamps item: {"start_time": s, "end_time": s}
    无有效时间戳时返回 (None, None)。
    """
    if not ts_list:
        return None, None
    first, last = ts_list[0], ts_list[-1]
    if isinstance(first, dict):
        return first.get("start_time"), last.get("end_time")
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        return round(first[0] / 1000.0, 3), round(last[1] / 1000.0, 3)
    return None, None


def _norm_timestamps(timestamps: list) -> list[tuple[float, float]]:
    """统一时间戳格式为 (start_ms, end_ms) 列表。"""
    if not timestamps:
        return []
    first = timestamps[0]
    if isinstance(first, dict):
        return [(t["start_time"] * 1000, t["end_time"] * 1000) for t in timestamps]
    return [(t[0], t[1]) for t in timestamps]


def _split_by_gap(text: str, timestamps: list, silence_gap_s: float) -> list[dict]:
    """
    将单条 VAD 段按静音间隔切分为多条话语。
    silence_gap_s <= 0 时直接整段返回（不切分）。
    支持 paraformer([[start_ms, end_ms]]) 和 Fun-ASR-Nano([{"start_time":s,"end_time":e}]) 格式。
    """
    if silence_gap_s <= 0 or not timestamps:
        start_s, end_s = _ts_bounds(timestamps)
        return [{"text": text, "start": start_s, "end": end_s}] if text else []

    norm_ts = _norm_timestamps(timestamps)
    silence_gap_ms = silence_gap_s * 1000.0
    chars = list(text)
    char_aligned = (len(chars) == len(norm_ts))

    utterances = []
    if char_aligned:
        # paraformer：字符与时间戳一一对齐，精确切分
        seg_tokens: list[str] = []
        seg_start_ms = norm_ts[0][0]
        prev_end_ms = norm_ts[0][1]
        for i, (start_ms, end_ms) in enumerate(norm_ts):
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
        # Fun-ASR-Nano：token 数与字符数不等，按时间边界切出段落，
        # inference_with_vad 将各 VAD 段文本以空格拼接，按空格拆分对应各段
        seg_ranges: list[tuple[float, float]] = []
        seg_start_ms = norm_ts[0][0]
        prev_end_ms = norm_ts[0][1]
        for start_ms, end_ms in norm_ts[1:]:
            if start_ms - prev_end_ms >= silence_gap_ms:
                seg_ranges.append((seg_start_ms, prev_end_ms))
                seg_start_ms = start_ms
            prev_end_ms = end_ms
        seg_ranges.append((seg_start_ms, prev_end_ms))

        parts = text.split(" ") if len(seg_ranges) > 1 else [text]
        for idx, (s_ms, e_ms) in enumerate(seg_ranges):
            seg_text = (parts[idx].strip() if idx < len(parts) else "")
            if seg_text:
                utterances.append({
                    "text": seg_text,
                    "start": round(s_ms / 1000.0, 3),
                    "end": round(e_ms / 1000.0, 3),
                })

    return [u for u in utterances if u["text"]]


def transcribe_file(model, audio_path: str, args) -> dict:
    """转写单个文件，返回结果 dict。text 字段为列表，每条含 text/start/end。"""
    audio_dur_s = _audio_duration(audio_path)

    generate_kwargs = {}
    if args.hotwords:
        generate_kwargs["hotword"] = args.hotwords
    if args.language:
        generate_kwargs["language"] = args.language
    if args.use_itn:
        generate_kwargs["use_itn"] = True
    if args.merge_vad:
        generate_kwargs["merge_vad"] = True
        generate_kwargs["merge_length_s"] = args.merge_length_s
    # 禁止 VAD 合并多段送入 batch，避免 Fun-ASR-Nano 报 batch decoding not implemented
    if args.vad_model:
        generate_kwargs["batch_size_threshold_s"] = 0
        generate_kwargs["batch_size_s"] = 0

    results, elapsed = _timed(
        f"transcribe {Path(audio_path).name}",
        model.generate,
        input=audio_path,
        **generate_kwargs,
        audio_dur_s=audio_dur_s,
    )

    text_list = []
    if results and isinstance(results, list):
        for item in results:
            raw = item.get("text", "").strip()
            if not raw:
                continue
            cleaned = _clean_text(raw)
            if not cleaned:
                continue
            ts_list = item.get("timestamp") or item.get("timestamps") or []
            segs = _split_by_gap(cleaned, ts_list, args.silence_gap)
            if segs:
                text_list.extend(segs)
            else:
                # 无时间戳（SenseVoice）：整条保留，start/end 为 null
                text_list.append({"text": cleaned, "start": None, "end": None})

    rtf = round(elapsed / audio_dur_s, 4) if audio_dur_s > 0 else None
    return {
        "source": audio_path,
        "filename": os.path.basename(audio_path),
        "text": text_list,
        "audio_dur_s": round(audio_dur_s, 3),
        "transcribe_s": round(elapsed, 3),
        "rtf": rtf,
        "rtfx": round(1 / rtf, 2) if rtf else None,
    }


def save_result(result: dict, audio_path: Path, output_dir: Path, channel: int | None = None):
    """将结果写入 JSON 文件。channel 不为 None 时在文件名中附加声道后缀。"""
    base = audio_path.stem
    if channel is not None:
        base = f"{base}_channel{channel}"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{base}.funasr.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[output] 已保存: %s", out_path)
    return out_path


def collect_files(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        return files
    else:
        logger.error("路径不存在: %s", input_path)
        sys.exit(1)


def main() -> None:
    args = parse_args()

    logger.info("[config] model=%s  vad=%s  punc=%s", args.model, args.vad_model, args.punc_model)
    logger.info("[config] hub=%s  device=%s  batch_size=%d", args.hub, args.device, args.batch_size)
    logger.info("[config] separate_channel=%s  language=%s  use_itn=%s  merge_vad=%s  silence_gap=%ss",
                args.separate_channel, args.language, args.use_itn, args.merge_vad, args.silence_gap)
    logger.info("[input]  %s", args.input)

    t0 = time.perf_counter()
    model = load_model(args)
    model_load_s = time.perf_counter() - t0
    logger.info("[timing] 模型加载: %.3fs", model_load_s)

    files = collect_files(args.input)
    logger.info("[info]   共 %d 个音频文件待处理", len(files))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_audio_s = 0.0
    total_transcribe_s = 0.0

    for i, f in enumerate(files, 1):
        logger.info("\n[%d/%d] 处理: %s", i, len(files), f.name)

        if args.separate_channel:
            import soundfile as sf
            audio_data, sample_rate = sf.read(str(f), always_2d=True)
            num_channels = audio_data.shape[1]
            logger.info("[channel] 检测到 %d 声道，分别转写", num_channels)

            with tempfile.TemporaryDirectory() as tmpdir:
                for ch in range(num_channels):
                    channel_audio = audio_data[:, ch]
                    tmp_wav = os.path.join(tmpdir, f"ch{ch}.wav")
                    sf.write(tmp_wav, channel_audio, sample_rate)

                    logger.info("[channel %d] 开始转写...", ch)
                    result = transcribe_file(model, tmp_wav, args)
                    result["source"] = str(f)
                    result["filename"] = f.name
                    result["channel"] = ch
                    logger.info("[channel %d] %d 条结果", ch, len(result["text"]))

                    save_result(result, f, output_dir, channel=ch)

                    total_audio_s += result["audio_dur_s"]
                    total_transcribe_s += result["transcribe_s"]
        else:
            result = transcribe_file(model, str(f), args)
            logger.info("[result] %d 条结果", len(result["text"]))
            save_result(result, f, output_dir)

            total_audio_s += result["audio_dur_s"]
            total_transcribe_s += result["transcribe_s"]

    # 汇总统计
    logger.info("\n[summary] 处理文件数: %d", len(files))
    logger.info("[summary] 总音频时长: %.1fs", total_audio_s)
    logger.info("[summary] 总转写耗时: %.1fs", total_transcribe_s)
    if total_audio_s > 0:
        overall_rtf = total_transcribe_s / total_audio_s
        logger.info("[summary] 整体 RTF: %.4f  RTFx: %.2f", overall_rtf, 1 / overall_rtf)


if __name__ == "__main__":
    main()
