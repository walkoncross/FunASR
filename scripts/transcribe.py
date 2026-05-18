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
    "audio_dur_s": 12.34,
    "transcribe_s": 1.23,
    "rtf": 0.1,
    "text": [
      {"text": "识别结果", "start": 0.0, "end": 5.0},
      ...
    ]
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


_SENSEVOICE_LANG_RE = re.compile(r"<\|(?:zh|en|yue|ja|ko|nospeech)\|>")


def _sensevoice_postprocess(text: str) -> str:
    """对单段 SenseVoice 原始文本做 rich_transcription_postprocess。"""
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        return rich_transcription_postprocess(text)
    except ImportError:
        return re.sub(r"<\|[^|]+\|>", "", text).strip()


def _clean_text(text: str) -> str:
    """清除 SenseVoice 原始标签，返回干净文本（单段）。"""
    if text and _SENSEVOICE_LANG_RE.search(text):
        return _sensevoice_postprocess(text)
    return text


def _split_sensevoice(text: str) -> list[str]:
    """
    将 SenseVoice 多 VAD 段拼接文本按语言标签切分为各段干净文本列表。
    inference_with_vad 用 ' ' 拼接各段，每段以 <|zh|> 等语言标签开头。
    返回各段经 postprocess 处理后的非空文本列表；
    若文本不含语言标签，直接返回 [text]（兼容无标签输出）。
    """
    if not _SENSEVOICE_LANG_RE.search(text):
        return [text.strip()] if text.strip() else []
    # 以 ' <|lang|>' 为分界点切开（保留各段首部的语言标签）
    parts = re.split(r"(?= <\|(?:zh|en|yue|ja|ko|nospeech)\|>)", text)
    result = []
    for part in parts:
        cleaned = _sensevoice_postprocess(part.strip())
        if cleaned:
            result.append(cleaned)
    return result


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
    silence_gap_s <= 0 或无时间戳时直接整段返回。

    支持三种输入情形：
      A) paraformer 无 punc：len(text) == len(timestamps)，字符与 ts 一一对齐
      B) paraformer + punc ：len(text) > len(timestamps)，punc 在原始字符间插入了标点
         => 跳过标点字符（不消耗 ts），非标点字符与 ts 一一对齐
      C) Fun-ASR-Nano      ：timestamps 为 {"start_time":s,"end_time":e} 字典列表，
         按时间边界切段，文本按空格拆分
    """
    if silence_gap_s <= 0 or not timestamps:
        start_s, end_s = _ts_bounds(timestamps)
        return [{"text": text, "start": start_s, "end": end_s}] if text else []

    norm_ts = _norm_timestamps(timestamps)
    silence_gap_ms = silence_gap_s * 1000.0

    # 判断格式：Fun-ASR-Nano 的 timestamps 是字典列表
    is_nano = isinstance(timestamps[0], dict)
    if is_nano:
        # C) Fun-ASR-Nano：按时间边界切段，文本按空格拆分对应各段
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
        utterances = []
        for idx, (s_ms, e_ms) in enumerate(seg_ranges):
            seg_text = (parts[idx].strip() if idx < len(parts) else "")
            if seg_text:
                utterances.append({
                    "text": seg_text,
                    "start": round(s_ms / 1000.0, 3),
                    "end": round(e_ms / 1000.0, 3),
                })
        return [u for u in utterances if u["text"]]

    # A/B) paraformer：[[start_ms, end_ms], ...]
    # punc 模型在原始字符间插入标点，导致 len(text) >= len(norm_ts)。
    # 遍历 text 时，跳过标点字符（不消耗 ts 索引），非标点字符与 ts 一一对齐。
    # 判断一个字符是否是 punc 插入的标点：不在 norm_ts 对应位置消耗时间戳
    ts_idx = 0
    n_ts = len(norm_ts)
    utterances = []
    seg_chars: list[str] = []
    seg_start_ms: float = norm_ts[0][0]
    prev_end_ms: float = norm_ts[0][1]

    for ch in text:
        if ts_idx >= n_ts:
            # 时间戳已用完，剩余字符（标点）追加到末尾段
            seg_chars.append(ch)
            continue

        # 判断当前字符是否消耗一个时间戳：
        # 若字符是 CJK / 字母 / 数字（即非纯标点），消耗一个 ts
        is_content = bool('\u4e00' <= ch <= '\u9fff'  # CJK
                          or ch.isalnum())
        if is_content:
            start_ms, end_ms = norm_ts[ts_idx]
            gap = start_ms - prev_end_ms
            if seg_chars and gap >= silence_gap_ms:
                utterances.append({
                    "text": "".join(seg_chars).strip(),
                    "start": round(seg_start_ms / 1000.0, 3),
                    "end": round(prev_end_ms / 1000.0, 3),
                })
                seg_chars = []
                seg_start_ms = start_ms
            seg_chars.append(ch)
            prev_end_ms = end_ms
            ts_idx += 1
        else:
            # 标点/空格：直接追加，不消耗 ts
            seg_chars.append(ch)

    if seg_chars:
        utterances.append({
            "text": "".join(seg_chars).strip(),
            "start": round(seg_start_ms / 1000.0, 3),
            "end": round(prev_end_ms / 1000.0, 3),
        })

    return [u for u in utterances if u["text"]]


def _transcribe_sensevoice_segments(model, audio_path: str, generate_kwargs: dict) -> list[dict]:
    """
    SenseVoice + VAD 专用：绕过 inference_with_vad 的聚合，
    先用 VAD 拿到分段时间列表，再对每段单独推理，保留各段的 start/end。
    返回 [{"text": ..., "start": s, "end": e}, ...] 列表。
    若 model 没有 vad_model，退回到整段推理（start/end=null）。
    """
    import soundfile as sf

    vad_model = getattr(model, "vad_model", None)
    if vad_model is None:
        # 无 VAD：直接整段推理
        results = model.generate(input=audio_path, **generate_kwargs)
        utterances = []
        for item in results or []:
            raw = item.get("text", "").strip()
            for seg in _split_sensevoice(raw):
                utterances.append({"text": seg, "start": None, "end": None})
        return utterances

    # step1: 单独跑 VAD，拿到 vadsegments（单位 ms）
    vad_kwargs = dict(model.vad_kwargs)
    vad_res = model.inference(audio_path, model=vad_model, kwargs=vad_kwargs)
    if not vad_res or not vad_res[0].get("value"):
        return []
    vadsegments = vad_res[0]["value"]  # [[start_ms, end_ms], ...]

    # step2: 加载原始音频，确保单声道 1D float32，16kHz
    speech, fs = sf.read(audio_path, dtype="float32", always_2d=False)
    if speech.ndim == 2:
        speech = speech.mean(axis=1)  # 立体声取均值降为单声道
    if fs != 16000:
        try:
            import resampy
            speech = resampy.resample(speech, fs, 16000)
        except ImportError:
            import librosa
            speech = librosa.resample(speech, orig_sr=fs, target_sr=16000)
        fs = 16000

    # step3: 对每个 VAD 段单独推理（临时禁用 vad_model，避免再次进入 inference_with_vad）
    orig_vad = model.vad_model
    model.vad_model = None
    try:
        utterances = []
        for seg_ms in vadsegments:
            start_ms, end_ms = int(seg_ms[0]), int(seg_ms[1])
            start_sample = int(start_ms / 1000 * fs)
            end_sample = int(end_ms / 1000 * fs)
            seg_audio = speech[start_sample:end_sample]
            if len(seg_audio) == 0:
                continue
            seg_results = model.generate(input=seg_audio, **generate_kwargs)
            for item in seg_results or []:
                raw = item.get("text", "").strip()
                for seg_text in _split_sensevoice(raw):
                    utterances.append({
                        "text": seg_text,
                        "start": round(start_ms / 1000.0, 3),
                        "end": round(end_ms / 1000.0, 3),
                    })
    finally:
        model.vad_model = orig_vad

    return utterances


def transcribe_file(model, audio_path: str, args) -> dict:
    """转写单个文件，返回结果 dict。text 字段为列表，每条含 text/start/end。"""
    audio_dur_s = _audio_duration(audio_path)

    generate_kwargs = {
        # paraformer: 启用字符级时间戳（其他模型会忽略此参数）
        "pred_timestamp": True,
    }
    if args.hotwords:
        generate_kwargs["hotword"] = args.hotwords
    if args.language:
        generate_kwargs["language"] = args.language
    if args.use_itn:
        generate_kwargs["use_itn"] = True
    if args.merge_vad:
        generate_kwargs["merge_vad"] = True
        generate_kwargs["merge_length_s"] = args.merge_length_s

    # SenseVoice + VAD：绕过 inference_with_vad 聚合，逐段单独推理，保留各段 start/end
    is_sensevoice = args.vad_model and re.search(r"sensevoice", args.model, re.IGNORECASE)
    if is_sensevoice:
        t0 = time.perf_counter()
        text_list = _transcribe_sensevoice_segments(model, audio_path, generate_kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("[timing] transcribe %s: %.3fs", Path(audio_path).name, elapsed)
        if audio_dur_s > 0:
            rtf = elapsed / audio_dur_s
            logger.info("[RTF]    RTF=%.4f  RTFx=%.2f", rtf, 1 / rtf)
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
            ts_list = item.get("timestamp") or item.get("timestamps") or []
            cleaned = _clean_text(raw)
            if not cleaned:
                continue
            segs = _split_by_gap(cleaned, ts_list, args.silence_gap)
            if segs:
                text_list.extend(segs)
            else:
                text_list.append({"text": cleaned, "start": None, "end": None})

    rtf = round(elapsed / audio_dur_s, 4) if audio_dur_s > 0 else None
    return {
        "source": audio_path,
        "filename": os.path.basename(audio_path),
        "audio_dur_s": round(audio_dur_s, 3),
        "transcribe_s": round(elapsed, 3),
        "rtf": rtf,
        "rtfx": round(1 / rtf, 2) if rtf else None,
        "model_name": Path(args.model).name or args.model,
        "vad_model": Path(args.vad_model).name if args.vad_model else None,
        "punc_model": Path(args.punc_model).name if args.punc_model else None,
        "text": text_list,
    }


def _model_tag(model_name: str, vad_model: str, punc_model: str) -> str:
    """
    生成文件名后缀标签，例如：
      iic/SenseVoiceSmall + fsmn-vad + ""  → "SenseVoiceSmall.fsmn-vad.no-punc"
      paraformer-zh        + fsmn-vad + ct-punc → "paraformer-zh.fsmn-vad.ct-punc"
    本地路径取最后一段目录名；名称中 / 替换为 -。
    """
    # 取模型名最后一段（兼容 HF 格式 org/model 和本地路径）
    name = Path(model_name).name or model_name
    name = name.replace("/", "-")
    vad_tag = Path(vad_model).name if vad_model else "no-vad"
    punc_tag = Path(punc_model).name if punc_model else "no-punc"
    return f"{name}.{vad_tag}.{punc_tag}"


def save_result(result: dict, audio_path: Path, output_dir: Path,
                args=None, channel: int | None = None):
    """将结果写入 JSON 文件。channel 不为 None 时在文件名中附加声道后缀。"""
    base = audio_path.stem
    if channel is not None:
        base = f"{base}_channel{channel}"
    if args is not None:
        tag = _model_tag(args.model, args.vad_model or "", args.punc_model or "")
        filename = f"{base}.{tag}.json"
    else:
        filename = f"{base}.funasr.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
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

                    save_result(result, f, output_dir, args=args, channel=ch)

                    total_audio_s += result["audio_dur_s"]
                    total_transcribe_s += result["transcribe_s"]
        else:
            result = transcribe_file(model, str(f), args)
            logger.info("[result] %d 条结果", len(result["text"]))
            save_result(result, f, output_dir, args=args)

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
