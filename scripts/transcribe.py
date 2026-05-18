# coding=utf-8
"""
FunASR speech transcription script (ModelScope / HuggingFace backend)

Usage:
  python scripts/transcribe.py -i <audio_file_or_dir> [OPTIONS]

  --model/-m              ASR model name or local path (default: paraformer-zh)
  --vad-model/-vm         VAD model name or path (default: fsmn-vad)
  --punc-model/-pm        Punctuation model name or path (default: ct-punc)
  --input/-i              Audio file or directory (required)
  --output/-o             Output directory (default: ./results/)
  --hub                   Model source: modelscope / hf (default: modelscope)
  --device/-d             Inference device: cpu / cuda:0 / mps (default: cpu)
  --batch-size/-bs        Inference batch size (default: 1)
  --separate-channel/-sc  Split channels and transcribe each separately
  --hotwords              Hotwords string, space-separated
  --enable-update         Enable FunASR version check (disabled by default)
  --language              Language code (SenseVoice): auto / zh / en / yue / ja / ko / nospeech
  --use-itn               Enable inverse text normalization ITN (SenseVoice)
  --merge-vad             Merge short VAD segments (SenseVoice)
  --merge-length-s        Max duration in seconds to merge VAD segments (default: 15.0, requires --merge-vad)
  --silence-gap/-sg       Silence gap threshold in seconds for splitting; 0 = no splitting (default: 0.5)

Output format (json):  <stem>.<model>.<vad>.<punc>.json
  {
    "source": "...",
    "filename": "...",
    "audio_dur_s": 12.34,
    "transcribe_s": 1.23,
    "rtf": 0.1,
    "rtfx": 10.0,
    "model_name": "paraformer-zh",
    "vad_model": "fsmn-vad",
    "punc_model": "ct-punc",
    "text": "recognition result one recognition result two",
    "segments": [
      {"text": "recognition result", "start": 0.0, "end": 5.0},
      ...
    ]
  }
  start / end are in seconds; both are null when no timestamps are available.
  With --silence-gap, each VAD segment is further split by silence gaps;
  default silence_gap=0.5s; set to 0 to keep each VAD segment as a single entry.
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
        description="FunASR speech transcription tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "-m", default="paraformer-zh", help="ASR model name or local path")
    parser.add_argument("--vad-model", "-vm", default="fsmn-vad", help="VAD model name or path")
    parser.add_argument("--punc-model", "-pm", default="ct-punc", help="Punctuation model name or path")
    parser.add_argument("--input", "-i", required=True, help="Audio file or directory")
    parser.add_argument("--output", "-o", default="./results/", help="Output directory")
    parser.add_argument("--hub", default="modelscope", choices=["modelscope", "hf"],
                        help="Model source: modelscope or hf (HuggingFace)")
    parser.add_argument("--device", "-d", default="cpu", help="Inference device: cpu / cuda:0 / mps")
    parser.add_argument("--batch-size", "-bs", type=int, default=1, help="Inference batch size")
    parser.add_argument("--separate-channel", "-sc", action="store_true", default=False,
                        help="Split channels and transcribe each separately")
    parser.add_argument("--hotwords", default=None, help="Hotwords string, space-separated")
    parser.add_argument("--enable-update", action="store_true", default=False,
                        help="Enable FunASR version check (disabled by default)")
    # SenseVoice-specific arguments
    parser.add_argument("--language", default=None,
                        help="Language code (SenseVoice): auto / zh / en / yue / ja / ko / nospeech")
    parser.add_argument("--use-itn", action="store_true", default=False,
                        help="Enable inverse text normalization ITN (SenseVoice)")
    parser.add_argument("--merge-vad", action="store_true", default=False,
                        help="Merge short VAD segments (SenseVoice)")
    parser.add_argument("--merge-length-s", type=float, default=15.0,
                        help="Max duration in seconds to merge VAD segments (SenseVoice, requires --merge-vad)")
    parser.add_argument("--silence-gap", "-sg", type=float, default=0.5, dest="silence_gap",
                        help="Silence gap threshold in seconds for splitting; 0 = no splitting")
    return parser.parse_args()


def _timed(label: str, fn, *args, audio_dur_s: float = 0.0, **kwargs):
    """Run fn and return (result, elapsed_s), logging elapsed time and RTF."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    logger.info("[timing] %s: %.3fs", label, elapsed)
    if audio_dur_s > 0:
        rtf = elapsed / audio_dur_s
        logger.info("[RTF]    RTF=%.4f  RTFx=%.2f", rtf, 1 / rtf)
    return result, elapsed


def _audio_duration(path: str) -> float:
    """Return audio duration in seconds using soundfile; returns 0 on failure."""
    try:
        import soundfile as sf
        return sf.info(path).duration
    except Exception:
        return 0.0


def load_model(args) -> "AutoModel":
    """Load FunASR AutoModel, supporting modelscope / hf backends."""
    from funasr import AutoModel

    kwargs = dict(
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        disable_update=not args.enable_update,
    )
    # Empty string means omit the argument to avoid AutoModel building an empty model name
    if args.vad_model:
        kwargs["vad_model"] = args.vad_model
    if args.punc_model:
        kwargs["punc_model"] = args.punc_model

    if args.hub == "hf":
        kwargs["hub"] = "hf"
        logger.info("Loading model from HuggingFace Hub")
    else:
        logger.info("Loading model from ModelScope")

    if args.hotwords:
        kwargs["hotword"] = args.hotwords

    return AutoModel(**kwargs)


_SENSEVOICE_LANG_RE = re.compile(r"<\|(?:zh|en|yue|ja|ko|nospeech)\|>")


def _sensevoice_postprocess(text: str) -> str:
    """Apply rich_transcription_postprocess to a single SenseVoice raw text segment."""
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
        return rich_transcription_postprocess(text)
    except ImportError:
        return re.sub(r"<\|[^|]+\|>", "", text).strip()


def _clean_text(text: str) -> str:
    """Strip SenseVoice raw tags and return clean text for a single segment."""
    if text and _SENSEVOICE_LANG_RE.search(text):
        return _sensevoice_postprocess(text)
    return text


def _split_sensevoice(text: str) -> list[str]:
    """
    Split SenseVoice concatenated VAD-segment text into individual clean segments.
    inference_with_vad joins segments with ' '; each segment starts with a language tag.
    Returns a list of postprocessed non-empty strings.
    If the text contains no language tags, returns [text] for compatibility.
    """
    if not _SENSEVOICE_LANG_RE.search(text):
        return [text.strip()] if text.strip() else []
    # Split at ' <|lang|>' boundaries (lookahead keeps the language tag on each segment)
    parts = re.split(r"(?= <\|(?:zh|en|yue|ja|ko|nospeech)\|>)", text)
    result = []
    for part in parts:
        cleaned = _sensevoice_postprocess(part.strip())
        if cleaned:
            result.append(cleaned)
    return result


def _ts_bounds(ts_list: list) -> tuple[float | None, float | None]:
    """
    Return (start_s, end_s) from the first and last timestamp entries.
    - paraformer timestamp item: [start_ms, end_ms]
    - Fun-ASR-Nano timestamps item: {"start_time": s, "end_time": s}
    Returns (None, None) if ts_list is empty or unrecognized.
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
    """Normalize timestamps to a list of (start_ms, end_ms) tuples."""
    if not timestamps:
        return []
    first = timestamps[0]
    if isinstance(first, dict):
        return [(t["start_time"] * 1000, t["end_time"] * 1000) for t in timestamps]
    return [(t[0], t[1]) for t in timestamps]


def _split_by_gap(text: str, timestamps: list, silence_gap_s: float) -> list[dict]:
    """
    Split a single VAD segment into multiple utterances by silence gaps.
    Returns the whole segment unsplit when silence_gap_s <= 0 or timestamps is empty.

    Three cases are handled:
      A) paraformer without punc: len(text) == len(timestamps), chars align 1:1 with ts
      B) paraformer + punc:       len(text) > len(timestamps), punc chars inserted between
         original chars; skip punctuation chars without consuming a ts index
      C) Fun-ASR-Nano:            timestamps are {"start_time": s, "end_time": e} dicts;
         split by time boundaries, text split by spaces
    """
    if silence_gap_s <= 0 or not timestamps:
        start_s, end_s = _ts_bounds(timestamps)
        return [{"text": text, "start": start_s, "end": end_s}] if text else []

    norm_ts = _norm_timestamps(timestamps)
    silence_gap_ms = silence_gap_s * 1000.0

    # Case C: Fun-ASR-Nano uses dict timestamps
    is_nano = isinstance(timestamps[0], dict)
    if is_nano:
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

    # Cases A/B: paraformer [[start_ms, end_ms], ...]
    # When punc model is used, len(text) >= len(norm_ts).
    # Iterate text chars: skip punctuation/space chars (no ts consumed),
    # align CJK/alnum chars 1:1 with ts entries.
    ts_idx = 0
    n_ts = len(norm_ts)
    utterances = []
    seg_chars: list[str] = []
    seg_start_ms: float = norm_ts[0][0]
    prev_end_ms: float = norm_ts[0][1]

    for ch in text:
        if ts_idx >= n_ts:
            # Timestamps exhausted; remaining chars (punctuation) go to the last segment
            seg_chars.append(ch)
            continue

        # CJK / alphanumeric chars consume a timestamp; punctuation/space do not
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
            # Punctuation / space: append without consuming a ts slot
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
    SenseVoice + VAD: bypass inference_with_vad aggregation.
    Run VAD separately to get segment timings, then infer each segment individually
    to preserve per-segment start/end.
    Returns [{"text": ..., "start": s, "end": e}, ...].
    Falls back to whole-file inference (start/end=null) if model has no vad_model.
    """
    import soundfile as sf

    vad_model = getattr(model, "vad_model", None)
    if vad_model is None:
        # No VAD: infer the whole file
        results = model.generate(input=audio_path, **generate_kwargs)
        utterances = []
        for item in results or []:
            raw = item.get("text", "").strip()
            for seg in _split_sensevoice(raw):
                utterances.append({"text": seg, "start": None, "end": None})
        return utterances

    # Step 1: run VAD to get segment list (in ms)
    vad_kwargs = dict(model.vad_kwargs)
    vad_res = model.inference(audio_path, model=vad_model, kwargs=vad_kwargs)
    if not vad_res or not vad_res[0].get("value"):
        return []
    vadsegments = vad_res[0]["value"]  # [[start_ms, end_ms], ...]

    # Step 2: load audio as mono float32 at 16kHz
    speech, fs = sf.read(audio_path, dtype="float32", always_2d=False)
    if speech.ndim == 2:
        speech = speech.mean(axis=1)  # stereo -> mono
    if fs != 16000:
        try:
            import resampy
            speech = resampy.resample(speech, fs, 16000)
        except ImportError:
            import librosa
            speech = librosa.resample(speech, orig_sr=fs, target_sr=16000)
        fs = 16000

    # Step 3: infer each VAD segment individually
    # Temporarily disable vad_model to avoid re-entering inference_with_vad
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
    """Transcribe a single file. Returns a dict with 'text' (joined string) and 'segments' (list of {text, start, end})."""
    audio_dur_s = _audio_duration(audio_path)

    generate_kwargs = {
        # paraformer: enable character-level timestamps (other models ignore this)
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

    # SenseVoice + VAD: bypass inference_with_vad aggregation, infer per segment
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
            "audio_dur_s": round(audio_dur_s, 3),
            "transcribe_s": round(elapsed, 3),
            "rtf": rtf,
            "rtfx": round(1 / rtf, 2) if rtf else None,
            "vad_s": None,
            "vad_rtf": None,
            "vad_rtfx": None,
            "punct_s": None,
            "punct_rtf": None,
            "punct_rtfx": None,
            "model_name": Path(args.model).name or args.model,
            "vad_model": Path(args.vad_model).name if args.vad_model else None,
            "punc_model": Path(args.punc_model).name if args.punc_model else None,
            "text": " ".join(s["text"] for s in text_list if s.get("text")),
            "segments": text_list,
        }

    # Prevent VAD from batching multiple segments to avoid "batch decoding not implemented"
    # errors with Fun-ASR-Nano
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
        "vad_s": None,
        "vad_rtf": None,
        "vad_rtfx": None,
        "punct_s": None,
        "punct_rtf": None,
        "punct_rtfx": None,
        "model_name": Path(args.model).name or args.model,
        "vad_model": Path(args.vad_model).name if args.vad_model else None,
        "punc_model": Path(args.punc_model).name if args.punc_model else None,
        "text": " ".join(s["text"] for s in text_list if s.get("text")),
        "segments": text_list,
    }


def _model_tag(model_name: str, vad_model: str, punc_model: str) -> str:
    """
    Generate a filename suffix tag, e.g.:
      iic/SenseVoiceSmall + fsmn-vad + ""      -> "SenseVoiceSmall.fsmn-vad.no-punc"
      paraformer-zh        + fsmn-vad + ct-punc -> "paraformer-zh.fsmn-vad.ct-punc"
    Uses the last path component for local paths; replaces / with - in the name.
    """
    name = Path(model_name).name or model_name
    name = name.replace("/", "-")
    vad_tag = Path(vad_model).name if vad_model else "no-vad"
    punc_tag = Path(punc_model).name if punc_model else "no-punc"
    return f"{name}.{vad_tag}.{punc_tag}"


def save_result(result: dict, audio_path: Path, output_dir: Path,
                args=None, channel: int | None = None):
    """Write result to a JSON file. Appends channel suffix when channel is not None."""
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
    logger.info("[output] saved: %s", out_path)
    return out_path


def collect_files(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        return files
    else:
        logger.error("path not found: %s", input_path)
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
    logger.info("[timing] model loaded: %.3fs", model_load_s)

    files = collect_files(args.input)
    logger.info("[info]   %d audio file(s) to process", len(files))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_audio_s = 0.0
    total_transcribe_s = 0.0

    for i, f in enumerate(files, 1):
        logger.info("\n[%d/%d] processing: %s", i, len(files), f.name)

        if args.separate_channel:
            import soundfile as sf
            audio_data, sample_rate = sf.read(str(f), always_2d=True)
            num_channels = audio_data.shape[1]
            logger.info("[channel] detected %d channel(s), transcribing each separately", num_channels)

            with tempfile.TemporaryDirectory() as tmpdir:
                for ch in range(num_channels):
                    channel_audio = audio_data[:, ch]
                    tmp_wav = os.path.join(tmpdir, f"ch{ch}.wav")
                    sf.write(tmp_wav, channel_audio, sample_rate)

                    logger.info("[channel %d] transcribing...", ch)
                    result = transcribe_file(model, tmp_wav, args)
                    result["source"] = str(f)
                    result["filename"] = f.name
                    result["channel"] = ch
                    logger.info("[channel %d] %d segment(s)", ch, len(result["segments"]))

                    save_result(result, f, output_dir, args=args, channel=ch)

                    total_audio_s += result["audio_dur_s"]
                    total_transcribe_s += result["transcribe_s"]
        else:
            result = transcribe_file(model, str(f), args)
            logger.info("[result] %d segment(s)", len(result["segments"]))
            save_result(result, f, output_dir, args=args)

            total_audio_s += result["audio_dur_s"]
            total_transcribe_s += result["transcribe_s"]

    logger.info("\n[summary] files processed: %d", len(files))
    logger.info("[summary] total audio duration: %.1fs", total_audio_s)
    logger.info("[summary] total transcribe time: %.1fs", total_transcribe_s)
    if total_audio_s > 0:
        overall_rtf = total_transcribe_s / total_audio_s
        logger.info("[summary] overall RTF: %.4f  RTFx: %.2f", overall_rtf, 1 / overall_rtf)


if __name__ == "__main__":
    main()
