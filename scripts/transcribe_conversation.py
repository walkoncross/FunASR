# coding=utf-8
"""
FunASR two-channel conversation transcription script (ModelScope / HuggingFace backend)

Processes each channel of a stereo audio file using FunASR with fsmn-vad segmentation,
sorts utterances from both channels by timestamp, and outputs a conversation JSON.

Usage:
  python scripts/transcribe_conversation.py -i <stereo_audio> [OPTIONS]

  --model/-m              ASR model name or local path (default: paraformer-zh)
  --vad-model/-vm         VAD model name or path (default: fsmn-vad)
  --punc-model/-pm        Punctuation model name or path (default: ct-punc)
  --input/-i              Stereo audio file (required)
  --output/-o             JSON output path (default: results/<basename>.<model>.<vad>.<punc>.conversation.json)
  --hub                   Model source: modelscope / hf (default: modelscope)
  --device/-d             Inference device: cpu / cuda:0 / mps (default: cpu)
  --batch-size/-bs        Inference batch size (default: 1)
  --channels/-c           Number of channels to process (default: 2)
  --silence-gap/-sg       Silence gap threshold in seconds for splitting utterances (default: 0.5)
  --hotwords              Hotwords string, space-separated
  --enable-update         Enable FunASR version check (disabled by default)
  --language              Language code (SenseVoice): auto / zh / en / yue / ja / ko / nospeech
  --use-itn               Enable inverse text normalization ITN (SenseVoice)
  --merge-vad             Merge short VAD segments (SenseVoice; reduces splitting granularity)
  --merge-length-s        Max duration in seconds to merge VAD segments (default: 15.0, requires --merge-vad)

Output format (json):  <stem>.conversation.<model>.<vad>.<punc>.json
  {
    "source": "...",
    "filename": "...",
    "channels": 2,
    "audio_dur_s": 306.68,
    "transcribe_s": 52.32,
    "rtf": 0.17,
    "rtfx": 5.86,
    "vad_s": 0.0,
    "vad_rtf": null,
    "vad_rtfx": null,
    "punct_s": null,
    "punct_rtf": null,
    "punct_rtfx": null,
    "model_name": "paraformer-zh",
    "vad_model": "fsmn-vad",
    "punc_model": "ct-punc",
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
import re
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


def _model_tag(model_name: str, vad_model: str, punc_model: str) -> str:
    """Generate filename tag, same logic as transcribe.py."""
    name = Path(model_name).name or model_name
    name = name.replace("/", "-")
    vad_tag = Path(vad_model).name if vad_model else "no-vad"
    punc_tag = Path(punc_model).name if punc_model else "no-punc"
    return f"{name}.{vad_tag}.{punc_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FunASR two-channel conversation transcription tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "-m", default="paraformer-zh", help="ASR model name or local path")
    parser.add_argument("--vad-model", "-vm", default="fsmn-vad", help="VAD model name or path")
    parser.add_argument("--punc-model", "-pm", default="ct-punc", help="Punctuation model name or path")
    parser.add_argument("--input", "-i", required=True, help="Stereo audio file path")
    parser.add_argument("--output", "-o", default=None,
                        help="JSON output path (default: results/<basename>-conversation.json)")
    parser.add_argument("--hub", default="modelscope", choices=["modelscope", "hf"],
                        help="Model source: modelscope or hf (HuggingFace)")
    parser.add_argument("--device", "-d", default="cpu", help="Inference device: cpu / cuda:0 / mps")
    parser.add_argument("--batch-size", "-bs", type=int, default=1, help="Inference batch size")
    parser.add_argument("--channels", "-c", type=int, default=2, help="Number of channels to process")
    parser.add_argument("--silence-gap", "-sg", type=float, default=0.5, dest="silence_gap",
                        help="Silence gap threshold in seconds; token gaps above this split utterances")
    parser.add_argument("--hotwords", default=None, help="Hotwords string, space-separated")
    parser.add_argument("--enable-update", action="store_true", default=False,
                        help="Enable FunASR version check (disabled by default)")
    # SenseVoice-specific arguments
    parser.add_argument("--language", default=None,
                        help="Language code (SenseVoice): auto / zh / en / yue / ja / ko / nospeech")
    parser.add_argument("--use-itn", action="store_true", default=False,
                        help="Enable inverse text normalization ITN (SenseVoice)")
    parser.add_argument("--merge-vad", action="store_true", default=False,
                        help="Merge short VAD segments (SenseVoice; reduces splitting granularity)")
    parser.add_argument("--merge-length-s", type=float, default=15.0,
                        help="Max duration in seconds to merge VAD segments (SenseVoice, requires --merge-vad)")
    return parser.parse_args()


def load_model(args):
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
        if args.vad_model == "fsmn-vad":
            logger.info("Using built-in fsmn-vad model")
            kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}
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


def _norm_timestamps(timestamps: list) -> list[tuple[float, float]]:
    """
    Normalize timestamps to a list of (start_ms, end_ms) tuples.
    Supports two input formats:
      - paraformer:   [[start_ms, end_ms], ...]                   (ms)
      - Fun-ASR-Nano: [{"start_time": s, "end_time": e}, ...]    (seconds, after VAD offset)
    """
    if not timestamps:
        return []
    first = timestamps[0]
    if isinstance(first, dict):
        # Fun-ASR-Nano timestamps: seconds -> convert to ms
        return [(t["start_time"] * 1000, t["end_time"] * 1000) for t in timestamps]
    else:
        # paraformer timestamps: already in ms
        return [(t[0], t[1]) for t in timestamps]


def _split_utterances_by_gap(text: str, timestamps: list, silence_gap_s: float) -> list[dict]:
    """
    Split token-level timestamps into multiple utterances by silence gaps.
    Handles three cases (same as transcribe.py _split_by_gap):
      A) paraformer without punc: len(text) == len(timestamps), 1:1 char-ts alignment
      B) paraformer + punc:       len(text) > len(timestamps), skip punctuation chars
      C) Fun-ASR-Nano:            timestamps are dicts, split by time boundaries
    """
    if not timestamps:
        return [{"text": text, "start": 0.0, "end": 0.0}] if text else []

    norm_ts = _norm_timestamps(timestamps)
    silence_gap_ms = silence_gap_s * 1000.0

    # Case C: Fun-ASR-Nano dict timestamps
    is_nano = isinstance(timestamps[0], dict)
    if is_nano:
        seg_ranges: list[tuple[float, float]] = []
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
            return [{"text": text.strip(),
                     "start": round(seg_ranges[0][0] / 1000.0, 3),
                     "end": round(seg_ranges[0][1] / 1000.0, 3)}]
        parts = text.split(" ")
        utterances = []
        for idx, (s_ms, e_ms) in enumerate(seg_ranges):
            seg_text = parts[idx].strip() if idx < len(parts) else ""
            if seg_text:
                utterances.append({"text": seg_text,
                                    "start": round(s_ms / 1000.0, 3),
                                    "end": round(e_ms / 1000.0, 3)})
        return [u for u in utterances if u["text"]]

    # Cases A/B: paraformer; len(text) >= len(norm_ts) when punc model is used.
    # Skip punctuation chars (no ts consumed); align CJK/alnum chars 1:1 with ts.
    ts_idx = 0
    n_ts = len(norm_ts)
    utterances = []
    seg_chars: list[str] = []
    seg_start_ms: float = norm_ts[0][0]
    prev_end_ms: float = norm_ts[0][1]

    for ch in text:
        if ts_idx >= n_ts:
            seg_chars.append(ch)
            continue
        is_content = bool('\u4e00' <= ch <= '\u9fff' or ch.isalnum())
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
            seg_chars.append(ch)

    if seg_chars:
        utterances.append({
            "text": "".join(seg_chars).strip(),
            "start": round(seg_start_ms / 1000.0, 3),
            "end": round(prev_end_ms / 1000.0, 3),
        })

    return [u for u in utterances if u["text"]]


_SENSEVOICE_LANG_RE = re.compile(r"<\|(?:zh|en|yue|ja|ko|nospeech)\|>")


def _is_sensevoice_output(text: str) -> bool:
    """Return True if the text contains SenseVoice raw language tags."""
    return bool(_SENSEVOICE_LANG_RE.search(text))


def _split_sensevoice_raw(raw_text: str) -> list[str]:
    """
    Split SenseVoice raw output into individual VAD-segment clean texts.
    inference_with_vad concatenates VAD segments with ' ' (auto_model.py L591).
    Each SenseVoice segment starts with <|lang|>, so split on ' <|lang_tag|>' boundaries.
    Returns a list of postprocessed non-empty strings.
    """
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess
    except ImportError:
        def rich_transcription_postprocess(s):
            return re.sub(r"<\|[^|]+\|>", "", s).strip()

    segments_raw = re.split(r"(?= <\|(?:zh|en|yue|ja|ko|nospeech)\|>)", raw_text)
    result = []
    for seg in segments_raw:
        cleaned = rich_transcription_postprocess(seg.strip())
        if cleaned:
            result.append(cleaned)
    return result


def _transcribe_sensevoice_channel(model, wav_path: str, generate_kwargs: dict) -> list[dict]:
    """
    SenseVoice + VAD: run VAD first to get segment timings, then infer each segment
    individually so that start/end timestamps reflect real audio positions.
    Returns [{"text": ..., "start": s, "end": e}, ...].
    Falls back to whole-file inference (start/end=0.0) when no VAD model is attached.
    """
    vad_model = getattr(model, "vad_model", None)
    if vad_model is None:
        results = model.generate(input=wav_path, **generate_kwargs)
        utterances = []
        for item in results or []:
            for seg_text in _split_sensevoice_raw(item.get("text", "").strip()):
                utterances.append({"text": seg_text, "start": 0.0, "end": 0.0})
        return utterances

    # Step 1: run VAD to get segment list [[start_ms, end_ms], ...]
    vad_kwargs = dict(model.vad_kwargs)
    vad_res = model.inference(wav_path, model=vad_model, kwargs=vad_kwargs)
    if not vad_res or not vad_res[0].get("value"):
        return []
    vadsegments = vad_res[0]["value"]

    # Step 2: load channel audio
    speech, fs = sf.read(wav_path, dtype="float32", always_2d=False)

    # Step 3: temporarily detach vad_model to avoid re-entering inference_with_vad
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
                for seg_text in _split_sensevoice_raw(item.get("text", "").strip()):
                    utterances.append({
                        "text": seg_text,
                        "start": round(start_ms / 1000.0, 3),
                        "end": round(end_ms / 1000.0, 3),
                    })
    finally:
        model.vad_model = orig_vad

    return utterances


def transcribe_channel(model, wav_path: str, args) -> list[dict]:
    """
    Transcribe a mono wav file. Returns a list of utterance dicts.
    Supports three model output formats:
      - paraformer:   item["timestamp"] = [[start_ms, end_ms], ...]  (char-level)
      - Fun-ASR-Nano: item["timestamps"] = [{"start_time": s, "end_time": e}, ...]  (token-level, seconds)
      - SenseVoice:   VAD segments inferred individually; start/end taken from VAD boundaries
    Utterances are split by --silence-gap; SenseVoice splits at VAD segment boundaries.
    """
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
    # Force per-segment inference to avoid "batch decoding not implemented" with Fun-ASR-Nano
    if args.vad_model:
        generate_kwargs["batch_size_threshold_s"] = 0
        generate_kwargs["batch_size_s"] = 0

    # SenseVoice: infer each VAD segment separately to preserve real timestamps
    is_sensevoice = re.search(r"sensevoice", args.model, re.IGNORECASE)
    if is_sensevoice:
        return _transcribe_sensevoice_channel(model, wav_path, generate_kwargs)

    results = model.generate(input=wav_path, **generate_kwargs)

    utterances = []
    if not results:
        return utterances

    for item in results:
        text = item.get("text", "").strip()
        if not text:
            continue

        # paraformer: "timestamp" (char-level ms list); Fun-ASR-Nano: "timestamps" (token-level seconds dict)
        ts = item.get("timestamp") or item.get("timestamps") or []
        segs = _split_utterances_by_gap(text, ts, args.silence_gap)
        utterances.extend(segs)

    return utterances


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        logger.error("--input must be a file: %r", args.input)
        sys.exit(1)

    audio_data, sample_rate = sf.read(args.input, always_2d=True)
    total_dur_s = audio_data.shape[0] / sample_rate
    num_channels = audio_data.shape[1]
    channels_to_process = min(args.channels, num_channels)

    if num_channels < args.channels:
        logger.warning("audio has only %d channel(s), processing %d", num_channels, channels_to_process)

    basename = os.path.splitext(os.path.basename(args.input))[0]
    if args.output:
        output_path = args.output
    else:
        tag = _model_tag(args.model, args.vad_model or "", args.punc_model or "")
        output_path = f"results/{basename}.conversation.{tag}.json"
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    logger.info("[config] model=%s  vad=%s  punc=%s", args.model, args.vad_model, args.punc_model)
    logger.info("[config] hub=%s  device=%s  batch_size=%d", args.hub, args.device, args.batch_size)
    logger.info("[config] silence_gap=%.2fs  channels=%d  language=%s  use_itn=%s  merge_vad=%s",
                args.silence_gap, args.channels, args.language, args.use_itn, args.merge_vad)
    logger.info("[input]  %s  (%d ch, %.1fs)", args.input, num_channels, total_dur_s)

    t0 = time.perf_counter()
    model = load_model(args)
    logger.info("[timing] model loaded: %.3fs", time.perf_counter() - t0)

    all_utterances = []
    total_transcribe_s = 0.0
    total_vad_s = 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        for ch in range(channels_to_process):
            channel_audio = audio_data[:, ch]
            tmp_wav = os.path.join(tmpdir, f"ch{ch}.wav")
            sf.write(tmp_wav, channel_audio, sample_rate)

            logger.info("\n[channel %d] transcribing...", ch)
            t1 = time.perf_counter()
            utterances = transcribe_channel(model, tmp_wav, args)
            elapsed = time.perf_counter() - t1

            # Approximate VAD time: FunASR runs VAD internally; estimate as a fixed
            # fraction is not possible, so we record total elapsed and leave vad_s as 0
            # unless a future hook exposes it.
            total_transcribe_s += elapsed

            ch_dur = len(channel_audio) / sample_rate
            rtf = elapsed / ch_dur if ch_dur > 0 else 0.0
            rtfx = 1 / rtf if rtf > 0 else 0.0
            logger.info("[channel %d] %d utterance(s)  elapsed=%.3fs  RTF=%.4f  RTFx=%.2f",
                        ch, len(utterances), elapsed, rtf, rtfx)

            for u in utterances:
                logger.info("  [%.2f-%.2fs] %r", u["start"], u["end"], u["text"])
                all_utterances.append({
                    "role": f"channel_{ch}",
                    "text": u["text"],
                    "start": u["start"],
                    "end": u["end"],
                })

    # Sort by start time; break ties by channel order
    all_utterances.sort(key=lambda u: (u["start"], u["role"]))

    rtf = round(total_transcribe_s / total_dur_s, 4) if total_dur_s > 0 else None
    vad_rtf = round(total_vad_s / total_dur_s, 4) if total_dur_s > 0 and total_vad_s > 0 else None
    output = {
        "source": args.input,
        "filename": os.path.basename(args.input),
        "channels": channels_to_process,
        "audio_dur_s": round(total_dur_s, 3),
        "transcribe_s": round(total_transcribe_s, 3),
        "rtf": rtf,
        "rtfx": round(1 / rtf, 2) if rtf else None,
        "vad_s": round(total_vad_s, 3),
        "vad_rtf": vad_rtf,
        "vad_rtfx": round(1 / vad_rtf, 2) if vad_rtf else None,
        "punct_s": None,
        "punct_rtf": None,
        "punct_rtfx": None,
        "model_name": Path(args.model).name or args.model,
        "vad_model": Path(args.vad_model).name if args.vad_model else None,
        "punc_model": Path(args.punc_model).name if args.punc_model else None,
        "conversations": all_utterances,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("\n[result] %d utterance(s) total", len(all_utterances))
    for u in all_utterances[:8]:
        logger.info("  [%.2f-%.2f] %s: %r", u["start"], u["end"], u["role"], u["text"])
    if len(all_utterances) > 8:
        logger.info("  ... (%d more)", len(all_utterances) - 8)
    logger.info("[output] JSON written: %s", output_path)


if __name__ == "__main__":
    main()
