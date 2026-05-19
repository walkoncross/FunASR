# coding=utf-8
"""
FunASR simulated-online streaming transcription with VAD (paraformer-zh-streaming + fsmn-vad)

Simulates a real-time AI outbound / AI customer service pipeline:
  audio stream → fsmn-vad (endpoint detection) → speech segments
               → paraformer-zh-streaming (streaming ASR) → utterance text

Two-channel audio is processed on a shared wall-clock timeline:
  channel 0 = user  (客户 / 用户)
  channel 1 = agent (人工坐席 / AI坐席)

Each channel has its own independent VAD state and ASR cache. The simulated
wall-clock advances in fixed chunk_stride steps; VAD runs on every chunk and
gates ASR input — only samples inside a detected speech segment are fed to ASR.
When VAD signals end-of-speech (is_end), ASR is flushed with is_final=True and
the completed utterance is emitted.

Usage:
  python scripts/transcribe_streaming_with_vad.py -i <audio_file_or_dir> [OPTIONS]

  --model/-m              Streaming ASR model (default: paraformer-zh-streaming)
  --vad-model/-vm         VAD model (default: fsmn-vad)
  --punc-model/-pm        Punctuation model; leave empty to skip (default: ct-punc)
  --input/-i              Audio file or directory (required)
  --output/-o             Output directory (default: ./results/)
  --hub                   Model source: modelscope / hf (default: modelscope)
  --device/-d             Inference device: cpu / cuda:0 / mps (default: cpu)
  --chunk-size            ASR chunk config [lookahead chunk shift] in frames (60ms each)
                          (default: 0 8 4 — 480ms chunk / 240ms look-ahead, real-time profile)
  --encoder-look-back     Encoder self-attention look-back chunks (default: 4)
  --decoder-look-back     Decoder cross-attention look-back encoder chunks (default: 1)
  --channels              Channel indices to transcribe (default: 0 1 — both channels)
                          Use --channels 0 for user-only (single-channel mode)
  --vad-chunk-ms          VAD input chunk size in ms (default: 200)
  --hotwords              Hotwords string, space-separated
  --enable-update         Enable FunASR version check (disabled by default)

Output format (json):  <stem>.streaming-vad.<model>.<vad>.<punc>.json
  {
    "source": "...",
    "filename": "...",
    "audio_dur_s": 306.68,
    "transcribe_s": 12.34,
    "rtf": 0.040,
    "rtfx": 24.87,
    "asr_model": "paraformer-zh-streaming",
    "vad_model": "fsmn-vad",
    "punc_model": "ct-punc",
    "conversations": [
      {
        "role": "user",      // channel 0
        "text": "你好，我想取消订单",
        "start_s": 17.17,
        "end_s": 19.50,
        "audio_dur_s": 2.33,
        "transcribe_s": 0.12,
        "rtf": 0.051,
        "rtfx": 19.4,
        "vad_s": 0.03,
        "vad_rtf": 0.013,
        "vad_rtfx": 77.5,
        "final_chunk_ms": 85.2,
        "seg_asr_ms": 120.4
      },
      {
        "role": "agent",     // channel 1
        "text": "好的，请稍等",
        "start_s": 20.10,
        "end_s": 21.30
      },
      ...
    ]
  }
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".aac"}
FRAME_MS = 60        # paraformer-zh-streaming: 60ms per frame
SAMPLE_RATE = 16000  # model fixed sample rate

ROLE_NAMES = {0: "user", 1: "agent"}


def _percentiles(values: list[float]) -> dict:
    """Compute P50/P90/P95/P99 from a list of float values. Returns {} if empty."""
    if not values:
        return {}
    s = sorted(values)
    n = len(s)

    def _p(pct: float) -> float:
        idx = pct / 100 * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 2)

    return {"p50": _p(50), "p90": _p(90), "p95": _p(95), "p99": _p(99)}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulated-online streaming ASR with VAD (AI outbound / customer service)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "-m", default="paraformer-zh-streaming",
                        help="Streaming ASR model name or local path")
    parser.add_argument("--vad-model", "-vm", default="fsmn-vad",
                        help="VAD model name or local path")
    parser.add_argument("--punc-model", "-pm", default="ct-punc",
                        help="Punctuation model name or path; leave empty to skip")
    parser.add_argument("--input", "-i", required=True, help="Audio file or directory")
    parser.add_argument("--output", "-o", default="./results/", help="Output directory")
    parser.add_argument("--hub", default="modelscope", choices=["modelscope", "hf"],
                        help="Model source: modelscope or hf (HuggingFace)")
    parser.add_argument("--device", "-d", default="cpu",
                        help="Inference device: cpu / cuda:0 / mps")
    parser.add_argument("--chunk-size", nargs=3, type=int, default=[0, 8, 4],
                        metavar=("LOOKAHEAD", "CHUNK", "SHIFT"),
                        help="ASR chunk config [lookahead chunk shift] in frames (60ms each). "
                             "[0,8,4]=480ms/240ms look-ahead (real-time); "
                             "[0,10,5]=600ms/300ms (balanced); "
                             "[0,16,8]=960ms/480ms (offline-batch)")
    parser.add_argument("--encoder-look-back", type=int, default=4,
                        help="Encoder self-attention look-back chunks")
    parser.add_argument("--decoder-look-back", type=int, default=1,
                        help="Decoder cross-attention look-back encoder chunks")
    parser.add_argument("--vad-chunk-ms", type=int, default=200,
                        help="VAD input chunk size in ms")
    parser.add_argument("--channels", nargs="+", type=int, default=[0, 1],
                        metavar="CH",
                        help="Channel indices to transcribe (0=user, 1=agent). "
                             "Use --channels 0 for single-channel (user only). "
                             "Default: 0 1 (both channels)")
    parser.add_argument("--latency-mode", default="fast", choices=["fast", "realtime"],
                        help="Latency measurement mode. "
                             "'fast': no sleep, maximum throughput (default). "
                             "'realtime': sleep per wall-clock step to simulate true real-time ingestion.")
    parser.add_argument("--hotwords", default=None, help="Hotwords string, space-separated")
    parser.add_argument("--enable-update", action="store_true", default=False,
                        help="Enable FunASR version check (disabled by default)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(args):
    """Load ASR model and VAD model separately. Returns (asr_model, vad_model)."""
    from funasr import AutoModel

    common = dict(device=args.device, disable_update=not args.enable_update)
    if args.hub == "hf":
        common["hub"] = "hf"
        logger.info("Loading models from HuggingFace Hub")
    else:
        logger.info("Loading models from ModelScope")

    # ASR model (with optional punctuation)
    asr_kwargs = dict(model=args.model, **common)
    if args.punc_model:
        asr_kwargs["punc_model"] = args.punc_model
    if args.hotwords:
        asr_kwargs["hotword"] = args.hotwords
    logger.info("Loading ASR model: %s", args.model)
    asr_model = AutoModel(**asr_kwargs)

    # VAD model (standalone, no punc)
    logger.info("Loading VAD model: %s", args.vad_model)
    vad_model = AutoModel(model=args.vad_model, **common)

    return asr_model, vad_model


# ---------------------------------------------------------------------------
# Per-channel streaming state
# ---------------------------------------------------------------------------

class ChannelState:
    """Holds all mutable state for one audio channel during online simulation."""

    def __init__(self, channel: int, role: str):
        self.channel = channel
        self.role = role
        self.asr_cache: dict = {}
        self.vad_cache: dict = {}
        # Wall-clock start sample of the current speech segment
        self.seg_start_sample: int | None = None
        # Whether VAD currently considers us inside a speech segment
        self.in_speech: bool = False
        # Accumulated ASR partial texts for the current segment
        self.seg_texts: list[str] = []
        # Cumulative ASR inference time for the current segment (ms)
        self.seg_asr_elapsed_ms: float = 0.0
        # Cumulative VAD inference time for the current segment (ms)
        self.seg_vad_elapsed_ms: float = 0.0
        # Completed utterances
        self.utterances: list[dict] = []

    def reset_segment(self):
        self.seg_start_sample = None
        self.in_speech = False
        self.seg_texts = []
        self.seg_asr_elapsed_ms = 0.0
        self.seg_vad_elapsed_ms = 0.0


# ---------------------------------------------------------------------------
# Core online simulation
# ---------------------------------------------------------------------------

def _run_asr_chunk(asr_model, state: ChannelState, pcm: np.ndarray,
                   is_final: bool, args) -> float:
    """
    Send one PCM chunk to ASR. Accumulates partial text into state.seg_texts.
    Returns the inference latency for this single chunk in ms.
    """
    t = time.perf_counter()
    try:
        res = asr_model.generate(
            input=pcm,
            cache=state.asr_cache,
            is_final=is_final,
            chunk_size=args.chunk_size,
            encoder_chunk_look_back=args.encoder_look_back,
            decoder_chunk_look_back=args.decoder_look_back,
        )
    except RuntimeError as e:
        logger.warning("[ch%d %s] ASR error (is_final=%s, %d samples): %s",
                       state.channel, state.role, is_final, len(pcm), e)
        state.asr_cache = {}
        return round((time.perf_counter() - t) * 1000, 2)
    latency_ms = round((time.perf_counter() - t) * 1000, 2)
    if res and isinstance(res, list):
        text = res[0].get("text", "").strip()
        if text:
            state.seg_texts.append(text)
    return latency_ms


def _emit_utterance(state: ChannelState, start_s: float, end_s: float,
                    final_chunk_ms: float) -> dict | None:
    """
    Finalise the current segment: reset ASR cache, build and return utterance dict.
    - final_chunk_ms: inference time of the is_final=True chunk
    - seg_asr_ms: cumulative ASR inference time = end-to-end latency for this model
    Returns None if no text was produced.
    """
    text = "".join(state.seg_texts)
    seg_asr_ms = state.seg_asr_elapsed_ms
    seg_vad_ms = state.seg_vad_elapsed_ms
    state.asr_cache = {}
    state.reset_segment()
    if not text:
        return None

    audio_dur_s = round(end_s - start_s, 3)
    transcribe_s = round(seg_asr_ms / 1000, 6)
    vad_s = round(seg_vad_ms / 1000, 6)
    rtf = round(transcribe_s / audio_dur_s, 4) if audio_dur_s > 0 else None
    rtfx = round(1 / rtf, 2) if rtf else None
    vad_rtf = round(vad_s / audio_dur_s, 4) if audio_dur_s > 0 else None
    vad_rtfx = round(1 / vad_rtf, 2) if vad_rtf else None

    return {
        "role": state.role,
        "text": text,
        "start_s": start_s,
        "end_s": end_s,
        "audio_dur_s": audio_dur_s,
        "transcribe_s": transcribe_s,
        "rtf": rtf,
        "rtfx": rtfx,
        "vad_s": vad_s,
        "vad_rtf": vad_rtf,
        "vad_rtfx": vad_rtfx,
        "final_chunk_ms": round(final_chunk_ms, 2),  # last is_final chunk inference time
        "seg_asr_ms": round(seg_asr_ms, 2),          # cumulative ASR time = end-to-end latency
    }


def simulate_online(asr_model, vad_model, audio_data: np.ndarray,
                    sample_rate: int, args) -> tuple[list[dict], float, dict]:
    """
    Simulate online two-channel streaming with VAD.

    audio_data: shape (n_samples, n_channels)

    Two latency modes (--latency-mode):
      fast     — no sleep; maximum throughput.
      realtime — sleeps per wall-clock step to simulate true real-time ingestion.

    Returns (conversations, elapsed_s, latency_stats).
      latency_stats keys:
        "final_chunk_ms"  — {p50,p90,p95,p99}: last is_final chunk inference time
        "seg_asr_ms"      — {p50,p90,p95,p99}: full-segment cumulative ASR time = end-to-end latency
    """
    assert audio_data.ndim == 2, "audio_data must be 2-D (n_samples, n_channels)"
    channels = sorted(set(args.channels))
    assert all(ch < audio_data.shape[1] for ch in channels), \
        f"requested channels {channels} but audio only has {audio_data.shape[1]} channel(s)"

    realtime = (args.latency_mode == "realtime")
    chunk_stride = args.chunk_size[1] * FRAME_MS * sample_rate // 1000
    vad_chunk_samples = int(args.vad_chunk_ms * sample_rate / 1000)
    wall_step = max(chunk_stride, vad_chunk_samples)
    wall_step_s = wall_step / sample_rate          # seconds per wall-clock step
    # Minimum speech samples to bother sending to ASR (avoids ct-punc conv1d error)
    min_speech_samples = int(0.2 * sample_rate)

    n_samples = audio_data.shape[0]
    states = [ChannelState(channel=ch, role=ROLE_NAMES.get(ch, f"ch{ch}")) for ch in channels]

    logger.info("[online-sim] samples=%d  duration=%.2fs  wall_step=%d samples (%.0fms)  mode=%s",
                n_samples, n_samples / sample_rate, wall_step, wall_step_s * 1000, args.latency_mode)

    t0 = time.perf_counter()
    # In realtime mode, track the "scheduled" wall-clock time for the next chunk
    next_wall_t = t0
    pos = 0

    while pos < n_samples:
        chunk_end = min(pos + wall_step, n_samples)
        is_last_wall_chunk = (chunk_end == n_samples)

        if realtime:
            # Sleep until this chunk's scheduled wall-clock time
            now = time.perf_counter()
            if next_wall_t > now:
                time.sleep(next_wall_t - now)
            next_wall_t += wall_step_s

        for state in states:
            ch = state.channel
            pcm = audio_data[pos:chunk_end, ch]

            # --- VAD ---
            t_vad = time.perf_counter()
            vad_res = vad_model.generate(
                input=pcm,
                cache=state.vad_cache,
                is_final=is_last_wall_chunk,
                chunk_size=args.vad_chunk_ms,
            )
            vad_chunk_ms = (time.perf_counter() - t_vad) * 1000

            # vad_res[0]["value"] is a list of [start_ms, end_ms] pairs.
            # start_ms >= 0, end_ms == -1 → speech started, not yet ended
            # start_ms == -1, end_ms >= 0 → ongoing speech ended in this chunk
            # start_ms >= 0, end_ms >= 0  → speech start and end in same chunk
            # empty list                  → silence
            segments = []
            if vad_res and isinstance(vad_res, list) and vad_res[0].get("value"):
                segments = vad_res[0]["value"]

            for seg in segments:
                seg_start_ms, seg_end_ms = seg

                if seg_start_ms >= 0 and not state.in_speech:
                    abs_start = pos + int(seg_start_ms * sample_rate / 1000)
                    state.in_speech = True
                    state.seg_start_sample = abs_start
                    logger.debug("[ch%d %s] speech start  %.3fs",
                                 ch, state.role, abs_start / sample_rate)

                if state.in_speech:
                    # Accumulate VAD time for this chunk while in speech
                    state.seg_vad_elapsed_ms += vad_chunk_ms

                    if seg_end_ms >= 0:
                        # VAD signals end-of-speech — flush ASR with is_final=True
                        abs_end = pos + int(seg_end_ms * sample_rate / 1000)
                        seg_samples = abs_end - state.seg_start_sample
                        if seg_samples >= min_speech_samples:
                            final_chunk_ms = _run_asr_chunk(asr_model, state, pcm, True, args)
                            state.seg_asr_elapsed_ms += final_chunk_ms
                        else:
                            logger.debug("[ch%d %s] segment too short (%d samples), skipping",
                                         ch, state.role, seg_samples)
                            state.asr_cache = {}
                            final_chunk_ms = 0.0

                        start_s = round(state.seg_start_sample / sample_rate, 3)
                        end_s = round(abs_end / sample_rate, 3)
                        utt = _emit_utterance(state, start_s, end_s, final_chunk_ms)
                        if utt:
                            state.utterances.append(utt)
                            logger.info("[ch%d %s] %.3fs–%.3fs  final_chunk=%.1fms  seg_asr=%.1fms  %s",
                                        ch, state.role, start_s, end_s,
                                        utt["final_chunk_ms"], utt["seg_asr_ms"], utt["text"])
                    else:
                        # Still inside speech — send intermediate ASR chunk (is_final=False)
                        latency_ms = _run_asr_chunk(asr_model, state, pcm, False, args)
                        state.seg_asr_elapsed_ms += latency_ms

            # Last wall chunk: force-flush any channel still in speech
            if is_last_wall_chunk and state.in_speech:
                abs_end = n_samples
                seg_samples = abs_end - (state.seg_start_sample or 0)
                state.seg_vad_elapsed_ms += vad_chunk_ms  # count last VAD chunk
                if seg_samples >= min_speech_samples:
                    final_chunk_ms = _run_asr_chunk(asr_model, state, pcm, True, args)
                    state.seg_asr_elapsed_ms += final_chunk_ms
                else:
                    state.asr_cache = {}
                    final_chunk_ms = 0.0
                start_s = round((state.seg_start_sample or 0) / sample_rate, 3)
                end_s = round(abs_end / sample_rate, 3)
                utt = _emit_utterance(state, start_s, end_s, final_chunk_ms)
                if utt:
                    state.utterances.append(utt)
                    logger.info("[ch%d %s] %.3fs–%.3fs  final_chunk=%.1fms  seg_asr=%.1fms  %s (force-flush)",
                                ch, state.role, start_s, end_s,
                                utt["final_chunk_ms"], utt["seg_asr_ms"], utt["text"])

        pos = chunk_end

    elapsed = time.perf_counter() - t0

    all_utterances = []
    for state in states:
        all_utterances.extend(state.utterances)
    all_utterances.sort(key=lambda u: u["start_s"])

    final_chunk_list = [u["final_chunk_ms"] for u in all_utterances]
    seg_asr_list = [u["seg_asr_ms"] for u in all_utterances]
    latency_stats = {
        "final_chunk_ms": _percentiles(final_chunk_list),
        "seg_asr_ms": _percentiles(seg_asr_list),
    }

    return all_utterances, elapsed, latency_stats


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _model_tag(asr_name: str, vad_name: str, punc_name: str) -> str:
    asr = Path(asr_name).name or asr_name
    vad = Path(vad_name).name or vad_name
    punc = Path(punc_name).name if punc_name else "no-punc"
    return f"{asr}.{vad}.{punc}"


def save_result(result: dict, audio_path: Path, output_dir: Path, args) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = _model_tag(args.model, args.vad_model, args.punc_model or "")
    stem = f"{audio_path.stem}.streaming-vad.{tag}"

    # JSON result
    out_path = output_dir / f"{stem}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[output] saved: %s", out_path)

    # Latency CSV — one row per utterance
    csv_path = output_dir / f"{stem}.latency.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["utt_id", "role", "start_s", "end_s",
                           "audio_dur_s", "transcribe_s", "rtf", "rtfx",
                           "vad_s", "vad_rtf", "vad_rtfx",
                           "final_chunk_ms", "seg_asr_ms", "text"])
        writer.writeheader()
        for i, utt in enumerate(result.get("conversations", [])):
            writer.writerow({
                "utt_id": i,
                "role": utt["role"],
                "start_s": utt["start_s"],
                "end_s": utt["end_s"],
                "audio_dur_s": utt.get("audio_dur_s", ""),
                "transcribe_s": utt.get("transcribe_s", ""),
                "rtf": utt.get("rtf", ""),
                "rtfx": utt.get("rtfx", ""),
                "vad_s": utt.get("vad_s", ""),
                "vad_rtf": utt.get("vad_rtf", ""),
                "vad_rtfx": utt.get("vad_rtfx", ""),
                "final_chunk_ms": utt.get("final_chunk_ms", ""),
                "seg_asr_ms": utt.get("seg_asr_ms", ""),
                "text": utt["text"],
            })
    logger.info("[output] saved: %s", csv_path)

    return out_path


def collect_files(input_path: str) -> list[Path]:
    p = Path(input_path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)
    else:
        logger.error("path not found: %s", input_path)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logger.info("[config] asr=%s  vad=%s  punc=%s", args.model, args.vad_model, args.punc_model)
    logger.info("[config] hub=%s  device=%s", args.hub, args.device)
    logger.info("[config] chunk_size=%s  encoder_look_back=%d  decoder_look_back=%d",
                args.chunk_size, args.encoder_look_back, args.decoder_look_back)
    logger.info("[config] vad_chunk_ms=%d  channels=%s  latency_mode=%s",
                args.vad_chunk_ms, args.channels, args.latency_mode)
    logger.info("[input]  %s", args.input)

    t0 = time.perf_counter()
    asr_model, vad_model = load_models(args)
    logger.info("[timing] models loaded: %.3fs", time.perf_counter() - t0)

    files = collect_files(args.input)
    logger.info("[info]   %d audio file(s) to process", len(files))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_audio_s = 0.0
    total_transcribe_s = 0.0

    for i, f in enumerate(files, 1):
        logger.info("\n[%d/%d] processing: %s", i, len(files), f.name)

        audio_data, sample_rate = sf.read(str(f), dtype="float32", always_2d=True)
        if sample_rate != SAMPLE_RATE:
            logger.warning("sample rate %d != %d; model may error; consider resampling first",
                           sample_rate, SAMPLE_RATE)
        needed = max(args.channels) + 1
        if audio_data.shape[1] < needed:
            logger.warning("file has %d channel(s) but channels %s requested; padding missing channels with silence",
                           audio_data.shape[1], args.channels)
            audio_data = np.pad(audio_data, ((0, 0), (0, needed - audio_data.shape[1])))

        audio_dur_s = audio_data.shape[0] / sample_rate

        conversations, elapsed, latency_stats = simulate_online(
            asr_model, vad_model, audio_data, sample_rate, args)

        rtf = round(elapsed / audio_dur_s, 4) if audio_dur_s > 0 else None
        result = {
            "source": str(f),
            "filename": f.name,
            "audio_dur_s": round(audio_dur_s, 3),
            "transcribe_s": round(elapsed, 3),
            "rtf": rtf,
            "rtfx": round(1 / rtf, 2) if rtf else None,
            "latency_mode": args.latency_mode,
            "asr_model": Path(args.model).name or args.model,
            "vad_model": Path(args.vad_model).name or args.vad_model,
            "punc_model": Path(args.punc_model).name if args.punc_model else None,
            "final_chunk_ms": latency_stats.get("final_chunk_ms", {}),
            "seg_asr_ms": latency_stats.get("seg_asr_ms", {}),
            "conversations": conversations,
        }

        fc = latency_stats.get("final_chunk_ms", {})
        sa = latency_stats.get("seg_asr_ms", {})
        logger.info("[result] RTF=%.4f  RTFx=%.2f  utterances=%d  "
                    "final_chunk p50=%.1fms p95=%.1fms  "
                    "seg_asr p50=%.1fms p95=%.1fms",
                    result["rtf"] or 0, result["rtfx"] or 0, len(conversations),
                    fc.get("p50", 0), fc.get("p95", 0),
                    sa.get("p50", 0), sa.get("p95", 0))
        save_result(result, f, output_dir, args)

        total_audio_s += audio_dur_s
        total_transcribe_s += elapsed

    logger.info("\n[summary] files processed: %d", len(files))
    logger.info("[summary] total audio duration: %.1fs", total_audio_s)
    logger.info("[summary] total transcribe time: %.1fs", total_transcribe_s)
    if total_audio_s > 0:
        overall_rtf = total_transcribe_s / total_audio_s
        logger.info("[summary] overall RTF: %.4f  RTFx: %.2f", overall_rtf, 1 / overall_rtf)


if __name__ == "__main__":
    main()
