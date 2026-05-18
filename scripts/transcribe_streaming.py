# coding=utf-8
"""
FunASR streaming speech transcription script (paraformer-zh-streaming)

Reads audio in fixed-size chunks and outputs recognition results in real time,
simulating a streaming ASR scenario. Each chunk corresponds to a fixed audio frame
duration; the last chunk sets is_final=True to flush remaining tokens.

Usage:
  python scripts/transcribe_streaming.py -i <audio_file_or_dir> [OPTIONS]

  --model/-m              Streaming ASR model name or local path (default: paraformer-zh-streaming)
  --punc-model/-pm        Punctuation model name or path; leave empty to skip (default: ct-punc)
  --input/-i              Audio file or directory (required)
  --output/-o             Output directory (default: ./results/)
  --hub                   Model source: modelscope / hf (default: modelscope)
  --device/-d             Inference device: cpu / cuda:0 / mps (default: cpu)
  --chunk-size            Streaming chunk config [lookahead, chunk, shift] in frames (60ms each)
                          (default: 0 10 5, i.e. 600ms chunk / 300ms lookahead)
  --encoder-look-back     Number of encoder self-attention look-back chunks (default: 4)
  --decoder-look-back     Number of decoder cross-attention look-back encoder chunks (default: 1)
  --hotwords              Hotwords string, space-separated
  --enable-update         Enable FunASR version check (disabled by default)
  --separate-channel/-sc  Split channels and transcribe each separately

Output format (json):  <stem>.<model>.no-vad.<punc>.json
  {
    "source": "...",
    "filename": "...",
    "audio_dur_s": 5.12,
    "transcribe_s": 1.23,
    "rtf": 0.24,
    "rtfx": 4.16,
    "model_name": "paraformer-zh-streaming",
    "vad_model": null,
    "punc_model": "ct-punc",
    "text": "full transcription text",
    "chunks": [
      {"chunk": 0, "is_final": false, "text": "partial result"},
      ...
    ]
  }
  With --separate-channel, filenames get a _channel0 / _channel1 suffix and
  the JSON includes "channel": 0.
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

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4", ".aac"}
FRAME_MS = 60          # paraformer-zh-streaming: 60ms per frame
SAMPLE_RATE = 16000    # model fixed sample rate


def _model_tag(model_name: str, punc_model: str) -> str:
    """Generate filename tag; streaming has no VAD, so only model name and punc info."""
    name = Path(model_name).name or model_name
    name = name.replace("/", "-")
    punc_tag = Path(punc_model).name if punc_model else "no-punc"
    return f"{name}.no-vad.{punc_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FunASR streaming speech transcription tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "-m", default="paraformer-zh-streaming",
                        help="Streaming ASR model name or local path")
    parser.add_argument("--punc-model", "-pm", default="ct-punc",
                        help="Punctuation model name or path; leave empty to skip")
    parser.add_argument("--input", "-i", required=True, help="Audio file or directory")
    parser.add_argument("--output", "-o", default="./results/", help="Output directory")
    parser.add_argument("--hub", default="modelscope", choices=["modelscope", "hf"],
                        help="Model source: modelscope or hf (HuggingFace)")
    parser.add_argument("--device", "-d", default="cpu",
                        help="Inference device: cpu / cuda:0 / mps")
    parser.add_argument("--chunk-size", nargs=3, type=int, default=[0, 10, 5],
                        metavar=("LOOKAHEAD", "CHUNK", "SHIFT"),
                        help="Streaming chunk config [lookahead chunk shift] in frames (60ms each). "
                             "[0,10,5]=600ms chunk/300ms lookahead; [0,8,4]=480ms/240ms")
    parser.add_argument("--encoder-look-back", type=int, default=4,
                        help="Number of encoder self-attention look-back chunks")
    parser.add_argument("--decoder-look-back", type=int, default=1,
                        help="Number of decoder cross-attention look-back encoder chunks")
    parser.add_argument("--hotwords", default=None, help="Hotwords string, space-separated")
    parser.add_argument("--enable-update", action="store_true", default=False,
                        help="Enable FunASR version check (disabled by default)")
    parser.add_argument("--separate-channel", "-sc", action="store_true", default=False,
                        help="Split channels and transcribe each separately")
    return parser.parse_args()


def load_model(args) -> "AutoModel":
    from funasr import AutoModel

    kwargs = dict(
        model=args.model,
        device=args.device,
        disable_update=not args.enable_update,
    )
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


def transcribe_streaming(model, audio_path: str, args) -> dict:
    """
    Stream-transcribe a single file.
    Feeds audio in chunk_size[1]*FRAME_MS ms blocks and collects per-chunk output.
    Returns a dict with the full text and per-chunk results.
    """
    speech, sample_rate = sf.read(audio_path, dtype="float32")
    if speech.ndim > 1:
        speech = speech[:, 0]  # take first channel
    if sample_rate != SAMPLE_RATE:
        logger.warning("sample rate %d != %d; model may error; consider resampling first",
                       sample_rate, SAMPLE_RATE)
    result = _stream_one_channel(model, speech, sample_rate, args)
    result["source"] = audio_path
    result["filename"] = os.path.basename(audio_path)
    return result


def _stream_one_channel(model, speech, sample_rate: int, args) -> dict:
    """
    Stream-transcribe a single mono numpy array.
    Returns a partial dict (without source/filename/channel).
    """
    chunk_size = args.chunk_size
    encoder_look_back = args.encoder_look_back
    decoder_look_back = args.decoder_look_back

    chunk_stride = chunk_size[1] * FRAME_MS * SAMPLE_RATE // 1000  # samples per chunk

    audio_dur_s = len(speech) / sample_rate
    total_chunk_num = int((len(speech) - 1) / chunk_stride + 1)

    logger.info("[stream] chunk_size=%s  chunk_stride=%d samples (%.0fms)  total_chunks=%d",
                chunk_size, chunk_stride, chunk_size[1] * FRAME_MS, total_chunk_num)

    cache = {}
    chunks_out = []

    t0 = time.perf_counter()
    for i in range(total_chunk_num):
        speech_chunk = speech[i * chunk_stride: (i + 1) * chunk_stride]
        is_final = (i == total_chunk_num - 1)

        res = model.generate(
            input=speech_chunk,
            cache=cache,
            is_final=is_final,
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_look_back,
            decoder_chunk_look_back=decoder_look_back,
        )

        text = ""
        if res and isinstance(res, list):
            text = res[0].get("text", "").strip()

        chunks_out.append({"chunk": i, "is_final": is_final, "text": text})

        if text:
            status = "[FINAL]" if is_final else f"[{i:04d}]"
            logger.info("  %s %s", status, text)

    elapsed = time.perf_counter() - t0

    # Final text: use the is_final chunk (complete output); fall back to joining all non-empty chunks
    final_text = chunks_out[-1]["text"] if chunks_out else ""
    if not final_text:
        final_text = " ".join(c["text"] for c in chunks_out if c["text"])

    rtf = round(elapsed / audio_dur_s, 4) if audio_dur_s > 0 else None
    return {
        "audio_dur_s": round(audio_dur_s, 3),
        "transcribe_s": round(elapsed, 3),
        "rtf": rtf,
        "rtfx": round(1 / rtf, 2) if rtf else None,
        "model_name": Path(args.model).name or args.model,
        "vad_model": None,
        "punc_model": Path(args.punc_model).name if args.punc_model else None,
        "text": final_text,
        "chunks": chunks_out,
    }


def save_result(result: dict, audio_path: Path, output_dir: Path, args=None,
                channel: int | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    base = audio_path.stem
    if channel is not None:
        base = f"{base}_channel{channel}"
    if args is not None:
        tag = _model_tag(args.model, args.punc_model or "")
        filename = f"{base}.{tag}.json"
    else:
        filename = f"{base}.streaming.json"
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

    logger.info("[config] model=%s  punc=%s", args.model, args.punc_model)
    logger.info("[config] hub=%s  device=%s", args.hub, args.device)
    logger.info("[config] chunk_size=%s  encoder_look_back=%d  decoder_look_back=%d",
                args.chunk_size, args.encoder_look_back, args.decoder_look_back)
    logger.info("[config] separate_channel=%s", args.separate_channel)
    logger.info("[input]  %s", args.input)

    t0 = time.perf_counter()
    model = load_model(args)
    logger.info("[timing] model loaded: %.3fs", time.perf_counter() - t0)

    files = collect_files(args.input)
    logger.info("[info]   %d audio file(s) to process", len(files))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_audio_s = 0.0
    total_transcribe_s = 0.0

    for i, f in enumerate(files, 1):
        logger.info("\n[%d/%d] processing: %s", i, len(files), f.name)

        if args.separate_channel:
            audio_data, sample_rate = sf.read(str(f), dtype="float32", always_2d=True)
            if sample_rate != SAMPLE_RATE:
                logger.warning("sample rate %d != %d; model may error; consider resampling first",
                               sample_rate, SAMPLE_RATE)
            num_channels = audio_data.shape[1]
            logger.info("[channel] detected %d channel(s), transcribing each separately", num_channels)

            with tempfile.TemporaryDirectory() as tmpdir:
                for ch in range(num_channels):
                    logger.info("[channel %d] transcribing...", ch)
                    result = _stream_one_channel(model, audio_data[:, ch], sample_rate, args)
                    result["source"] = str(f)
                    result["filename"] = f.name
                    result["channel"] = ch
                    logger.info("[channel %d] RTF=%.4f  RTFx=%.2f  text: %s",
                                ch, result["rtf"] or 0, result["rtfx"] or 0, result["text"])
                    save_result(result, f, output_dir, args=args, channel=ch)

                    total_audio_s += result["audio_dur_s"]
                    total_transcribe_s += result["transcribe_s"]
        else:
            result = transcribe_streaming(model, str(f), args)
            logger.info("[result] RTF=%.4f  RTFx=%.2f  text: %s",
                        result["rtf"] or 0, result["rtfx"] or 0, result["text"])
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
