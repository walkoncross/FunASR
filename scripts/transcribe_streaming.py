# coding=utf-8
"""
FunASR 流式语音转写脚本（paraformer-zh-streaming）

逐块读取音频并实时输出识别结果，模拟流式 ASR 场景。
每个 chunk 对应一段固定时长的音频帧，最后一块设置 is_final=True 刷出剩余词。

Usage:
  python scripts/transcribe_streaming.py -i <音频文件或目录> [OPTIONS]

  --model/-m              ASR 流式模型名称或本地路径 (default: paraformer-zh-streaming)
  --punc-model/-pm        标点模型名称或路径，留空则不加标点 (default: ct-punc)
  --input/-i              音频文件或目录（必填）
  --output/-o             输出目录 (default: ./results/)
  --output-format/-f      输出格式: txt / json (default: json)
  --hub                   模型来源: modelscope / hf (default: modelscope)
  --device/-d             推理设备: cpu / cuda:0 / mps (default: cpu)
  --chunk-size            流式分块配置 [lookahead, chunk, shift]，单位帧(60ms)
                          (default: 0 10 5，即 600ms chunk / 300ms lookahead)
  --encoder-look-back     Encoder self-attention 回看 chunk 数 (default: 4)
  --decoder-look-back     Decoder cross-attention 回看 encoder chunk 数 (default: 1)
  --hotwords              热词字符串，空格分隔
  --enable-update         启用 FunASR 版本检查（默认禁用）

Output format (json):
  {
    "source": "...",
    "filename": "...",
    "text": "完整转写文本",
    "chunks": [
      {"chunk": 0, "is_final": false, "text": "部分识别"},
      ...
    ],
    "audio_dur_s": 5.12,
    "transcribe_s": 1.23,
    "rtf": 0.24
  }
"""

import argparse
import json
import logging
import os
import sys
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
FRAME_MS = 60          # paraformer-zh-streaming 每帧 60ms
SAMPLE_RATE = 16000    # 模型固定采样率


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FunASR 流式语音转写工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", "-m", default="paraformer-zh-streaming",
                        help="流式 ASR 模型名称或本地路径")
    parser.add_argument("--punc-model", "-pm", default="ct-punc",
                        help="标点模型名称或路径，留空则跳过标点恢复")
    parser.add_argument("--input", "-i", required=True, help="音频文件或目录")
    parser.add_argument("--output", "-o", default="./results/", help="输出目录")
    parser.add_argument("--output-format", "-f", default="json", choices=["txt", "json"],
                        help="输出格式")
    parser.add_argument("--hub", default="modelscope", choices=["modelscope", "hf"],
                        help="模型来源：modelscope 或 hf（HuggingFace）")
    parser.add_argument("--device", "-d", default="cpu",
                        help="推理设备：cpu / cuda:0 / mps")
    parser.add_argument("--chunk-size", nargs=3, type=int, default=[0, 10, 5],
                        metavar=("LOOKAHEAD", "CHUNK", "SHIFT"),
                        help="流式分块配置 [lookahead chunk shift]，单位帧(60ms)。"
                             "[0,10,5]=600ms chunk/300ms lookahead；[0,8,4]=480ms/240ms")
    parser.add_argument("--encoder-look-back", type=int, default=4,
                        help="Encoder self-attention 回看的 chunk 数")
    parser.add_argument("--decoder-look-back", type=int, default=1,
                        help="Decoder cross-attention 回看的 encoder chunk 数")
    parser.add_argument("--hotwords", default=None, help="热词字符串，空格分隔")
    parser.add_argument("--enable-update", action="store_true", default=False,
                        help="启用 FunASR 版本检查（默认禁用）")
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
        logger.info("使用 HuggingFace Hub 加载模型")
    else:
        logger.info("使用 ModelScope 加载模型")
    if args.hotwords:
        kwargs["hotword"] = args.hotwords

    return AutoModel(**kwargs)


def transcribe_streaming(model, audio_path: str, args) -> dict:
    """
    流式转写单个文件。
    将音频按 chunk_size[1]*FRAME_MS ms 分块逐块送入模型，收集每块输出。
    返回包含完整文本和逐块结果的 dict。
    """
    chunk_size = args.chunk_size
    encoder_look_back = args.encoder_look_back
    decoder_look_back = args.decoder_look_back

    # chunk 步长：chunk_size[1] 帧 × FRAME_MS ms × sample_rate / 1000
    chunk_stride = chunk_size[1] * FRAME_MS * SAMPLE_RATE // 1000  # samples per chunk

    speech, sample_rate = sf.read(audio_path, dtype="float32")
    if speech.ndim > 1:
        speech = speech[:, 0]  # 取第一声道
    if sample_rate != SAMPLE_RATE:
        logger.warning("采样率 %d != %d，模型可能报错，建议先重采样", sample_rate, SAMPLE_RATE)

    audio_dur_s = len(speech) / sample_rate
    total_chunk_num = int((len(speech) - 1) / chunk_stride + 1)

    logger.info("[stream] chunk_size=%s  chunk_stride=%d samples (%.0fms)  总块数=%d",
                chunk_size, chunk_stride, chunk_size[1] * FRAME_MS, total_chunk_num)

    cache = {}
    chunks_out = []
    full_text_parts = []

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
            if is_final:
                full_text_parts.append(text)
            else:
                # 流式输出是累积前缀，取最后一个非空块作为最终结果时会被 is_final 覆盖
                # 这里先保留逐块文本，最终文本以 is_final 块为准
                pass

    elapsed = time.perf_counter() - t0

    # 最终文本：取 is_final 块的文本（完整输出），若为空则拼接所有非空块
    final_text = chunks_out[-1]["text"] if chunks_out else ""
    if not final_text:
        final_text = " ".join(c["text"] for c in chunks_out if c["text"])

    return {
        "source": audio_path,
        "filename": os.path.basename(audio_path),
        "text": final_text,
        "chunks": chunks_out,
        "audio_dur_s": round(audio_dur_s, 3),
        "transcribe_s": round(elapsed, 3),
        "rtf": round(elapsed / audio_dur_s, 4) if audio_dur_s > 0 else None,
    }


def save_result(result: dict, audio_path: Path, output_dir: Path, fmt: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    base = audio_path.stem

    if fmt == "json":
        out_path = output_dir / f"{base}.streaming.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        out_path = output_dir / f"{base}.streaming.txt"
        out_path.write_text(result["text"], encoding="utf-8")

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

    logger.info("[config] model=%s  punc=%s", args.model, args.punc_model)
    logger.info("[config] hub=%s  device=%s", args.hub, args.device)
    logger.info("[config] chunk_size=%s  encoder_look_back=%d  decoder_look_back=%d",
                args.chunk_size, args.encoder_look_back, args.decoder_look_back)
    logger.info("[input]  %s", args.input)

    t0 = time.perf_counter()
    model = load_model(args)
    logger.info("[timing] 模型加载: %.3fs", time.perf_counter() - t0)

    files = collect_files(args.input)
    logger.info("[info]   共 %d 个音频文件待处理", len(files))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_audio_s = 0.0
    total_transcribe_s = 0.0

    for i, f in enumerate(files, 1):
        logger.info("\n[%d/%d] 处理: %s", i, len(files), f.name)
        result = transcribe_streaming(model, str(f), args)
        logger.info("[result] RTF=%.4f  文本: %s", result["rtf"] or 0, result["text"])
        save_result(result, f, output_dir, args.output_format)

        total_audio_s += result["audio_dur_s"]
        total_transcribe_s += result["transcribe_s"]

    logger.info("\n[summary] 处理文件数: %d", len(files))
    logger.info("[summary] 总音频时长: %.1fs", total_audio_s)
    logger.info("[summary] 总转写耗时: %.1fs", total_transcribe_s)
    if total_audio_s > 0:
        logger.info("[summary] 整体 RTF: %.4f", total_transcribe_s / total_audio_s)


if __name__ == "__main__":
    main()
