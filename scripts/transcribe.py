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
  --output-format/-f  输出格式: txt / json (default: txt)
  --hub               模型来源: modelscope / hf (default: modelscope)
  --device/-d         推理设备: cpu / cuda:0 / mps (default: cpu)
  --batch-size/-bs    推理 batch size (default: 1)
  --separate-channel/-sc  分离声道分别转录
  --hotwords          热词字符串，空格分隔
  --enable-update     启用 FunASR 版本检查（默认禁用）
"""

import argparse
import json
import logging
import os
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
    parser.add_argument("--output-format", "-f", default="json", choices=["txt", "json"],
                        help="输出格式")
    parser.add_argument("--hub", default="modelscope", choices=["modelscope", "hf"],
                        help="模型来源：modelscope 或 hf（HuggingFace）")
    parser.add_argument("--device", "-d", default="cpu", help="推理设备：cpu / cuda:0 / mps")
    parser.add_argument("--batch-size", "-bs", type=int, default=1, help="推理 batch size")
    parser.add_argument("--separate-channel", "-sc", action="store_true", default=False,
                        help="分离声道分别转录，每声道独立输出")
    parser.add_argument("--hotwords", default=None, help="热词字符串，空格分隔")
    parser.add_argument("--enable-update", action="store_true", default=False,
                        help="启用 FunASR 版本检查（默认禁用）")
    return parser.parse_args()


def _timed(label: str, fn, *args, audio_dur_s: float = 0.0, **kwargs):
    """执行 fn 并返回 (result, elapsed_s)，打印耗时和 RTF。"""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    logger.info("[timing] %s: %.3fs", label, elapsed)
    if audio_dur_s > 0:
        rtf = elapsed / audio_dur_s
        logger.info("[RTF]    RTF=%.4f，即每秒可处理 %.2f 秒音频", rtf, 1 / rtf)
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
        vad_model=args.vad_model,
        punc_model=args.punc_model,
        device=args.device,
        batch_size=args.batch_size,
        disable_update=not args.enable_update,
    )

    if args.hub == "hf":
        # HuggingFace Hub：需要 model_id 形如 "funasr/paraformer-zh"
        kwargs["hub"] = "hf"
        logger.info("使用 HuggingFace Hub 加载模型")
    else:
        logger.info("使用 ModelScope 加载模型")

    if args.hotwords:
        kwargs["hotword"] = args.hotwords

    return AutoModel(**kwargs)


def transcribe_file(model, audio_path: str, args) -> dict:
    """转写单个文件，返回结果 dict。"""
    audio_dur_s = _audio_duration(audio_path)

    generate_kwargs = {}
    if args.hotwords:
        generate_kwargs["hotword"] = args.hotwords

    results, elapsed = _timed(
        f"transcribe {Path(audio_path).name}",
        model.generate,
        input=audio_path,
        **generate_kwargs,
        audio_dur_s=audio_dur_s,
    )

    text = ""
    if results and isinstance(results, list):
        text = results[0].get("text", "")

    return {
        "source": audio_path,
        "filename": os.path.basename(audio_path),
        "text": text,
        "audio_dur_s": round(audio_dur_s, 3),
        "transcribe_s": round(elapsed, 3),
        "rtf": round(elapsed / audio_dur_s, 4) if audio_dur_s > 0 else None,
    }


def save_result(result: dict, audio_path: Path, output_dir: Path, fmt: str, channel: int | None = None):
    """将结果写入文件。channel 不为 None 时在文件名中附加声道后缀。"""
    base = audio_path.stem
    if channel is not None:
        base = f"{base}_channel{channel}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        out_path = output_dir / f"{base}.funasr.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        out_path = output_dir / f"{base}.funasr.txt"
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

    logger.info("[config] model=%s  vad=%s  punc=%s", args.model, args.vad_model, args.punc_model)
    logger.info("[config] hub=%s  device=%s  batch_size=%d", args.hub, args.device, args.batch_size)
    logger.info("[config] separate_channel=%s", args.separate_channel)
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
                    logger.info("[channel %d] %s", ch, result["text"])

                    save_result(result, f, output_dir, args.output_format, channel=ch)

                    total_audio_s += result["audio_dur_s"]
                    total_transcribe_s += result["transcribe_s"]
        else:
            result = transcribe_file(model, str(f), args)
            logger.info("[result] %s", result["text"])
            save_result(result, f, output_dir, args.output_format)

            total_audio_s += result["audio_dur_s"]
            total_transcribe_s += result["transcribe_s"]

    # 汇总统计
    logger.info("\n[summary] 处理文件数: %d", len(files))
    logger.info("[summary] 总音频时长: %.1fs", total_audio_s)
    logger.info("[summary] 总转写耗时: %.1fs", total_transcribe_s)
    if total_audio_s > 0:
        overall_rtf = total_transcribe_s / total_audio_s
        logger.info("[summary] 整体 RTF: %.4f", overall_rtf)


if __name__ == "__main__":
    main()
