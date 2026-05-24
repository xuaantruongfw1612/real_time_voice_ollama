"""
transcribe_file.py — Phiên âm file ghi âm (iPhone, Android, máy tính...) bằng faster-whisper.
Nhanh hơn 2-4x, ít RAM hơn so với Whisper gốc. Stream từng segment ngay khi xong.
Output cùng định dạng log với transcribe.py → dùng được ngay với polish.py.

Hỗ trợ: m4a, mp3, wav, ogg, flac, aac, mp4, mov, wma, webm
Yêu cầu:
    pip install faster-whisper
    ffmpeg cài trên hệ thống (brew/apt install ffmpeg)

Cách dùng:
    python transcribe_file.py recording.m4a
    python transcribe_file.py recording.m4a --model large-v3
    python transcribe_file.py recording.m4a --lang vi --out my_output.txt
    python transcribe_file.py recording.m4a --no-timestamps
    python transcribe_file.py recording.m4a --debug
"""

import sys, argparse, time
from pathlib import Path

from transcribe_common import (
    WHISPER_MODEL, LANGUAGE, BEAM_SIZE,
    NO_SPEECH_THRESH, LOGPROB_THRESH,
    SUPPORTED_EXT, ROOT_DIR, LOG_DIR,
    C, fmt_duration, secs_to_ts,
    make_log_path, init_file_log, append_log, save_plain,
    check_base_imports, check_ffmpeg,
)

check_base_imports()
check_ffmpeg()

# Fix lỗi symlink trên Windows (không cần quyền admin)
import os as _os
_os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
_os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY", "error")
if _os.name == "nt":  # Windows
    _os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    _os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Kiểm tra faster-whisper
try:
    from faster_whisper import WhisperModel
except ImportError:
    print(f"{C.RD}Thiếu faster-whisper. Cài đặt:{C.R}")
    print("  pip install faster-whisper")
    sys.exit(1)

import torch
USE_GPU = torch.cuda.is_available()


# ── Hallucination filter (faster-whisper dùng segment object, không phải dict) ──
HALLUC_STRINGS = {
    "ghiền mì gõ", "subscribe", "like và subscribe",
    "đăng ký kênh", "cảm ơn các bạn đã xem",
    "xin chào các bạn", "hẹn gặp lại",
    "thanks for watching", "please subscribe",
}

def is_hallucination(text: str, seg) -> tuple[bool, str]:
    if not text or len(text.strip()) < 2:
        return True, "too short"
    tl = text.lower()
    for h in HALLUC_STRINGS:
        if h in tl:
            return True, f"hallucination string: {h!r}"
    # faster-whisper: seg.no_speech_prob, seg.avg_logprob
    ns = getattr(seg, "no_speech_prob", 0) or 0
    lp = getattr(seg, "avg_logprob",   0) or 0
    if ns > 0.95 and lp < -1.5:
        return True, f"noise rõ: no_speech={ns:.2f} logprob={lp:.2f}"
    return False, ""


# Core transcribe
def transcribe_audio(
    audio_path: Path,
    model_name: str,
    language: str,
    log_path: Path,
    verbose_debug: bool = False,
) -> list[dict]:
    # Chọn device + compute type
    if USE_GPU:
        device       = "cuda"
        compute_type = "float16"
    else:
        device       = "cpu"
        compute_type = "float32"  # int8 giảm độ chính xác với tiếng Việt

    # Tải model vào thư mục local — tránh lỗi symlink trên Windows
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(exist_ok=True)
    local_path = models_dir / model_name.replace("/", "-")

    if not local_path.exists():
        print(f"\n{C.YL}Đang tải model \'{model_name}\' lần đầu về {local_path}...{C.R}", flush=True)
        print(f"  {C.GY}(lần sau dùng cache local, không tải lại){C.R}", flush=True)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=f"Systran/faster-whisper-{model_name}",
            local_dir=str(local_path),
            local_dir_use_symlinks=False,
        )

    print(f"\n{C.YL}Đang load \'{model_name}\' ({device}/{compute_type})...{C.R}", flush=True)
    t0    = time.time()
    model = WhisperModel(str(local_path), device=device, compute_type=compute_type)
    print(f"  {C.GR}Tải xong ({time.time()-t0:.1f}s). Bắt đầu phiên âm...{C.R}\n", flush=True)

    lang_arg = None if language == "auto" else language

    # faster-whisper trả về generator thật — yield segment ngay khi xong
    segments_gen, info = model.transcribe(
        str(audio_path),
        language=lang_arg,
        beam_size=BEAM_SIZE,
        condition_on_previous_text=True,
        initial_prompt="Đây là bài giảng tiếng Việt.",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
        ),
    )

    print(
        f"  Ngôn ngữ nhận ra: {C.BOLD}{info.language}{C.R}"
        f"  (xác suất {info.language_probability:.0%})"
        f"  |  Thời lượng: {C.BOLD}{fmt_duration(info.duration)}{C.R}\n",
        flush=True,
    )

    output = []
    for seg in segments_gen:          # ← generator thật, yield từng segment ngay
        text     = seg.text.strip()
        ts       = secs_to_ts(seg.start)
        is_hall, reason = is_hallucination(text, seg)

        if verbose_debug:
            ns = getattr(seg, "no_speech_prob", 0) or 0
            lp = getattr(seg, "avg_logprob",   0) or 0
            print(
                f"  {C.YL}[DBG]{C.R} {fmt_duration(seg.start)}"
                f"  no_speech={C.YL}{ns:.2f}{C.R}"
                f"  logprob={C.CY}{lp:.2f}{C.R}"
                f"  {'[SKIP] ' + reason if is_hall else ''}"
                f"  {repr(text[:60])}",
                flush=True,
            )

        if is_hall:
            if not verbose_debug:
                print(f"  {C.GY}[SKIP] {ts}  {reason}{C.R}", flush=True)
            continue

        # In terminal + ghi file ngay lập tức
        print(f"  {C.CY}[{ts}]{C.R} {text}", flush=True)
        append_log(log_path, ts, text)

        output.append({
            "ts":    ts,
            "text":  text,
            "start": seg.start,
            "end":   seg.end,
        })

    return output


# CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Phiên âm file ghi âm bằng faster-whisper (iPhone, Android, v.v.)"
    )
    parser.add_argument("audio", help="Đường dẫn file âm thanh")
    parser.add_argument("--model", "-m", default=WHISPER_MODEL,
        help=f"Model (tiny/base/small/medium/large-v3). Mặc định: {WHISPER_MODEL}")
    parser.add_argument("--lang", "-l", default=LANGUAGE,
        help=f"Ngôn ngữ (vi/en/auto). Mặc định: {LANGUAGE}")
    parser.add_argument("--out", "-o", default=None,
        help="Đường dẫn file output (mặc định: lecture/<tên>_<datetime>.txt)")
    parser.add_argument("--plain", action="store_true",
        help="Lưu thêm bản văn thuần (không timestamp) .plain.txt")
    parser.add_argument("--no-timestamps", action="store_true", dest="no_ts",
        help="Gộp tất cả thành 1 đoạn văn, không chia theo thời gian")
    parser.add_argument("--debug", action="store_true",
        help="Hiện thông số no_speech_prob / avg_logprob từng đoạn")
    return parser.parse_args()


# Main
def main():
    args       = parse_args()
    audio_path = Path(args.audio)

    if not audio_path.exists():
        print(f"{C.RD}Không tìm thấy file: {audio_path}{C.R}")
        sys.exit(1)

    if audio_path.suffix.lower() not in SUPPORTED_EXT:
        print(f"{C.YL}Cảnh báo: định dạng '{audio_path.suffix}' chưa được kiểm thử.")
        print(f"Hỗ trợ chính thức: {', '.join(sorted(SUPPORTED_EXT))}{C.R}")

    log_path = Path(args.out) if args.out else make_log_path(audio_path.stem)
    init_file_log(log_path, audio_path)

    print(f"\n{C.CY}{C.BOLD}  Transcribe File  {C.GY}(faster-whisper){C.R}")
    print(f"  File    : {C.CY}{audio_path}{C.R}")
    print(f"  Model   : {C.YL}{args.model}{C.R}")
    print(f"  Ngôn ngữ: {C.YL}{args.lang}{C.R}")
    print(f"  Device  : {C.GR}{'GPU (float16)' if USE_GPU else 'CPU (float32)'}{C.R}")
    print(f"  Log     : {C.CY}{log_path}{C.R}")

    t_start  = time.time()
    segments = transcribe_audio(audio_path, args.model, args.lang, log_path, args.debug)
    elapsed  = time.time() - t_start

    if not segments:
        print(f"\n{C.YL}Không nhận ra đoạn nào trong file.{C.R}")
        sys.exit(0)

    if args.no_ts:
        combined = " ".join(s["text"] for s in segments)
        init_file_log(log_path, audio_path)
        append_log(log_path, "00:00:00", combined)
        segments = [{"ts": "00:00:00", "text": combined, "start": 0, "end": 0}]

    print(f"\n{C.GR}  Hoàn tất!{C.R}")
    print(f"  Thời gian  : {C.BOLD}{elapsed:.1f}s{C.R}")
    print(f"  Đoạn nhận  : {C.BOLD}{len(segments)}{C.R}")
    print(f"  Log lưu tại: {C.CY}{log_path}{C.R}")

    if args.plain:
        plain_path = save_plain(segments, log_path)
        print(f"  Plain text : {C.CY}{plain_path}{C.R}")

    print(f"\n  {C.GY}Để sửa lỗi chính tả, chạy tiếp:{C.R}")
    print(f"  {C.BOLD}python polish.py {log_path}{C.R}\n")


if __name__ == "__main__":
    main()
