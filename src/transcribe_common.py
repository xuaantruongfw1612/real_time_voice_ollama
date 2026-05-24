"""
transcribe_common.py — Dùng chung cho transcribe.py và transcribe_file.py.

Chứa:
  - Config (đọc từ .env)
  - Class màu C
  - Hallucination filter
  - Helper format thời gian
  - Ghi file log (định dạng chuẩn → polish.py đọc được)
"""

import os, sys, shutil
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# Paths
ROOT_DIR = Path(__file__).parent.parent   # VOICE_REAL_TIME/
LOG_DIR  = ROOT_DIR / "lecture"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# Config từ .env
WHISPER_MODEL    = os.getenv("WHISPER_MODEL",  "medium")
SAMPLE_RATE      = int(os.getenv("SAMPLE_RATE", "16000"))
LANGUAGE         = os.getenv("LANGUAGE", "vi")
BEAM_SIZE        = int(os.getenv("BEAM_SIZE", "5"))
CONTEXT_WIN      = int(os.getenv("CONTEXT_WINDOW", "4"))

SILENCE_SEC      = float(os.getenv("SILENCE_SEC",    "1.0"))
MIN_SPEECH_SEC   = float(os.getenv("MIN_SPEECH_SEC", "0.4"))
CALIBRATE_SEC    = float(os.getenv("CALIBRATE_SEC",  "2.0"))
RMS_SMOOTH       = int(os.getenv("RMS_SMOOTH_BLOCKS", "4"))
MAX_QUEUE        = int(os.getenv("MAX_QUEUE_SIZE", "6"))

NO_SPEECH_THRESH = float(os.getenv("NO_SPEECH_THRESH", "0.6"))
LOGPROB_THRESH   = float(os.getenv("LOGPROB_THRESH",  "-1.0"))

_te = os.getenv("SILENCE_THRESH", "").strip()
FORCE_THRESH = float(_te) if _te else None

SUPPORTED_EXT = {".m4a", ".mp3", ".wav", ".ogg", ".flac",
                 ".aac", ".mp4", ".mov", ".wma", ".webm"}

# Chuỗi Whisper hay hallucinate
HALLUC_STRINGS = {
    "ghiền mì gõ", "subscribe", "like và subscribe",
    "đăng ký kênh", "cảm ơn các bạn đã xem",
    "xin chào các bạn", "hẹn gặp lại",
    "thanks for watching", "please subscribe",
}


# Màu terminal
class C:
    R    = "\033[0m";  BOLD = "\033[1m"; DIM = "\033[2m"
    GR   = "\033[92m"; CY   = "\033[96m"; YL  = "\033[93m"
    RD   = "\033[91m"; GY   = "\033[90m"


# Kiểm tra dependencies
def check_base_imports():
    """Kiểm tra các package bắt buộc cho cả 2 script."""
    missing = []
    for pkg, imp in [
        ("openai-whisper", "whisper"),
        ("numpy",          "numpy"),
        ("python-dotenv",  "dotenv"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Thiếu package: pip install {' '.join(missing)}")
        sys.exit(1)


def check_mic_imports():
    """Kiểm tra thêm package cho transcribe.py (mic realtime)."""
    missing = []
    for pkg, imp in [
        ("sounddevice", "sounddevice"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Thiếu package: pip install {' '.join(missing)}")
        sys.exit(1)


def check_ffmpeg():
    """Kiểm tra ffmpeg cho transcribe_file.py."""
    if not shutil.which("ffmpeg"):
        print("Thiếu ffmpeg. Cài đặt:")
        print("  macOS : brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        print("  Win   : https://ffmpeg.org/download.html")
        sys.exit(1)


# Helpers thời gian
def fmt_duration(secs: float) -> str:
    """Giây → MM:SS hoặc HH:MM:SS (dùng để hiển thị)."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def secs_to_ts(secs: float) -> str:
    """Giây → HH:MM:SS (dùng làm timestamp trong log)."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# Hallucination filter
def is_hallucination(text: str, seg: dict) -> tuple[bool, str]:
    """
    Trả về (True, lý do) nếu đoạn bị coi là hallucination/noise.
    seg: dict với các key 'no_speech_prob', 'avg_logprob' (từ Whisper segments).
    """
    if not text or len(text.strip()) < 2:
        return True, "too short"

    tl = text.lower()
    for h in HALLUC_STRINGS:
        if h in tl:
            return True, f"hallucination string: {h!r}"

    ns = seg.get("no_speech_prob", 0)
    lp = seg.get("avg_logprob", 0)
    if ns > 0.95 and lp < -1.5:
        return True, f"noise rõ: no_speech={ns:.2f} logprob={lp:.2f}"

    return False, ""


# Ghi file log
def make_log_path(stem: str) -> Path:
    """Tạo đường dẫn log mặc định trong LOG_DIR."""
    return LOG_DIR / f"lecture_{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


def write_log_header(f, extra_lines: list[str], thresh_line: str):
    """
    Ghi phần header chuẩn vào file log đã mở.
    extra_lines: các dòng thêm vào giữa dòng đầu và dòng ngưỡng
                 (vd: ['Nguon    : /path/to/file.m4a'])
    """
    import torch
    dev = f"GPU:{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"

    f.write(f"BUOI HOC : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    for line in extra_lines:
        f.write(line + "\n")
    f.write(f"Model    : Whisper {WHISPER_MODEL}  |  Ngon ngu: {LANGUAGE}\n")
    f.write(f"Thiet bi : {dev}\n")
    f.write(thresh_line + "\n")
    f.write("=" * 60 + "\n\n")


def init_mic_log(log_path: str, thresh: float):
    """Khởi tạo file log cho transcribe.py (mic realtime)."""
    with open(log_path, "w", encoding="utf-8") as f:
        write_log_header(
            f,
            extra_lines=[],
            thresh_line=(
                f"Nguong   : VAD={thresh:.5f}"
                f"  no_speech<{NO_SPEECH_THRESH}"
                f"  logprob>{LOGPROB_THRESH}"
            ),
        )


def init_file_log(log_path: Path, audio_path: Path):
    """Khởi tạo file log cho transcribe_file.py."""
    with log_path.open("w", encoding="utf-8") as f:
        write_log_header(
            f,
            extra_lines=[f"Nguon    : {audio_path.resolve()}"],
            thresh_line=(
                f"Nguong   : no_speech<{NO_SPEECH_THRESH}"
                f"  logprob>{LOGPROB_THRESH}"
            ),
        )


def append_log(log_path: str | Path, ts: str, text: str):
    """Ghi thêm 1 dòng vào file log."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {text}\n")


def save_plain(segments: list[dict], log_path: Path) -> Path:
    """Lưu bản plain text (không timestamp) bên cạnh file log."""
    plain_path = log_path.with_suffix(".plain.txt")
    with plain_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(s["text"] for s in segments))
    return plain_path
