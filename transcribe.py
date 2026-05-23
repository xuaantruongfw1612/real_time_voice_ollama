import os, sys, queue, threading, collections
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# ── Kiem tra thu vien ─────────────────────────────────────────────────────────
def check_imports():
    missing = []
    for pkg, imp in [
        ("sounddevice", "sounddevice"),
        ("numpy", "numpy"),
        ("openai-whisper", "whisper"),
        ("python-dotenv", "dotenv"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Thieu: pip install {' '.join(missing)}")
        sys.exit(1)


check_imports()

import sounddevice as sd
import whisper
import torch


# ── Cau hinh ─────────────────────────────────────────────────────────────────
WHISPER_MODEL  = os.getenv("WHISPER_MODEL",    "small")
SAMPLE_RATE    = int(os.getenv("SAMPLE_RATE",      "16000"))
SILENCE_THRESH = float(os.getenv("SILENCE_THRESH",  "0.006"))
SILENCE_SEC    = float(os.getenv("SILENCE_SEC",     "1.5"))
MIN_SPEECH_SEC = float(os.getenv("MIN_SPEECH_SEC",  "0.5"))
LANGUAGE       = os.getenv("LANGUAGE", "vi")        # vi | en | auto
LOG_FILE       = os.getenv("LOG_FILE",
    f"lecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "8"))

CONTEXT_WINDOW = int(os.getenv("CONTEXT_WINDOW", "3"))

ENERGY_SMOOTH  = int(os.getenv("ENERGY_SMOOTH", "3"))

# Auto-detect fp16
USE_FP16 = torch.cuda.is_available()

SKIP = {
    "", ".", " ", "..", "...",
    "cảm ơn", "cảm ơn bạn", "cảm ơn bạn đã xem", "xin cảm ơn",
    "bye", "thank you", "thanks", "okay", "ok",
    "[blank_audio]", "(blank)", "(nhạc)", "(tiếng nhạc)",
}


class C:
    RESET  = "\033[0m";  BOLD   = "\033[1m"
    GREEN  = "\033[92m"; CYAN   = "\033[96m"
    GRAY   = "\033[90m"; YELLOW = "\033[93m"
    RED    = "\033[91m"


# ── Thread-safe printing ──────────────────────────────────────────────────────
_print_lock   = threading.Lock()
_amp_suppress = threading.Event()   # set = dang in transcript, khong in amp


def safe_print(line: str):
    """In transcript: xoa dong amp hien tai truoc, giu lock trong khi in."""
    with _print_lock:
        print(f"\r{' ' * 60}\r", end="", flush=True)
        print(line)


# ── Log file ──────────────────────────────────────────────────────────────────
def init_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"BUOI HOC : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model    : Whisper {WHISPER_MODEL}  |  Ngon ngu: {LANGUAGE}\n")
        f.write(f"GPU/fp16 : {USE_FP16}\n")
        f.write("=" * 55 + "\n\n")
    print(f"  Log -> {C.CYAN}{LOG_FILE}{C.RESET}")


def append_log(ts: str, text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {text}\n")


# ── STT Worker ────────────────────────────────────────────────────────────────
_audio_q: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=MAX_QUEUE_SIZE)
_recent_texts: "collections.deque[str]" = collections.deque(maxlen=CONTEXT_WINDOW)


def stt_worker(model: whisper.Whisper):
    while True:
        audio = _audio_q.get()
        if audio is None:
            break

        prompt = " ".join(_recent_texts) if _recent_texts else None

        result = model.transcribe(
            audio.astype(np.float32),
            language=None if LANGUAGE == "auto" else LANGUAGE,
            fp16=USE_FP16,
            condition_on_previous_text=False,
            initial_prompt=prompt,
        )
        text = result["text"].strip()

        if text.lower() not in SKIP and len(text) >= 2:
            _recent_texts.append(text)
            ts = datetime.now().strftime("%H:%M:%S")
            _amp_suppress.set()
            safe_print(f"{C.CYAN}[{ts}]{C.RESET} {text}")
            _amp_suppress.clear()
            append_log(ts, text)

        _audio_q.task_done()


# ── Audio Callback ────────────────────────────────────────────────────────────
def make_callback():
    """
    FIX: Toan bo trang thai (buf, counters) duoc bao ve bang 1 lock rieng
    de tranh race condition voi queue.put() va buf.clear().
    """
    _cb_lock  = threading.Lock()
    buf       = []
    silent    = [0]
    speech    = [0]
    _energy_history: "collections.deque[float]" = collections.deque(
        maxlen=ENERGY_SMOOTH, iterable=[0.0] * ENERGY_SMOOTH
    )

    SBLOCKS = max(1, int(SILENCE_SEC    / 0.05))
    MBLOCKS = max(1, int(MIN_SPEECH_SEC / 0.05))

    def cb(indata, frames, t, status):
        chunk = indata[:, 0].copy()
        raw_amp = float(np.abs(chunk).mean())

        _energy_history.append(raw_amp)
        amp = float(np.mean(_energy_history))

        is_speech = amp > SILENCE_THRESH

        segment_to_enqueue = None

        with _cb_lock:
            if is_speech:
                speech[0] += 1
                silent[0]  = 0
                buf.append(chunk)
            elif speech[0] > 0:
                silent[0] += 1
                buf.append(chunk)
                if silent[0] >= SBLOCKS:
                    if speech[0] >= MBLOCKS and buf:
                        segment_to_enqueue = np.concatenate(buf)
                    buf.clear()
                    silent[0] = 0
                    speech[0] = 0

        if segment_to_enqueue is not None:
            try:
                _audio_q.put_nowait(segment_to_enqueue)
            except queue.Full:
                if _print_lock.acquire(blocking=False):
                    try:
                        print(f"\r{C.RED}[!] Queue day ({MAX_QUEUE_SIZE}), bo qua segment{C.RESET}",
                              flush=True)
                    finally:
                        _print_lock.release()

        if not _amp_suppress.is_set():
            bars  = min(int(amp * 500), 35)
            color = C.GREEN if is_speech else C.GRAY
            q_sz  = _audio_q.qsize()
            wait  = f" {C.YELLOW}(xu ly: {q_sz}/{MAX_QUEUE_SIZE}){C.RESET}" if q_sz else ""
            if _print_lock.acquire(blocking=False):
                try:
                    print(
                        f"\r  {color}{'█' * bars:<35}{C.RESET} {amp:.4f}{wait}",
                        end="", flush=True,
                    )
                finally:
                    _print_lock.release()

    return cb


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device_info = "GPU (fp16)" if USE_FP16 else "CPU"
    print(f"""
{C.CYAN}{C.BOLD}  Lecture Transcriber  (100% Local)
  Whisper : {WHISPER_MODEL}  |  Ngon ngu: {LANGUAGE}
  Thiet bi: {device_info}
  Nhan Enter de dung va luu{C.RESET}
""")

    init_log()

    print(f"\n{C.YELLOW}Dang tai Whisper ({WHISPER_MODEL})...{C.RESET}", flush=True)
    model = whisper.load_model(WHISPER_MODEL)
    print(f"{C.GREEN}San sang. Bat dau nghe...{C.RESET}\n")

    worker = threading.Thread(target=stt_worker, args=(model,), daemon=True)
    worker.start()

    print(f"{C.YELLOW}  Nhan Enter de dung va luu...{C.RESET}\n")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * 0.05),
        callback=make_callback(),
    ):
        input()
    print(f"\n{C.YELLOW}Dang xu ly phan con lai...{C.RESET}")
    _audio_q.put(None)
    worker.join(timeout=30)

    print(f"{C.GREEN}Da luu: {LOG_FILE}{C.RESET}\n")


if __name__ == "__main__":
    main()
