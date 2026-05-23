import os, sys, queue, threading, collections, time
import numpy as np
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_imports():
    missing = []
    for pkg, imp in [
        ("sounddevice",    "sounddevice"),
        ("numpy",          "numpy"),
        ("openai-whisper", "whisper"),
        ("python-dotenv",  "dotenv"),
    ]:
        try: __import__(imp)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"Thiếu: pip install {' '.join(missing)}")
        sys.exit(1)

check_imports()

import sounddevice as sd
import whisper
import torch


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent   # VOICE_REAL_TIME/
LOG_DIR  = ROOT_DIR / "lecture"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
WHISPER_MODEL  = os.getenv("WHISPER_MODEL",  "medium")
SAMPLE_RATE    = int(os.getenv("SAMPLE_RATE", "16000"))
LANGUAGE       = os.getenv("LANGUAGE", "vi")
LOG_FILE       = str(LOG_DIR / os.getenv("LOG_FILE",
    f"lecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"))
MAX_QUEUE      = int(os.getenv("MAX_QUEUE_SIZE", "6"))
CONTEXT_WIN    = int(os.getenv("CONTEXT_WINDOW", "4"))
BEAM_SIZE      = int(os.getenv("BEAM_SIZE", "5"))
SILENCE_SEC    = float(os.getenv("SILENCE_SEC",    "1.0"))
MIN_SPEECH_SEC = float(os.getenv("MIN_SPEECH_SEC", "0.4"))
CALIBRATE_SEC  = float(os.getenv("CALIBRATE_SEC",  "2.0"))
RMS_SMOOTH     = int(os.getenv("RMS_SMOOTH_BLOCKS", "4"))

# Hallucination filter — 2 nguong cua Whisper:
# no_speech_prob > 0.6  : Whisper nghi doan nay la im lang / nhieu
# avg_logprob   < -1.0  : Whisper khong tu tin vao ket qua
NO_SPEECH_THRESH = float(os.getenv("NO_SPEECH_THRESH", "0.6"))
LOGPROB_THRESH   = float(os.getenv("LOGPROB_THRESH",  "-1.0"))

_te = os.getenv("SILENCE_THRESH", "").strip()
FORCE_THRESH = float(_te) if _te else None

USE_GPU  = torch.cuda.is_available()
USE_FP16 = USE_GPU

# Chuoi Whisper hay hallucinate — filter them tang thu 2
HALLUC_STRINGS = {
    "ghiền mì gõ", "subscribe", "like và subscribe",
    "đăng ký kênh", "cảm ơn các bạn đã xem",
    "xin chào các bạn", "hẹn gặp lại",
}


class C:
    R="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
    GR="\033[92m"; CY="\033[96m"; YL="\033[93m"
    RD="\033[91m"; GY="\033[90m"


_lock     = threading.Lock()
_suppress = threading.Event()
_stop     = threading.Event()
_audio_q  = queue.Queue(maxsize=MAX_QUEUE)
_recent   = collections.deque(maxlen=CONTEXT_WIN)


def _clear():
    sys.stdout.write(f"\r{' ' * 85}\r")
    sys.stdout.flush()


def sprint(line: str):
    with _lock:
        _clear()
        print(line, flush=True)


def init_log(thresh: float):
    dev = f"GPU:{torch.cuda.get_device_name(0)}" if USE_GPU else "CPU"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"BUOI HOC : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model    : Whisper {WHISPER_MODEL}  |  Ngon ngu: {LANGUAGE}\n")
        f.write(f"Thiet bi : {dev}\n")
        f.write(f"Nguong   : VAD={thresh:.5f}  no_speech<{NO_SPEECH_THRESH}  logprob>{LOGPROB_THRESH}\n")
        f.write("=" * 60 + "\n\n")
    sprint(f"  Log -> {C.CY}{LOG_FILE}{C.R}")


def log(ts: str, text: str):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {text}\n")


def calibrate() -> float:
    if FORCE_THRESH is not None:
        sprint(f"  VAD ngưỡng (tu .env): {C.YL}{FORCE_THRESH:.5f}{C.R}")
        return FORCE_THRESH
    sprint(f"{C.YL}[Cal] Giữ im lặng {CALIBRATE_SEC:.0f}s...{C.R}")
    buf = sd.rec(int(SAMPLE_RATE * CALIBRATE_SEC),
                 samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    rms    = float(np.sqrt(np.mean(buf ** 2)))
    thresh = max(0.002, round(rms * 3.5, 5))
    sprint(f"  noise rms={rms:.5f}  ->  VAD ngưỡng={C.GR}{thresh:.5f}{C.R}")
    return thresh


# ── STT worker ────────────────────────────────────────────────────────────────
def stt_worker(model):
    sprint(f"{C.GR}[Worker] sẵn sàng{C.R}")
    while True:
        try:
            audio = _audio_q.get(timeout=1)
        except queue.Empty:
            if _stop.is_set():
                break
            continue

        if audio is None:
            _audio_q.task_done()
            break

        try:
            dur = len(audio) / SAMPLE_RATE

            prompt = " ".join(_recent) if _recent else None

            res = model.transcribe(
                audio.astype(np.float32),
                language=None if LANGUAGE == "auto" else LANGUAGE,
                fp16=USE_FP16,
                condition_on_previous_text=False,
                initial_prompt=prompt,
                beam_size=BEAM_SIZE,
                best_of=BEAM_SIZE,
                temperature=[0.0, 0.2, 0.4],
            )

            text = res["text"].strip()

            # ── Lay confidence metrics tu segments ───────────────────────────
            segs = res.get("segments", [])
            if segs:
                avg_no_speech = np.mean([s.get("no_speech_prob", 0) for s in segs])
                avg_logprob   = np.mean([s.get("avg_logprob", 0)    for s in segs])
            else:
                avg_no_speech = 1.0
                avg_logprob   = -2.0

            # DEBUG: hien thi day du de chon nguong
            sprint(
                f"{C.YL}[DBG]{C.R} "
                f"dur={dur:.1f}s  "
                f"no_speech={C.YL}{avg_no_speech:.2f}{C.R}  "
                f"logprob={C.CY}{avg_logprob:.2f}{C.R}  "
                f"text={repr(text[:60])}"
            )

            # ── Filter ───────────────────────────────────────────────────────
            # Không dùng no_speech/logprob làm ngưỡng cứng vì không đáng tin
            # với tiếng Việt / giọng nhỏ / CPU — chỉ skip khi chắc chắn noise
            reason = None
            if not text or len(text) < 2:
                reason = "too short"
            elif any(h in text.lower() for h in HALLUC_STRINGS):
                reason = "hallucination string"
            elif avg_no_speech > 0.95 and avg_logprob < -1.5:
                reason = f"noise ro: no_speech={avg_no_speech:.2f} logprob={avg_logprob:.2f}"

            if reason:
                sprint(f"{C.GY}[SKIP] {reason}{C.R}")
            else:
                _recent.append(text)
                ts = datetime.now().strftime("%H:%M:%S")

                # In tung ky tu
                _suppress.set()
                with _lock:
                    _clear()
                    sys.stdout.write(f"{C.CY}[{ts}]{C.R} ")
                    for ch in text:
                        sys.stdout.write(ch)
                        sys.stdout.flush()
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                _suppress.clear()

                log(ts, text)

        except Exception as e:
            sprint(f"{C.RD}[ERR] {e}{C.R}")
        finally:
            _audio_q.task_done()


# ── Audio callback ────────────────────────────────────────────────────────────
def make_callback(thresh: float):
    buf    = []
    silent = [0]
    speech = [0]
    peak   = [0.0]
    hist   = collections.deque([0.0] * RMS_SMOOTH, maxlen=RMS_SMOOTH)

    SBLK = max(1, int(SILENCE_SEC    / 0.05))
    MBLK = max(1, int(MIN_SPEECH_SEC / 0.05))

    def cb(indata, frames, t, status):
        chunk = indata[:, 0].copy()
        hist.append(float(np.sqrt(np.mean(chunk ** 2))))
        rms = float(np.mean(hist))

        peak[0] = rms if rms > peak[0] else peak[0] * 0.997
        is_sp   = rms > thresh
        seg     = None

        if is_sp:
            speech[0] += 1; silent[0] = 0; buf.append(chunk)
        elif speech[0] > 0:
            silent[0] += 1; buf.append(chunk)
            if silent[0] >= SBLK:
                if speech[0] >= MBLK and buf:
                    seg = np.concatenate(buf)
                buf.clear(); silent[0] = 0; speech[0] = 0

        if seg is not None:
            try:
                _audio_q.put_nowait(seg)
                if _lock.acquire(blocking=False):
                    try:
                        _clear()
                        print(f"  {C.GR}[SEG] {len(seg)/SAMPLE_RATE:.1f}s -> queue{C.R}",
                              flush=True)
                    finally:
                        _lock.release()
            except queue.Full:
                sprint(f"{C.RD}[!] queue day{C.R}")

        if not _suppress.is_set() and _lock.acquire(blocking=False):
            try:
                bars = min(int(rms * 1000), 30)
                col  = C.GR if is_sp else C.GY
                sys.stdout.write(
                    f"\r  {col}{'█'*bars:<30}{C.R}"
                    f" rms={C.BOLD}{rms:.5f}{C.R}"
                    f" peak={C.YL}{peak[0]:.5f}{C.R}"
                    f" thr={C.CY}{thresh:.5f}{C.R}"
                    f" sp={speech[0]:03d} si={silent[0]:02d}"
                    + (f" {C.YL}q={_audio_q.qsize()}{C.R}" if _audio_q.qsize() else "")
                )
                sys.stdout.flush()
            finally:
                _lock.release()

    return cb


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    dev = f"GPU:{torch.cuda.get_device_name(0)}" if USE_GPU else "CPU"
    print(f"\n{C.CY}{C.BOLD}  Lecture Transcriber"
          f"\n  Model:{WHISPER_MODEL}  Lang:{LANGUAGE}  Device:{dev}"
          f"\n  Nhấn ENTER để dừng{C.R}\n")

    thresh = calibrate()
    init_log(thresh)

    print(f"{C.YL}Đang tải Whisper {WHISPER_MODEL}...{C.R}", flush=True)
    model = whisper.load_model(WHISPER_MODEL, device="cuda" if USE_GPU else "cpu")
    print(f"{C.GR}OK — Nói gì đó, xem [DBG] no_speech va logprob.{C.R}")
    print(f"{C.DIM}  Ghi khi: no_speech < {NO_SPEECH_THRESH}  va  logprob > {LOGPROB_THRESH}{C.R}\n")

    worker = threading.Thread(target=stt_worker, args=(model,), daemon=False)
    worker.start()

    threading.Thread(target=lambda: (input(), _stop.set()), daemon=True).start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=int(SAMPLE_RATE * 0.05),
                        callback=make_callback(thresh)):
        while not _stop.is_set():
            time.sleep(0.05)

    print(f"\n{C.YL}Đừng, chờ xử lý xong...{C.R}")
    _audio_q.put(None)
    worker.join(timeout=120)
    print(f"{C.GR}Đã lưu: {LOG_FILE}{C.R}\n")


if __name__ == "__main__":
    main()
