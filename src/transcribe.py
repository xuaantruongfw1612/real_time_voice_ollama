"""
transcribe.py — Phiên âm realtime từ microphone bằng Whisper.
Nhấn ENTER để dừng. Output log dùng được với polish.py.

Cách dùng:
    python transcribe.py
"""

import os, sys, queue, threading, collections, time
import numpy as np
from datetime import datetime
from pathlib import Path

from transcribe_common import (
    # config
    WHISPER_MODEL, SAMPLE_RATE, LANGUAGE, BEAM_SIZE, CONTEXT_WIN,
    SILENCE_SEC, MIN_SPEECH_SEC, CALIBRATE_SEC, RMS_SMOOTH, MAX_QUEUE,
    NO_SPEECH_THRESH, LOGPROB_THRESH, FORCE_THRESH,
    HALLUC_STRINGS,
    # helpers
    C, is_hallucination,
    # log
    LOG_DIR, make_log_path, init_mic_log, append_log,
    # checks
    check_base_imports, check_mic_imports,
)

check_base_imports()
check_mic_imports()

import sounddevice as sd
import whisper
import torch

USE_GPU  = torch.cuda.is_available()
USE_FP16 = USE_GPU

LOG_FILE = str(LOG_DIR / os.getenv("LOG_FILE",
    f"lecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"))

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


def calibrate() -> float:
    if FORCE_THRESH is not None:
        sprint(f"  VAD ngưỡng (từ .env): {C.YL}{FORCE_THRESH:.5f}{C.R}")
        return FORCE_THRESH
    sprint(f"{C.YL}[Cal] Giữ im lặng {CALIBRATE_SEC:.0f}s...{C.R}")
    buf = sd.rec(int(SAMPLE_RATE * CALIBRATE_SEC),
                 samplerate=SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    rms    = float(np.sqrt(np.mean(buf ** 2)))
    thresh = max(0.002, round(rms * 3.5, 5))
    sprint(f"  noise rms={rms:.5f}  ->  VAD ngưỡng={C.GR}{thresh:.5f}{C.R}")
    return thresh


# STT worker
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
            dur    = len(audio) / SAMPLE_RATE
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
            segs = res.get("segments", [])

            if segs:
                avg_no_speech = np.mean([s.get("no_speech_prob", 0) for s in segs])
                avg_logprob   = np.mean([s.get("avg_logprob", 0)    for s in segs])
            else:
                avg_no_speech, avg_logprob = 1.0, -2.0

            sprint(
                f"{C.YL}[DBG]{C.R} "
                f"dur={dur:.1f}s  "
                f"no_speech={C.YL}{avg_no_speech:.2f}{C.R}  "
                f"logprob={C.CY}{avg_logprob:.2f}{C.R}  "
                f"text={repr(text[:60])}"
            )

            # Dùng is_hallucination từ common với pseudo-seg dict
            pseudo_seg = {"no_speech_prob": avg_no_speech, "avg_logprob": avg_logprob}
            is_hall, reason = is_hallucination(text, pseudo_seg)

            if is_hall:
                sprint(f"{C.GY}[SKIP] {reason}{C.R}")
            else:
                _recent.append(text)
                ts = datetime.now().strftime("%H:%M:%S")

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

                append_log(LOG_FILE, ts, text)

        except Exception as e:
            sprint(f"{C.RD}[ERR] {e}{C.R}")
        finally:
            _audio_q.task_done()


# Audio callback
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
                sprint(f"{C.RD}[!] queue đầy{C.R}")

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


# Main
def main():
    dev = f"GPU:{torch.cuda.get_device_name(0)}" if USE_GPU else "CPU"
    print(f"\n{C.CY}{C.BOLD}  Lecture Transcriber"
          f"\n  Model:{WHISPER_MODEL}  Lang:{LANGUAGE}  Device:{dev}"
          f"\n  Nhấn ENTER để dừng{C.R}\n")

    thresh = calibrate()
    init_mic_log(LOG_FILE, thresh)

    print(f"{C.YL}Đang tải Whisper {WHISPER_MODEL}...{C.R}", flush=True)
    model = whisper.load_model(WHISPER_MODEL, device="cuda" if USE_GPU else "cpu")
    print(f"{C.GR}OK — Nói gì đó, xem [DBG] no_speech và logprob.{C.R}")
    print(f"{C.DIM}  Ghi khi: no_speech < {NO_SPEECH_THRESH}  và  logprob > {LOGPROB_THRESH}{C.R}\n")

    worker = threading.Thread(target=stt_worker, args=(model,), daemon=False)
    worker.start()

    threading.Thread(target=lambda: (input(), _stop.set()), daemon=True).start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=int(SAMPLE_RATE * 0.05),
                        callback=make_callback(thresh)):
        while not _stop.is_set():
            time.sleep(0.05)

    print(f"\n{C.YL}Dừng, chờ xử lý xong...{C.R}")
    _audio_q.put(None)
    worker.join(timeout=120)
    print(f"{C.GR}Đã lưu: {LOG_FILE}{C.R}\n")


if __name__ == "__main__":
    main()
