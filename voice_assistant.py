"""
Vietnamese Real-time Voice Assistant - 100% Local
STT : Whisper (OpenAI, chay local)
AI  : Ollama (chay local, khong can internet)
Chay: python voice_assistant.py
"""

import os, sys, queue, threading, json, urllib.request
from dotenv import load_dotenv
load_dotenv()

def check_imports():
    missing = []
    for pkg, imp in [("sounddevice","sounddevice"),("numpy","numpy"),("openai-whisper","whisper"),("python-dotenv","dotenv")]:
        try: __import__(imp)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"\nThieu: {', '.join(missing)}")
        print(f"Chay: pip install {' '.join(missing)}\n")
        sys.exit(1)

check_imports()

import numpy as np
import sounddevice as sd
import whisper

# ── Doc .env ─────────────────────────────────────────────────────────────────
WHISPER_MODEL   = os.getenv("WHISPER_MODEL",   "small")
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "gemma3:4b")
SAMPLE_RATE     = int(os.getenv("SAMPLE_RATE",    "16000"))
SILENCE_THRESH  = float(os.getenv("SILENCE_THRESH", "0.008"))
SILENCE_BLOCKS  = int(os.getenv("SILENCE_BLOCKS",  "4"))
BLOCK_SEC       = float(os.getenv("BLOCK_SEC",      "0.5"))
SYSTEM_PROMPT   = os.getenv("SYSTEM_PROMPT",
    "Ban la tro ly AI thong minh noi tieng Viet. "
    "Tra loi ngan gon, tu nhien, than thien. Toi da 3-4 cau moi lan.")

class C:
    RESET="\033[0m"; BOLD="\033[1m"; GREEN="\033[92m"; YELLOW="\033[93m"
    CYAN="\033[96m"; GRAY="\033[90m"; WHITE="\033[97m"; RED="\033[91m"

def banner():
    print(f"""
{C.CYAN}{C.BOLD}  Vietnamese Voice Assistant  (100% Local)
  STT : Whisper {WHISPER_MODEL}
  AI  : Ollama  {OLLAMA_MODEL}
  Ctrl+C de thoat{C.RESET}
""")

# ── Kiem tra Ollama ───────────────────────────────────────────────────────────
def check_ollama():
    try:
        r = urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3)
        data = json.loads(r.read())
        models = [m["name"] for m in data.get("models", [])]
        base = OLLAMA_MODEL.split(":")[0]
        matched = [m for m in models if m.startswith(base)]
        if not matched:
            print(f"{C.YELLOW}Model '{OLLAMA_MODEL}' chua co. Dang tai...{C.RESET}")
            print(f"  (Co the mat vai phut lan dau)\n")
            # Pull model
            req = urllib.request.Request(
                f"{OLLAMA_URL}/api/pull",
                data=json.dumps({"name": OLLAMA_MODEL}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                while True:
                    line = resp.readline()
                    if not line: break
                    try:
                        obj = json.loads(line)
                        status = obj.get("status","")
                        if "pulling" in status or "downloading" in status:
                            pct = obj.get("completed",0)
                            tot = obj.get("total",1)
                            if tot:
                                p = int(pct/tot*30)
                                print(f"\r  [{'█'*p}{'░'*(30-p)}] {pct/tot:.0%}", end="", flush=True)
                        elif status == "success":
                            print(f"\r{C.GREEN}  Model da tai xong!{C.RESET}      ")
                    except: pass
        else:
            print(f"{C.GREEN}Ollama san sang  ({matched[0]}){C.RESET}")
    except Exception as e:
        print(f"\n{C.RED}Khong ket noi duoc Ollama.{C.RESET}")
        print(f"  Cai dat Ollama tai: https://ollama.com/download")
        print(f"  Sau do chay: ollama serve")
        print(f"  Loi: {e}\n")
        sys.exit(1)

# ── Goi Ollama (streaming) ────────────────────────────────────────────────────
def ask_ollama(messages, on_chunk):
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    full = ""
    with urllib.request.urlopen(req, timeout=60) as resp:
        while True:
            line = resp.readline()
            if not line: break
            try:
                obj = json.loads(line)
                token = obj.get("message", {}).get("content", "")
                if token:
                    on_chunk(token)
                    full += token
                if obj.get("done"): break
            except: pass
    return full

# ── Worker ────────────────────────────────────────────────────────────────────
result_queue  = queue.Queue()
display_queue = queue.Queue()

def transcribe_worker(model, history_ref):
    while True:
        item = result_queue.get()
        if item is None: break

        audio_f32 = item.astype(np.float32)
        result = model.transcribe(audio_f32, language="vi", fp16=False)
        text = result["text"].strip()
        if not text:
            result_queue.task_done()
            continue

        display_queue.put(("user", text))

        lower = text.lower()
        if any(w in lower for w in ["xoa lich su", "bat dau lai", "reset"]):
            history_ref.clear()
            display_queue.put(("system", "Da xoa lich su."))
            result_queue.task_done()
            continue

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += list(history_ref)
        messages.append({"role": "user", "content": text})

        display_queue.put(("ai_start", ""))
        full = ""
        try:
            def on_chunk(token):
                display_queue.put(("ai_chunk", token))
            full = ask_ollama(messages, on_chunk)
            display_queue.put(("ai_end", ""))
            history_ref.append({"role": "user",      "content": text})
            history_ref.append({"role": "assistant", "content": full})
            # Giu toi da 20 luot
            if len(history_ref) > 40:
                history_ref[:] = history_ref[-40:]
        except Exception as e:
            display_queue.put(("ai_end", ""))
            display_queue.put(("system", f"Loi Ollama: {e}"))

        result_queue.task_done()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    banner()

    print(f"{C.YELLOW}Dang tai Whisper ({WHISPER_MODEL})...{C.RESET}", flush=True)
    model = whisper.load_model(WHISPER_MODEL)
    print(f"{C.GREEN}Whisper san sang.{C.RESET}")

    print(f"{C.YELLOW}Dang kiem tra Ollama...{C.RESET}", flush=True)
    check_ollama()
    print()

    history = []
    worker = threading.Thread(target=transcribe_worker, args=(model, history), daemon=True)
    worker.start()

    block_size   = int(SAMPLE_RATE * BLOCK_SEC)
    audio_buf    = []
    silent_count = 0
    speech_count = 0

    print(f"{C.CYAN}Dang nghe — noi bat cu luc nao...{C.RESET}\n")

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buf, silent_count, speech_count
        chunk = indata[:, 0].copy()
        amp   = float(np.abs(chunk).mean())

        if amp > SILENCE_THRESH:
            speech_count += 1
            silent_count  = 0
            audio_buf.append(chunk)
        else:
            if speech_count > 0:
                silent_count += 1
                audio_buf.append(chunk)
                if silent_count >= SILENCE_BLOCKS:
                    if speech_count >= 1:
                        result_queue.put(np.concatenate(audio_buf))
                    audio_buf    = []
                    silent_count = 0
                    speech_count = 0

        bars  = int(amp * 400)
        bar   = ("█" * min(bars, 30)).ljust(30)
        color = C.GREEN if amp > SILENCE_THRESH else C.GRAY
        print(f"\r  {color}{bar}{C.RESET} amp={amp:.4f} thresh={SILENCE_THRESH}", end="", flush=True)

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=block_size, callback=audio_callback):
            while True:
                while not display_queue.empty():
                    kind, data = display_queue.get()
                    if kind == "user":
                        print(f"\r{' '*36}\r")
                        print(f"{C.WHITE}{C.BOLD}Ban:{C.RESET} {data}")
                    elif kind == "ai_start":
                        print(f"\r{' '*36}\r")
                        print(f"{C.CYAN}AI:{C.RESET} ", end="", flush=True)
                    elif kind == "ai_chunk":
                        print(data, end="", flush=True)
                    elif kind == "ai_end":
                        print()
                    elif kind == "system":
                        print(f"\r{C.YELLOW}{data}{C.RESET}")
                threading.Event().wait(0.05)
    except KeyboardInterrupt:
        result_queue.put(None)
        print(f"\n\n{C.CYAN}Tam biet!{C.RESET}\n")

if __name__ == "__main__":
    main()
