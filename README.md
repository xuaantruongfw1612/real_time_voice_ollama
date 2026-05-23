# 🎙️ Lecture Transcriber

> Real-time Vietnamese speech-to-text for lectures, with an offline LLM polish pass.

<p align="right">
  🌐 <a href="i18n/README.vi.md">Đọc bằng tiếng Việt</a>
</p>

---

## Overview

Two-stage pipeline that captures live audio, transcribes it with [OpenAI Whisper](https://github.com/openai/whisper), then cleans up the raw transcript with a local [Ollama](https://ollama.com) model — all offline, no cloud required.

```
Microphone → transcribe.py (Whisper) → lecture_YYYYMMDD.txt
                                              ↓
                                        polish.py (Ollama)
                                              ↓
                                  lecture_YYYYMMDD_polished.txt
```

---

## Features

- **Live transcription** — segments speech on silence, feeds chunks to Whisper
- **Hallucination filter** — drops garbage output using `no_speech_prob` and `avg_logprob`
- **VAD auto-calibration** — measures ambient noise at startup, sets threshold automatically
- **Accumulative dictionary** — `dictionary.txt` grows with every polish run; corrections are applied before Ollama even sees the text
- **GPU-aware** — uses CUDA if available, falls back to CPU silently
- **NixOS support** — includes `shell.nix` for a reproducible dev environment

---

## Requirements

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| FFmpeg | any recent |
| Ollama | any recent |
| CUDA *(optional)* | 11.8+ |

Python packages (install via `pip install -r requirements.txt`):

```
sounddevice>=0.4.6
numpy>=1.24.0
openai-whisper>=20231117
python-dotenv>=1.0.0
requests  # for polish.py
```

---

## Quick Start

### 1 — Clone & install

```bash
git clone <your-repo>
cd <your-repo>

pip install -r requirements.txt
```

On NixOS, use the provided shell instead:

```bash
nix-shell shell.nix
pip install openai-whisper requests
```

### 2 — Configure

Copy the example env file and edit as needed:

```bash
cp _env .env
```

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `medium` | Whisper model size (`tiny` / `base` / `small` / `medium` / `large`) |
| `LANGUAGE` | `vi` | Language code, or `auto` for detection |
| `SILENCE_THRESH` | *(auto)* | Leave blank to auto-calibrate, or set a fixed RMS value |
| `SILENCE_SEC` | `1.0` | Seconds of silence before a segment is cut |
| `MIN_SPEECH_SEC` | `0.4` | Minimum speech duration to keep a segment |
| `OLLAMA_MODEL` | `gemma3:4b` | Ollama model used for polishing |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |

### 3 — Transcribe

```bash
python transcribe.py
```

- Hold still for the 2-second noise calibration at startup.
- Speak normally. Each segment is printed live and appended to `lecture_<timestamp>.txt`.
- Press **Enter** to stop.

### 4 — Polish

After stopping the transcriber (to free RAM for Ollama):

```bash
# Auto-picks the newest lecture_*.txt
python polish.py

# Or specify a file
python polish.py lecture_20250523_143000.txt
```

Ollama will be started automatically if it isn't already running. The polished output is saved next to the original as `lecture_<timestamp>_polished.txt`.

---

## Output Format

### Raw log (`lecture_*.txt`)

```
BUOI HOC : 2025-05-23 14:30:00
Model    : Whisper medium  |  Ngon ngu: vi
...
============================

[14:30:12] Hôm nay chúng ta sẽ học về...
[14:30:45] ...cấu trúc dữ liệu cây nhị phân...
```

### Polished log (`lecture_*_polished.txt`)

```
TIMESTAMP   RAW (Whisper)                             POLISHED (Ollama)
------------------------------------------------------------------------
[14:30:12]  Hôm nay chúng ta sẽ học về...          ✓ Hôm nay chúng ta học về...
[14:30:45]  cấu trúc dữ liệu cây nhị phân...         (unchanged)
```

### Dictionary (`dictionary.txt`)

Auto-created and updated. You can also edit it manually at any time:

```
# Format: wrong -> correct  (one pair per line)
huyền thại -> huyền thoại
whisper -> Whisper
```

---

## File Structure

```
.
├── transcribe.py        # Live transcription (Whisper)
├── polish.py            # Post-processing (Ollama)
├── requirements.txt     # Python dependencies
├── shell.nix            # NixOS dev shell
├── _env                 # Example environment file (rename to .env)
└── dictionary.txt       # Auto-generated correction dictionary
```

---

## Tips

- **Low-resource machine?** Use `WHISPER_MODEL=small` or `tiny`. Quality drops slightly but it's much faster.
- **Lots of domain-specific terms?** Pre-populate `dictionary.txt` with the correct forms before running `polish.py`.
- **Audio too noisy?** Set `SILENCE_THRESH` manually in `.env` (e.g. `0.040`) instead of relying on auto-calibration.
- **Ollama model choice:** `gemma3:4b` balances quality and speed well. For a faster machine, try `llama3.2:3b`; for higher accuracy, try `mistral` or `qwen2.5:7b`.

---

## License

MIT
