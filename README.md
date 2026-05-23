# 🎙 Vietnamese Real-time Voice Assistant

Trợ lý giọng nói tiếng Việt chạy trên terminal.
- **STT**: Whisper (OpenAI) — nhận diện tiếng Việt
- **AI**: Claude (Anthropic) — hội thoại thông minh
- **Tuỳ chọn TTS**: pyttsx3

---

## ⚡ Cài đặt nhanh

### 1. Tạo môi trường ảo (khuyến nghị)
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 2. Cài thư viện
```bash
pip install -r requirements.txt
```

> **macOS**: nếu lỗi sounddevice, cài thêm: `brew install portaudio`  
> **Ubuntu/Debian**: `sudo apt install portaudio19-dev python3-dev`  
> **Windows**: cài [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### 3. Đặt API key
```bash
export ANTHROPIC_API_KEY=sk-ant-...        # Linux/macOS
set ANTHROPIC_API_KEY=sk-ant-...           # Windows CMD
$env:ANTHROPIC_API_KEY="sk-ant-..."        # Windows PowerShell
```

> Không có key thì vẫn chạy được ở chế độ STT-only.

### 4. Chạy
```bash
python voice_assistant.py
```

---

## 🎮 Cách dùng

| Thao tác | Kết quả |
|---|---|
| Nhấn **Enter** | Bắt đầu ghi âm |
| Im lặng 2 giây | Tự động dừng ghi |
| Nói **"thoát"** | Kết thúc chương trình |
| Nói **"xóa"** | Xóa lịch sử hội thoại |
| **Ctrl+C** | Thoát ngay |

---

## ⚙️ Tuỳ chỉnh trong `voice_assistant.py`

```python
WHISPER_MODEL = "small"   # tiny / base / small / medium
SILENCE_LIMIT = 2.0       # giây im lặng để kết thúc
SILENCE_THRESH = 0.01     # ngưỡng âm lượng (0.005–0.02)
```

| Model | RAM | Tốc độ | Độ chính xác |
|---|---|---|---|
| tiny | ~1GB | rất nhanh | thấp |
| base | ~1GB | nhanh | trung bình |
| **small** | ~2GB | **khuyến nghị** | tốt |
| medium | ~5GB | chậm | rất tốt |

---

## 🔧 Bật TTS (phát giọng nói)

```bash
pip install pyttsx3
```

Trong `voice_assistant.py`, bỏ comment dòng:
```python
# try_speak(response)
```

---

## 📁 Cấu trúc

```
voice_vi/
├── voice_assistant.py   # File chính
├── requirements.txt     # Thư viện cần cài
└── README.md
```
