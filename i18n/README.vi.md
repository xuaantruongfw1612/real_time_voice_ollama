# 🎙️ Lecture Transcriber

> Nhận diện giọng nói tiếng Việt theo thời gian thực cho bài giảng, kết hợp với LLM offline để làm sạch transcript.

<p align="right">
  🌐 <a href="README.md">Read in English</a>
</p>

---

## Tổng quan

Pipeline 2 bước: thu âm trực tiếp, phiên âm bằng [OpenAI Whisper](https://github.com/openai/whisper), sau đó làm sạch bản thô bằng model [Ollama](https://ollama.com) chạy hoàn toàn offline — không cần cloud.

```
Microphone → transcribe.py (Whisper) → lecture_YYYYMMDD.txt
                                              ↓
                                        polish.py (Ollama)
                                              ↓
                                  lecture_YYYYMMDD_polished.txt
```

---

## Tính năng

- **Phiên âm trực tiếp** — tách đoạn theo khoảng lặng, đẩy từng chunk cho Whisper
- **Lọc ảo giác (hallucination filter)** — loại bỏ kết quả rác dựa trên `no_speech_prob` và `avg_logprob`
- **Tự động hiệu chỉnh VAD** — đo tiếng ồn môi trường lúc khởi động, tự đặt ngưỡng
- **Từ điển tích lũy** — `dictionary.txt` tự lớn lên sau mỗi lần chạy polish; sửa trước khi Ollama xử lý
- **Hỗ trợ GPU** — dùng CUDA nếu có, tự chuyển sang CPU nếu không
- **Hỗ trợ NixOS** — có sẵn `shell.nix` để dựng môi trường reproducible

---

## Yêu cầu

| Công cụ | Phiên bản |
|---------|-----------|
| Python | 3.11+ |
| FFmpeg | bất kỳ |
| Ollama | bất kỳ |
| CUDA *(tuỳ chọn)* | 11.8+ |

Các gói Python (cài qua `pip install -r requirements.txt`):

```
sounddevice>=0.4.6
numpy>=1.24.0
openai-whisper>=20231117
python-dotenv>=1.0.0
requests  # dùng bởi polish.py
```

---

## Bắt đầu nhanh

### 1 — Clone & cài đặt

```bash
git clone <your-repo>
cd <your-repo>

pip install -r requirements.txt
```

Trên NixOS, dùng shell có sẵn:

```bash
nix-shell shell.nix
pip install openai-whisper requests
```

### 2 — Cấu hình

Sao chép file env mẫu và chỉnh theo nhu cầu:

```bash
cp _env .env
```

Các biến quan trọng trong `.env`:

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `WHISPER_MODEL` | `medium` | Kích thước model (`tiny` / `base` / `small` / `medium` / `large`) |
| `LANGUAGE` | `vi` | Mã ngôn ngữ, hoặc `auto` để tự phát hiện |
| `SILENCE_THRESH` | *(tự đo)* | Để trống để tự hiệu chỉnh, hoặc đặt giá trị RMS cố định |
| `SILENCE_SEC` | `1.0` | Số giây im lặng để cắt đoạn |
| `MIN_SPEECH_SEC` | `0.4` | Thời lượng nói tối thiểu để giữ đoạn |
| `OLLAMA_MODEL` | `gemma3:4b` | Model Ollama dùng để polish |
| `OLLAMA_URL` | `http://localhost:11434` | Địa chỉ server Ollama |

### 3 — Phiên âm

```bash
python transcribe.py
```

- Giữ yên trong 2 giây để hiệu chỉnh tiếng ồn lúc khởi động.
- Nói bình thường. Mỗi đoạn được in ra ngay và ghi vào `lecture_<timestamp>.txt`.
- Nhấn **Enter** để dừng.

### 4 — Làm sạch transcript

Sau khi dừng transcribe (để giải phóng RAM cho Ollama):

```bash
# Tự tìm file lecture_*.txt mới nhất
python polish.py

# Hoặc chỉ định file cụ thể
python polish.py lecture_20250523_143000.txt
```

Ollama sẽ tự khởi động nếu chưa chạy. Kết quả được lưu cạnh file gốc với tên `lecture_<timestamp>_polished.txt`.

---

## Định dạng đầu ra

### Log thô (`lecture_*.txt`)

```
BUOI HOC : 2025-05-23 14:30:00
Model    : Whisper medium  |  Ngon ngu: vi
...
============================

[14:30:12] Hôm nay chúng ta sẽ học về...
[14:30:45] ...cấu trúc dữ liệu cây nhị phân...
```

### Log đã polish (`lecture_*_polished.txt`)

```
TIMESTAMP   RAW (Whisper)                             POLISHED (Ollama)
------------------------------------------------------------------------
[14:30:12]  Hôm nay chúng ta sẽ học về...          ✓ Hôm nay chúng ta học về...
[14:30:45]  cấu trúc dữ liệu cây nhị phân...         (không đổi)
```

### Từ điển (`dictionary.txt`)

Tự tạo và cập nhật. Có thể sửa tay bất cứ lúc nào:

```
# Định dạng: sai -> đúng  (mỗi dòng 1 cặp)
huyền thoại -> huyền thoại
whisper -> Whisper
```

---

## Cấu trúc file

```
.
├── transcribe.py        # Phiên âm trực tiếp (Whisper)
├── polish.py            # Hậu xử lý (Ollama)
├── requirements.txt     # Các gói Python
├── shell.nix            # Dev shell cho NixOS
├── _env                 # File env mẫu (đổi tên thành .env)
└── dictionary.txt       # Từ điển sửa lỗi (tự động tạo)
```

---

## Mẹo

- **Máy yếu?** Dùng `WHISPER_MODEL=small` hoặc `tiny`. Chất lượng giảm nhẹ nhưng nhanh hơn nhiều.
- **Nhiều thuật ngữ chuyên ngành?** Điền trước vào `dictionary.txt` với dạng đúng trước khi chạy `polish.py`.
- **Môi trường ồn?** Đặt `SILENCE_THRESH` thủ công trong `.env` (ví dụ `0.040`) thay vì dùng tự hiệu chỉnh.
- **Chọn model Ollama:** `gemma3:4b` cân bằng tốt giữa chất lượng và tốc độ. Máy mạnh hơn thử `qwen2.5:7b`; cần nhanh hơn thử `llama3.2:3b`.

---

## Giấy phép

MIT
