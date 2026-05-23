# Software Requirements Specification
# Lecture Transcriber v6

**Version:** 1.0  
**Date:** 2026-05-23  
**Author:** — (tự động sinh)  
**Status:** Draft

---

## 1. Giới thiệu

### 1.1 Mục đích

Tài liệu này mô tả đầy đủ các yêu cầu phần mềm cho hệ thống **Lecture Transcriber v6** — một ứng dụng ghi âm và chuyển đổi giọng nói thành văn bản (Speech-to-Text) theo thời gian thực, được tối ưu hóa cho môi trường bài giảng tiếng Việt. Tài liệu dành cho nhà phát triển, người kiểm thử và người dùng cuối.

### 1.2 Phạm vi

Hệ thống thu âm thanh từ microphone hệ thống, phát hiện hoạt động giọng nói (Voice Activity Detection — VAD), phân đoạn âm thanh, chạy mô hình Whisper offline để nhận dạng ngôn ngữ, lọc ảo giác (hallucination), in kết quả trực tiếp lên terminal và lưu file log theo phiên.

**Trong phạm vi:**
- Thu âm và phân đoạn real-time
- VAD dựa trên RMS với ngưỡng tự hiệu chỉnh (calibration)
- Nhận dạng giọng nói via OpenAI Whisper (local)
- Lọc hallucination hai tầng (no_speech_prob + avg_logprob)
- Context window để cải thiện độ chính xác
- Logging có timestamp theo phiên
- Hỗ trợ GPU (CUDA) và CPU fallback

**Ngoài phạm vi:**
- Giao diện đồ họa (GUI)
- API server hoặc streaming qua mạng
- Dịch thuật đa ngôn ngữ tự động
- Tích hợp cloud STT (Google, Azure, AWS)
- Lưu trữ cơ sở dữ liệu

### 1.3 Định nghĩa & Từ viết tắt

| Thuật ngữ | Định nghĩa |
|---|---|
| VAD | Voice Activity Detection — phát hiện có/không có tiếng nói |
| RMS | Root Mean Square — phép đo năng lượng tín hiệu âm thanh |
| STT | Speech-To-Text — chuyển giọng nói thành văn bản |
| Hallucination | Văn bản sai/vô nghĩa do mô hình tự sinh ra không có trong audio |
| no_speech_prob | Xác suất Whisper đánh giá đoạn audio là im lặng/nhiễu |
| avg_logprob | Log probability trung bình của các token — đo độ tự tin mô hình |
| MBLK | Minimum speech blocks — số block tối thiểu để xác nhận có giọng nói |
| SBLK | Silence blocks — số block im lặng liên tiếp để kết thúc phân đoạn |
| FP16 | Half-precision float (16-bit) — chỉ dùng khi có GPU |
| Context Window | Danh sách N câu gần nhất dùng làm prompt cho Whisper |

### 1.4 Tài liệu tham khảo

- OpenAI Whisper paper: "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)
- Python `sounddevice` documentation: https://python-sounddevice.readthedocs.io/
- OpenAI Whisper GitHub: https://github.com/openai/whisper

---

## 2. Tổng quan hệ thống

### 2.1 Bối cảnh sản phẩm

Lecture Transcriber v6 là công cụ CLI chạy trên máy tính cá nhân (Windows/Linux/macOS), không cần kết nối internet trong lúc vận hành. Người dùng mục tiêu là sinh viên hoặc giảng viên muốn tạo bản text của bài giảng tiếng Việt theo thời gian thực, không phụ thuộc cloud.

### 2.2 Kiến trúc luồng dữ liệu

```
[Microphone]
     |
     v
[sounddevice InputStream]  -- blocksize = 50ms --
     |
     v
[make_callback()]
  - Tính RMS sliding window (RMS_SMOOTH blocks)
  - So sánh với ngưỡng VAD (thresh)
  - Tích lũy audio khi speech > MBLK, kết thúc khi silence > SBLK
  - Đẩy segment vào audio_queue
     |
     v
[audio_queue (maxsize=MAX_QUEUE)]
     |
     v (thread riêng biệt)
[stt_worker()]
  - model.transcribe() với context prompt
  - Trích xuất no_speech_prob, avg_logprob từ segments
  - Lọc hallucination tầng 1: no_speech_prob + avg_logprob
  - Lọc hallucination tầng 2: HALLUC_STRINGS
  - In ra terminal (char-by-char)
  - Ghi log file
```

### 2.3 Môi trường triển khai

| Thành phần | Yêu cầu |
|---|---|
| Hệ điều hành | Windows 10+, Ubuntu 20.04+, macOS 12+ |
| Python | 3.9 – 3.12 |
| RAM tối thiểu | 4 GB (model small); 8 GB (model medium); 16 GB (model large) |
| GPU (tùy chọn) | NVIDIA với CUDA 11.8+ để tăng tốc |
| Microphone | Bất kỳ thiết bị thu âm hỗ trợ 16000 Hz, mono |
| Disk | ~1.5 GB (model medium) |

---

## 3. Yêu cầu chức năng (Functional Requirements)

### FR-01: Kiểm tra thư viện khi khởi động

**Mô tả:** Hệ thống phải kiểm tra đầy đủ các thư viện bắt buộc trước khi thực thi bất kỳ logic nào.

**Đầu vào:** Danh sách package cần thiết: `sounddevice`, `numpy`, `whisper`, `dotenv`.

**Đầu ra:**
- Nếu đủ: tiếp tục khởi động bình thường.
- Nếu thiếu: in lệnh `pip install ...` và `sys.exit(1)`.

**Độ ưu tiên:** Bắt buộc (MUST)

---

### FR-02: Tải cấu hình từ biến môi trường

**Mô tả:** Mọi tham số vận hành phải có thể điều chỉnh qua file `.env` hoặc biến môi trường. Chương trình không được hardcode giá trị cấu hình trong code.

**Các biến cấu hình bắt buộc:**

| Biến | Mặc định | Kiểu | Mô tả |
|---|---|---|---|
| `WHISPER_MODEL` | `medium` | str | Tên model Whisper |
| `SAMPLE_RATE` | `16000` | int | Tần số lấy mẫu (Hz) |
| `LANGUAGE` | `vi` | str | Ngôn ngữ STT (`vi`, `en`, `auto`) |
| `LOG_FILE` | auto-datetime | str | Đường dẫn file log |
| `MAX_QUEUE_SIZE` | `6` | int | Giới hạn hàng đợi audio |
| `CONTEXT_WINDOW` | `4` | int | Số câu context cho Whisper |
| `BEAM_SIZE` | `5` | int | Beam search width |
| `SILENCE_SEC` | `1.0` | float | Thời gian im lặng để kết thúc đoạn (giây) |
| `MIN_SPEECH_SEC` | `0.4` | float | Thời gian giọng nói tối thiểu hợp lệ (giây) |
| `CALIBRATE_SEC` | `2.0` | float | Thời gian thu âm nền để hiệu chỉnh (giây) |
| `RMS_SMOOTH_BLOCKS` | `4` | int | Số block để làm mượt RMS |
| `NO_SPEECH_THRESH` | `0.6` | float | Ngưỡng lọc no_speech_prob |
| `LOGPROB_THRESH` | `-1.0` | float | Ngưỡng lọc avg_logprob |
| `SILENCE_THRESH` | (auto) | float | Ghi đè ngưỡng VAD thủ công |

**Độ ưu tiên:** Bắt buộc (MUST)

---

### FR-03: Hiệu chỉnh ngưỡng VAD (Calibration)

**Mô tả:** Hệ thống phải tự động xác định ngưỡng VAD dựa trên nhiễu nền thực tế của môi trường.

**Quy trình:**
1. Nếu `SILENCE_THRESH` được đặt trong `.env` → dùng trực tiếp (bỏ qua calibration).
2. Nếu không → thu `CALIBRATE_SEC` giây âm thanh khi im lặng.
3. Tính `rms = sqrt(mean(samples²))`.
4. Tính `thresh = max(0.002, round(rms × 3.5, 5))`.
5. In giá trị noise rms và ngưỡng kết quả.

**Điều kiện biên:**
- Ngưỡng tối thiểu phải ≥ 0.002 để tránh trigger liên tục trong môi trường cực im lặng.

**Độ ưu tiên:** Bắt buộc (MUST)

---

### FR-04: Thu âm liên tục và phân đoạn âm thanh (Audio Segmentation)

**Mô tả:** Hệ thống phải thu âm liên tục từ microphone mặc định và tự động tách các đoạn có giọng nói.

**Tham số block:**
- Block size = `SAMPLE_RATE × 0.05` samples (tương đương 50ms mỗi block)
- `SBLK = ceil(SILENCE_SEC / 0.05)` — số block im lặng để kết thúc đoạn
- `MBLK = ceil(MIN_SPEECH_SEC / 0.05)` — số block tối thiểu cần có giọng nói

**Thuật toán VAD (trong callback):**

```
Với mỗi block audio 50ms:
  1. Cập nhật hist (sliding window RMS_SMOOTH blocks)
  2. rms = mean(hist)
  3. Cập nhật peak = max(peak, rms) * 0.997 (decay factor)
  4. is_speech = (rms > thresh)
  
  Nếu is_speech:
    - speech_count += 1
    - silent_count = 0
    - Thêm block vào buf
  
  Ngược lại nếu đang theo dõi giọng nói (speech_count > 0):
    - silent_count += 1
    - Thêm block vào buf
    - Nếu silent_count >= SBLK:
        Nếu speech_count >= MBLK và buf không rỗng:
          seg = concatenate(buf)  → đẩy vào queue
        Xóa buf, reset counters
```

**Điều kiện lỗi:**
- Nếu queue đầy (`queue.Full`): in cảnh báo `[!] queue day`, bỏ segment đó (không block callback).

**Độ ưu tiên:** Bắt buộc (MUST)

---

### FR-05: Nhận dạng giọng nói (STT Worker)

**Mô tả:** Thread STT đọc từ queue và gọi `model.transcribe()` để chuyển audio thành text.

**Tham số transcription:**

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `language` | `LANGUAGE` hoặc `None` nếu "auto" | |
| `fp16` | `True` nếu có GPU | Half-precision |
| `condition_on_previous_text` | `False` | Tránh phụ thuộc context lỗi |
| `initial_prompt` | `" ".join(_recent)` hoặc `None` | Context window |
| `beam_size` | `BEAM_SIZE` | Beam search width |
| `best_of` | `BEAM_SIZE` | Sampling fallback |
| `temperature` | `[0.0, 0.2, 0.4]` | Danh sách fallback temperatures |

**Luồng xử lý sau transcription:**
1. Trích xuất `segments` từ kết quả.
2. Tính `avg_no_speech = mean([s.no_speech_prob for s in segments])`.
3. Tính `avg_logprob = mean([s.avg_logprob for s in segments])`.
4. Nếu không có segments: `avg_no_speech = 1.0`, `avg_logprob = -2.0`.

**Độ ưu tiên:** Bắt buộc (MUST)

---

### FR-06: Lọc Hallucination

**Mô tả:** Hệ thống phải loại bỏ kết quả STT không đáng tin cậy qua hai tầng lọc.

**Tầng 1 — Confidence filtering:**

| Điều kiện | Lý do bỏ qua |
|---|---|
| `len(text) < 2` hoặc text rỗng | Quá ngắn, không có nghĩa |
| `avg_no_speech > NO_SPEECH_THRESH` | Whisper cho rằng không có tiếng nói |
| `avg_logprob < LOGPROB_THRESH` | Độ tự tin thấp, kết quả không đáng tin |

**Tầng 2 — String matching:**

Bỏ qua nếu text chứa bất kỳ chuỗi nào trong `HALLUC_STRINGS` (case-insensitive):
```
{"ghiền mì gõ", "subscribe", "like và subscribe",
 "đăng ký kênh", "cảm ơn các bạn đã xem",
 "xin chào các bạn", "hẹn gặp lại"}
```

**Đầu ra khi lọc:** In `[SKIP] <lý do>` bằng màu xám.

**Độ ưu tiên:** Bắt buộc (MUST)

---

### FR-07: Hiển thị kết quả và logging

**Mô tả:** Kết quả hợp lệ phải được hiển thị và lưu log.

**Hiển thị terminal:**
- In từng ký tự liên tiếp (char-by-char) để tạo cảm giác real-time.
- Format: `[HH:MM:SS] <text>`
- Dùng ANSI color codes: timestamp màu cyan, text màu trắng.
- Xóa dòng progress bar trước khi in text.

**Logging:**
- File log được tạo khi khởi động với header chứa: datetime, model, device, ngưỡng VAD.
- Mỗi entry: `[HH:MM:SS] <text>` append vào file.
- Encoding: UTF-8 (bắt buộc để hỗ trợ tiếng Việt).
- Tên file mặc định: `lecture_YYYYMMDD_HHMMSS.txt`.

**Cập nhật context window:**
- Sau khi ghi nhận text hợp lệ: thêm vào `_recent` deque (maxlen = `CONTEXT_WIN`).

**Độ ưu tiên:** Bắt buộc (MUST)

---

### FR-08: Hiển thị progress bar real-time

**Mô tả:** Trong lúc chờ giọng nói, terminal phải hiển thị thanh mức âm lượng cập nhật liên tục.

**Format:**
```
  [█████████                     ] rms=0.00312 peak=0.00850 thr=0.04000 sp=000 si=00
```

- Thanh: 30 ký tự `█`, số ký tự = `min(int(rms × 1000), 30)`.
- Màu thanh: xanh lá (`GR`) nếu đang nhận speech, xám (`GY`) nếu im lặng.
- Hiển thị: rms, peak, threshold, speech count, silence count.
- Hiển thị kích thước queue nếu > 0.
- Dùng `\r` để overwrite dòng hiện tại (không tạo newline).
- Bị ẩn (`suppress`) trong khi đang in text kết quả.

**Độ ưu tiên:** Nên có (SHOULD)

---

### FR-09: Thoát an toàn (Graceful Shutdown)

**Mô tả:** Người dùng nhấn Enter để dừng. Hệ thống phải xử lý hết queue trước khi thoát.

**Quy trình:**
1. Thread lắng nghe `input()` → set `_stop` event khi Enter được nhấn.
2. Vòng lặp main thoát khi `_stop` được set.
3. Đẩy sentinel `None` vào queue để báo STT worker dừng.
4. `worker.join(timeout=120)` — chờ tối đa 2 phút để xử lý hết.
5. In đường dẫn file log.

**Độ ưu tiên:** Bắt buộc (MUST)

---

## 4. Yêu cầu phi chức năng (Non-Functional Requirements)

### NFR-01: Hiệu năng (Performance)

| Chỉ số | Yêu cầu |
|---|---|
| Audio callback latency | < 10ms (không được block vòng lặp audio) |
| Queue wait time (GPU) | < 3s/segment với model medium trên RTX 3060+ |
| Queue wait time (CPU) | < 30s/segment với model small |
| Memory usage (model medium, GPU) | < 6 GB VRAM |
| Drop rate khi queue đầy | 0% block — chỉ log cảnh báo và bỏ segment |

### NFR-02: Độ tin cậy (Reliability)

- Hệ thống không được crash khi gặp exception trong STT worker — phải bắt và log lỗi.
- Thread callback không được throw exception (sẽ làm dừng audio stream).
- File log phải được flush sau mỗi entry (`with open(...) as f: f.write(...)`).

### NFR-03: Khả năng cấu hình (Configurability)

- Tất cả tham số vận hành phải điều chỉnh được qua `.env` mà không cần sửa code.
- Thay đổi `.env` có hiệu lực ở lần khởi động tiếp theo.

### NFR-04: Khả năng di động (Portability)

- Chạy được trên Windows 10+, Ubuntu 20.04+, macOS 12+.
- Không phụ thuộc phần cứng cụ thể; GPU là tùy chọn.

### NFR-05: Bảo mật (Security)

- Toàn bộ xử lý diễn ra offline, trên máy cục bộ.
- Không gửi dữ liệu âm thanh hoặc text ra ngoài.
- File log được lưu tại thư mục làm việc hiện tại (không phải thư mục hệ thống).

### NFR-06: Khả năng bảo trì (Maintainability)

- Tất cả hằng số cấu hình phải nằm trong block `# Config` ở đầu file.
- Mỗi module logic (calibration, callback, stt_worker, main) phải là hàm riêng.
- HALLUC_STRINGS phải là set để tra cứu O(1).

---

## 5. Yêu cầu giao diện hệ thống (System Interface Requirements)

### 5.1 Giao diện phần cứng

- **Microphone:** Bất kỳ thiết bị âm thanh nào được nhận diện bởi `sounddevice` (PortAudio backend).
- **GPU (tùy chọn):** NVIDIA GPU với driver CUDA tương thích.

### 5.2 Giao diện phần mềm

| Thư viện | Version tối thiểu | Mục đích |
|---|---|---|
| `sounddevice` | 0.4.6 | Thu âm từ microphone |
| `numpy` | 1.24.0 | Xử lý mảng số, tính RMS |
| `openai-whisper` | 20231117 | Mô hình STT |
| `python-dotenv` | 1.0.0 | Đọc file `.env` |
| `torch` | (kèm whisper) | Backend ML, phát hiện CUDA |

### 5.3 Giao diện người dùng

- **Input:** Terminal/console tiêu chuẩn (stdin cho lệnh Enter để dừng).
- **Output:** Terminal hỗ trợ ANSI escape codes.
- **File output:** File text UTF-8 tại thư mục làm việc.

---

## 6. Luồng xử lý lỗi (Error Handling)

| Tình huống | Hành vi |
|---|---|
| Thiếu thư viện khi khởi động | In lệnh install, `sys.exit(1)` |
| Không có microphone | `sounddevice` raise exception, in lỗi, thoát |
| Queue đầy | Log cảnh báo `[!] queue day`, bỏ segment hiện tại |
| Exception trong `stt_worker` | Bắt tất cả Exception, in `[ERR] <message>`, tiếp tục |
| Exception trong audio callback | Không được raise — callback phải return bình thường |
| STT worker timeout khi shutdown | `worker.join(timeout=120)` — nếu quá 2 phút thì thoát |
| File log không thể ghi | Chưa xử lý trong v6 — cần xử lý trong v7 |

---

## 7. Ràng buộc thiết kế (Design Constraints)

- **Single-file application:** Toàn bộ logic nằm trong một file `.py` duy nhất để dễ triển khai.
- **Offline-first:** Không có bất kỳ HTTP call nào trong luồng vận hành chính.
- **Thread model:** Đúng 2 thread chính — main thread (audio callback) + STT worker thread. Thread thứ 3 chỉ lắng nghe stdin.
- **No GUI:** Giao diện hoàn toàn là CLI/terminal.
- **Python standard library + listed deps only:** Không thêm dependency ngoài danh sách.

---

## 8. Ma trận yêu cầu — Truy xuất nguồn gốc

| ID Yêu cầu | Mô đun triển khai | Test case |
|---|---|---|
| FR-01 | `check_imports()` | Xóa một thư viện, kiểm tra exit message |
| FR-02 | Block `# Config`, `load_dotenv()` | Đặt giá trị trong `.env`, kiểm tra biến runtime |
| FR-03 | `calibrate()` | Đo với môi trường im lặng và môi trường ồn ào |
| FR-04 | `make_callback()` | Nói một từ ngắn, kiểm tra segment trong queue |
| FR-05 | `stt_worker()` | Feed audio mẫu biết trước, so sánh transcript |
| FR-06 | `stt_worker()` — phần filter | Feed audio nhiễu, kiểm tra `[SKIP]` output |
| FR-07 | `sprint()`, `log()`, `init_log()` | Kiểm tra file log sau phiên |
| FR-08 | `make_callback()` — phần display | Quan sát terminal khi thu âm |
| FR-09 | `main()` — shutdown logic | Nhấn Enter, kiểm tra file log được lưu đầy đủ |

---

## 9. Giả định và phụ thuộc

1. Máy có ít nhất một thiết bị âm thanh input được nhận diện bởi PortAudio.
2. Người dùng đã cài đặt đúng phiên bản Python và pip.
3. Model Whisper được tải về tự động lần đầu qua internet (sau đó cached offline).
4. Terminal hỗ trợ ANSI escape codes (mọi terminal hiện đại đều hỗ trợ).
5. Với Windows: cần chạy trong terminal hỗ trợ ANSI (Windows Terminal, PowerShell 7+, không dùng cmd.exe cũ).

---

## 10. Hướng phát triển tương lai (v7+)

- **FR-T01:** Xử lý lỗi khi file log không thể ghi (disk full, permission denied).
- **FR-T02:** Hỗ trợ chọn thiết bị microphone cụ thể qua tham số `DEVICE_ID`.
- **FR-T03:** Xuất file log định dạng SRT/VTT cho subtitle.
- **FR-T04:** GUI tối giản (systray icon + text area) bằng PyQt hoặc tkinter.
- **FR-T05:** Whisper faster-whisper backend để giảm thời gian xử lý trên CPU.
- **FR-T06:** WebSocket server để stream transcript sang thiết bị khác trên LAN.

