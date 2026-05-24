"""
polish.py — Dùng Ollama sửa lại text từ file log của transcribe.py
Chạy SAU KHI đã dừng transcribe.py để tiết kiệm RAM.

Cách dùng:
    python polish.py                        # tự tìm file log mới nhất
    python polish.py lecture_20250523.txt   # chỉ định file cụ thể

Từ điển tích lũy:
    dictionary.txt  — tự động cập nhật sau mỗi lần chạy
                      có thể sửa tay bất cứ lúc nào
    Định dạng: sai -> đúng   (mỗi dòng 1 cặp)
"""

import os, sys, time, subprocess, requests, json, re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
ROOT_DIR  = Path(__file__).parent.parent   # VOICE_REAL_TIME/
LOG_DIR   = ROOT_DIR / "lecture"

# Config
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL",   "gemma3:4b")
OLLAMA_URL     = os.getenv("OLLAMA_URL",     "http://localhost:11434")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
OLLAMA_BOOT    = int(os.getenv("OLLAMA_BOOT_SEC", "15"))
DICT_FILE      = Path(os.getenv("DICT_FILE", str(ROOT_DIR / "dictionary.txt")))


class C:
    R="\033[0m"; BOLD="\033[1m"
    GR="\033[92m"; CY="\033[96m"; YL="\033[93m"
    RD="\033[91m"; GY="\033[90m"


# Từ điển tích lũy
def load_dict() -> dict[str, str]:
    """Đọc dictionary.txt → {sai: đúng}"""
    d = {}
    if not DICT_FILE.exists():
        return d
    for line in DICT_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "->" not in line:
            continue
        parts = line.split("->", 1)
        wrong, correct = parts[0].strip().lower(), parts[1].strip()
        if wrong and correct:
            d[wrong] = correct
    return d


def save_dict(d: dict[str, str]):
    """Ghi lại dictionary.txt, sắp xếp theo alphabet."""
    lines = ["# Từ điển tích lũy — sai -> đúng", "# Sửa tay bất cứ lúc nào", ""]
    for wrong in sorted(d):
        lines.append(f"{wrong} -> {d[wrong]}")
    DICT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_dict(text: str, d: dict[str, str]) -> str:
    """Replace cứng các từ đã biết chắc trước khi gửi Ollama."""
    for wrong, correct in d.items():
        # dùng word boundary để tránh replace nhầm giữa câu
        text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
    return text


def merge_new_words(d: dict[str, str], new_words: list[dict]) -> tuple[dict, int]:
    """Thêm từ mới Ollama phát hiện vào từ điển, trả về (dict mới, số từ thêm)."""
    added = 0
    for item in new_words:
        wrong   = str(item.get("wrong",   "")).strip().lower()
        correct = str(item.get("correct", "")).strip()
        if wrong and correct and wrong != correct and wrong not in d:
            d[wrong] = correct
            added += 1
    return d, added


# Ollama server
def is_ollama_running() -> bool:
    try:
        return requests.get(OLLAMA_URL, timeout=2).status_code == 200
    except Exception:
        return False


def start_ollama():
    if is_ollama_running():
        print(f"  {C.GR}Ollama đang chạy rồi.{C.R}")
        return True

    print(f"  {C.YL}Đang khởi động Ollama...{C.R}", flush=True)
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for i in range(OLLAMA_BOOT * 2):
        time.sleep(0.5)
        if is_ollama_running():
            print(f"  {C.GR}Ollama sẵn sàng sau {(i+1)*0.5:.1f}s.{C.R}")
            return True
        sys.stdout.write(f"\r  Chờ Ollama... {(i+1)*0.5:.1f}s")
        sys.stdout.flush()

    print(f"\n  {C.RD}Không thể khởi động Ollama.{C.R}")
    return False


def pull_model_if_needed(model: str):
    try:
        r = requests.post(f"{OLLAMA_URL}/api/show", json={"name": model}, timeout=5)
        if r.status_code == 200:
            print(f"  {C.GR}Model {model} đã có sẵn.{C.R}")
            return
    except Exception:
        pass
    print(f"  {C.YL}Model {model} chưa có — đang pull...{C.R}", flush=True)
    result = subprocess.run(["ollama", "pull", model])
    if result.returncode != 0:
        print(f"  {C.RD}Pull thất bại. Thử: ollama pull {model}{C.R}")
        sys.exit(1)


# Polish
def build_system_prompt(d: dict[str, str]) -> str:
    base = (
        "Bạn là trợ lý chỉnh sửa văn bản tiếng Việt được phiên âm từ giọng nói.\n"
        "Nhiệm vụ: sửa lỗi chính tả, dấu câu, từ ngữ bị nghe nhầm. "
        "Giữ nguyên ý nghĩa, không thêm bớt thông tin.\n\n"
        "Trả về JSON với đúng 2 trường, không giải thích thêm:\n"
        "{\n"
        '  "fixed": "câu đã sửa hoàn chỉnh",\n'
        '  "new_words": [\n'
        '    {"wrong": "từ sai trong câu gốc", "correct": "từ đúng"}\n'
        "  ]\n"
        "}\n"
        "new_words chỉ chứa từ bị nghe nhầm mà bạn phát hiện THÊM "
        "(không có trong từ điển bên dưới). "
        "Nếu không có từ mới thì new_words = [].\n"
    )
    if d:
        dict_str = "\n".join(f"  {w} -> {c}" for w, c in sorted(d.items()))
        base += f"\nTừ điển đã biết (đã được áp dụng trước, KHÔNG lặp lại):\n{dict_str}\n"
    return base


def polish_line(raw: str, d: dict[str, str]) -> tuple[str, list[dict]]:
    """
    Gửi 1 dòng cho Ollama.
    Trả về (text đã sửa, danh sách từ mới phát hiện).
    """
    # Bước 1: apply từ điển cứng trước
    pre = apply_dict(raw, d)

    payload = {
        "model": OLLAMA_MODEL,
        "system": build_system_prompt(d),
        "prompt": pre,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
        },
    }
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        raw_response = r.json().get("response", "").strip()

        # Parse JSON từ Ollama
        # Đôi khi model bọc thêm ```json ... ``` nên strip ra
        clean = re.sub(r"^```json|```$", "", raw_response, flags=re.MULTILINE).strip()
        data  = json.loads(clean)

        fixed     = str(data.get("fixed", pre)).strip() or pre
        new_words = data.get("new_words", [])
        if not isinstance(new_words, list):
            new_words = []
        return fixed, new_words

    except (json.JSONDecodeError, KeyError):
        # Ollama không trả JSON hợp lệ — giữ bản pre-processed
        return pre, []
    except Exception as e:
        print(f"\n  {C.RD}[ERR] {e}{C.R}")
        return pre, []


# File log
def find_latest_log() -> Path | None:
    # bỏ qua file _polished để không đọc nhầm
    logs = sorted(
        [p for p in LOG_DIR.glob("lecture_*.txt") if "_polished" not in p.name],
        key=lambda p: p.stat().st_mtime,
    )
    return logs[-1] if logs else None


def parse_log(path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    header, entries = [], []
    in_header = True
    for line in path.read_text(encoding="utf-8").splitlines():
        if in_header:
            header.append(line)
            if line.startswith("=" * 10):
                in_header = False
            continue
        if line.startswith("[") and "]" in line:
            ts  = line[1:9]
            txt = line[11:].strip()
            entries.append((ts, txt))
        elif line.strip():
            entries.append(("", line.strip()))
    return header, entries


def save_polished(original: Path, header: list[str], polished: list[tuple[str, str, str]]):
    out = original.with_stem(original.stem + "_polished")
    with out.open("w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n\n")
        f.write(f"{'TIMESTAMP':<10}  {'RAW (Whisper)':<55}  POLISHED (Ollama)\n")
        f.write("-" * 120 + "\n")
        for ts, raw, fixed in polished:
            changed = "✓" if fixed != raw else " "
            f.write(f"[{ts}]  {raw:<55}  {changed} {fixed}\n")
    return out


# Main
def main():
    print(f"\n{C.CY}{C.BOLD}  Polish — Ollama text fixer (với từ điển tích lũy){C.R}")
    print(f"  Model: {C.YL}{OLLAMA_MODEL}{C.R}\n")

    # Load từ điển
    dictionary = load_dict()
    print(f"  Từ điển: {C.CY}{DICT_FILE}{C.R} — {C.BOLD}{len(dictionary)}{C.R} từ đã biết.")

    # Chọn file log
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_path = find_latest_log()
        if not log_path:
            print(f"  {C.RD}Không tìm thấy file lecture_*.txt nào trong {LOG_DIR}{C.R}")
            sys.exit(1)
        print(f"  File log:  {C.CY}{log_path}{C.R}")

    if not log_path.exists():
        print(f"  {C.RD}File không tồn tại: {log_path}{C.R}")
        sys.exit(1)

    header, entries = parse_log(log_path)
    print(f"  Đoạn cần xử lý: {C.BOLD}{len(entries)}{C.R}\n")

    if not entries:
        print(f"  {C.YL}File log trống.{C.R}")
        sys.exit(0)

    # Khởi động Ollama
    if not start_ollama():
        sys.exit(1)
    pull_model_if_needed(OLLAMA_MODEL)
    print()

    # Polish từng dòng
    results      = []
    total_new    = 0

    for i, (ts, raw) in enumerate(entries, 1):
        sys.stdout.write(f"\r  [{i}/{len(entries)}] Đang xử lý...{' '*20}")
        sys.stdout.flush()

        fixed, new_words = polish_line(raw, dictionary)

        # Cập nhật từ điển ngay với từ mới phát hiện
        if new_words:
            dictionary, added = merge_new_words(dictionary, new_words)
            total_new += added
            if added:
                save_dict(dictionary)  # ghi ngay để không mất nếu crash
                for w in new_words:
                    print(f"\n  {C.YL}[Dict+] {w.get('wrong')} -> {w.get('correct')}{C.R}", end="")

        results.append((ts, raw, fixed))

        if fixed != raw:
            print(f"\r  {C.CY}[{ts}]{C.R} {C.GY}{raw[:55]}{C.R}")
            print(f"  {' '*10}{C.GR}→ {fixed[:80]}{C.R}")
        else:
            print(f"\r  {C.CY}[{ts}]{C.R} (không đổi) {C.GY}{raw[:55]}{C.R}")

    # Lưu file polished
    out_path = save_polished(log_path, header, results)
    changed  = sum(1 for _, r, f in results if r != f)

    print(f"\n  {C.GR}Xong!{C.R}")
    print(f"  Đã sửa   : {C.BOLD}{changed}/{len(results)}{C.R} đoạn")
    print(f"  Từ mới   : {C.YL}{total_new}{C.R} từ thêm vào {DICT_FILE}")
    print(f"  Lưu tại  : {C.CY}{out_path}{C.R}\n")


if __name__ == "__main__":
    main()
