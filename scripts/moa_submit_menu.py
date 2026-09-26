#!/usr/bin/env python3
"""Vietnamese menu for the MOA loop; launched by NOP_BAI.cmd."""
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

import moa_submit_loop as runner


def choose(title, labels, default=1):
    print(f"\n{title}")
    for index, label in enumerate(labels, 1):
        print(f"  {index}. {label}")
    print("  0. Thoát")
    while True:
        value = input(f"Chọn số [{default}]: ").strip() or str(default)
        if value.isdigit() and 0 <= int(value) <= len(labels):
            return int(value)
        print("Hãy nhập một số trong danh sách.")


def number(title, default, integer=False):
    while True:
        value = input(f"{title} [{default}]: ").strip() or str(default)
        try:
            parsed = int(value) if integer else float(value.replace(",", "."))
            if math.isfinite(parsed) and (parsed >= 0 if integer else parsed > 0):
                return parsed
        except ValueError:
            pass
        print("Giá trị không hợp lệ. Hãy nhập lại.")


def version_path(text, root):
    value = text.strip().strip('"').strip("'")
    if not value:
        raise ValueError("Chưa nhập thư mục.")
    if re.match(r"^[A-Za-z]:[\\/]", value) or value.startswith("\\\\"):
        value = subprocess.check_output(["wslpath", "-u", value], text=True).strip()
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def sessions(staging):
    found = []
    for path in sorted(staging.glob("*/state.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("source") and "hashes" in state:
                found.append((path.parent, state))
        except (OSError, ValueError):
            continue
    return found


def select_source(root, remembered=None):
    candidates = []
    baseline = root / "research/furiosa-score-20260923/baseline-source"
    if (baseline / "Cargo.toml").exists():
        candidates.append(("Snapshot baseline-source của nghiên cứu Arena 78017", baseline))
    candidates.append(("Workspace hiện tại (có thể đang được chỉnh sửa)", root))
    if remembered and Path(remembered).exists() and Path(remembered) not in [p for _, p in candidates]:
        candidates.insert(0, (f"Phiên bản đã chọn lần trước: {remembered}", Path(remembered)))
    candidates += [(p.name, p) for p in sorted(root.glob("delivery_*"))
                   if p.is_dir() and (p / "Cargo.toml").is_file()]
    while True:
        default = next((i for i, (_, path) in enumerate(candidates, 1) if str(path) == remembered), 1)
        choice = choose("Chọn phiên bản muốn nộp", [label for label, _ in candidates] + ["Dán đường dẫn thư mục khác"], default)
        if choice == 0:
            return None
        try:
            source = (candidates[choice - 1][1] if choice <= len(candidates) else
                      version_path(input("Dán đường dẫn Windows hoặc WSL: "), root))
            runner.source_hashes(source)
            return source
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"Không đọc được phiên bản: {exc}")


def main(root=None):
    root = (root or Path(__file__).resolve().parents[1]).resolve()
    staging = root / "submission_staging"
    preferences = staging / "moa-menu.json"
    try:
        remembered = json.loads(preferences.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        remembered = {}
    print("\nNỘP BÀI MOA TỰ ĐỘNG")
    print("Chờ bài trước kết thúc rồi mới nộp tiếp. Kiểm tra mỗi 5 phút.")
    saved = sessions(staging)
    labels = [f"Tiếp tục {folder.name} | đã nộp {state.get('attempts', 0)} | "
              f"điểm cao nhất {state.get('best_score')} | ID đang chờ {state.get('active_id') or '-'}"
              for folder, state in saved]
    labels.append("Tạo phiên nộp mới / chọn phiên bản khác")
    choice = choose("Chọn phiên nộp", labels)
    if choice == 0:
        return 0
    recovery = None
    if choice <= len(saved):
        directory, state = saved[choice - 1]
        source = Path(state["source"])
        defaults = remembered if remembered.get("source") == str(source) else {}
        print(f"Tiếp tục dùng source đã đóng băng: {directory / 'source'}")
        if state["phase"] == "submitting":
            print("Lần gửi trước chưa rõ kết quả. Cần đối chiếu ID trên MOA trước khi tiếp tục.")
            recovery = input("ID đã xác minh của lần gửi trước (Enter để thoát): ").strip()
            if not recovery:
                return 0
    else:
        source = select_source(root, remembered.get("source"))
        if source is None:
            return 0
        directory = staging / f"moa-menu-{source.name}-{uuid.uuid4().hex[:10]}"
        defaults = remembered
    target = number("Đạt bao nhiêu điểm thì dừng?", defaults.get("target_score", 8.0))
    limit = number("Tối đa bao nhiêu lần nộp trong phiên? (0 = không giới hạn)",
                   defaults.get("max_submissions", 0), integer=True)
    print(f"\nPhiên bản: {source}\nĐiểm dừng: {target}\nGiới hạn: {limit or 'Không giới hạn'}")
    print("Kiểm tra: 5 phút/lần. Gặp bài lỗi: dừng để xem log.")
    print(f"Log và lịch sử: {directory / 'logs'}")
    action = choose("Sẵn sàng", ["Bắt đầu nộp / tiếp tục theo dõi", "Chỉ kiểm tra cấu hình, chưa nộp"], default=2)
    if action == 0:
        return 0
    cli = shutil.which("moa-submitter") or str(Path.home() / ".cargo/bin/moa-submitter")
    argv = ["--source", str(source), "--target-score", str(target), "--max-submissions", str(limit),
            "--state-dir", str(directory), "--cli", cli]
    if recovery:
        argv += ["--recover-id", recovery]
    if action == 2:
        argv += ["--dry-run"]
    else:
        staging.mkdir(parents=True, exist_ok=True)
        runner.save(preferences, {"source": str(source), "target_score": target, "max_submissions": limit})
    print("\nNhấn Ctrl+C để dừng. Mở lại NOP_BAI để tiếp tục theo dõi bài đang chạy.")
    return runner.main(argv)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nĐã dừng. Bài đã gửi lên server vẫn tiếp tục chạy.")
        sys.exit(130)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"\nKhông thể tiếp tục: {error}")
        print("Có thể mở lại NOP_BAI và chọn phiên cũ để tiếp tục.")
        sys.exit(2)
