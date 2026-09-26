# Nộp MOA tuần tự đến điểm mục tiêu

## Cách dễ nhất: nhấp đúp

Trong thư mục repo, nhấp đúp **NOP_BAI.cmd**. Menu tiếng Việt sẽ hướng dẫn:

1. Chọn phiên cũ để tiếp tục, hoặc tạo phiên mới.
2. Chọn phiên bản (snapshot Arena 78017, workspace, thư mục delivery hoặc dán đường dẫn).
3. Nhập điểm mục tiêu và số lần nộp tối đa; nhập 0 để không giới hạn.
4. Chọn **Bắt đầu nộp**. Có lựa chọn chỉ kiểm tra cấu hình trước.

Lựa chọn phiên bản/điểm/giới hạn được nhớ cho lần sau. Script tự mở WSL,
kiểm tra mỗi 5 phút và lưu log. Nhấn Ctrl+C để dừng, mở lại để tiếp tục.
Khi tiếp tục một phiên, giới hạn là tổng số lần nộp của phiên đó, không phải số lượt bổ sung.
Giữ cửa sổ mở và máy thức trong khi chạy.

Chạy bằng WSL/Linux, Python 3 và `moa-submitter` đã đăng nhập.
Script sử dụng điểm `Score:` chính thức do CLI trả về (hiện làm tròn 4 số thập phân).
Ngưỡng áp dụng cho các bài của **phiên chạy này**, không lấy điểm lịch sử của phiên bản khác.

## Chạy

Từ WSL, trong repo:

```bash
bash scripts/submit_leader_loop.sh \
  --source research/furiosa-score-20260923/baseline-source \
  --target-score 8.0 \
  --state-dir submission_staging/moa-78017-target8 \
  --poll-seconds 300
```

`8.0` là ví dụ; thay bằng điểm muốn đạt. `--source` là thư mục phiên bản muốn nộp,
có `Cargo.toml`, `src/ops.rs` và `src/device/`. Có thể chọn `--source .` cho workspace,
hoặc thư mục delivery/snapshot khác. Với bản 78017, đối chiếu source với manifest
`research/furiosa-score-20260923/source-hashes.json` trước khi chọn nếu snapshot đã thay đổi.
Script không tự chứng minh tính đúng luật hay correctness của phiên bản được chọn.

Từ PowerShell, gọi cùng lệnh qua WSL:

```powershell
wsl bash -lc 'cd /mnt/d/Project/furiosa-opt-gemma4-12B-main && bash scripts/submit_leader_loop.sh --source research/furiosa-score-20260923/baseline-source --target-score 8.0 --state-dir submission_staging/moa-78017-target8'
```

Thêm `--dry-run` để kiểm tra cấu hình/source mà không gọi mạng, không tạo phiên chạy,
không gửi bài. Chỉ khi bỏ `--dry-run` script mới bắt đầu nộp.

## Cách hoạt động

1. Sao chép source nộp, Cargo và toolchain sang `<state-dir>/source/`, lưu SHA256.
   Các lần nộp trong cùng phiên dùng bản đóng băng này, kể cả khi workspace gốc được sửa.
2. Kiểm tra `status --all`; nếu tài khoản có bài chưa kết thúc thì đợi.
3. Nộp một bài, lưu ID ngay; đọc status và toàn bộ log mỗi 300 giây mặc định.
4. Khi `completed`, ghi điểm/kết quả. Đạt `score >= target` thì dừng.
   Chưa đạt thì đợi 30 giây và nộp tiếp sau khi kiểm tra tài khoản lần nữa.

Không có giới hạn thời gian chờ build/evaluate. Mất mạng khi đọc status/log chỉ khiến
script đợi rồi thử đọc lại. Có khóa chung cho các phiên script chạy cùng Linux user;
khóa không ngăn người khác hoặc CLI thủ công nộp đồng thời trên máy khác.

Tùy chọn:

- `--max-submissions 20`: tối đa 20 lần gửi trong state-dir, vẫn chờ bài cuối kết thúc.
  Mặc định `0` nghĩa là tiếp tục cho đến khi đạt điểm hoặc người dùng dừng.
- `--poll-seconds 300`: khoảng cách giữa các lần đọc status/log khi bài đang chạy.
- `--cooldown-seconds 30`: khoảng nghỉ tối thiểu sau khi nhận kết quả trước lần gửi kế.
- `--continue-on-failure`: tiếp tục sau `failed/rejected/cancelled`; mặc định dừng.
- `--max-consecutive-failures 3`: dừng sau ba lỗi liên tiếp khi bật tiếp tục sau lỗi.
- `--cli /path/to/moa-submitter`: chọn CLI cụ thể.

## Dừng, tiếp tục và đổi phiên bản

Nhấn Ctrl+C để dừng script. Bài đang chạy trên server vẫn tiếp tục.
Chạy lại cùng lệnh/cùng `--state-dir` để theo dõi ID cũ; không nộp lại ID đang chờ.
Có thể đổi `--target-score` khi tiếp tục. Muốn dùng source phiên bản mới, chọn một
`--state-dir` mới; chạy lại cùng state-dir luôn dùng snapshot cũ.

`state.json` lưu source hashes, ID đang chạy, số lần gửi, lịch sử và điểm cao nhất.
`logs/` lưu phản hồi submit, status gần nhất và log đầy đủ gần nhất cho từng ID.

Nếu timeout hoặc dừng đúng lúc gửi, script giữ trạng thái `submitting` và **không tự
gửi lại**, vì server có thể đã nhận bài. Kiểm tra `moa-submitter status --all` và log,
xác định ID của lần gửi đó, rồi thêm `--recover-id ID` vào cùng lệnh để tiếp tục.
ID phải thuộc tài khoản và chưa có trước lần gửi chưa rõ kết quả. Người dùng cần đối
chiếu thời gian/source để chọn đúng bài. Nếu đã xác nhận server không nhận bài nào,
giữ lại state/log cũ và bắt đầu với state-dir mới.

## Kiểm tra offline

```bash
python3 -m unittest discover -s scripts -p test_moa_submit_loop.py -v
```

Các test giả lập CLI: tuần tự, ngưỡng điểm, giới hạn lượt, mạng lỗi, resume,
timeout khi submit, source bị sửa và parser lỗi. Không nộp lên dịch vụ thật.
