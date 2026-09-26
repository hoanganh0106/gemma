# Đo workspace hiện tại trên RNGD — Arena 78017

Job `78017`, tên `research-current-sdk081-three-seeds`, chạy 23/09/2026 06:51:51–06:51:58 UTC. Trạng thái `SUCCEEDED`, exit code **0**. Wrapper bật `FURIOSA_OPT_PROFILE=info`, dùng fixture ba seed mới và không pin CPU.

## Correctness và cycles

**15/15 dòng so sánh output PASS**, mỗi dòng `within tol=100.00%`; cuối log `all 3 tests passed`.

- K1: `[102213, 90703, 91270]` → median **91.270 cycles**.
- K2: `[44471, 44409, 44622]` → median **44.471 cycles**.
- K3: `[224052, 220925, 219896]` → median **220.925 cycles**.

Max absolute error trên các output/case: K1 0,03125; K2 0,03125; K3 0,01172. K3 vẫn PASS dù max error lớn hơn atol 0,01 vì tolerance còn thành phần `0.01 × abs(expected)`. Log hiện không báo max normalized tolerance ratio; không suy biên an toàn từ max absolute error đơn lẻ.

Với tuple baseline API quan sát lúc 06:49:42 UTC `250514 / 404633 / 3703473`, geometric mean speedup tính từ tuple này là **7,4808490442**. Đây là **điểm tham khảo tính lại từ public Arena**, chưa phải kết quả MOA submission hay cam kết rerun của BTC. Nghiên cứu không gửi submission MOA.

So với một dòng leaderboard SDK 0.8.1 cùng snapshot có tuple `73366 / 33020 / 201948`, khoảng cách từng median là 19,62% / 25,75% / 8,59% nếu lấy workspace này làm mẫu số. Đạt đồng thời tuple tham khảo đó sẽ tăng score 22,38%. Khác job, điều kiện có thể khác và source của dòng đó chưa xác minh; đây là mốc tham khảo về quy mô khoảng cách, không phải phép A/B.

## Static không phải hardware

Lịch compiler mới của cùng source là `42539 / 24718 / 110520`, thấp hơn số Arena với tỷ lệ hardware/static lần lượt khoảng `2,146 / 1,799 / 1,999`. Không có hệ số chuyển đổi chung, và một mẫu không đủ tách ảnh hưởng clock, startup, runtime scheduling hay môi trường. Dùng static để phân tích dependency và sàng lọc; dùng harness runtime để quyết định score.

## Định danh

- Binary `test_runtime`: SHA256 `021b9009886682add70e20ee3481bf5bcd9711b4b3535d1cba0821c46a37a39d`.
- Fixture `fixtures.safetensors`: SHA256 `338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359`.
- Source: `source-hashes.json`, toolchain/harness manifest: `evidence-hashes.json`.
- Log nguyên bản: [arena-78017.log](arena-78017.log).
- Trạng thái cuối: [arena-78017-status.json](arena-78017-status.json).

Đây là một invocation đầy đủ của harness, mỗi kernel có median ba input case. Chưa đo lặp A/B để khẳng định độ ổn định. Kết quả chỉ xác minh public fixture của frozen tuple trên; không chứng minh đúng với mọi đầu vào có thể có.
