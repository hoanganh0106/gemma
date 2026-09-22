# Tối ưu K2: thử bỏ scalar input RMS, giữ weight và hai term f8

Hãy triển khai và đánh giá giả thuyết dưới đây đến khi có kết luận có bằng chứng. Đây là thí nghiệm; không mặc định phép biến đổi đúng hoặc nhanh hơn trước khi kiểm tra.

## Bối cảnh môi trường đã kiểm tra ngày 2026-09-18

- Máy Windows, shell mặc định PowerShell; build Rust/Furiosa qua WSL Bash. Workspace chính: `D:\Project\furiosa-opt-gemma4-12B-main`, tương ứng `/mnt/d/Project/furiosa-opt-gemma4-12B-main`.
- Remote origin: `https://github.com/hoanganh0106/furiosa-opt.git`.
- Base yêu cầu: `origin/stage1-candidate-noop-reshape`, đã resolve thành `23d4493e7f5b37d49e729a66f27fa63ea87235c1`. Dùng commit cố định này; xác minh ref trước khi bắt đầu.
- Workspace chính hiện ở nhánh `kernel2-optimization`, HEAD `cae1a94`, có nhiều thay đổi, file xóa và file untracked của người dùng. Không dùng working tree này làm base thí nghiệm và không reset/restore/clean các thay đổi đó.
- Đã có worktree `D:\Project\furiosa-opt-gemma4-qkv-drop-input-rms` (WSL: `/mnt/d/Project/furiosa-opt-gemma4-qkv-drop-input-rms`), nhánh `qkv-drop-input-rms`, HEAD `23d4493`. File `src/device/qkv/xnorm8.rs` trong đó đang modified, chưa commit. Đọc status/diff và artifact hiện có trước; tiếp tục phần phù hợp, không ghi đè hoặc tạo thí nghiệm trùng. Nếu diff không liên quan, dùng worktree mới tách từ commit cố định và giữ nguyên worktree cũ.
- WSL có `/home/hoanganh/.cargo/bin/cargo`, `furiosa-arena`; `cargo-furiosa-opt 0.6.0`. Toolchain repo: `nightly-2026-05-01`, rustc quan sát: `1.97.0-nightly (f53b654a8 2026-04-30)`.
- Phần cứng được đo qua Arena. Không mặc định WSL có NPU local, quyền chọn chip hoặc CPU affinity trên worker. Các job Arena trước đã chạy thành công; kiểm tra lại truy cập hiện tại.
- Đọc AGENTS.md áp dụng. Khi repo có `.codegraph/`, dùng CodeGraph trước khi tìm/đọc implementation; kiểm tra phạm vi index vì workspace chứa nhiều bản code. Không tự index repo chưa có index.
- Gọi Bash thông qua `wsl.exe bash -lc ...`; tránh để PowerShell nội suy biến Bash. Nếu script CRLF gây lỗi, chuẩn hóa bản sao script dùng cho thí nghiệm sang LF, ghi nhận việc này.

## Ba phiên bản và bằng chứng có sẵn

Đọc `D:\Project\furiosa-opt-gemma4-12B-main\STAGE1_THREE_VERSION_COMPARISON.md`, ưu tiên phần đính chính correctness K2 đầu file. Đọc thêm OPTIMIZATION-NOTES.md ở đúng base và ledger/log liên quan; các kết luận cũ phải được đối chiếu source/artifact.

- Merge baseline: `bc9a3cc225604fff119bc6858939becb8a9ccfae`, trước đây đo tại repo con `furiosa-opt-gemma4-12B-stage1-merged-arena`.
- Merge candidate/base của thí nghiệm này: K1 bỏ reshape no-op, vẫn dùng weight và hai term f8 ở K2. Xác minh nội dung commit `23d4493` thay vì suy ra chỉ từ tên nhánh.
- All3: snapshot `all3_job53647_81909_38271_210595/source`, README ghi base `024b5e2`; K1 chia tile 104+16, còn merge dùng tile 120. K3 giống source trong phép so sánh đã làm.
- K2 All3 bỏ input weight, dùng `BitAnd(-0.5)` và một term. Fixture dựng `x ≈ signs / input_rms_weight` khiến đường này PASS nhưng không bảo toàn ngữ nghĩa trên input tổng quát. Không lấy All3 K2 làm mẫu hay làm frontier correctness. Không giả định local branch `stage1-all3-job53647` tồn tại nếu chưa kiểm tra.

Kết quả lịch sử 10 lần, thứ tự K2/K1/K3:

- Merge baseline median: `86239 / 45946.5 / 221496`; min: `83861 / 43646 / 214171`.
- Merge candidate median: `83910 / 42295.5 / 217392`; min: `80155 / 40159 / 212351`.
- All3 median: `88847.5 / 46380 / 221036`; min: `76677 / 38432 / 212391`. K2 này chỉ là timing của implementation không đúng tổng quát.

Ba nhóm đo theo từng đợt, không xen kẽ ABBA. Chúng không chứng minh bỏ reshape K1 làm K2/K3 nhanh hơn; source K2/K3 giữa merge baseline và candidate giống nhau. QKV candidate từng dao động `80155–109759`, vì vậy không mặc định nhiễu chỉ ±1–3K.

Mục tiêu người dùng là điểm tốt nhất qua nhiều lần nộp. Báo cáo cả median lẫn min/tail và bộ ba cycles của cùng một job; không ghép min khác job thành điểm giả. Việc benchmark Arena được phép; không tự official-submit hoặc publish/push.

## Phạm vi sửa

- Chỉ sửa implementation trong `src/device/qkv/xnorm8.rs`.
- Không sửa `src/ops.rs`, tên/signature hàm `#[device]`, K1, K3, tolerance hoặc fixture chính thức.
- Được thêm tài liệu kết quả trong `OPTIMIZATION-NOTES.md` và công cụ/test chẩn đoán độc lập trong thư mục evidence của thí nghiệm. Những file phụ này không thuộc payload kernel.

## Giả thuyết toán học và giới hạn

Xác minh Q, K, V đều qua head RMSNorm sau projection; V dùng norm không weight. Nếu bỏ qua lượng tử hóa, làm tròn và epsilon, projection tuyến tính và head normalization triệt tiêu scalar dương chung `1/rms(x)`.

Tuy nhiên với `HN_eps(y) = y / sqrt(mean(y²) + eps)` và c > 0:

`HN_eps(c*y) = y / sqrt(mean(y²) + eps/c²)`.

Vì vậy không có đẳng thức chính xác ở epsilon cố định, và sai số không bị chặn đơn giản bởi `eps=1e-6`. Head có năng lượng nhỏ hoặc input magnitude cực đoan có thể khác đáng kể. Việc lượng tử hóa f8/bf16 trước head norm còn tạo sai số phụ thuộc scale.

Giữ phép nhân `input_rms_weight`: weight theo phần tử thay đổi hướng vector. Giữ cả hai term f8; không dùng signfast hoặc shortcut đặc thù fixture. Hai term cũng không tự bảo đảm đúng với mọi input khi có overflow/underflow.

## Thay đổi cần thử

Trong `normalize_everywhere_f8`:

1. Xóa chuỗi `mean_square`, `reduced`, `rms`, reshape của rms và `rms_vrf`.
2. Xóa `.vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)` trong pass `normalized`.
3. Giữ `xpool`: weight tile 1 ghi trước, x tile 0 ghi sau; giữ `weight_vrf`.
4. Giữ hai phép nhân với `weight_vrf` và `X_PRESCALE`; đầu ra pass normalized sẽ là `x ⊙ w * X_PRESCALE` ở f32.
5. Giữ `q1_pieces`, residual `normalized - q1`, và cả hai `gather_f8` vào tensor có trục Term.
6. Xóa `H_F32`, import EPS nếu hết dùng. Sửa doc-comment để nói rõ scale cancellation là xấp xỉ chịu ảnh hưởng epsilon và quantization.

Xác minh `X_PRESCALE` thực tế trước khi dùng giá trị 8. Với giá trị 8 và f8e4m3 max 448, điều kiện `|x ⊙ w| ≤ 56` chỉ là kiểm tra miền giá trị cho term chính, không chứng minh độ chính xác tổng thể. Term dư có thể bằng 0 hoặc nhỏ tùy ý; không khẳng định residual luôn ≥ 2^-6. Kiểm tra overflow, subnormal/underflow và sai số head năng lượng thấp. Không áp đặt miền input mới chỉ bằng comment nếu spec không cho phép.

Nếu bằng chứng cho thấy phép bỏ RMS không đạt correctness trên miền input yêu cầu, ghi REJECT hoặc giới hạn chưa giải quyết; không đổi fixture/tolerance để làm nó PASS.

## Build, correctness và schedule

- Lưu diff, commit base, hash source/binary/package, toolchain, fixture, runner và flags cho A/B. A = nguyên commit `23d4493`; B = đúng delta thí nghiệm. Build vào target/package riêng, tránh dùng nhầm binary qua `--no-build` chọn file mới nhất.
- Build theo workflow đang có, ví dụ `cargo furiosa-opt test --release --test test_kernels --no-run --message-format=json`. Có thể chạy `cargo furiosa-opt test --release --test test_kernels` nếu môi trường thực sự hỗ trợ execution; build PASS không đồng nghĩa NPU correctness PASS.
- Chạy fixture chính thức trên Arena cho cả ba kernel. Giữ tolerance hiện hành, xác minh từ tests (prompt ban đầu kỳ vọng atol 0.04, rtol 1e-2). Ghi max|Δ|, mean|Δ|, phần trăm within tolerance của Q/K/V đối với reference, và B−A nếu harness cho phép.
- Thêm kiểm tra chẩn đoán độc lập: x và w không bị ràng buộc `x=signs/w`, nhiều seed, magnitude nhỏ/lớn, head projection có năng lượng nhỏ, vùng gần giới hạn f8. Dùng reference đầy đủ input RMSNorm. Giữ fixture chính thức nguyên vẹn; tách kết quả chẩn đoán khỏi benchmark chính thức.
- Kiểm tra `cargo furiosa-opt --help`/subcommand help trước khi dùng lệnh dump schedule. Cú pháp dự kiến cần xác minh: `cargo furiosa-opt compile ops::sliding_project_qkv --exact --dump-schedule target/schedules/qkv_after.json`.
- Dump A/B với cùng cấu hình; đếm instruction, cross-resource wait, makespan. Mức giảm 4 pass và ≥1 wait là giả thuyết, không phải kết quả bắt buộc. Schedule không tự chứng minh speedup.
- Tìm `dmalign.py` trước khi chạy; hiện chưa xác minh script này có trong checkout. Nếu có, đọc usage và kiểm tra DM lớn ≥200 KB/512-byte alignment theo quy tắc thật của toolchain. Nếu không có, dùng thông tin allocator/schedule tương đương và ghi phần chưa xác minh; không bịa công cụ hoặc kết quả.

## Arena và đo ghép cặp

Runner hiện có: `scripts/rngd_test.sh`, entrypoint `scripts/rngd/remote_entrypoint.sh`, fixture `ref/fixtures.safetensors`.

Fixture đã dùng trước có SHA-256 `9801C5732D9F5BEF89D544631C37F74DAE1603DF35B67A2CFDD6377915B7599E`, có tại workspace chính và repo merge. Nếu worktree thiếu fixture, copy đúng artifact này và kiểm tra hash.

Server trong phiên trước giới hạn timeout job 70 giây, script mặc định 1800 giây bị từ chối. Dùng `RNGD_TIMEOUT=70` khi giới hạn còn áp dụng. Script dùng cùng biến này cho thời gian poll phía client; client hết thời gian không đồng nghĩa job fail. Kiểm tra lại `furiosa-arena status/logs` trước khi submit thay thế để tránh chạy thừa.

Entrypoint hiện đặt `TUC_PROFILE_LEVEL=info` mặc định và chưa pin NUMA. Biến ở máy client không mặc nhiên truyền sang worker. Cấu hình profile/NUMA trong entrypoint package chung cho A/B; chỉ dùng `numactl --cpunodebind=0 --membind=0` nếu worker hỗ trợ và node 0 phù hợp. Ghi cấu hình thực tế.

- Sau correctness, chạy tối thiểu 7 block ABBA = 28 executions = 14 cặp A/B; ưu tiên 10 block = 40 executions = 20 cặp. Mỗi block ghép A đầu với B kế tiếp và B tiếp theo với A cuối. Ghi quy tắc pairing trước khi xem kết quả.
- Ưu tiên cùng chip/worker và cùng runner. Nếu Arena không cho đảm bảo cùng chip, ghi device metadata và giới hạn đó; không gọi là same-chip khi chưa có bằng chứng.
- Ghi job ID, timestamp, worker/chip nếu có, correctness, cycles cả ba kernel, hash package. Không tự loại outlier; giữ raw samples và lý do cho mọi mẫu không hợp lệ.
- Tính paired delta K2 `B−A`: median và trimmed mean 10% hai đầu, bỏ floor(0.1*n) delta nhỏ nhất/lớn nhất. Báo thêm mean/min/max, phân vị thấp, số cặp B thắng và độ bất định nếu tính được.
- Điều kiện tối thiểu để cân nhắc chấp nhận: correctness đầy đủ PASS, paired median và trimmed mean đều âm. Nếu mức cải thiện chưa tách khỏi nhiễu thì INCONCLUSIVE. Kỳ vọng −2K đến −5K chỉ là giả thuyết; không suy ra chi phí mỗi pass/host round-trip khi chưa có trace.
- Do mục tiêu lấy điểm cao nhất, đánh giá thêm kết quả tốt nhất với ngân sách số lần bằng nhau, dùng bộ ba của cùng job và công thức score đã xác minh. Không coi best-of-N là bảo đảm cho các lần sau.
- Nếu trace được hỗ trợ và nằm trong giới hạn worker, chạy một job trace mỗi bản riêng khỏi tập timing info để kiểm tra dependency từ x load tới gather; không suy luận DMA song song chỉ từ span chồng nhau.

## Bàn giao

- Giữ kernel delta trong đúng một file; giữ nguyên thay đổi có sẵn của người dùng.
- Commit riêng trên nhánh thí nghiệm từ base, sau khi kiểm tra diff. Message gợi ý: `qkv: test dropping input RMS scalar while retaining weight and two f8 terms`.
- Ghi trong OPTIMIZATION-NOTES.md: parent/candidate hashes, giả thuyết và giới hạn toán học, diff, build/correctness, dữ liệu chẩn đoán, schedule/alignment, toàn bộ job/package, số cặp, median/trimmed mean, min/max và kết luận PROMOTE/REJECT/INCONCLUSIVE.
- Nếu không thắng hoặc correctness chưa đạt, giữ commit/dữ liệu trên nhánh thí nghiệm, không merge. Không tự push/publish hoặc nộp bài chấm chính thức.
- Nếu thiếu phần cứng/công cụ, hoàn tất phần phân tích/build/test độc lập hữu ích và chỉ rõ phép kiểm tra còn thiếu. Không báo hoàn thành đo khi mới build.
- Chỉ cân nhắc thay X_PRESCALE hoặc hướng tiếp theo sau kết luận thí nghiệm này; không bỏ q2 trong phạm vi hiện tại.
