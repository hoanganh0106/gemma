# Stage 1: so sánh ba phiên bản

Ngày tổng hợp: 2026-09-18

## Kết luận ngắn

Ba phiên bản được so sánh:

1. **All3 Job 53647** — snapshot tại `all3_job53647_81909_38271_210595/source`, base được ghi là `hyun/stage1-optimized` commit `024b5e2`.
2. **Merge baseline** — repo `furiosa-opt-gemma4-12B-stage1-merged-arena`, commit `bc9a3cc225604fff119bc6858939becb8a9ccfae`.
3. **Merge candidate** — merge baseline cộng thay đổi xóa một `reshape` no-op cuối K1 trong `src/device/sliding/output31.rs`.

Theo **median 10 lần**, merge candidate tốt nhất trên cả K2, K1 và K3. Theo chiến lược **nộp nhiều lần và giữ điểm cao nhất**, All3 có tail tốt hơn ở K2 và K1, còn candidate chỉ nhỉnh hơn All3 40 cycles ở K3 best-case.

All3 và merge **không phải cùng source**. Trong 30 file dưới `src/device/`, có 27 file giống nội dung và 3 file khác: `qkv/xnorm8.rs`, `qkv/proj8.rs`, `sliding/output31.rs`. `src/ops.rs` cũng khác ở phần nối K2. K3 dùng cùng implementation.

Fixture của All3 và merge giống hệt nhau, SHA-256:

`9801C5732D9F5BEF89D544631C37F74DAE1603DF35B67A2CFDD6377915B7599E`

## Khác biệt source

### K2 — `sliding_project_qkv`

**All3 Job 53647**

- `normalize_everywhere_f8` nhận `_rms_weight` nhưng không đọc tensor weight này.
- Đường tiền xử lý dùng phép `BitAnd` với `-0.5`, sau đó all-gather một biểu diễn f8 đơn.
- Projection nhận tensor dạng `m![H]`; contraction dùng một lane/term, không có bước cộng hai term.
- Các file liên quan: `all3_job53647_81909_38271_210595/source/src/device/qkv/xnorm8.rs`, `qkv/proj8.rs`, và `src/ops.rs`.

**Merge baseline/candidate**

- Thực hiện RMSNorm với `input_rms_weight`.
- Tạo khai triển hai term f8: `q1 = f8(x)` và `q2 = f8(x - q1)`.
- Projection nhận `m![Term, H]`; contraction xử lý cả hai term rồi cộng chúng bằng intra-slice reduction.
- Baseline và candidate giống nhau ở K2.

Ý nghĩa: đây là hai thuật toán/schedule K2 khác nhau, không phải cùng binary chỉ dao động timing.

### K1 — `sliding_attention_output`

**All3 Job 53647**

- Chia weight mỗi slice thành hai tile `104 + 16` hàng.
- Phát hai DMA load và hai contraction: `contract_tile_a` và `contract_tile_b`.
- Thiết kế này từng tạo các mẫu tail rất thấp, gồm K1 `38,432` cycles trong retest 10 lần.

**Merge baseline**

- Tải một tile đủ `120` hàng bằng một DMA command.
- Chạy một contraction `contract_tile` cho toàn bộ tile.
- Có một `reshape` cuối trước khi store ra HBM; type trước và sau reshape thực tế giống nhau.

**Merge candidate**

- Giống merge baseline về tile `120` và toàn bộ thuật toán.
- Chỉ xóa `reshape` no-op cuối hàm trước HBM store.

Ý nghĩa: khác biệt lớn giữa All3 và merge là schedule `104+16` so với `120`. Khác biệt baseline và candidate chỉ là một điểm chờ/lệnh reshape không đổi dữ liệu.

### K3 — `decoder_feedforward`

- Cả ba bản dùng cùng `src/device/shared/ffn7.rs` và cùng thân `decoder_feedforward`.
- Không có khác biệt nội dung K3 giữa All3 và merge trong phép so sánh source hiện tại.
- Chênh lệch cycle K3 giữa các job vì vậy phù hợp với nhiễu/tình trạng phần cứng hơn là thay đổi implementation.

## Thống kê Arena 10 lần

Đơn vị: cycles; thấp hơn là tốt hơn. Độ lệch chuẩn dùng population standard deviation trên 10 mẫu.

| Phiên bản | Kernel | Min | Median | Mean | Max | Std. dev. |
|---|---|---:|---:|---:|---:|---:|
| All3 Job 53647 | K2 | 76,677 | 88,847.5 | 87,118.1 | 91,823 | 4,967.3 |
| All3 Job 53647 | K1 | 38,432 | 46,380 | 45,371.2 | 49,998 | 3,959.9 |
| All3 Job 53647 | K3 | 212,391 | 221,036 | 220,472.8 | 228,565 | 3,929.0 |
| Merge baseline | K2 | 83,861 | 86,239 | 87,952.8 | 98,139 | 4,746.2 |
| Merge baseline | K1 | 43,646 | 45,946.5 | 46,324.2 | 49,467 | 2,188.9 |
| Merge baseline | K3 | 214,171 | 221,496 | 220,652.6 | 225,867 | 3,368.1 |
| Merge candidate | K2 | 80,155 | 83,910 | 88,511.8 | 109,759 | 10,659.5 |
| Merge candidate | K1 | 40,159 | 42,295.5 | 43,206.5 | 47,969 | 2,921.7 |
| Merge candidate | K3 | 212,351 | 217,392 | 218,389.4 | 225,755 | 4,398.2 |

## So sánh theo mục tiêu

### Nếu ưu tiên độ ổn định/median

Merge candidate là frontier hiện tại:

- K2: `83,910`, thấp hơn merge baseline khoảng 2.7% và All3 khoảng 5.6%.
- K1: `42,295.5`, thấp hơn merge baseline khoảng 7.9% và All3 khoảng 8.8%.
- K3: `217,392`, thấp hơn merge baseline khoảng 1.9% và All3 khoảng 1.6%.

### Nếu Arena giữ điểm tốt nhất qua nhiều lần nộp

Min/tail quan trọng hơn median:

- K2 tốt nhất: **All3**, `76,677` so với candidate `80,155` và baseline `83,861`.
- K1 tốt nhất: **All3**, `38,432` so với candidate `40,159` và baseline `43,646`.
- K3 tốt nhất: **candidate**, `212,351`; All3 gần như hòa ở `212,391` (chênh 40 cycles).

Do đó không có một frontier duy nhất cho cả hai mục tiêu:

- **Frontier median/ổn định:** merge candidate.
- **Frontier săn best-run/leaderboard:** All3 Job 53647 cho K2 và K1; candidate chỉ có lợi thế K3 rất nhỏ trong tập 10 mẫu này.

Record lịch sử duy nhất được lưu trong README của snapshot All3 là `81,909 / 38,271 / 210,595` tại Job 53647. Đây là một job lịch sử, không phải median; retest 10 lần không tái tạo đồng thời cả ba con số đó.

## Raw samples và job IDs

Thứ tự mỗi mẫu: `K2/K1/K3`.

### All3 Job 53647 snapshot

- Jobs: `56604`, `56609`, `56614`, `56618`, `56621`, `56622`, `56625`, `56628`, `56631`, `56636`.
- Samples:
  - `87532/49452/223309`
  - `86284/46078/221231`
  - `90187/49998/218259`
  - `91164/48800/219811`
  - `78872/38432/212391`
  - `90947/46682/220939`
  - `89230/38642/228565`
  - `91823/45448/221447`
  - `76677/42726/217643`
  - `88465/47454/221133`
- Correctness: 10/10 jobs PASS cả ba kernel.

### Merge baseline

- Jobs: `56548`, `56552`, `56555`, `56559`, `56562`, `56565`, `56569`, `56572`, `56576`, `56579`.
- Samples:
  - `85021/44089/215137`
  - `96045/49467/225867`
  - `86133/48517/221413`
  - `87941/46082/214171`
  - `87011/45811/220723`
  - `84887/49172/222963`
  - `84145/48103/222585`
  - `86345/43826/222269`
  - `98139/43646/221579`
  - `83861/44529/219819`
- Correctness: 10/10 jobs PASS cả ba kernel.

### Merge candidate bỏ reshape no-op

- Jobs: `56467`, `56469`, `56472`, `56475`, `56480`, `56482`, `56484`, `56487`, `56490`, `56493`.
- Samples:
  - `108939/46203/219159`
  - `87877/47266/222337`
  - `109759/40592/213429`
  - `83649/40609/218135`
  - `81203/43876/215405`
  - `84171/40579/216649`
  - `85713/47969/224939`
  - `80155/40159/215735`
  - `83267/44097/225755`
  - `80385/40715/212351`
- Correctness: 10/10 jobs PASS cả ba kernel.

## Giới hạn của kết luận

- Ba nhóm được chạy liên tiếp trong các cửa sổ thời gian gần nhau nhưng không phải ABBA xen kẽ từng job, nên vẫn có thể có drift theo host/phần cứng.
- Best-of-10 là thống kê tail nhạy với số lần thử; tăng số lần nộp có thể đổi thứ tự best-run.
- Không được ghép min của từng kernel từ các job khác nhau thành một “job tổng hợp” giả. Điểm leaderboard phải lấy bộ ba cycle từ cùng một submission/job theo đúng công thức chấm.
- Chưa có official final submission hoặc publish nào được thực hiện trong các phép đo này.
