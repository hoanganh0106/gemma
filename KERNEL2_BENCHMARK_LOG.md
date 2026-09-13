# Kernel 2 benchmark log

Workspace: D:\Project\furiosa-opt-gemma4-12B-main

Mọi lần test đều thực hiện quy trình: compile `cargo furiosa-opt compile` sau `rm -rf target/furiosa-opt`, sau đó benchmark 3 lần với `RNGD_TIMEOUT=70 ./scripts/rngd_test.sh --no-build`.

## E5 early partial add

- Source change: đổi cách cộng partial: `p01 = add_partials(p0,p1)` ngay sau `p0,p1`, rồi `p23`, rồi cộng `p01,p23`.
- Backup: `backups/kernel2_current_20260912_131556`
- Cycles: 339376, 332864, 336175
- Median: 336175ttttttttttttfaqrrrrrrrr
- Status: PASS 3 kernels, candidate giữ để so sánh thêm.

## E6 mapping16 partial reduction (CHUNK = 768)

- Source change: `const CHUNK: usize = 768` trong `project_output`.
- Backup: `backups/kernel2_e6_tile768_20260912_1351`
- Cycles: 339690, 341519, 339213
- Median: 339690
- Status: PASS 3 kernels, không có cải thiện so với baseline E5.

## E7 swap DMA/Main order in `output_partial`

- Source change: đổi thứ tự đường dữ liệu trong `output_partial` để đưa path x trước path weight.
- Backup: `backups/kernel2_e7_swap_dma_main_20260912_1352`
- Cycles: 335979, 333951, 343409
- Median: 335979
- Status: PASS 3 kernels, biên dao động nhỏ hơn chút nhưng chưa rõ lợi thế ổn định.

## E8 alternate contract order

- Source change: đổi thứ tự gọi `contract_outer / contract_time / contract_packet`.
- Backup: `backups/kernel2_e8_alt_contract_order_20260912_1400`
- Cycles: 340408, 335585, 340573
- Median: 340408
- Status: PASS 3 kernels, không tốt hơn E5.

## E9 LaneMode Sequential

- Source change: thử `LaneMode::Sequential` cho `contract_lane`.
- Backup: `backups/kernel2_e9_lane_seq_20260912_1345`
- Cycles: 335749, 339477, 336469
- Median: 336469
- Status: PASS 3 kernels, không thắng áp đảo so với E5.

## E10 split 30/30/15 (cải tiến `vector_narrow_split`)

- Source change: thử phân tách `vector_narrow_split` theo nhánh 30/30/15 trên `apply_output_channel_scale`.
- Backup: `backups/kernel2_e10_split3015_20260912_1400`
- Cycles: 336264, 341772, 342147
- Median: 341772
- Status: PASS 3 kernels.

## E11 cast/fma order in `apply_output_channel_scale`

- Source change: thử thay đổi thứ tự cast/FMA theo phương án đã thử trước đó.
- Backup: `backups/kernel2_e11_mul1_scale_20260912_1404`
- Cycles: 343024, 343119, 347365
- Median: 343119
- Status: PASS 3 kernels, chậm hơn các bản khác.

## E12 streaming add pattern

- Source change: pipeline partial add: ghi partial trước rồi cộng dồn.
- Backup: `backups/kernel2_e12_streaming_add_20260912_1354`
- Cycles: 334915, 337977, 337085
- Median: 337085
- Status: PASS 3 kernels, có cải thiện nhẹ so với baseline E5 nhưng không vượt trội ổn định.

## E13 CHUNK = 1536

- Source change: `const CHUNK: usize = 1536`.
- Backup: `backups/kernel2_e13_chunk1536_20260912_1356`
- Cycles: 343334, 338423, 336128
- Median: 338423
- Status: PASS 3 kernels, không vượt trội.

## E14 `contract_lane` split 1#4

- Source change: thử `contract_lane` với `m![H % 120], m![1 # 4]`.
- Backup: `backups/kernel2_e14_lane4_20260912_1358`
- Cycles: 341121, 338437, 340695
- Median: 340695
- Status: PASS 3 kernels.

## E15 rollback baseline

- Source change: quay về bản baseline đã benchmark được an toàn.
- Backup: `backups/kernel2_e15_rollback_baseline_20260912_1402`
- Cycles: 340935, 335080, 337406
- Median: 337406
- Status: PASS 3 kernels.

## Nhận xét nhanh

- Trong số các biến thể đã benchmark từ E5–E15, bản tốt nhất theo trung vị hiện tại là: `E7 (335979)`.
- Số đo có nhiễu nhẹ giữa các lần đo, nên cần 2–3 lần chạy lặp lại khi chốt bản cuối cùng.
- Kiểm tra khớp lại snapshot: một số backup hiện diện dưới dạng file riêng (`backups/kernel2_*.`), trong đó chỉ có `e6_tile768` và `e13_chunk1536` chắc chắn có khác biệt rõ về `CHUNK`; các backup còn lại gần như giống snapshot gốc E5 (có thể do lần chép snapshot trước/sau chưa trộn trượt đúng biến thể). Nếu muốn khóa chốt nghiêm ngặt, cần chạy lại E7/E8/E9/E10/E11/E12/E14/E15 trực tiếp trong một phiên WSL để xác nhận lại cycle mới.

## Danh sách phương pháp còn lại cần chốt (theo nghiên cứu)

### Tier 1
- `E1-CHUNK-2048` (đã có backup đã sửa bug `kernel2_e1_chunk2048_fixed_20260912_1600`)
- `E2-FP8-STREAM-DECODE` (đã là code hiện tại của `output_partial`, đã được benchmark trong nhiều phiên trước đó)

### Tier 2
- `E2-SCALE-RMS-FUSION` (gộp scale trước RMSNorm trong helper mới)
- `E3-F32-PARTIAL-ACCUM` (dùng accumulator F32 cho partial add)
- `E4-TILE-SWEEP` (`CHUNK` sweep 1024/1536/2048/4096 theo khả dụng compiler)

### Cần chạy lại để khóa chốt (đã có backup)
- `E5`…`E15`: đã có log, nhưng nên chạy lại 1 vòng 3 lần nữa khi có WSL để xác nhận lại phương pháp `E7,E8,E9,E10,E11,E12,E14,E15`.

## Chạy sweep tự động (để không quên)

Đã tạo script mới: `scripts/run_kernel2_methods.sh`.

- Mặc định: mỗi phương pháp 3 lần run, `RNGD_TIMEOUT=70`, `rm -rf target/furiosa-opt` + `cargo furiosa-opt compile` trước khi test.
- Mỗi phiên ghi vào `target/kernel2_bench_sweeps/<timestamp>/`.
- Sau khi chạy xong, copy kết quả `summary.csv` vào phần log tương ứng và dán lại vào cuối file này.

```sh
chmod +x scripts/run_kernel2_methods.sh
KERNEL2_RUNS=3 KERNEL2_TIMEOUT=70 ./scripts/run_kernel2_methods.sh
```

## Phiên bản chạy đủ tất cả methods

- Danh sách phương pháp đầy đủ: [scripts/kernel2_method_backlog.csv](scripts/kernel2_method_backlog.csv)
- Script tổng quát mới nhất: [scripts/run_kernel2_all_methods.sh](scripts/run_kernel2_all_methods.sh)

```sh
chmod +x scripts/run_kernel2_all_methods.sh
KERNEL2_RUNS=3 KERNEL2_TIMEOUT=70 ./scripts/run_kernel2_all_methods.sh
```

Script sẽ ghi:
- `target/kernel2_bench_sweeps/<timestamp>/summary.csv` (trạng thái từng run)
- `target/kernel2_bench_sweeps/<timestamp>/<E?.log>` (log chi tiết mỗi method)

Trong vòng sweep này:
- `missing_backup` = method chưa có bản backup/implementation để chạy ngay (E2-FUSION)
- `compile` = bước compile lỗi
- `submit_error` = lỗi submit/arena trong lúc nộp job
- `done` = chạy đủ các vòng, có thể trích cycles từ log

## Bảng theo dõi còn lại (điểm cần chốt/đã chờ)

| Method | File backup | Trạng thái | Ghi chú |
|---|---|---|---|
| E1-CHUNK-2048 | `kernel2_e1_chunk2048_fixed_20260912_1600` | ✅ Sẵn sàng chạy | Sửa bug duplicate `p1` |
| E2-FP8-STREAM-DECODE | `kernel2_e2_fp8_stream_20260912_1415` | ✅ Sẵn sàng chạy | Đang dùng làm baseline variant |
| E2-SCALE-RMS-FUSION | - | ⏸️ Chưa implement | Chưa có biến thể mã |
| E3-F32-PARTIAL-ACCUM | `kernel2_e3_f32_partial_20260912_1500` | ✅ Sẵn sàng chạy | F32 partial add |
| E4-TILE-SWEEP | `kernel2_e4_tile4096_20260912_1500` | ✅ Sẵn sàng chạy | `CHUNK=4096` |
| E5-EARLY-PARTIAL-ADD | `kernel2_current_20260912_131556` | ✅ Đã chạy (trước đây) | Median 336175 |
| E6-CHUNK-768 | `kernel2_e6_tile768_20260912_1351` | ✅ Đã chạy (trước đây) | Median 339690 |
| E7-SWAP-DMA-MAIN | `kernel2_e7_swap_dma_main_20260912_1352` | ✅ Đã chạy (trước đây) | Median 335979 |
| E8-CONTRACT-ORDER-ALT | `kernel2_e8_alt_contract_order_20260912_1400` | ✅ Đã chạy (trước đây) | |
| E9-LANEMODE-SEQUENTIAL | `kernel2_e9_lane_seq_20260912_1345` | ✅ Đã chạy (trước đây) | |
| E10-SPLIT-3015 | `kernel2_e10_split3015_20260912_1400` | ✅ Đã chạy (trước đây) | |
| E11-MUL1-SCALE | `kernel2_e11_mul1_scale_20260912_1404` | ✅ Đã chạy (trước đây) | |
| E12-STREAMING-ADD | `kernel2_e12_streaming_add_20260912_1354` | ✅ Đã chạy (trước đây) | |
| E13-CHUNK-1536 | `kernel2_e13_chunk1536_20260912_1356` | ✅ Đã chạy (trước đây) | |
| E14-LANE-4 | `kernel2_e14_lane4_20260912_1358` | ✅ Đã chạy (trước nữa) | |
| E15-ROLLBACK | `kernel2_e15_rollback_baseline_20260912_1402` | ✅ Đã chạy (trước đây) | |

## Phiên thử tiếp (đang bị chặn hạ tầng)

- Thời điểm: 2026-09-12 ~16:00
- Lỗi môi trường: `Access is denied` khi gọi `bash`/`wsl`
  - `wsl -l -v` và `bash -lc ...` đều trả về `Wsl/EnumerateDistributions/Service/E_ACCESSDENIED`.
  - `bash` và `wsl` đều không khởi tạo được môi trường WSL.
- Hệ quả: không thể chạy `KERNEL2_RUNS=3 KERNEL2_TIMEOUT=70 ./scripts/run_kernel2_all_methods.sh` ở thời điểm này.
- Khi môi trường mở lại, chạy đúng lệnh sau đây rồi đính kèm:
  - `KERNEL2_RUNS=3 KERNEL2_TIMEOUT=70 ./scripts/run_kernel2_all_methods.sh`
- Ưu tiên method để thử tiếp:
  - `E1-CHUNK-2048` (backup vừa sửa)
  - `E2-FUSION` (chờ implement)
  - `E3-F32-PARTIAL-ACCUM`
  - `E4-TILE-SWEEP`

## Kernel2 sweep session 20260912_194450

- Created: `target/kernel2_bench_sweeps/20260912_194450`
- Runs per method: `3`
- Timeout per job: `70s`
- Script: `scripts/run_kernel2_all_methods.sh`

### New method runs (append by hand to this section)

method,run,status,cycles,note
E1,1,pass,
E1,2,pass,
E1,3,pass,
E1,all,done,339716
E2,1,pass,
E2,2,pass,
E2,3,pass,
E2,all,done,339678
E2-FUSION,manual,missing_backup,,
E3,compile,failed,,
E4,1,pass,
E4,2,pass,
E4,3,pass,
E4,all,done,338435
E5,manual,backup_not_found,,
E6,1,pass,
E6,2,pass,
E6,3,pass,
E6,all,done,334481
E7,1,pass,
E7,2,pass,
E7,3,pass,
E7,all,done,339751
E8,1,pass,
E8,2,pass,
E8,3,pass,
E8,all,done,345210
E9,1,pass,
E9,2,pass,
E9,3,pass,
E9,all,done,338637
E10,1,pass,
E10,2,pass,
E10,3,pass,
E10,all,done,337809
E11,1,pass,
E11,2,pass,
E11,3,pass,
E11,all,done,343369
E12,1,pass,
E12,2,submit_error,,
E12,3,pass,
E12,all,incomplete,
E13,1,pass,
E13,2,pass,
E13,3,pass,
E13,all,done,338889
E14,1,pass,
E14,2,pass,
E14,3,pass,
E14,all,done,341721
E15,1,pass,
E15,2,pass,
E15,3,pass,
E15,all,done,336862


### E16 — E2 FP8 stream + early partial add
- Backup: `backups/kernel2_e2_plus_early_partial_20260912_201356` (created 2026-09-12T20:13:56)
- Single combined candidate for follow-up validation: preserves E2 direct FP8 lookup and moves p0+p1 reduction before p2/p3.
- Pending Arena measurement.


### E16 — E2 FP8 stream + early partial add
- Source: src/device/sliding/projection.rs
- Backup: backups/kernel2_e2_plus_early_partial_20260912_201356/projection.rs
- Change: reduce p0+p1 before launching p2/p3; FP8 direct stream remains unchanged.

### E16 measurement reported by user
- Run 1: 340099 cycles (Arena result pasted by user; job ID and full PASS log not provided).
- One measurement only; improvement and stability are not established.
- E2 sweep summary was 339678 cycles: this run is 421 cycles (0.124%) higher; different sessions and insufficient repeats prevent a firm comparison.
### E16 — three user-reported cycle measurements
- Cycles: 340099, 341931, 346356.
- Median: 341931; mean: 342795.33; min/max: 340099 / 346356.
- Range: 6257 cycles (1.83% of median).
- Compared with the earlier E2 sweep summary (339678), E16 median is 2253 cycles higher (+0.66%). Compared with E6 summary (334481), it is 7450 higher (+2.23%). These are cross-session comparisons; the earlier aggregation and individual runs have not been verified here.
- No improvement demonstrated. No measurement is excluded as an outlier. Full PASS logs and job IDs were not provided with these cycle-only reports.

### E17 — weight DMA first
E17: weight DMA before x TRF preparation. Parent: current E16. Only block order changed inside output_partial. Tile 1024, offsets 0/1024/2048/3072, FP8 stream, reduction and BF16 rounding preserved. Not yet compiled or measured. Parent and candidate snapshots in D:\Project\furiosa-opt-gemma4-12B-main\backups\kernel2_e17_weight_dma_first_20260912_202129.
### E17 — three user-reported measurements
- Cycles: 332923, 339041, 340905.
- Median: 339041; mean: 337623; range: 7982 (2.35% of median).
- E16 median: 341931. E17 median is 2890 cycles lower (0.85%).
- E2 previous sweep summary: 339678; E17 median is 637 lower (0.19%), insufficient evidence of a reliable gain across sessions.
- Keep all observations; no outlier discarded. Full PASS logs and job IDs have not been supplied for these reports. Stability and correctness confirmation remain pending.

### E18 — reverse partial reduction order
- Backup: D:\Project\furiosa-opt-gemma4-12B-main\backups\kernel2_e18_reverse_reduction_20260912_202528
- Single change: compute p2+p3 before p0+p1; tile, FP8 path, DMA order, and arithmetic unchanged.
- Pending Arena measurement.


### E19 — contract packet 2
- Backup: D:\Project\furiosa-opt-gemma4-12B-main\backups\kernel2_e19_packet2_20260912_203132
- Single change: contract_packet m![1] -> m![2]; all tiling, reduction, DMA, and numeric operations unchanged.
- Pending Arena measurement.


### E19 result
- Compile failed: OutPacket 2 is not a valid contraction.
- Reverted source to E18 parent.

### E20 — sequential lane mode
- Backup: D:\Project\furiosa-opt-gemma4-12B-main\backups\kernel2_e20_lane_sequential_20260912_203549
- Single change: sliding output contract lane Interleaved -> Sequential.
- Pending Arena measurement.


### E21 tile 512
# E21: tile 512, eight partials
Parent: E17 candidate. Hypothesis: smaller tiles reduce live memory pressure.
Change: output projection uses eight 512-wide tiles and seven balanced partial additions, replacing four 1024-wide tiles and three additions.
Coverage: [0,512), [512,1024), [1024,1536), [1536,2048), [2048,2560), [2560,3072), [3072,3584), [3584,4096).
FP8 stream, weight-first DMA, Interleaved lane mode and other functions are unchanged from E17.
Additional BF16 rounding boundaries require official correctness validation.
Not compiled or submitted yet. before_e20.rs preserves the source previously on disk; parent_e17.rs is the comparison source.
Backup: D:\Project\furiosa-opt-gemma4-12B-main\backups\kernel2_e21_tile512_20260912_204108

### E22 — E17 + F32 partial accumulation
- Backup: D:\Project\furiosa-opt-gemma4-12B-main\backups\kernel2_e22_e17_f32_20260912_204612
- Change: output partial cast BF16 -> F32; E17 schedule preserved.
- Pending compile/Arena measurement.

