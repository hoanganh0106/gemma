# Nghiên cứu và tối ưu sliding_attention_output trên Furiosa RNGD

## 1. Tóm tắt dành riêng cho người làm Kernel 2

Kernel xử lý vector attention 4096 phần tử BF16, O projection 3840×4096 FP8, RMSNorm toàn cục trên 3840 phần tử và residual add. Mã ban đầu chia chiều giảm thành bốn tile 1024; mỗi tile chuẩn bị activation vào TRF, lookup FP8→BF16, contraction rồi cộng ba lần qua tensor BF16.

Đã thử tile 2048 nhưng compiler từ chối vì allocation SRAM 125,829,120 byte; thay đổi đó đã được hoàn tác. Candidate hiện tại nối FP8 table lookup trực tiếp vào stream contraction với tile 1024, giữ nguyên cây partial và các biên làm tròn BF16. Candidate mới đã compile và benchmark Arena; các job có một outlier cần theo dõi.

Ưu tiên tiếp theo: xác nhận thêm paired runs cho candidate FP8 stream; sau đó scale+RMSNorm fusion bằng helper riêng; tiếp theo là accumulator F32 hoặc tối ưu các kernel khác trong src/device. Giữ nguyên per-channel scale trong RMS statistic, RMSNorm toàn H, residual sau normalization và API.

## 2. Những gì đã biết và chưa biết

### PROVIDED CONTEXT

Shapes cố định: x=[8,2,256] (4096 BF16), o_weight=[3840,4096] FP8 e4m3, hai vector scale/RMS BF16, residual BF16. Correctness gate: atol 0.05, rtol 1e-2. Scoring dùng device cycles thật và geometric mean ba kernel. Chỉ được đổi implementation trong src/device và thân device function.

### FACT

RNGD là Tensor Contraction Processor, công bố 256 TFLOPS BF16, 512 TFLOPS FP8, HBM3 1.5 TB/s và 256 MB SRAM ([RNGD overview](https://developer.furiosa.ai/latest/en/overview/rngd.html)). Tensor Unit gồm Fetch, Switch, Collect, Contraction, Vector, Cast, Transpose, Commit; Main chạy toàn pipeline, Sub thường chuẩn bị TRF/VRF, DMA chạy DMA riêng ([Computing Tensors](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/index.html)). Contraction dùng operand stationary trong TRF và operand stream; TRF preparation thường ở Sub ([Contraction Engine](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/index.html)). Quick Start công bố TRF 8 KiB/lane, VRF 8 KiB/slice, DM 512 KiB/slice ([Kernel Design](https://developer.furiosa.ai/furiosa-opt/book/quick-start/kernel-design.html)). Schedule dump là kế hoạch tĩnh, không thay thế device benchmark ([Kernel Optimizer](https://developer.furiosa.ai/furiosa-opt/book/tools/kernel-optimizer.html)).

TCP paper mô tả input broadcast, input-buffer reuse và compiler tìm lowered shape/tactic ([TCP ISCA 2024](https://furiosa.ai/download/FuriosaAI-tensor-contraction-processor-isca24)). Tài liệu Tiling yêu cầu tile vừa bộ nhớ, căn flit và chừa tài nguyên overlap ([Tiling](https://developer.furiosa.ai/furiosa-opt/book/kernel-examples/tiling.html)).

### INFERENCE

Tile 2048 có thể giảm setup, commit và partial reduction; tile lớn hơn cũng có thể làm giảm overlap hoặc tăng resource pressure. Scale/RMSNorm hoặc normalize/residual fusion có thể bỏ DM round-trip nếu compiler giữ stream trên chip, nhưng phải kiểm rounding và live range.

### UNKNOWN

Baseline/candidate cycles, variance, mapping/bank placement, thực tế HBM→DM, LUT latency/setup reuse, spill, inter-slice switch cost, compiler acceptance của mapping 2048/4096, và mọi rounding boundary chưa được xác nhận. Cần commit, toolchain, fixture checksum, lệnh build/test/dump schedule, timeline Main/Sub/DMA, transfer size, contraction mapping, temporary lifetime và span aggregation từ môi trường RNGD.

## 3. Bản đồ chi phí theo từng stage

Byte count là lower bound logic, không phải DMA đo được.

| Stage | Chi phí logic | Rủi ro/bottleneck cần đo |
|---|---:|---|
| Load x | 4096 BF16 = 8192 B HBM | DMA và layout |
| Broadcast | tới 256 slice × 8192 B logic | Switch/ring; không suy ra HBM round-trip |
| FP8 weight | 3840×4096 = 15,728,640 FP8 B | HBM, LUT, decoded BF16 staging |
| Contraction | 15,728,640 MAC | Main CE mapping và TRF reuse |
| Partial add | baseline 3 full-H passes; candidate 1 | Vector/Cast/Commit dependency |
| Scale | 3840 BF16 = 7680 B, F32 multiply rồi BF16 | Một full-H pass trước RMS |
| RMSNorm | square/reduce F32, inter-slice reduce, sqrt, divide, weight | global reduction critical path |
| Residual | 8 tiles × 480, residual load/store tổng 15,360 B | Sub VRF load, Main add, Commit |

Scale phải nằm trước RMS statistic; residual không được nằm trong statistic.

## 4. Kiến thức Furiosa/RNGD liên quan trực tiếp

Fetch đọc DM vào stream và Commit ghi về DM; Switch đổi Slice; Vector xử lý elementwise/reduction; Contraction chạy trên TRF. DM có hierarchy cluster/DMN/slice/bank và access pattern ảnh hưởng throughput ([Memory Performance](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html)). DMA có packet và phân hoạch theo cluster/slice; HBM↔DM thường là ứng viên bandwidth bottleneck ([DMA Engine](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html)). Fetch Adapter cung cấp table lookup/cast nhưng table lookup có giới hạn context ([Fetch Adapter](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/fetch-adapter.html)). Main/Sub/DMA chỉ overlap khi không có resource/dependency conflict ([Schedule](https://developer.furiosa.ai/furiosa-opt/book/scheduling/schedule.html)). Vì vậy source-level helper boundary không chứng minh launch hay HBM round-trip; chỉ schedule mới cho biết.

Candidate 2048 dùng activation tile 2048×2 B = 4096 B, dưới 8 KiB/lane. Weight tile trên HiddenRows có 32×2048 FP8 = 64 KiB logic trước decode, dưới DM 512 KiB/slice theo shape suy ra; allocation/lane thực tế vẫn UNKNOWN. Không áp dụng giả định CUDA warp, occupancy hoặc shared-memory bank.

## 5. Paper và implementation đáng nghiên cứu

- TCP ISCA 2024: broadcast, input-buffer reuse, compiler tactic search; áp dụng cho broadcast_sliding_heads và output_partial. DIRECTLY TRANSFERABLE ở nguyên lý, chưa chứng minh nhanh.
- Furiosa Contraction/Tiling/Memory docs: GEMV mapping, TRF stationary operand, tile/flit/resource constraints; DIRECTLY TRANSFERABLE, phải khớp furiosa-opt-std 0.6.0.
- CUTLASS custom epilogue ([official docs](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/001_gemm_with_fused_epilogue.html)): scale/residual fusion và FP32 accumulator; CONCEPTUALLY TRANSFERABLE, CUDA implementation không port trực tiếp.
- SGLang Gemma4 fused Triton RMSNorm+residual ([source](https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/layernorm/gemma4_fused_ops.py)): cùng semantics residual sau norm; CONCEPTUALLY TRANSFERABLE, không có RNGD speed claim.
- FLUTE LUT GEMM ([paper](https://arxiv.org/abs/2407.10960)): LUT/dequant staging; CONCEPTUAL và có thể vi phạm cấm prepacking/cache. Không sao chép code; chỉ chuyển nguyên lý.

## 6. Bảng so sánh và xếp hạng phương pháp

| Method | Core idea | Expected speedup | Memory impact | Difficulty | Hardware dependency | Risk | Evidence/source | Suitability |
|---|---|---|---|---|---|---|---|---|
| FP8 lookup→contraction stream (đã benchmark) | Bỏ decoded BF16 DM temporary, giữ tile 1024 | Khoảng 16.7% theo median 4 job; cần tiếp tục paired runs | Giảm temporary decoded weight khoảng 245,760 B/tile/layout logic | Trung bình | Main Fetch Adapter + CE | Biến thiên job/outlier | Fetch Adapter/CE + Arena jobs 23804/23806/23808/23810 | Cao |
| Tile 2048 | 2 partial thay 4, 1 add thay 3 | Unknown — benchmark required. | Allocation SRAM 125,829,120 B trong compile thử | Trung bình | TRF/DM mapping | Đã compile fail | Tiling/CE | Đã loại |
| Tile sweep 1024/2048/4096 | tìm mapping hợp lệ tối ưu | Unknown — benchmark required. | 4096 gần TRF 8 KiB | Trung bình | allocator/compiler | pressure/spill | Tiling | Cao sau E1 |
| F32 partial accumulator | cast một lần | Unknown — benchmark required. | tăng storage/VRF | Cao | Vector/VRF | rounding/live range | Vector docs | Trung bình |
| Scale+RMS fusion | giữ scale và F32 trước statistic | Unknown — benchmark required. | có thể bỏ full-H pass | Cao | Main/VRF chain | sai rounding/statistic | code + pipeline | Trung bình |
| Normalize+residual fusion | bỏ normalized temporary | Unknown — benchmark required. | giảm DM nhưng live range tăng | Cao | Cast/Vector ordering | đổi BF16 boundary | pipeline + SGLang | Trung bình |
| Bỏ broadcast/remap | phân phối lại output rows | Unknown — benchmark required. | thay đổi switch/weight movement | Rất cao | Switch/layout | sai contract | TCP paper | Thấp |
| Approximate RMS/prepack | đổi semantics hoặc preprocessing | Unknown — benchmark required. | không rõ | Rất cao | rules/numeric | correctness/legality | không đủ | Không nên làm |

Tier 1: baseline/schedule/cycle và E2 FP8 stream. Tier 2: scale+RMS fusion riêng; F32 partial. Tier 3: normalize+residual, TRF reuse, remap broadcast. Tier 4: CUDA tricks, hardcode fixture, partial RMS, đổi format/prepack/cache, đổi tolerance.

## 7. Kế hoạch thí nghiệm cho coding agent

### Experiment 0: baseline

Ghi commit/toolchain/fixture; chạy build/test và dump schedule exact; xác nhận correctness; đo paired nhiều lần; lưu median/spread; tách host/compile khỏi span::npu; đánh dấu UNVERIFIED nếu không có RNGD.

### Experiment ID: E1-CHUNK-2048
Priority tier: Tier 1
Evidence/source: Furiosa Contraction Engine, Tiling, src/device/sliding/projection.rs
Transferability classification: DIRECTLY TRANSFERABLE (cơ chế), chưa chứng minh nhanh.

Observation: baseline có bốn output_partial và ba add_partials.
Hypothesis: hai tile 2048 giảm hai partial passes và setup boundary.
Code region: project_output/output_partial.
Single change: CHUNK=2048; offsets 0/2048; một add_partials.
Prerequisites or legality checks: compile mapping, TRF/DM, flit alignment, signature bất biến.
Expected mechanism: ít Fetch/Collect/Commit/vector add.
Expected schedule change: partial tree ngắn hơn, contraction mỗi tile dài hơn.
Expected device-cycle effect: Unknown — benchmark required.
Metric to measure: schedule makespan, paired span::npu cycles, Main/Sub/DMA.
Correctness checks: official atol 0.05, rtol 1e-2, finite output.
Benchmark case: public fixed shape.
Shared-code impact: sliding projection only.
Regression checks: build và ba Stage 1 kernels.
KEEP if: correctness pass và cycle gain vượt noise lặp lại.
REVERT if: compile/correctness fail hoặc chậm reproducibly.
INVESTIGATE if: schedule tốt nhưng cycles không cải thiện.

### Experiment ID: E2-FP8-STREAM-DECODE
Priority tier: Tier 1
Evidence/source: [Fetch Adapter](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/fetch-adapter.html), [Contraction Engine](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/index.html), local furiosa-opt-std 0.6.0 API và Arena jobs 23804/23806/23808/23810.
Transferability classification: DIRECTLY TRANSFERABLE (API đã kiểm tra cục bộ; cycle evidence trên fixture đã thu được).
Observation: output_partial decode FP8→BF16 rồi commit decoded tensor vào DM và fetch lại cho contraction.
Hypothesis: stream lookup BF16 trực tiếp vào contract_outer sẽ bỏ commit/fetch và temporary decoded weight, không đổi tile hoặc số lần partial.
Code region: src/device/sliding/projection.rs::output_partial.
Single change: bỏ DmTensor weight BF16 trung gian; chain fetch_table_lookup→collect→contract_outer trên weight_f8.
Prerequisites or legality checks: lookup phải ở Main; mapping fetch raw FP8 dùng packet 32 byte và collect 16 phần tử; compiler phải chấp nhận stream type.
Expected mechanism: giảm DM write/read và allocation; FP8 decode vẫn xảy ra một lần khi stream qua Main.
Expected schedule change: biến mất decoded-weight Commit/Fetch nodes; CE dependency có thể sát hơn vào LUT.
Expected device-cycle effect: giảm nếu decoded-DM movement là critical path; kết quả Arena đã có ở phụ lục.
Metric to measure: compile resource, schedule makespan, paired span::npu cycles, Main/DMA overlap.
Correctness checks: official sliding_attention_output atol 0.05/rtol 1e-2; full three-kernel regression.
Benchmark case: public fixed shape.
Shared-code impact: sliding projection only.
KEEP if: compile, PASS correctness và cycle gain lặp lại vượt noise.
REVERT if: compile fail, numerical fail hoặc cycles chậm hơn.
INVESTIGATE if: schedule giảm nhưng cycles không giảm.

### Experiment ID: E2-SCALE-RMS-FUSION
Priority tier: Tier 2
Evidence/source: projection.rs + shared/rmsnorm.rs.
Transferability classification: UNKNOWN WITHOUT RNGD EXPERIMENT.
Observation: scale là full-H pass ngay trước RMSNorm.
Hypothesis: helper sliding riêng giữ F32 scale và giảm materialization.
Code region: helper mới trong src/device/sliding; không sửa generic trước.
Single change: scale rồi square/reduce/normalize với statistic của vector đã scale.
Prerequisites or legality checks: chứng minh rounding và global reduction.
Expected mechanism/schedule: bỏ một full-H movement nếu chain được.
Expected device-cycle effect: Unknown — benchmark required.
Metric: cycles, reducer critical path, temporary lifetime.
Correctness/benchmark: official fixture + diagnostic random; paired runs.
Shared-code impact: không ảnh hưởng consumer khác.
KEEP/REVERT/INVESTIGATE: cùng tiêu chí E1.

### Experiment ID: E3-F32-PARTIAL-ACCUM
Priority tier: Tier 2/3
Evidence/source: add_partials + Vector docs.
Transferability classification: UNKNOWN WITHOUT RNGD EXPERIMENT.
Observation: ba BF16 round-trip add.
Hypothesis: accumulator F32 và một cast cuối giảm conversion.
Single change: F32 partial/add path.
Prerequisites: API output type, VRF/DM capacity.
Expected effect: Unknown — benchmark required; rounding khác.
Correctness: official + stress; KEEP chỉ khi nhanh và pass.

### Experiment ID: E4-TILE-SWEEP
Priority tier: Tier 3
Evidence/source: Tiling/Memory Performance.
Transferability classification: DIRECTLY TRANSFERABLE as measurement method.
Single change: thử riêng 1024, 2048, 4096; mapping tương ứng; compile/schedule/cycle từng candidate.
KEEP/REVERT/INVESTIGATE theo E1.

### Experiment ID: E5-EARLY-PARTIAL-ADD
Priority tier: Tier 2
Evidence/source: TCP paper về overlap context và code project_output.
Transferability classification: UNKNOWN WITHOUT RNGD EXPERIMENT.
Observation: source hiện tạo cả p0..p3 trước khi cộng.
Hypothesis: cộng p0+p1 ngay sau p1 có thể chồng Vector work với contraction p2/p3 và rút ngắn lifetime temporary.
Code region: project_output trong src/device/sliding/projection.rs (candidate baseline 4 tile).
Single change: chỉ đổi thứ tự xây p01/p23; không đổi tile, precision hoặc toán học.
Prerequisites or legality checks: compiler không serialize thêm dependency và vẫn giữ kết quả tree.
Expected mechanism: overlap Sub/Main/Vector, giảm live range DM.
Expected schedule change: p01 lifetime kết thúc sớm hơn; có thể không đổi nếu queue serialize.
Expected device-cycle effect: Unknown — benchmark required.
Metric to measure: schedule lifetime/context overlap và paired span::npu cycles.
Correctness checks: official fixture atol 0.05, rtol 1e-2.
Benchmark case: public fixed shape.
Shared-code impact: sliding projection only.
Regression checks: ba Stage 1 kernels.
KEEP if: correctness pass và cycle gain reproducible; REVERT nếu không gain hoặc schedule xấu; INVESTIGATE nếu chỉ schedule thay đổi.

## 8. Dữ liệu profiling cần thu thập

Makespan/context occupancy để biết Main, Sub hay DMA là critical path; LUT node/lifetime để phát hiện decode/staging lặp; transfer size/địa chỉ DM để tìm padding/non-contiguous/bank contention; TRF/VRF/DM allocation để phát hiện spill; Switch/inter-slice region để định lượng broadcast; reducer/sqrt dependency; temporary lifetime để biết có DM round-trip; paired cycles và variance. Nếu counter không có, dùng schedule JSON và controlled one-change experiments; không nhận diện bottleneck từ utilization đơn lẻ.

## 9. Những hướng không nên tốn thời gian

Không dùng GPU warp/tensor-core/occupancy assumptions, không hardcode fixture ±1, không đổi FP8 format hoặc external cache/prepacking khi chưa có rule, không bỏ scale khỏi RMS, không đưa residual vào statistic, không normalize chunk-local. Tile lớn/fusion/double buffering có thể làm chậm vì resource pressure, live range, mất overlap hoặc compiler serialization. Schedule ngắn hơn không đủ để KEEP nếu device cycles không giảm.

## 10. Ba việc nên làm tiếp theo

1. Biên dịch và dump schedule E1-CHUNK-2048, rồi chạy paired RNGD cycles cùng correctness.
2. Nếu E1 pass, thu timeline Main/Sub/DMA và temporary lifetime để quyết định E2.
3. Lưu baseline/candidate JSON và cycle variance; chỉ bắt đầu E3/E4 khi nhiễu đo được đủ nhỏ.

## Phụ lục — Những gì đã sửa trong workspace

- Đã thử src/device/sliding/projection.rs với CHUNK 2048 nhưng compiler báo thiếu SRAM (allocation 125,829,120 byte); thay đổi đã được hoàn tác.
- Candidate hiện tại trong output_partial bỏ decoded BF16 DmTensor trung gian và nối FP8 lookup trực tiếp vào contraction; tile 1024, bốn partial, cây cộng và các stage sau giữ nguyên.
- scripts/rngd_test.sh không nằm trong thay đổi cuối cùng vì README loại scripts khỏi phạm vi grading.
- Baseline Arena đã PASS; Kernel 2 median ba lần là 406471 cycles (404031, 406471, 409657).
- Candidate Arena jobs: 23804 = 407875, 23806 = 340282, 23808 = 334422, 23810 = 337012 cycles; cả ba kernel đều PASS correctness. Median cả bốn candidate = 338647 cycles, thấp hơn baseline median khoảng 16.7%; median ba job thấp (23806/23808/23810) = 337012, thấp hơn khoảng 20.6%. Job 23804 là outlier và phải được theo dõi trong paired runs tiếp theo.

