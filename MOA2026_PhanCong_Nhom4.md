```{=html}
<!-- Trang 1 -->
```
# MICRO 2026 -- NPU MODEL OPTIMIZATION COMPETITION

## BẢNG PHÂN CÔNG NHIỆM VỤ

## VÀ MÔ HÌNH VẬN HÀNH NHÓM

Kernel Optimization Round -- Gemma 4 12B trên FuriosaAI RNGD

Phạm vi Tối ưu ba kernel mục tiêu của Round 1: sliding_project_qkv,
sliding_attention_output, decoder_feedforward. Quy mô nhóm 04 thành
viên, tổ chức theo ownership ngang hàng và chuyên môn hóa sâu. Nguyên
tắc Shared system understanding + deep individual specialization +
evidence-based integration. Không áp dụng Không phân cấp "nhóm
trưởng/thành viên" trong phân công kỹ thuật; khác biệt nằm ở phạm vi
ownership và trách nhiệm bàn giao.

Tài liệu nội bộ phục vụ tổ chức kỹ thuật, phân công ownership, kiểm thử,
benchmark và tích hợp submission Round 1.

------------------------------------------------------------------------

```{=html}
<!-- Trang 2 -->
```
### Mục lục

1 Mục tiêu và phạm vi 2

2 Nguyên tắc phân công 2

3 Cơ cấu nhóm và vai trò tổng quan 3 3.1 Sơ đồ ownership và tích hợp . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

4 Phân công chi tiết 4 4.1 TV1 -- QKV Kernel Engineer . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . 4 4.2 TV2 -- Attention &
Layout Engineer . . . . . . . . . . . . . . . . . . . . . . . . . . 6
4.3 TV3 -- FFN / NVFP4 Kernel Engineer . . . . . . . . . . . . . . . . .
. . . . . . . . 7 4.4 TV4 -- Performance Verification & Integration
Engineer . . . . . . . . . . . . . . 8

5 Ma trận ownership và giao diện bàn giao 10

6 Quy trình kỹ thuật chuẩn cho mọi experiment 11 6.1 Experiment record
chuẩn . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

7 Quản lý mã nguồn và tích hợp 11 7.1 Branch model . . . . . . . . . . .
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 11 7.2 Quy tắc
commit . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
. . . . . 12 7.3 Continuous integration rule . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . 12

8 Sản phẩm đầu ra bắt buộc của toàn nhóm 12

9 Tiêu chí nghiệm thu và chỉ số đánh giá 13

10 Bản đồ kiến thức chung và chuyên môn hóa 14

11Tài liệu tham chiếu kỹ thuật 14

------------------------------------------------------------------------

```{=html}
<!-- Trang 3 -->
```
## 1. Mục tiêu và phạm vi

Mục tiêu của nhóm là tối ưu hiệu năng ba kernel được đánh giá trong
Kernel Optimization Round trên baseline Gemma 4 12B cho FuriosaAI RNGD,
đồng thời duy trì tính đúng đắn số học và khả năng tái lập kết quả. Mỗi
thành viên sở hữu một vùng kỹ thuật rõ ràng, nhưng toàn bộ nhóm cùng sử
dụng một baseline, một chuẩn kiểm thử và một cơ chế tích hợp thống nhất.
Ba cấu phần kỹ thuật chính: • QKV Projection Path: RMSNorm đầu vào,
broadcast hidden, Q/K/V projection, Q/K/V normalization, RoPE, ghi Q và
KV cache. • Attention Output Path: head broadcast, O projection,
post-attention RMSNorm và residual update. • Feed-Forward Path: pre-FF
RMSNorm, NVFP4 up/gate projection, GeGLU, down projection,
post-processing và residual/layer gate. Một cấu phần ngang hệ thống: •
Performance Verification & Integration: baseline authority, correctness
gate, static schedule analysis, RNGD benchmark, regression, merge và
submission candidate.

### Nguyên tắc cốt lõi

Tài liệu này tổ chức nhóm theo ownership kỹ thuật ổn định; nguồn lực có
thể hỗ trợ chéo khi cần, nhưng quyền sở hữu chuyên môn của từng vùng
không thay đổi. Mọi thay đổi chỉ được tích hợp khi có bằng chứng:
correctness PASS + đo hiệu năng + khả năng tái lập + mô tả tác động.

## 2. Nguyên tắc phân công

Nhóm gồm 04 vai trò ngang hàng về trách nhiệm dự án, khác nhau về chiều
sâu chuyên môn. Mỗi vai trò phải hiểu toàn bộ flow Round 1 ở mức hệ
thống, sau đó đào sâu một workstream riêng. Cấu trúc này tránh hai cực
đoan: "ai cũng biết một ít nhưng không ai đủ sâu" và "mỗi người làm một
mảnh hoàn toàn tách rời". \## 1. Một baseline chung: cùng commit, cùng
toolchain, cùng fixture, cùng command benchmark. \## 2. Một owner cho
mỗi vùng kỹ thuật: owner chịu trách nhiệm cuối cùng về hiểu biết,
implementation, experiment log và handoff của vùng đó. \## 3.
Shared-code có reviewer bắt buộc: thay đổi ở helper dùng chung phải được
đánh giá regression trên cả ba kernel. \## 4. Experiment là đơn vị công
việc: mỗi experiment phải có Observation → Hypothesis → ONE Change →
Verification. \## 5. Integration diễn ra liên tục: candidate tốt được
đưa vào integration branch sớm; không chờ "làm xong hết" mới ghép. \##
6. Evidence trước opinion: schedule là bằng chứng chẩn đoán; RNGD cycles
và correctness là bằng chứng quyết định.

------------------------------------------------------------------------

```{=html}
<!-- Trang 4 -->
```
*Bảng 1: Bảng định vị vai trò, độ khó và tầm quan trọng*

Mã Vai trò chuyên Ownership Độ khó Tầm Giá trị đối với submission môn
chính quan trọng sliding_project\_ TV1 QKV Kernel qkv 4/5 5/5 Sở hữu
trực tiếp một thành Engineer phần của geometric-mean score; chuyên gia
projection/contraction QKV. sliding_attention\_ TV2 Attention & output
3.5/5 5/5 Sở hữu trực tiếp một thành Layout Engineer phần score; đồng
thời kiểm soát layout/data movement dùng chung với projection path.
decoder\_ TV3 FFN / NVFP4 feedforward 5/5 5/5 Sở hữu trực tiếp một thành
Kernel Engineer phần score; workstream compute phức tạp nhất, nhiều
không gian tối ưu. TV4 Performance Correctness, 4.5/5 5/5 Bảo đảm
speedup của ba Verification & schedule, workstream thực sự có giá
Integration RNGD, trị khi ghép; giữ single Engineer integration, source
of truth cho release benchmark và release.

## 3. Cơ cấu nhóm và vai trò tổng quan

### Cách đọc bảng trên

Tầm quan trọng của cả bốn vai trò đều là 5/5. Ba kernel owner tác động
trực tiếp lên ba số hạng của score; vai trò Verification/Integration
quyết định liệu các speedup đó có còn đúng, tái lập và không regression
khi ghép thành submission hay không.

------------------------------------------------------------------------

```{=html}
<!-- Trang 5 -->
```
### 3.1. Sơ đồ ownership và tích hợp

                                              COMMON BASELINE
                                              commit + toolchain
                                                  + fixture +
                                                baseline cycles
                Projection / Attention pair                         FFN / Verification pair


            TV1                        TV2                                                 TV4
                                                               TV3
         QKV Kernel                 Attention                                         Performance
                                                           FFN / NVFP4
          Engineer                   & Layout                                          Verification
                                                              MLP &
       Q/K/V mapping               O-projection                                        Integration
                                                           contractions
       & contraction               & movement                                           & release


                                           INTEGRATION GATE
                                         Correctness + Schedule
                                          + RNGD + Regression


                                              RELEASE CANDIDATE
                                                Geometric-mean
                                               score + submission

*Hình 1: Bốn vai trò ngang hàng, chuyên môn hóa riêng nhưng hội tụ qua
một integration gate* chung.

## 4. Phân công chi tiết

### 4.1. TV1 -- QKV Kernel Engineer

### Vai trò

Sở hữu toàn bộ workstream sliding_project_qkv. Chịu trách nhiệm biến
pipeline QKV từ một "black box" thành một bản đồ tensor/dataflow có thể
profile, giải thích và tối ưu có kiểm soát. TV1 là người hiểu sâu nhất
team về Q/K/V projection, contraction mapping và reuse/staging của
hidden input trong sliding-attention path.

### Phạm vi kỹ thuật

• Entry point: src/ops.rs::sliding_project_qkv. • Primary code:
src/device/sliding/projection.rs, rmsnorm.rs, rope.rs. • Shared
dependencies: src/device/layout.rs, src/device/shared/rmsnorm.rs. •
Pipeline phải hiểu đầy đủ: input RMSNorm → broadcast hidden → Q/K/V
projection → Q/K/V norm → RoPE → Q output + KV cache write.

### Nhiệm vụ chi tiết

TT Công việc Sản phẩm / bằng chứng bắt buộc

------------------------------------------------------------------------

```{=html}
<!-- Trang 6 -->
```
1.1 Dựng tensor contract của QKV: shape, dtype, QKV_KERNEL_SPEC.md; sơ
memory tier, producer/consumer, output layout. đồ dataflow và bảng
shape/dtype. 1.2 Đọc và giải thích mapping hiện tại của Q, K, V; xác Bản
chú giải code theo định phần nào dùng chung input/staging và phần
function; danh sách nào độc lập. assumptions đã xác minh. 1.3 Dump và
phân tích baseline schedule của QKV. QKV_PROFILE.md; dominant context,
longest region, DMA/Main/SubContext observations. 1.4 Lập backlog
hypothesis theo bằng chứng, ưu tiên Danh sách hypothesis có mapping,
staging, contraction, scale path, mức ưu tiên và expected movement và
output/cache write. effect. 1.5 Triển khai experiment theo nguyên tắc
ONE Commit nhỏ, compile change. được, experiment record đầy đủ. 1.6 Chạy
correctness và performance validation cho PASS/FAIL rõ ràng; static
candidate. schedule trước/sau; RNGD cycles trước/sau. 1.7 Bàn giao
candidate tốt nhất cho integration. Best commit + handoff note +
shared-code impact.

### Sản phẩm đầu ra

• QKV kernel specification và baseline performance map. • Experiment log
có thể truy ngược từ code → hypothesis → measurement. • Best QKV
candidate đã PASS correctness và được đo trên RNGD. • Handoff note mô tả
mọi thay đổi vào shared code và rủi ro regression.

### Yêu cầu năng lực / kỹ năng sẽ hình thành

Tensor shape reasoning; contraction mapping; BF16/FP8 data path; TRF/DM
staging; schedule interpretation; Rust DSL của furiosa-opt; performance
experiment discipline.

### Giao diện phối hợp

• Phối hợp chặt với TV2 tại sliding/projection.rs và layout.rs; TV1 sở
hữu Q/K/V functions, TV2 sở hữu output projection functions. • Mọi thay
đổi shared helper phải báo TV2 và TV4 trước khi merge. • Khi cần review
contraction/mapping cho FFN, TV1 hỗ trợ TV3 nhưng không thay đổi
ownership FFN.

------------------------------------------------------------------------

```{=html}
<!-- Trang 7 -->
```
### Definition of Done -- TV1

Build PASS; QKV correctness PASS; baseline và optimized schedule được
lưu; RNGD speedup được đo; experiment có reasoning rõ; candidate được
TV4 reproduce; full regression sau integration không làm hỏng hai kernel
còn lại.

### 4.2. TV2 -- Attention & Layout Engineer

### Vai trò

Sở hữu workstream sliding_attention_output và trở thành chuyên gia của
team về layout, broadcast, reshape/switch và HBM↔DM movement trong
projection path. Vai trò này vừa có score trực tiếp, vừa có leverage
xuyên kernel vì nhiều helper layout/projection được dùng chung.

### Phạm vi kỹ thuật

• Entry point: src/ops.rs::sliding_attention_output. • Primary code:
output projection trong src/device/sliding/projection.rs. • Shared code:
src/device/layout.rs, src/device/shared/rmsnorm.rs, residual.rs. •
Pipeline: attention result → head broadcast → O projection →
post-attention RMSNorm → residual update → HBM.

### Nhiệm vụ chi tiết

TT Công việc Sản phẩm / bằng chứng bắt buộc

2.1 Dựng tensor/data-movement map của Attention ATTN_OUTPUT_SPEC.md;
Output. shape, dtype, layout và memory path. 2.2 Giải thích
broadcast_sliding_heads, output Annotated code notes; projection
chunking/partials và residual path. function ownership map. 2.3 Phân
tích schedule để phân biệt compute cost, ATTN_PROFILE.md. movement cost,
idle/dependency gaps. 2.4 Xây dựng hypothesis về layout, broadcast,
output Prioritized hypothesis split/chunk, partial reduction, transfer
boundary, backlog. RMSNorm/residual scheduling. 2.5 Triển khai và kiểm
thử từng experiment. Commit nhỏ + correctness + schedule + RNGD
evidence. 2.6 Theo dõi ảnh hưởng chéo của thay đổi Cross-kernel impact
projection.rs/layout.rs lên QKV. note. 2.7 Bàn giao candidate tối ưu cho
integration. Best commit + handoff note + regression risk.

------------------------------------------------------------------------

```{=html}
<!-- Trang 8 -->
```
### Sản phẩm đầu ra

• Attention Output baseline anatomy và movement/layout profile. • Best
candidate đã được đo, cùng experiment log. • Tài liệu shared-layout
behavior để TV1/TV3 có thể tham khảo khi chạm shared helper.

### Yêu cầu năng lực / kỹ năng sẽ hình thành

Tensor layout; broadcast/switch semantics; DMA/memory movement; output
contraction; partial reduction; shared-helper impact analysis;
cross-kernel regression awareness.

### Giao diện phối hợp

• TV1 và TV2 là projection/attention collaboration pair; cross-review
bắt buộc khi sửa projection.rs. • TV2 là reviewer chuyên môn layout nếu
TV3 hoặc TV4 đề xuất thay đổi data representation/movement dùng chung.

### Definition of Done -- TV2

Attention Output correctness PASS; optimized candidate có measurement;
mọi thay đổi shared projection/layout đã được QKV regression test;
integration branch reproduce được kết quả.

### 4.3. TV3 -- FFN / NVFP4 Kernel Engineer

### Vai trò

Sở hữu workstream decoder_feedforward. Đây là vùng compute phức tạp
nhất: NVFP4 packed weights, local/global scales, up/gate contractions,
GeGLU và down projection. TV3 là người hiểu sâu nhất team về MLP
dataflow, weight decode/scale path và contraction của FFN.

### Phạm vi kỹ thuật

• Entry point: src/ops.rs::decoder_feedforward. • Primary code:
src/device/shared/mlp.rs. • Shared code: src/device/shared/rmsnorm.rs,
residual.rs. • Pipeline: pre-FF RMSNorm → up/gate projection → GeGLU →
down projection → post- processing → residual/layer gate.

### Nhiệm vụ chi tiết

TT Công việc Sản phẩm / bằng chứng bắt buộc

3.1 Dựng FFN tensor contract và decomposition theo FFN_KERNEL_SPEC.md;
sơ stage. đồ up/gate/GeGLU/down.

------------------------------------------------------------------------

```{=html}
<!-- Trang 9 -->
```
3.2 Phân rã cost của weight movement, Cost map / unpack/decode, local
scale, global scale, source-to-schedule map. contraction, activation,
reduction và write-back. 3.3 Dump và đọc schedule baseline, xác định
dominant FFN_PROFILE.md. contexts và critical path. 3.4 Xây dựng
hypothesis về tiling/mapping, Prioritized hypothesis decode-scale
fusion, staging, contraction backlog. scheduling,
activation/down-projection interaction. 3.5 Triển khai ONE-change
experiments và kiểm soát Small commits + numerical drift. correctness
evidence. 3.6 Phối hợp TV4 để đánh giá schedule, reproducibility Joint
verification notes. và regression. 3.7 Bàn giao candidate FFN tốt nhất
cho integration. Best commit + handoff note.

### Sản phẩm đầu ra

• FFN architecture/performance map có thể giải thích từng stage. •
Experiment log cho NVFP4/MLP optimizations. • Best FFN candidate đã PASS
correctness và RNGD measurement.

### Yêu cầu năng lực / kỹ năng sẽ hình thành

NVFP4 data path; packed-weight handling; scale/dequant pipeline; GeGLU;
large contraction scheduling; numerical correctness; deep schedule
analysis.

### Giao diện phối hợp

• TV3 + TV4 là FFN/verification collaboration pair: TV3 sở hữu
implementation, TV4 sở hữu measurement/reproduction. • Khi vấn đề thiên
về contraction mapping, TV1 có thể cross-review; khi thiên về
layout/movement, TV2 có thể cross-review.

### Definition of Done -- TV3

FFN correctness PASS; candidate được TV4 reproduce; schedule/RNGD
evidence cho thấy lợi ích; numerical drift nằm trong official tolerance;
integration không gây regression chéo.

### 4.4. TV4 -- Performance Verification & Integration Engineer

### Vai trò

Sở hữu single source of truth cho baseline, correctness, schedule
archive, RNGD benchmark, experiment registry, integration branch và
release candidate. Đây là ownership kỹ thuật chứ không phải cấp bậc quản
lý. TV4 bảo đảm ba speedup độc lập vẫn đúng và có lợi khi ghép thành
submission hoàn chỉnh.

------------------------------------------------------------------------

```{=html}
<!-- Trang 10 -->
```
### Phạm vi kỹ thuật

• Baseline pinning: commit, toolchain, fixture, benchmark commands. •
Correctness harness và tolerance tracking. • Static schedule
capture/compare. • RNGD cycle measurement và reproducibility. •
Integration branch, full three-kernel regression, geometric-mean score
và submission candidate. • Secondary technical ownership của FFN để
tránh bus factor = 1.

### Nhiệm vụ chi tiết

TT Công việc Sản phẩm / bằng chứng bắt buộc

4.1 Khóa baseline chính thức và lưu đầy đủ BASELINE.md; commit,
environment metadata. toolchain, fixture hash, commands. 4.2 Xây dựng
Performance Registry cho toàn team. Bảng experiment ID, owner, commit,
correctness, schedule, RNGD, speedup, decision. 4.3 Reproduce candidate
do TV1--TV3 bàn giao. VERIFIED / NOT VERIFIED record. 4.4 Quản lý
integration branch và merge theo từng Clean integration history;
candidate nhỏ. rollback point rõ. 4.5 Chạy full regression cả ba kernel
sau shared-code Regression report; change. cross-kernel before/after.
4.6 Tính score tổng và chọn release candidate dựa trên Release
scorecard. geometric mean, không dựa trên một kernel riêng. 4.7 Chuẩn
hóa source tree và thực hiện Release candidate + submission/release
checklist. submission log. 4.8 Đồng sở hữu schedule diagnosis với TV3 ở
FFN. FFN verification notes và secondary understanding.

### Sản phẩm đầu ra

• Baseline authority và performance registry. • Integration branch luôn
ở trạng thái gần-release. • Full regression report cho mỗi release
candidate. • Submission scorecard và reproducible release note.

------------------------------------------------------------------------

```{=html}
<!-- Trang 11 -->
```
### Yêu cầu năng lực / kỹ năng sẽ hình thành

Benchmark hygiene; correctness engineering; static schedule vs measured
cycles; Git integration; regression analysis; score optimization;
release engineering cho accelerator kernels.

### Giao diện phối hợp

• Không thay thế owner kernel trong quyết định implementation; TV4 xác
minh evidence và compatibility. • Mọi candidate muốn vào integration đều
đi qua TV4 verification checklist. • TV4 phải đủ hiểu code cả ba kernel
để phát hiện regression và hỗ trợ debug, nhưng không "ôm" ownership của
TV1--TV3.

### Definition of Done -- TV4

Baseline tái lập; mọi best candidate được reproduce; integration PASS
correctness cho cả ba kernel; cycle table đầy đủ; score tổng được tính;
release candidate có rollback point và submission log.

## 5. Ma trận ownership và giao diện bàn giao

*Bảng 6: Ownership matrix theo component*

Component / vùng code Primary Mandatory Quy tắc owner reviewer

sliding_project_qkv TV1 TV4 TV1 quyết định implementation; TV4 xác minh
correctness/performance. sliding_attention_output TV2 TV4 TV2 quyết định
implementation; TV4 xác minh. decoder_feedforward TV3 TV4 TV3 quyết định
implementation; TV4 đồng sở hữu verification. Q/K/V functions trong TV1
TV2 Cross-review vì chung sliding/projection.rs file/path. Output
projection functions TV2 TV1 Cross-review vì chung projection
infrastructure. device/layout.rs TV2 TV4 Mọi thay đổi phải regression
QKV + AttnOut, và FFN nếu có dependency. shared/mlp.rs TV3 TV4
FFN-owned. shared/rmsnorm.rs TV4 (shared Kernel owner Không merge nếu
chưa full quality) liên quan regression. shared/residual.rs TV4 (shared
TV2 / TV3 Regression các kernel dùng quality) helper. Benchmark /
registry / TV4 TV1 hoặc owner Reproduce trước khi integration kernel
liên quan merge/release.

------------------------------------------------------------------------

```{=html}
<!-- Trang 12 -->
```
### Shared-code rule

Nếu một experiment chạm shared helper, experiment record bắt buộc ghi
SHARED CODE = YES. Candidate chỉ được KEEP sau khi chạy regression trên
mọi kernel có dependency. Không được suy luận "tôi chỉ tối ưu QKV nên
kernel khác không đổi".

## 6. Quy trình kỹ thuật chuẩn cho mọi experiment

## 1. OBSERVE

## 2. HYPOTHESIS 3. ONE CHANGE

                  schedule /
                                       bottleneck cụ thể     commit nhỏ
                 cycles / code

## 6. RNGD 5. SCHEDULE 4. CORRECTNESS

                  real cycles            before / after        PASS / FAIL

## 7. REVIEW

## 8. INTEGRATE

                                       KEEP / REVERT
                                                               full regression
                                       / INVESTIGATE

*Hình 2: Pipeline bắt buộc cho mọi optimization experiment.*

### 6.1. Experiment record chuẩn

EXPERIMENT ID: OWNER / KERNEL / BASELINE COMMIT / EXPERIMENT COMMIT
OBSERVATION: schedule/profiler/code cho thấy gì? HYPOTHESIS: bottleneck
là gì, vì sao? ONE CHANGE: thay chính xác cái gì? FILES CHANGED: ...
SHARED CODE: YES / NO CORRECTNESS: PASS / FAIL; atol / rtol; max error
nếu có STATIC SCHEDULE: baseline / optimized; dominant context
before/after RNGD: baseline cycles / optimized cycles / speedup
INTERPRETATION: evidence có ủng hộ hypothesis không? DECISION: KEEP /
REVERT / INVESTIGATE

### Merge criterion

Một thay đổi chỉ được vào integration khi có: Correctness PASS +
performance evidence + reproducibility + handoff note. "Code nhìn có vẻ
nhanh", "schedule ngắn hơn" hoặc "compile được" đều chưa đủ.

## 7. Quản lý mã nguồn và tích hợp

### 7.1. Branch model

official-baseline \| \|-- kernel/qkv \|-- kernel/attention-output \|--
kernel/ffn

------------------------------------------------------------------------

```{=html}
<!-- Trang 13 -->
```
\`-- integration

Optional experiment branches: exp/qkv/`<hypothesis>`{=html}
exp/attn/`<hypothesis>`{=html} exp/ffn/`<hypothesis>`{=html}

### 7.2. Quy tắc commit

• Một commit thành công nên tương ứng một hypothesis rõ ràng. • Commit
message mô tả vùng + thay đổi, ví dụ: qkv: adjust QueryRows mapping. •
Không trộn cleanup/refactor không liên quan vào performance experiment.
• Candidate KEEP mới được cherry-pick vào kernel branch; candidate
VERIFIED mới được cherry-pick vào integration.

### 7.3. Continuous integration rule

Integration không phải bước cuối. Mỗi best candidate nên được ghép sớm
để phát hiện interaction. Trình tự an toàn:

       best QKV → test all 3 → + best AttnOut → test all 3 → + best FFN → test all 3.

## 8. Sản phẩm đầu ra bắt buộc của toàn nhóm

TT Sản phẩm Mô tả / yêu cầu Owner

1 Common Baseline Commit, toolchain, fixture, test TV4 Specification
commands, baseline cycles, baseline schedule IDs. 2 QKV Kernel
Specification Shape/dtype/layout/dataflow + TV1 source-to-schedule map.
3 QKV Best Candidate Correctness PASS + schedule + RNGD TV1 cycles +
experiment history. 4 Attention Output Layout/movement/projection map +
TV2 Specification baseline profile. 5 Attention Output Best Correctness
PASS + schedule + RNGD TV2 Candidate cycles + shared-impact note. 6 FFN
Kernel Specification NVFP4/MLP decomposition + TV3 source-to-schedule
map. 7 FFN Best Candidate Correctness PASS + schedule + RNGD TV3
cycles + numerical note. 8 Performance Registry Toàn bộ experiment ID,
commit, TV4 correctness, cycles, speedup, decision. 9 Integration Branch
Best verified candidates được ghép TV4 theo lịch sử sạch, có rollback
points. 10 Regression Report Kết quả correctness/cycles cả ba TV4 kernel
sau shared-code/integration changes.

------------------------------------------------------------------------

```{=html}
<!-- Trang 14 -->
```
11 Release Scorecard Speedup từng kernel + TV4 geometric-mean score +
candidate identifier. 12 Technical Handoff Pack Mỗi owner mô tả "what
changed / why Cả nhóm / evidence / risks / rollback".

## 9. Tiêu chí nghiệm thu và chỉ số đánh giá

Tiêu chí Điều kiện nghiệm thu Trạng thái chuẩn

Correctness Ba kernel giữ đúng official tolerance sau 3 PASS
integration. Reproducibility Một thành viên khác có thể checkout commit
và 3 Required reproduce kết quả. Evidence traceability Mỗi optimization
truy được từ observation → 3 Required hypothesis → code → measurement.
Static schedule archive Baseline và candidate schedules được lưu, có
notes 3 Required về dominant contexts. RNGD measurement Candidate KEEP
có real RNGD cycles, không chỉ 3 Required static makespan. Shared-code
regression Mọi helper change được test trên tất cả kernel liên 3
Required quan. Integration quality Integration branch build/test sạch và
có rollback 3 Required points. Overall performance Geometric-mean score
của release candidate tốt 3 Target hơn baseline. Documentation Spec,
experiment log và handoff đủ để người khác 3 Required tiếp tục
workstream.

### Anti-patterns bị loại bỏ

• Không merge code do AI/Codex sinh ra nếu owner không giải thích được
tensor/dataflow và hypothesis. • Không thay nhiều biến cùng lúc rồi đo
một con số duy nhất. • Không dùng "util thấp" hoặc "AI cao" như kết luận
bottleneck nếu chưa có evidence kết hợp. • Không thay official
correctness threshold để làm candidate PASS. • Không để shared helper
trở thành "vùng vô chủ". • Không chờ đến cuối mới tích hợp ba kernel.

------------------------------------------------------------------------

```{=html}
<!-- Trang 15 -->
```
*Bảng 8: Mức hiểu biết tối thiểu và chiều sâu chuyên môn*

Nội dung TV1 TV2 TV3 TV4

Competition scoring / Strong Strong Strong Master correctness /
submission Tensor shapes / axes Master Strong Strong Strong QKV
projection mapping Master Strong Working Strong Layout / broadcast /
Strong Master Working Strong movement FFN / NVFP4 / GeGLU Working
Working Master Strong Schedule viewer / bottleneck Strong Strong Strong
Master classification Benchmark hygiene / Strong Strong Strong Master
reproducibility Git integration / regression / Working Working Working
Master release Cross-kernel system Strong Strong Strong Master
understanding

## 10. Bản đồ kiến thức chung và chuyên môn hóa

### Mục tiêu tổ chức cuối cùng

          Shared system understanding + Deep individual
                         specialization

Mỗi người đủ hiểu hệ thống để phối hợp, nhưng có một vùng mà mình là
người chịu trách nhiệm kỹ thuật sâu nhất.

## 11. Tài liệu tham chiếu kỹ thuật

• Official baseline repository • Competition site • Furiosa-opt
programming guide • Các file nội bộ cần đọc theo role: README.md,
ARCHITECTURE.md, OPTIMIZATION.md, src/axes.rs, src/ops.rs,
src/device/\*\*, tests/test_kernels.rs.

            — MOA 2026 Round 1 – Team Responsibility & Integration Specification —

------------------------------------------------------------------------
