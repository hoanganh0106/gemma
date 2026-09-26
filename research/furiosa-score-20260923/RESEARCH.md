# Nghiên cứu tối ưu Gemma-4 trên Furiosa RNGD

Ngày nghiên cứu: 23/09/2026 UTC. Phạm vi chính: Round 1, ba kernel; SDK `furiosa-opt-std = 0.8.1`, compiler `cargo-furiosa-opt 0.8.1`, Rust `nightly-2026-05-01`. Tài liệu được đọc trực tiếp, mã được tra bằng CodeGraph và lịch được biên dịch mới từ workspace. Các giả thuyết bên dưới được phân biệt với số đã đo.

## 1. Quyết định tối ưu nên thay đổi ở đâu

Đòn bẩy có cơ sở nhất hiện nay là **giảm các lần materialize trung gian và đồng bộ của kernel**, rồi tối ưu cách đọc weight/scale. Việc chỉ đổi thứ tự dòng Rust không đủ. Cần xét lại những hướng từng bị loại trên SDK cũ vì 0.8.1 đã có đường Vector → VRF và primitive chuyển DM giữa cluster.

Ba probe được làm riêng, giữ source chính nguyên trạng:

- K1: tách tải Q/K/V weight ra trước các projection. Biên dịch thành công nhưng lịch vẫn **42.539 cycles / 154 instructions**. Thời điểm ba weight DMA không thay đổi. Đây là kết quả âm có ích: compiler giữ cùng lịch dù source gọi prefetch sớm.
- K2: bỏ HBM hop, dùng local gather + peer `cluster_swap` + DM relayout. Biên dịch thành công nhưng **24.718 → 26.142 cycles**, tăng 5,76%; số instruction 44 → 55. Chi phí đồng bộ mới lớn hơn phần tiết kiệm truyền dữ liệu.
- K2: ghi inverse-RMS trực tiếp bằng `.vector_final().to_vrf(&mut device.sub)`. Biên dịch thành công, **24.718 → 24.451 cycles**, giảm 267 cycles, tương đương **1,080%**; 44 → 43 instructions. Đã bỏ đúng Sub preload 267 cycles trên đường phụ thuộc cuối. **Bốn job A–B–B–A đều PASS 15/15 outputs**. K2 giảm ở cả hai cặp so sánh **0,371% và 1,980%**, trung bình nhân tỷ lệ cycles tương ứng giảm **1,179%**. Đây là candidate có kết quả hardware thuận hướng, chưa đủ chứng minh speedup ổn định hoặc optimum. Xem [runtime probe](probes/k2-direct-vrf/RUNTIME.md).

Nếu giảm 1,080% K2 được giữ nguyên trên số đo chính thức và K1/K3 không đổi, score tăng khoảng **0,363%**. Đây là suy luận có điều kiện, chưa phải điểm đã đạt. Probe xác nhận direct VRF dùng được tại scalar inverse-RMS này; áp dụng tại các site RMSNorm/GeGLU/RoPE khác vẫn là giả thuyết cần kiểm tra layout, register capacity, schedule và correctness.

## 2. Luật chấm quyết định hàm mục tiêu

Thông báo ngày 23/09 xác nhận: mỗi kernel chạy ba input case, mọi case phải đúng, dùng median cycles; hạn Round 1 là 30/09/2026 23:59 AoE. **Chung kết Round 1 chấm submission cuối trước hạn**. Leaderboard hiển thị best lịch sử hợp lệ; vì vậy bản cuối phải là champion đã kiểm chứng. Round 2 bắt đầu 02/10; khoảng 70% đội tiến tiếp và 3x–5x chỉ là dự đoán cutoff. [Thông báo BTC](https://micro2026-moa.github.io/index.html).

Stage 1 chỉ nhận implementation trong `src/device/` và function bodies của `src/ops.rs`. Giữ tên, signature các `#[device]` và module paths. Tối ưu host, API, test harness, axes không được đưa vào phép chấm Stage 1. Source dự thi không được công khai trước hạn. [README chính thức](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B#stage-1-rules-skeleton-contract).

Score:

`G = ((B1/S1) × (B2/S2) × (B3/S3))^(1/3)`

Trong đó `Si` là median của ba case thuộc kernel i; `Bi` là baseline evaluator áp dụng cho submission. API được kiểm tra hiện chỉ trả một tuple chung, chưa công bố baseline map theo version; tuple này đã đối chiếu khớp hai kết quả 0.8.1 bên dưới. Tối đa G tương đương tối thiểu **S1 × S2 × S3**, hoặc tổng `ln(Si)`. Một kernel giảm 10% cycles cho +3,57% score; giảm 20% cho +7,72%; giảm 50% cho +25,99%. Cả ba giảm 10% cho +11,11%. Vì thế không tự động dành toàn bộ thời gian cho K3 chỉ vì K3 dài nhất. [Leaderboard và scoring](https://micro2026-moa.github.io/leaderboard.html).

Cách xếp ưu tiên nghiên cứu đề xuất: với một candidate cần t giờ và ước lượng giảm tỷ lệ r của kernel, xét `-ln(1-r)/(3t)`, rồi điều chỉnh theo rủi ro correctness và khả năng triển khai. Đây là thước đo phân bổ công sức, không phải công thức của BTC và không có giá trị nếu r chỉ là phỏng đoán thiếu cơ sở.

Snapshot API leaderboard lúc 06:49:42 UTC có baseline `250514 / 404633 / 3703473`. Đối chiếu một submission 0.8.1 đã hoàn thành, `7cffa30b`, các median `92419 / 45109 / 221526` cho đúng score `7.407717941...`; log của nó có đủ ba case mỗi kernel. Một dòng top 0.8.1 `73366 / 33020 / 201948` cho `9.155112220...`, cũng khớp. Đây là xác minh formula trên dữ liệu hiện thời; snapshot không công bố bản đồ baseline riêng từng version và không chứng minh nguồn của hai submission đó giống workspace này. Xem `leaderboard-observation.md` và JSON snapshot kèm theo.

Không suy score chính thức từ schedule. Không ghép minima K1/K2/K3 ở ba job khác nhau. Không dùng best-of-N để thay median ba case. Chạy lặp nhiều process có ích để biết noise, nhưng mỗi process vẫn phải giữ nguyên bộ đo chính thức.

## 3. Correctness là điều kiện đầu tiên

Harness hiện tại kiểm tra mọi phần tử finite và:

`abs(actual - expected) <= atol + 0.01 × abs(expected)`

Atol: K1 0,04; K2 0,05; K3 0,01. Không có ngân sách cho một tỷ lệ nhỏ phần tử sai. Chín kernel/input checks tạo **15 dòng so sánh output**: Q/K/V × ba case, K2 × ba, K3 × ba. Một run hợp lệ cần cả 15 PASS, ba median đều có đủ ba cycles, exit 0 và `all 3 tests passed`. [Harness](../../src/bin/test_kernels.rs).

Các case độc lập với nhau nhưng deterministic giữa các invocation. K2 hiện dùng activation BF16 liên tục trong [-1,1), vị trí RoPE/cache thay đổi và khác 0, global scale FFN thay đổi theo case. BTC đã công bố các thay đổi này để loại đường tính chỉ đúng với fixture cũ. [Changelog chính thức](https://github.com/micro2026-moa/furiosa-opt-gemma4-12B/blob/main/CHANGES.md).

Ba điểm cần kiểm chứng riêng trong source đang chọn:

1. K1 đã có weighted input RMSNorm trong `qkv_head_local.rs`. Tuy nhiên representation hai FP8 components vẫn là xấp xỉ của activation BF16 và cần đo numerical margin sau Q/K normalization, RoPE. Không được xem việc đã dùng đủ tham số là runtime PASS.
2. K2 dùng reconstruction hai FP8 components cho activation. Fixture mới không cho phép thay activation bằng dấu ±1; gần-zero và cancellation có thể lộ sai số dù average error nhỏ.
3. K3 dùng Erf GELU trong `ffn7.rs`, trong khi reference là GELU dạng tanh; code cũng thay vị trí một số scale và BF16 rounding boundaries. Đây là các sai khác số học cần lượng hóa. Phải bảo toàn ảnh hưởng của global scale và input RMS weights; bỏ hoặc dời scale chỉ khi xét đủ EPS và BF16 rounding boundaries. Fixed `UP_GAIN=128` và `DOWN_GAIN=0.25` cần kiểm tra saturation/underflow trên khoảng đầu vào rộng hơn public fixture.

Nên lưu `max(abs_error/(atol + rtol*abs(expected)))`, số nonfinite, phần tử lỗi, max abs error và phân bố theo output. Tỷ lệ sai số chuẩn hóa ≤1 mới đủ PASS; cách chỉ báo mean/max relative error có thể bỏ sót phần tử gần zero.

Stress tests hữu ích: input zero/gần-zero; dấu trộn; một outlier; RMS weight không đồng nhất; global scales ở hai đầu khoảng; vị trí RoPE và cache wrap; Q/K có cancellation; activation GeGLU lớn/nhỏ. Các test phụ giúp giữ semantics model, không thay bộ chấm chính thức. Chưa có tài liệu đủ để kết luận chính sách hidden tests.

## 4. Bằng chứng của đúng workspace hiện tại

HEAD là `670eaad17556ac3c28c6ae442c446a7e78bb4030`, nhưng workspace có nhiều thay đổi staged/unstaged trước nghiên cứu. **HEAD không định danh toàn bộ candidate**. Manifest `source-hashes.json` ghi mọi file dưới `src/`; `evidence-hashes.json` ghi thêm Cargo/toolchain/fixture. Thư mục `baseline-source/` đóng băng toàn bộ 53 file source, Cargo.toml, Cargo.lock và toolchain; kiểm tra lại có **0 hash mismatch**. Ba probe nằm ngoài source chính. Cargo.lock của static probes ban đầu được resolver tạo riêng; bản K2 đem đo runtime đã khóa lại byte-identical với Cargo.lock baseline.

Source lựa chọn:

- `src/ops.rs`: `C75F578DC24603450EF7EBA136B12AA90CFD8DB6E928B3B320C8B34C1BD09F1C`.
- K1 `qkv_head_local.rs`: `E40F4ABFE84DEE3F3BBA2776B164E52382261D83130C7F71193E2FE4A6981C84`.
- K2 `output31.rs`: `BCA4C687EB1BC735E806D0FA5A0E093358B172C91C990D532B53C9A0F0E79E95`.
- K3 `ffn7.rs`: `087F4A4FB7E8E051CF8965052C8C8FC230F6E5732AB2ADA6E8D7123436776C3D`.

Đo lại bằng `CARGO_INCREMENTAL=0 cargo furiosa-opt compile ops::<kernel> --exact --dump-schedule ...`, tuần tự từng kernel:

- **K1: 42.539 cycles / 154 instructions**. Hợp các khoảng chiếm DmaEngine: 38.174; MainContext 16.737; SubContext 7.166; VectorEngine 12.249.
- **K2: 24.718 / 44**. DmaEngine 18.052; MainContext/VectorEngine 5.846; SubContext 2.541.
- **K3: 110.520 / 170**. DmaEngine 100.780; MainContext 44.073; SubContext 27.063; VectorEngine 41.922.

Các context chồng lấp, vì vậy không cộng chúng thành tổng runtime. Một instruction có thể chiếm nhiều context. Đây là occupancy của lịch compiler, không phải đo mức sử dụng vật lý của RNGD. `schedule-analysis.json` và ba schedule gốc là nguồn.

Hash schedule K1 `80b57687b69df536c4b4b02cec0a9e5cdd0dd1f67648c2d52b8f6a39da15e022`; K2 `57ce0b6b86663c39c3bc519728c0954754fadf5bc2001444ce296bc35d4cb895`; K3 `55e32ca5824c8d02dda44da6afa8b9c20d3aca12679ccb3c0a97cf4a3b47e84f`.

Các số 22.843 cho K2 và 106.908 cho K3 trong ledger cũ không phải baseline của lần biên dịch này. K3 hiện đã dùng ba down tiles 5+5+5; artifact cũ có hai DMA down 10+5. Không suy regression do một thay đổi duy nhất khi source/toolchain đồng thời khác nhau.

Phát hiện bộ đo local bị lệch: `ref/fixtures.safetensors` cũ chỉ 67.336 bytes, không có `run0/run1/run2`; launcher cũ dùng `TUC_PROFILE_LEVEL`, harness mới cần `FURIOSA_OPT_PROFILE`. Nghiên cứu đã tạo fixture ba seed và launcher riêng trong thư mục này. Bộ chuẩn bị không thay fixture hoặc launcher chính.

Fresh full NPU build `cargo furiosa-opt build --release --bin test_kernels` thành công. Binary frozen SHA256 `021b9009886682add70e20ee3481bf5bcd9711b4b3535d1cba0821c46a37a39d`; fixture SHA256 `338c0a8c06458a528b29d58f0f65a5ea13af79a14a9929a06c571b8adb002359`. Public Arena job **78017** đã **SUCCEEDED, exit 0, 15/15 output comparisons PASS**, `all 3 tests passed`, không pin CPU. K1 median **91.270** từ `[102213,90703,91270]`; K2 **44.471** từ `[44471,44409,44622]`; K3 **220.925** từ `[224052,220925,219896]`. Score tham khảo theo baseline API vừa quan sát là **7,480849**; chưa phải MOA submission. Frozen tuple, log và giới hạn suy luận nằm trong [arena-result.md](arena-result.md).

Hardware/static tương ứng `2,146 / 1,799 / 1,999`, nên không có cơ sở dùng một hệ số chung đổi static sang score. Mốc leaderboard 0.8.1 nêu trên thấp hơn workspace 19,62% / 25,75% / 8,59% theo từng median; K2 có khoảng cách tỷ lệ lớn nhất dù K3 có số cycles tuyệt đối lớn nhất. Đây là tham khảo giữa các job, không phải A/B cùng điều kiện hay bằng chứng về source của đối thủ.

## 5. Mô hình phần cứng dùng cho quyết định

RNGD có hai cluster, mỗi cluster 256 slice; mỗi slice có Tensor Unit và DM riêng. DM tổng 256 MiB, tương đương 512 KiB/slice. HBM công bố 48 GB, băng thông peak khoảng 1,5 TB/s. Mapping quyết định dữ liệu nằm ở cluster/slice/lane nào, và trục nào chạy theo Time hoặc Packet. [Kernel Design](https://developer.furiosa.ai/furiosa-opt/book/quick-start/kernel-design.html), [Memory Performance](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html).

Trong schedule hiện tại, với durations và resource assignments do compiler gán, makespan tĩnh không thấp hơn `max(tổng duration của từng resource tuần tự, độ dài critical path)`. Đây là cận trong mô hình compiler; NPU cycles phải đo riêng và không suy occupancy của tám DMA engines vật lý từ một resource `DmaEngine` trong JSON. Tối ưu một đoạn đã nằm dưới DMA dài có thể không giảm makespan; cắt một store/sync trên đuôi dependency có thể có ích hơn giảm hàng nghìn phép toán bị che khuất.

HBM làm việc theo transaction 256 bytes. Địa chỉ/độ dài đọc lệch boundary tốn thêm transaction; partial write có thể cần read-modify-write rất đắt. Hãy ưu tiên các chunk contiguous và đủ 256 bytes khi store qua HBM. DMN cũng cần phân bố truy cập giữa các nhóm slice; tổng byte nhỏ không đảm bảo DMA nhanh. [Memory Performance](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/memory-performance.html).

Có tám DMA engines/chip. Một tensor move được compiler phân phối qua các engine theo mapping; không tương đương tám dòng `to_dm` tự động chạy song song. Tensor DMA hỗ trợ HBM↔DM và DM↔DM. Đối với chuyển SRAM thuần túy, Fetch/Commit có thể hiệu quả hơn DMA; startup/mapping và đồng bộ phải được tính cùng bytes. [DMA Engine](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/dma-engine.html).

Main/sub không hoàn toàn độc lập: chúng chia sẻ pipeline resource và memory banks. Compiler có thể serialize DMA với truy cập TU gây bank starvation; 64 lần truy cập liên tiếp cùng bank là giới hạn cần tránh. Source order không phải lịch thực thi. [Schedule](https://developer.furiosa.ai/furiosa-opt/book/scheduling/schedule.html).

Contraction gồm outer, packet reduction, time reduction, lane folding. Outer có tối đa tám lanes và packet 64 bytes; đặt toàn bộ trục reduction vào Time có thể làm phần lớn MAC nhàn. Với GEMV decode, ưu tiên phần reduction trong Packet, phần còn lại trong Time và chọn lane mapping phù hợp representation. Không giả định packet rộng nhất luôn tốt: register banks, padding và legality của reducer có thể đảo kết quả. [Contraction Engine](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/contraction-engine/index.html).

TRF/VRF giới hạn bộ nhớ cục bộ. TRF có half modes để compiler double-buffer; hai nửa vẫn chia sẻ banks nên overlap có tranh chấp. Dung lượng logic theo dtype/active lanes không nên đánh đồng với dung lượng SRAM vật lý 80 KiB/slice mô tả trong chương chi tiết. Dùng mapping thực và compiler allocator để kiểm tra fit. [Register Files](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/register-files.html).

Switch hoạt động trên ring trong cluster; ước tính cơ bản tỷ lệ với ring size × số time steps × số flit/packet. Ưu tiên giữ head local, chọn ring nhỏ hợp lý và giảm số lần đổi layout. Switch không đổi Cluster. `reshape` chỉ relabel, không chuyển dữ liệu hoặc khởi tạo bản sao bị padding. [Switch Engine](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/switch-engine.html), [DmTensor 0.8.1](https://docs.rs/furiosa-opt-std/0.8.1/furiosa_opt_std/prelude/struct.DmTensor.html).

Fetch main có thể đọc rộng hơn sub; sub đọc/ghi 8 bytes một lượt, Collect chuẩn hóa flit 32 bytes. Packet 24 bytes có thể bị tách thành nhiều lượt; tăng padding đôi khi nhanh hơn nhưng làm tăng storage và lưu lượng. Cần đo mapping cụ thể. [Fetch Engine](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/fetch-engine.html), [Commit Engine](https://developer.furiosa.ai/furiosa-opt/book/moving-tensors/commit-engine.html).

Paper ISCA 2024 của Furiosa mô tả FFN có weight traffic lớn, padding do kích thước không phải power-of-two; dùng tiling và prefetch phần sau khi tính phần trước. Điều có thể chuyển sang Gemma là phương pháp overlap và giữ activation trong chip. Số tile tối ưu trong LLaMA của paper không chứng minh số tile tối ưu cho Gemma GeGLU. [Paper, mục VI.C](https://furiosa.ai/download/FuriosaAI-tensor-contraction-processor-isca24).

## 6. Lưu lượng tối thiểu cho một invocation

Từ axes hiện tại H=3840, Qs=4096, Ps=2048, L=15360; đây là tính toán của nghiên cứu từ tensor shapes, giả sử mỗi weight/scale đọc đúng một lần, chưa tính activation, metadata, padding hoặc repeat transfer:

- K1: `(4096 + 2048 + 2048) × 3840 × 1 byte = 31.457.280 bytes` FP8 weights, khoảng 31,46 MB.
- K2: `3840 × 4096 × 1 = 15.728.640 bytes`, khoảng 15,73 MB.
- K3: ba ma trận FP4, mỗi ma trận 58.982.400 phần tử: packed weights tổng **88.473.600 bytes**. FP8 scale mỗi 16 phần tử thêm **11.059.200 bytes**; tổng chính **99.532.800 bytes**, khoảng 99,53 MB.

Chia cho peak 1,5 TB/s cho cận bandwidth lý tưởng lần lượt **20,97 μs / 10,49 μs / 66,36 μs**. Không đổi trực tiếp các số này sang schedule cycles khi chưa xác minh clock domain. Đây không phải dự báo latency thực tế: cận dưới chỉ áp dụng khi giả định lượng dữ liệu bắt buộc đọc là đúng; DMA startup, bank conflicts, compute, layout, sync và utilization chưa được tính.

Hệ quả: Stage 1 single invocation không hưởng lợi từ weight caching liên token như E2E. Tạo một bản weight unpacked trong HBM ngay trong kernel thường tăng traffic. Nếu dequant/prepack có lợi, nên kiểm tra giữ dữ liệu trong DM và bảo đảm chi phí chuẩn bị nằm trong phép đo.

## 7. K1: ưu tiên giảm DMA nhỏ và materialization

### Bổ sung: A010 là đối chứng cần đưa vào nghiên cứu

Sau khi người dùng nhắc A010, đã đối chiếu trực tiếp `experiments/auto_k1_current_38723/AUTO_K1_LOG.md` và source còn lưu. SHA256 `qkv69382.rs` khớp **7CEAF8DE01E78CF9624F63CDA71CB219683309D19466A9FFE10E13874EA5FC21**, đúng candidate AUTO_K1 A010. Ledger ghi job **69485**, K1 **80.940 cycles**, Q/K/V và cả ba package tests PASS; static **38.482 / 168**, giảm 241 cycles so với parent B000 38.723. Đây là bằng chứng lịch sử đáng giữ; phần nghiên cứu ban đầu chưa đưa đối chứng này vào đủ rõ.

Cơ chế riêng của A010 là `project_one_kv_matrix_scaled` tạo đồng thời V BF16 trên Main và mean-square trên Sub từ cùng rows/scale, rồi `normalize_value_from_mean_square` dùng scalar đã tính. Parent đọc lại V BF16 để tính RMS; A010 chuyển reduction sang nhánh song song. Ledger ghi Main giảm 345 cycles, Sub tăng 348 nhưng chạy chồng lấp, makespan giảm 241. Có thay đổi numerical boundary: mean-square lấy từ scaled FP32 trước cast BF16, nên phải kiểm chứng tolerance khi chuyển ý tưởng sang source mới.

**Chưa thể so trực tiếp 80.940 với 91.270 của job 78017.** Cargo.toml A010 dùng SDK `0.6.0`; fixture lịch sử SHA `9801C573...` khác fixture ba seed hiện tại `338C0A8C...`. Ngoài ra source đã hash-verify cho thấy `normalize_native_input` tại dòng 824 nhận `_weight` nhưng không dùng, quantize x trực tiếp bằng `BitAnd(-0.5)` và một FP8 component; nó không thực hiện weighted input RMSNorm như K1 hiện tại. PASS trên fixture cũ không chứng minh biến đổi này bảo toàn kết quả cho input weights tổng quát. Không quy toàn bộ khoảng cách cycles giữa hai job cho riêng tối ưu A010.

**Điều chỉnh ưu tiên K1:** đưa cơ chế nhánh V/RMS song song của A010 lên trước các rewrite RoPE lớn. Tạo đối chứng trên SDK 0.8.1 giữ đầy đủ weighted input RMSNorm, cùng representation activation và fixture ba seed; thay riêng nhánh V/RMS, sau đó thử direct VRF cho scalar RMS. Chỉ gọi candidate kết hợp là thắng sau khi đo correctness và runtime với parent cùng điều kiện. A010 nguyên bản vẫn được giữ làm artifact lịch sử, chưa được promote thành baseline hiện hành.

Ba weight DMA lớn chiếm 27.572 cycles: Q `2848..16360`, K `23786..30816`, V `31267..38297`. Khoảng giữa Q và K không phải toàn bộ idle: nó chứa scale loads, RoPE gathers, stores/reloads của cos/sin. Tổng 19 instruction chiếm DmaEngine là 38.174 cycles trên makespan 42.539. Nếu giữ nguyên tất cả operations, resource assignments và duration của schedule này, reorder lý tưởng cũng không xuống dưới 38.174; mức cải thiện K1 tối đa theo cận này khoảng 10,26%, còn cận có thể không đạt do dependency. Đây không phải cận kiến trúc RNGD tổng quát. Cần loại hoặc rút ngắn DMA để có bước nhảy lớn hơn.

**K1-A: chuyển RoPE row trong DM.** `apply_rope` hiện gather một dòng cos/sin rồi ghi HBM và load lại theo head mapping. Thử chuyển từ gather output sang head layout bằng DM→DM hoặc broadcast hợp lệ, tận dụng API 0.8.1. Phải giữ offsets động, signs của sine và cache writes đúng. Đo số sync mới; kết quả K2 cho thấy SRAM transfer cũng có thể đắt. Không đặt mục tiêu chỉ xóa bytes HBM trên source.

**K1-B: Vector → VRF ở các scalar và RoPE.** RMS root và `sin_product` đang commit DM rồi sub đọc VRF. Candidate direct VRF có thể bỏ intermediate buffer và command, nhưng main producer sẽ chiếm cả main+sub. Với scalar reuse dài, rủi ro VRF live range tăng; với `sin_product` 256 f32/head cần xem cùng lúc cos/sin/RMS operands có vượt VRF không. Bắt đầu từng site để biết chi phí thật.

**K1-C: fuse projection scale/head normalization.** Giảm pass trung gian có triển vọng vì scale/head tensors nhỏ bị startup chi phối. Tuy nhiên reference có BF16 rounding giữa projection và norm; bỏ boundary không tự động tương đương. Cần preserve boundary trong stream nếu API cho phép, hoặc chứng minh full elementwise tolerance margin trên ba case và stress.

**K1-D: cải thiện representation hai FP8 terms.** Uniform high hiện chia theo biên 16, residual FP8 và scale động. Thử mantissa-high + residual có thể giảm lỗi cùng số contractions; lợi ích có thể là cho phép fuse mạnh hơn mà vẫn PASS, không nhất thiết giảm cycles trực tiếp. Không bỏ weighted RMSNorm hoặc dựa vào fixture x gần ±1. Scalar RMS có thể triệt tiêu xấp xỉ trong head norm nhưng EPS, per-coordinate weights và rounding khiến việc bỏ nó không phải phép biến đổi chính xác.

**K1-E: prefetch chỉ đáng mở lại khi thay dependency vật lý.** Probe đã chứng minh upfront loads đơn thuần không đổi lịch. Bước tiếp theo phải tách/bố trí buffer lifetimes, tile boundary hoặc grouping hợp lệ để compiler có lịch khác. Ba weight tensors tổng 30 MiB, 60 KiB/slice với mapping hiện tại, nhưng phải cộng scratch và kiểm tra allocator. Không dùng lặp lại source-order sweep như một hướng mới.

## 8. K2: đuôi RMSNorm và đồng bộ quan trọng

Weight O DMA 13.383 cycles (`1936..15319`) là phần lớn đầu kernel. Contraction 1.374 cycles. Đoạn RMS inverse ở `output31.rs:227` chiếm 2.579 cycles; tiếp theo preload VRF 267 và final vector 401. Các barrier, HBM hop/reload và `DramReuse` làm đuôi kéo dài.

**K2-A — ưu tiên đầu tiên: direct VRF.** Probe inverse-RMS giữ đúng toán học/reduction, bỏ một preload và thắng tĩnh. Bản source `probes/k2-direct-vrf/src/device/sliding/output31.rs`, schedule `probes/k2-direct-vrf/k2.json`. SHA source `02630EA698CC31D5749AE62248C3E0688A50E185DE511B409E9AFF16FC649879`; SHA schedule `433C1390E2557E5AD0661B8CA4161AD489EFDB5CF9312DE4C70D22CF252A39C3`. Tiếp theo thử factor `sw`/scalar site khác riêng lẻ; site nằm dưới DMA có thể chỉ giảm instruction mà không giảm makespan.

Runtime đầu tiên của candidate **78085**, binary SHA `46c68fe538f01ef4f72dbf31913fb1d69cc2b60a0be0a8bb5b6395683521f94d`, có full tuple **92.772 / 44.306 / 220.747**, score tham khảo **7,451483**, thấp hơn baseline 7,480849 ở lượt này. Cả 15 dòng thống kê numerical error in ra đều giống baseline; điều này không chứng minh output bitwise identical vì log làm tròn. K1/K3 source không thay nhưng medians vẫn dao động, minh họa vì sao không suy causality hoặc promotion từ một job. Chỉ riêng K2 giảm 0,371% sẽ cho khoảng +0,124% score nếu hai kernel kia giữ nguyên; tuple thực của candidate phải được tính nguyên bộ. Biên dịch lại K2 sau khi khóa đúng Cargo.lock baseline cho schedule **byte-identical**, vẫn **24.451 / 43**, SHA không đổi (`k2.locked.json`).

Đã làm thêm hai server-side reruns của đúng frozen packages để hoàn tất thứ tự A–B–B–A, giữ nguyên batching, timeout 70 giây, profiling và no-CPU-pin:

- A1 **78017**: `91270 / 44471 / 220925`, score tham khảo 7,480849.
- B1 **78085**: `92772 / 44306 / 220747`, score tham khảo 7,451483.
- B2 **78108**: `91859 / 43960 / 222051`, score tham khảo 7,480951.
- A2 **78115**: `92098 / 44848 / 222520`, score tham khảo 7,419592.

Cả bốn job `SUCCEEDED`, exit 0, 15/15 output PASS mỗi job; đây là lặp lại cùng bộ public case, không phải 60 bộ input độc lập. Các dòng numerical-error in ra giống nhau. K2 `B1/A1 = 0,9962897`, `B2/A2 = 0,9801998`; geometric mean ratio `0,9882120` tương đương **giảm 1,1788% cycles**, speedup **1,011929x**. Nếu chỉ K2 giảm như vậy và hai kernel kia cố định, tác động score khoảng +0,3960%.

Tính nguyên bộ từng job rồi lấy geometric mean theo nhóm A/B cho 7,450158 và 7,466202, chênh **+0,2154%** quan sát. Đây là thống kê thăm dò của bốn job, không phải điểm nộp MOA hoặc cách BTC chấm nhiều invocation; không chọn minimum từng kernel để ghép tuple. Chưa có confidence interval đáng tin từ hai cặp, các job cách nhau về thời gian và controller không trả physical worker ID. Quyết định: **giữ candidate để tiếp tục thử nghiệm; chưa promote vào source chính hoặc khẳng định stable score win**.

**K2-B — đổi kiến trúc reduction.** Probe allgather toàn z bằng ba SRAM moves bị chậm. Hướng khác có cơ sở hơn: giữ 1920 z values tại mỗi cluster, tính local sumsq rồi trao đổi chỉ scalar sum giữa cluster bằng một `cluster_swap`, cộng hai scalar và normalize local rows. Mục tiêu giảm số synchronization boundaries, không chỉ giảm bytes. Cần padding/alignment hợp lệ, mọi scalar replica thực sự được khởi tạo, không vô tình double-count 32/256 copies. Chưa triển khai hướng này; confidence trung bình-thấp.

**K2-C — bỏ no-op reshape trước final store.** A028 trước đây giảm instruction nhưng không giảm static cycles nên bị revert. Tài liệu repo có số đo cũ cho thấy xóa một wait point có thể tác động runtime, song các batch còn trái dấu và khác SDK. Vì thế đây là probe giá rẻ cần đo lại, không phải speedup đã chứng minh. Không bỏ qua candidate chỉ vì max-end tĩnh bằng nhau khi nó thực sự giảm host/control work.

**K2-D — tile lại O projection chỉ khi có nguyên nhân mới.** Ledger có nhiều sweep ngõ cụt: phân mảnh weight load, layout `H240`, packet/lane constraints, VRF overflow. SDK mới có thể thay đổi legality nhưng không nên lặp toàn bộ search mù. Dùng schedule để quyết định cần che contraction bằng tile tiếp, giảm barriers hay cải thiện DMA; giảm instruction count riêng lẻ không phải mục tiêu cuối.

## 9. K3: weight/scale traffic và lookup provisioning

Hai weight DMA up/gate mỗi cái 24.612 cycles; ba down DMA mỗi cái 8.650. Scale loads khoảng 4.880/4.880/4.961 cycles, utilization compiler khoảng 0,55 so với weight khoảng 0,84–0,88. Tổng thời lượng trên resource DmaEngine của schedule là 100.780 trên makespan 110.520; các op này không overlap với nhau trong lịch. Điều đó gợi ý compute thuần có thể bị che khuất; không có nghĩa bandwidth HBM thực đạt 91% peak.

**K3-A: direct VRF cho scalar/GeGLU.** Cụ thể `normalize_quantize` RMS, `gate_factor_vrf`, output Erf của GeGLU và post-FF RMS. Sub-produced GeGLU có thể dùng direct Sub VRF mà không thêm Main occupancy, nhưng cần giữ đủ register capacity và không vô tình kéo dài đời operand. Mỗi site là một candidate; sau đó mới kết hợp các site thắng.

**K3-B: lookup setup là chi phí cần kiểm tra trong compiler hiện hành.** Schedule có năm anonymous DmaLoad, mỗi cái 838 cycles, tạo SRAM 4 KiB. Mỗi tensor đi qua Sub op 1.293 cycles thành tensor SRAM 8 MiB, rồi trở thành input thứ ba cho các Main `.fetch_table_lookup::<f8e4m3>()` chains. Các nhóm lặp với up/gate và ba down tiles; có trường hợp provisioning sau bắt đầu khi tensor LUT trước còn sống. Vì vậy không được bỏ qua chi phí này chỉ vì table nhỏ.

Book hiện mô tả table decode là constant gắn với fetch configuration, không cần staging mỗi invocation; artifact 0.8.1 cho thấy provisioning cụ thể. Public `.fetch_table_lookup::<OutD>()` chọn table theo dtype và không nhận table handle để ép reuse. Cơ hội là giảm số chain lookup/nhóm tile hoặc tìm cơ chế compiler reuse; chưa có cách source-level hợp lệ được chứng minh để cache table. Không quy toàn bộ 5×(838+1293) thành speedup vì các phần overlap. [Fetch Adapter](https://developer.furiosa.ai/furiosa-opt/book/computing-tensors/fetch-adapter.html).

**K3-C: tách DMA tile khỏi compute tile.** Current down dùng 5+5+5 để phù hợp lowering. Thử load weight vào một DM tensor lớn hợp lệ, rồi compute bằng views 10+5 hoặc các nhóm khác; nếu cần, decode một lần rồi dùng views. Có thể giảm startup/lookup setup nhưng mất overlap hoặc tăng DM footprint. Phải xét cả max-end và lượng SRAM sống đồng thời. Không dùng schedule 10+5 cũ như bằng chứng rằng current SDK chấp nhận candidate mới.

**K3-D: trace packetization của scale DMA.** Mỗi slice hiện có 7.200 bytes scale. Việc tổng này không chia hết 256 không đủ giải thích utilization 0,55: nếu cả chunk contiguous được coalesce, 29 transactions × 256 bytes đã chứa đủ 7.200 bytes, tỷ lệ useful bytes gần 97%. Cần xem lowering theo từng scale row 240 bytes. Trong mô hình mỗi row là một transfer riêng, các offset tăng 240 mod 256: 14 trong 16 rows cắt boundary, 16 × 240 useful bytes cần 30 × 256 transaction bytes, hiệu suất chỉ 50%. Điều này phù hợp về quy mô với utilization compiler nhưng **chưa chứng minh compiler thật sự phát các transaction theo cách đó**.

Thí nghiệm phù hợp là trace stride/packetization, rồi thử coalesce nhiều row liên tiếp thành chunk bội 256 trước khi phân phối trong DM. Ví dụ 16 rows contiguous tạo 3.840 bytes, đúng 15 transactions, nhưng chi phí relayout, phân bố slice và đồng bộ có thể triệt tiêu lợi ích. Tách microcase scale transfer khỏi full FFN để biết mapping nào giảm duration thực; sau đó mới ghép lại. Không suy alignment là nút thắt đã xác nhận chỉ từ 7.200 bytes/slice hoặc utilization ước lượng.

**K3-E: bỏ HBM rendezvous của partial sums.** 0.8.1 `cluster_swap` cho phép đổi hai cluster qua SRAM DMA. Với down partial mỗi cluster, swap + VE add có thể thay store/reload HBM, nhưng final norm cần mapping hợp lệ và đồng bộ vẫn còn. K2 negative probe là lời nhắc phải đánh giá exact schedule, không suy từ tên memory tier.

**K3-F: pipeline up/gate theo nhóm output nếu còn đủ room.** Process một nhóm up và gate, tính GeGLU, tích lũy down với FP32 lâu hơn có thể tiết kiệm activation gather/working set. Tradeoff là reload down tiles, nhiều DMA commands, đổi reduction order và scale handling. Đây là hướng kiến trúc tốn công, chỉ làm sau các primitive cải tiến nhỏ đã được đo. Tài liệu tuning cho thấy prefetch chỉ overlap khi compiler thấy producer/consumer trong cùng scheduling unit; rolled loops có thể mất overlap. [Tuning](https://developer.furiosa.ai/furiosa-opt/book/scheduling/tuning.html).

## 10. Những thứ nên tránh

- Bỏ RMS input weights, global scales, RoPE, cache writes hoặc thay nội dung dựa vào seed/fixture; có thể nhanh nhưng không giữ hợp đồng tính toán.
- Dùng `unsafe reshape` để thay transpose hoặc tạo dữ liệu ở padded lanes. Các API mới vẫn yêu cầu wire-order và replica initialization đúng.
- Tối ưu chỉ theo số instruction, chỉ static makespan, hoặc chỉ một minimum hardware đẹp. Cả correctness và official median mới quyết định.
- Thử endless source-order/permutation khi emitted schedule không đổi. Cần một thay đổi dependency/resource cụ thể.
- Đưa CPU affinity khác nhau vào A/B. Các kết quả pinned cũ không nằm trong tập so sánh của nghiên cứu.
- Sửa host để cache weights ngoài kernel nhằm làm đẹp số Stage 1: phần đó không nằm trong upload scope và phép đo chính thức.
- Tự đổi activation/weight quantization sang FP8 một thành phần hoặc integer recoding mà chưa phân tích error. Tolerance K3 nhỏ nhất và BF16 rounding có thể khuếch đại ở cancellation.

## 11. Trình tự tối ưu để tiến tới score cao

1. Chốt một baseline hiện hành: source hashes + toolchain + public fixture mới + full binary + all-nine correctness + ba medians. Nghiên cứu đã hoàn tất gate public fixture này ở job 78017. Với lần đổi source/toolchain/fixture tiếp theo, xác minh lại trước khi dùng tốc độ làm frontier.
2. Giữ K2 direct VRF làm candidate ưu tiên. Gate public correctness và bounded A–B–B–A đã hoàn tất; hai cặp đều thuận hướng. Trước promotion chính thức, dùng thêm bằng chứng lặp cùng điều kiện nếu mức lợi nhỏ vẫn nằm trong noise budget. Đây là thay đổi ít ảnh hưởng số học, ưu tiên hơn rewrite lớn.
3. Áp dụng direct VRF từng site ở K3/K1 và đo lại. Ghi site nào bị che bởi DMA, site nào giảm exposed dependency; chỉ kết hợp sau khi mỗi site có evidence.
4. Thử K1 RoPE DM transfer, K3 partial-sum cluster swap và K2 scalar exchange riêng biệt. Ngừng một topology nếu sync tăng vượt budget trước khi tốn job runtime.
5. Định lượng và giảm K3 lookup provisioning thông qua tile/chain structure; tiếp theo scale DMA alignment. Đây là nhóm cần nghiên cứu compiler nhiều nhất nhưng có thể mở trần bandwidth hiện tại.
6. Với candidate gần nhau, đóng băng A/B thành hai package, giữ nguyên SDK, fixture, profiling, CPU policy và full official batching. Có thể chạy thứ tự process A-B-B-A để giảm bias thời gian; không sửa thứ tự case bên trong official harness. Dùng phân bố tỷ lệ median hoặc log-ratio để thấy noise.
7. Promotion cần đủ điều kiện: build, full correctness, official median tuple, hash khớp, improvement lớn hơn noise hoặc tái lập. Stress test là gate phụ cho những biến đổi số học.
8. Chốt package thống nhất cả ba kernel. Kiểm tra final submission ID, trạng thái terminal, log và hash package. Vì BTC dùng bản cuối, ngừng submit candidate chưa xác minh trước deadline; re-submit champion khi cần.

Mỗi experiment nên ghi: giả thuyết; parent identity; thay đổi duy nhất; source hash; static max-end/contexts/sync count; runtime output; numerical margin; official median tuple; quyết định; cơ chế giải thích kết quả. Các kết quả âm trong nghiên cứu này giúp thu hẹp search mà không biến một hạn chế cũ thành kết luận vĩnh viễn.

## 12. Phần chưa thể kết luận

Không có bằng chứng rằng kernel hiện tại hoặc một probe đạt optimum toàn cục. Bandwidth peak chỉ cho cận lý tưởng; compiler/resource constraints và chính thức rerun vẫn quyết định. Các ước lượng phần trăm trong roadmap là độ nhạy của score hoặc cận của một mô hình, không phải hứa hẹn speedup.

Stage 2 cho phép host orchestration, batching, weight residency, prefill/decode specialization và tối ưu KV-cache ở cấp model. Tuy nhiên metric và benchmark cụ thể chưa được chốt trong tài liệu được kiểm tra; các hướng đó có giá trị dài hạn nhưng không được tính như lợi ích score Stage 1 hiện tại.

Artifacts quan trọng cùng thư mục: `schedule-analysis.json`, `k1.schedule.json`, `k2.schedule.json`, `k3.schedule.json`, `source-hashes.json`, `evidence-hashes.json`, `runtime-sha256.txt`, `fixture-generation.log`, `public-build.log`, `leaderboard-observation.md`, `arena-result.md` và ba thư mục `probes/`. Script `compile-current.sh` và `analyze_schedules.py` cho phép tái lập phép đo tĩnh trên cùng source.
