from pathlib import Path


path = Path("src/device/shared/mlp.rs")
raw = path.read_bytes().decode("utf-8")
nl = "\r\n" if "\r\n" in raw else "\n"

function_anchor = raw.index("fn project_down_matrix(")
anchor = raw.index("let native_tile:", function_anchor)
start = raw.index("{" + nl + "let block_scale:", anchor)
suffix = "}" + nl + "}" + nl + nl + nl + "    // Keep both compensation parts"
suffix_at = raw.index(suffix, start)
end = suffix_at + len("}" + nl)  # consume the final microtile block, keep the outer native-tile brace


def block(rows: int, offset: int) -> str:
    lines = [
        "{",
        f"let block_scale: VrfTensor<f32, Chip, DownCluster, DownRowsByColumns, m![H % 15 = {rows}, L / 16 % 480]> = ctx.sub",
        f"            .begin(scale.view().tile::<m![H % 15], {rows}, m![H % 15 = {rows} # 15, L / 16 % 480]>({offset}))",
        f"            .fetch::<m![H % 15 = {rows}], m![L / 16 % 480]>()",
        "            .fetch_cast::<f32>()",
        f"            .collect::<m![H % 15 = {rows}, L / 128 % 60], m![L / 16 % 8]>()",
        "            .to_vrf();",
        "        ctx.main",
        f"            .begin(native_tile.view().tile::<m![H % 15 = 15], {rows}, m![H % 15 = {rows} # 15, L % 7680]>({offset}))",
        f"            .fetch::<m![H % 15 = {rows}, L / 32 % 240], m![L % 32]>()",
        f"            .collect::<m![H % 15 = {rows}, L / 32 % 240], m![L % 32]>()",
        f"            .contract_outer::<m![H % 15 = {rows}, L / 64 % 120], m![L % 64], _, _, _>(x_trf)",
        "            .contract_packet::<m![L / 16 % 4]>()",
        f"            .contract_time::<m![H % 15 = {rows}, L / 64 % 120]>()",
        f"            .contract_lane::<m![H % 15 = {rows}, L / 64 % 120, Dummy2], m![L / 16 % 4 # 8]>(LaneMode::Sequential)",
        "            .vector_init()",
        "            .vector_intra_slice_tag(TagMode::Zero)",
        "            .vector_narrow_trim::<m![L / 16 % 4]>()",
        "            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &block_scale)",
        "            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &input_scale_vrf)",
        f"            .vector_intra_slice_reduce::<L, m![H % 15 = {rows}, Dummy2], m![1 # 4]>(IntraSliceReduceOpF32::Add)",
        "            .vector_widen_pad::<m![1 # 8]>()",
        "            .vector_final()",
        f"            .transpose::<m![H % 15 = {rows}], m![Dummy2 # 8]>()",
        "            .commit_trim::<m![Dummy2]>()",
        f"            .commit_view(partials.view_mut().tile::<m![H % 15], {rows}, m![H % 15 = {rows} #{{!}} 15, Dummy2]>({offset}));",
        "    ",
        "}",
    ]
    return nl.join(lines) + nl


replacement = "".join(block(rows, offset) for rows, offset in ((4, 0), (4, 4), (4, 8), (3, 12)))
updated = raw[:start] + replacement + raw[end:]
if updated == raw:
    raise SystemExit("no change")
path.write_bytes(updated.encode("utf-8"))
