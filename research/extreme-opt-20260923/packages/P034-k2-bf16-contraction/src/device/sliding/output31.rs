//! `ops::sliding_attention_output`: output19's projection and hop, with a RE-SHAPED TAIL.
//!
//! output19's tail after the hop reload is four serial Main/Sub commands whose total the device
//! trace puts at 4,445 cycles with the DMA idle:
//!   sum of squares (1,063) -> ring-16 reduce + eps + sqrt (661) -> `to_vrf` of the rms (439)
//!   -> final pass `z * s / rms * w + r` (2,276).
//! The last one is by far the biggest, and the only thing it does that the (same-length) sum of
//! squares pass does not is the **`DivF` by the rms**. So this module removes the division from the
//! final pass:
//!   * the rms pass computes 1 / rms instead of rms. `t = ms + eps` is stashed before the square
//!     root, and the FpDiv stage then divides the root by the stash: `sqrt(t) / t = 1 / sqrt(t)`.
//!     The stage order is Fp -> IntraSliceReduce -> FpDiv, so the root (Fp, FpFpu) and the division
//!     (a separate stage) live in ONE pass. That pass writes eight values per slice, so whatever the
//!     divider costs per element it costs it 8 times instead of 240 times.
//!   * the final pass then needs THREE multiplies (s, 1/rms, w) and the vector engine has two
//!     (`Mul0`, `Mul1`), so `sw = s * w` is precomputed in one small pass placed between the hop
//!     store and the reload -- i.e. inside the cross-cluster sync's wait, where the machine is idle
//!     anyway (the sync span ends when everything already issued has finished, so this work is
//!     free and it shortens the exposed part of the wait).
//! The sum of squares pass still needs `s` on its own, so the channel-scale VRF stays.

use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Gs, H, Ns, Qs};
use crate::device::layout::{OutputClusters, SlidingOutputColumns, SlidingOutputRows};
use crate::{Chip, EPS};

axes![Lq = 2, Vr = 16];
/// The tail runs on cluster 0 only (the cluster that stores): the final sync is then the cheap kind.
type Vc = m![1 # 2];

const H_F32: f32 = H::SIZE as f32;
const X_SCALE: f32 = 16.0;
/// z stays 16 times too large to the end; the RMSNorm divides that out, given a matching epsilon.
const EPS_SCALED: f32 = EPS * X_SCALE * X_SCALE;

type Levels = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![Lq, Qs % 256]>;

pub(crate) type Rows = SlidingOutputRows;
/// The tail's home: all 256 slices of cluster 0, each logical group covering 480 rows.
pub(crate) type Tail = m![1 # 16, H / 240];
pub(crate) type TailDm<D> = DmTensor<D, Chip, Vc, Tail, m![H % 240]>;
pub(crate) type TailVrf = VrfTensor<f32, Chip, Vc, Tail, m![H % 240]>;

pub(crate) type XTrf = TrfTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![1], m![Qs % 256]>;
pub(crate) type Z = DmTensor<bf16, Chip, OutputClusters, Rows, m![H % 120]>;

pub(crate) type WeightTile = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![H % 120, Qs % 256]>;

/// Every one of the 120 rows of every row slice: the two levels summed inside the pass.
pub(crate) fn contract_tile(device: &mut Device, weight: &WeightTile, x_trf: &XTrf, z: &mut Z) {
    let weight: DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![H % 120, Qs % 256]> = device.sub
        .begin(weight.view())
        .fetch::<m![H % 120, Qs / 8 % 32], m![Qs % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H % 120, Qs / 8 % 32], m![Qs % 8]>()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    device.main
        .begin(weight.view())
        .fetch::<m![H % 120, Qs / 16 % 16], m![Qs % 16]>()
        .collect::<m![H % 120, Qs / 16 % 16], m![Qs % 16]>()
        .contract_outer::<m![H % 120, Qs / 32 % 8], m![Qs % 32], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120]>()
        .contract_lane::<m![H % 120], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<Rows, m![H % 120]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 / 4], m![H % 120 % 4 # 16]>()
        .commit_trim::<m![H % 120 % 4]>()
        .commit_view(z.view_mut());
}

/// Preserve the power-of-two gain and BF16 activation directly in the TRF.
pub(crate) fn quantise_x(
    device: &mut Device,
    x: &DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]>,
) -> XTrf {
    let scaled: DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]> = device.main
        .begin(x.view())
        .fetch::<m![1], m![Qs % 256]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), X_SCALE)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<bf16, m![Qs % 8 # 16]>()
        .commit_trim::<m![Qs % 8]>()
        .commit();
    device.sub
        .begin(scaled.view())
        .fetch::<m![1], m![Qs % 256]>()
        .collect::<m![Qs / 16 % 16], m![Qs % 16]>()
        .to_trf()
}

pub(crate) fn project_normalize_add(
    device: &mut Device,
    x: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    // The whole weight in ONE DMA command: 120 rows per slice.
    let weight_dm: WeightTile = weight.view().to_dm(&mut device.tdma);

    // One DMA load hands every slice the 256 columns it contracts, once per row group.
    let x: DmTensor<bf16, Chip, OutputClusters, m![H / 120 % 16, Ns, Gs], m![Ds]> = x.to_dm(&mut device.tdma);
    let x: DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]> = unsafe { x.reshape() };
    let x_trf = quantise_x(device, &x);

    // 16 * (weight @ x), bf16, 120 rows per row slice.
    let mut z: Z = DmTensor::new();
    contract_tile(device, &weight_dm, &x_trf, &mut z);

    // Load each tail operand into an independent allocation, removing pool version dependencies.
    let residual_dm: TailDm<bf16> = residual_hbm.to_dm(&mut device.tdma);
    let residual: TailVrf = device
        .sub
        .begin(residual_dm.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();

    // The hop: all 3,840 projected values go to HBM in row order and return in H480 tail groups.
    let mut hop: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    let z_gathered: DmTensor<bf16, Chip, OutputClusters, m![1 # 256], m![H % 1920]> =
        z.to_dm(&mut device.tdma);
    z_gathered.view().to_hbm_view(&mut device.tdma, hop.view_mut());

    // These two are what the model puts between the store and the reload: they cover the sync.
    let scale_dm: TailDm<bf16> = weight_scale.to_dm(&mut device.tdma);
    let scale: TailVrf = device
        .sub
        .begin(scale_dm.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();
    // Keep RMS weights in an independent allocation so this read does not advance the shared
    // residual/channel-scale pool's version chain.
    let rms_dm: TailDm<bf16> = rms_weight.to_dm(&mut device.tdma);
    let rms_weight: TailVrf = device
        .sub
        .begin(rms_dm.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf();

    // sw = channel scale * RMSNorm weight, so that the final pass needs only two multiplies
    // (`sw` and `1 / rms`) and no division. Listed before the reload: the issuer hands it ove
    // during the cross-cluster sync's wait, where both the DMA and the tensor unit are idle.
    let sw: TailVrf = device
        .main
        .begin(scale_dm.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &rms_weight)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);

    let z_tail: TailDm<bf16> = hop.to_dm(&mut device.tdma);

    // Sum of squares of the 480 scaled values of each logical tail group, over H.
    let partial: DmTensor<f32, Chip, Vc, Tail, m![1 # 8]> = device
        .main
        .begin(z_tail.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    // Gather the 16 group partials explicitly within each 16-slice ring.
    // Each active tail slice receives all 16 H-group partials as time values.
    let mean: DmTensor<f32, Chip, Vc, m![1 # 16, Vr], m![1 # 8]> = device
        .main.begin(partial.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![1 # 16, Vr], m![H / 240]>(SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 })
        .collect::<m![H / 240], m![1 # 8]>()
        .vector_init().vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final().commit_trim::<m![1 # 8]>().commit();
    // The scalar arithmetic is unchanged; only the cross-slice sum is expressed explicitly.
    let inv_rms: VrfTensor<f32, Chip, Vc, m![1 # 16, Vr], m![1 # 8]> = device
        .main
        .begin(mean.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS_SCALED)
        .vector_stash()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div(Stash)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .to_vrf(&mut device.sub);
    // Same physical slice grouping as the former DM reshape followed by Sub fetch.
    let inv_rms_vrf: VrfTensor<f32, Chip, Vc, Tail, m![1 # 8]> = unsafe { inv_rms.reshape() };

    let out: TailDm<bf16> = device
        .main
        .begin(z_tail.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &sw)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &inv_rms_vrf)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &residual)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    out.view().to_hbm_view(&mut device.tdma, residual_hbm.view_mut());
}
