//! RMSNorm on genuinely replicated 240-element pieces, gathered by the Switch into every slice.
//!
//! The hidden state is loaded as 16 pieces of 240 elements, 16 real copies of each (`Rep16` is a
//! real axis, not `1 # 16`), so all 256 slices of a cluster hold live data from the first load on.
//! Each group of 16 slices normalizes the whole vector on its own, in parallel with the other 15.
//! The last pass gathers along the ring with `Broadcast1`: every slice of a ring ends up with all
//! of that ring's pieces, which is a real copy in all 256 slices for ~500 MainContext cycles.
use furiosa_opt_std::prelude::*;

use crate::axes::H;
use super::{Rep16, Ring16, Ring4, Ring8};
use crate::{Chip, EPS};

const H_F32: f32 = H::SIZE as f32;

macro_rules! normalize_gathered {
    ($name:ident, $pieces:ty, $gathered:ty, $time:ty, $cfg:expr, $collect_time:ty, $element:ty) => {
        pub(crate) fn $name<Cluster: M>(
            ctx: &mut Context,
            x: &HbmTensor<bf16, Chip, m![H]>,
            rms_weight: &HbmTensor<bf16, Chip, m![H]>,
        ) -> DmTensor<bf16, Chip, Cluster, $gathered, $element> {
            let x: DmTensor<bf16, Chip, Cluster, $pieces, m![H % 240]> = x.to_dm(&mut ctx.tdma);

            let mean_square: DmTensor<f32, Chip, Cluster, $pieces, m![1 # 8]> = ctx
                .main
                .begin(x.view())
                .fetch::<m![H / 16 % 15], m![H % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![H / 8 % 30], m![H % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
                .vector_stash()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
                .vector_fp_div(H_F32)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            let reduced: DmTensor<f32, Chip, Cluster, m![Rep16, Ring16], m![1 # 8]> = ctx
                .main
                .begin(mean_square.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .vector_init()
                .vector_inter_slice_reduce::<m![Rep16, Ring16], m![1]>(InterSliceReduceOpF32::Add)
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_clip(ClipBinaryOpF32::Add, EPS)
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            let rms: DmTensor<f32, Chip, Cluster, m![Rep16, Ring16], m![1 # 8]> = ctx
                .main
                .begin(reduced.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_trim::<m![1 # 4]>()
                .vector_fp_unary(FpUnaryOp::Sqrt)
                .vector_widen_pad::<m![1 # 8]>()
                .vector_final()
                .commit_trim::<m![1 # 8]>()
                .commit();
            let rms: DmTensor<f32, Chip, Cluster, $pieces, m![1 # 8]> = unsafe { rms.reshape() };

            let weight_dm: DmTensor<bf16, Chip, Cluster, $pieces, m![H % 240]> = rms_weight.to_dm(&mut ctx.tdma);
            let weight_vrf: VrfTensor<f32, Chip, Cluster, $pieces, m![H % 240]> = ctx
                .sub
                .begin(weight_dm.view())
                .fetch::<m![H / 16 % 15], m![H % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![H / 8 % 30], m![H % 8]>()
                .to_vrf();
            let rms_vrf: VrfTensor<f32, Chip, Cluster, $pieces, m![1 # 8]> = ctx
                .sub
                .begin(rms.view())
                .fetch::<m![1], m![1 # 8]>()
                .collect::<m![1], m![1 # 8]>()
                .to_vrf();
            let normalized: DmTensor<f32, Chip, Cluster, $pieces, m![H % 240]> = ctx
                .main
                .begin(x.view())
                .fetch::<m![H / 16 % 15], m![H % 16]>()
                .fetch_cast::<f32>()
                .collect::<m![H / 8 % 30], m![H % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
                .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
                .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
                .vector_final()
                .commit_trim::<m![H % 8]>()
                .commit();

            ctx.main
                .begin(normalized.view())
                .fetch::<m![1], m![H % 240]>()
                .switch::<$gathered, $time>($cfg)
                .collect::<$collect_time, m![H % 8]>()
                .cast::<bf16, m![H % 8 # 16]>()
                .commit_trim::<m![H % 8]>()
                .commit()
        }
    };
}


// Quarters of H in every slice; physically the same slice order as `m![<64 row groups>, H / 960]`.
normalize_gathered!(
    normalize_quarters,
    m![Rep16, H / 240 % 4, H / 960],
    m![Rep16, Ring4, H / 960],
    m![H / 240 % 4],
    SwitchConfig::Broadcast1 { slice1: 4, slice0: 4 },
    m![H / 8 % 120],
    m![H % 960]
);
// Halves of H in every slice; the same slice order as `m![<128 row groups>, H / 1920]`.
normalize_gathered!(
    normalize_halves,
    m![Rep16, H / 240 % 8, H / 1920],
    m![Rep16, Ring8, H / 1920],
    m![H / 240 % 8],
    SwitchConfig::Broadcast1 { slice1: 8, slice0: 2 },
    m![H / 8 % 240],
    m![H % 1920]
);
// The whole of H in every slice.
normalize_gathered!(
    normalize_everywhere,
    m![Rep16, H / 240],
    m![Rep16, Ring16],
    m![H / 240],
    SwitchConfig::Broadcast1 { slice1: 16, slice0: 1 },
    m![H / 8],
    m![H]
);
