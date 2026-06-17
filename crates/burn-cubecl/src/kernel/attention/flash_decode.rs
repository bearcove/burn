//! Fused flash-DECODE attention: a single kernel for the streaming-decode shape
//! (few query rows, `n_q` small) that the general flash kernel loses on (it's tuned
//! for `SeqQ >> 1` tiling). One unit per `(query_row, head)` runs the whole online-
//! softmax over the cached key window — no score materialization, no transposes, no
//! GQA expand, no separate softmax reduction. Replaces the
//! transpose+scores-matmul+softmax+value-matmul chain (~6 launches/layer) with ONE.
//!
//! Layout: `q [n_q, n_heads, head_dim]`, `k`/`v [n_k, n_kv, head_dim]`, additive
//! `mask [n_q, n_k]` (`-inf` where masked), `out [n_q, n_heads, head_dim]`. GQA: query
//! head `h` reads kv head `h / groups` (heads grouped consecutively per kv head).

use crate::{CubeRuntime, tensor::CubeTensor};
use crate::ops::numeric::empty_device_dtype;
use burn_backend::{DType, Shape, TensorMetadata};
use cubecl::{CubeCount, CubeDim, prelude::*};

#[cube(launch)]
fn flash_decode_kernel<F: Float>(
    q: &Tensor<F>,
    k: &Tensor<F>,
    v: &Tensor<F>,
    mask: &Tensor<F>,
    out: &mut Tensor<F>,
    n_q: usize,
    n_k: usize,
    scale: f32,
    #[comptime] n_heads: usize,
    #[comptime] n_kv: usize,
    #[comptime] head_dim: usize,
    #[comptime] plane: usize,
) {
    // One PLANE (warp) per (query_row, head): the `plane` lanes split `head_dim`, the
    // q·k dot is a `plane_sum`, and each lane keeps the online-softmax accumulator for
    // its own `head_dim/plane` channels. 32× the threads of a scalar unit-per-head, and
    // the K/V cache read is parallel across the warp.
    let qh = CUBE_POS_X as usize; // (query_row, head) index = one cube = one plane
    let total = n_q * n_heads;
    if qh >= total {
        terminate!();
    }
    let i = qh / n_heads; // query row
    let h = qh % n_heads; // query head
    let groups = comptime!(n_heads / n_kv);
    let kv = h / groups;
    let lane = UNIT_POS_X as usize; // 0..plane
    let per_lane = comptime!(head_dim / plane); // channels this lane owns

    let q_off = i * n_heads * head_dim + h * head_dim;
    let scale_f = F::cast_from(scale);

    // Masked keys carry additive −inf ⇒ exp(s−m)=0, no branch (key 0 always visible).
    let mut m = F::cast_from(-1.0e30_f32);
    let mut l = F::from_int(0);
    let mut acc = Array::<F>::new(per_lane);
    #[unroll]
    for t in 0..per_lane {
        acc[t] = F::from_int(0);
    }

    for j in 0..n_k {
        let masked = mask[i * n_k + j];
        let k_off = j * n_kv * head_dim + kv * head_dim;
        let mut partial = F::from_int(0);
        #[unroll]
        for t in 0..per_lane {
            let d = lane + t * plane;
            partial += q[q_off + d] * k[k_off + d];
        }
        let dot = plane_sum(partial); // full q·k across the warp
        let s = dot * scale_f + masked;

        let m_new = if s > m { s } else { m };
        let alpha = F::exp(m - m_new);
        let p = F::exp(s - m_new);
        l = l * alpha + p;
        let v_off = j * n_kv * head_dim + kv * head_dim;
        #[unroll]
        for t in 0..per_lane {
            let d = lane + t * plane;
            acc[t] = acc[t] * alpha + p * v[v_off + d];
        }
        m = m_new;
    }

    let out_off = i * n_heads * head_dim + h * head_dim;
    let inv_l = F::recip(l);
    #[unroll]
    for t in 0..per_lane {
        let d = lane + t * plane;
        out[out_off + d] = acc[t] * inv_l;
    }
}

/// Launch the fused flash-decode attention. Returns `out [n_q, n_heads, head_dim]`.
pub fn flash_decode_attention<R: CubeRuntime>(
    q: CubeTensor<R>,
    k: CubeTensor<R>,
    v: CubeTensor<R>,
    mask: CubeTensor<R>,
    scale: f32,
) -> CubeTensor<R> {
    let q = crate::kernel::into_contiguous(q);
    let k = crate::kernel::into_contiguous(k);
    let v = crate::kernel::into_contiguous(v);
    let mask = crate::kernel::into_contiguous(mask);

    let client = q.client.clone();
    let dtype = q.dtype;
    let qs = q.shape();
    let ks = k.shape();
    let (n_q, n_heads, head_dim) = (qs[0], qs[1], qs[2]);
    let (n_k, n_kv) = (ks[0], ks[1]);

    let out = empty_device_dtype(
        client.clone(),
        q.device.clone(),
        Shape::new([n_q, n_heads, head_dim]),
        dtype,
    );

    let total = n_q * n_heads;
    // One plane (warp) per (query, head). 32 lanes = a CUDA warp; `plane_sum` reduces
    // across exactly the cube's units, so cube_dim must equal the plane width.
    let plane: usize = 32;
    let cube_dim = CubeDim::new_1d(plane as u32);
    let cube_count = CubeCount::Static(total as u32, 1, 1);

    macro_rules! launch {
        ($f:ty) => {
            flash_decode_kernel::launch::<$f, R>(
                &client,
                cube_count,
                cube_dim,
                q.into_tensor_arg(),
                k.into_tensor_arg(),
                v.into_tensor_arg(),
                mask.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                n_q,
                n_k,
                scale,
                n_heads,
                n_kv,
                head_dim,
                plane,
            )
        };
    }
    match dtype {
        DType::F32 => launch!(f32),
        DType::F16 => launch!(burn_backend::f16),
        other => panic!("flash_decode_attention: unsupported dtype {other:?}"),
    }
    out
}
