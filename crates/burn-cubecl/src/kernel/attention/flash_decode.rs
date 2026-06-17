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
use cubecl::{CubeDim, CubeCount, prelude::*};

#[cube(launch)]
fn flash_decode_kernel<F: Numeric>(
    q: &Tensor<F>,
    k: &Tensor<F>,
    v: &Tensor<F>,
    mask: &Tensor<F>,
    out: &mut Tensor<F>,
    n_q: u32,
    n_k: u32,
    #[comptime] n_heads: u32,
    #[comptime] n_kv: u32,
    #[comptime] head_dim: u32,
    #[comptime] scale: f32,
) {
    // One unit per (query_row, head).
    let total = n_q * n_heads;
    if ABSOLUTE_POS >= total {
        terminate!();
    }
    let i = ABSOLUTE_POS / n_heads; // query row
    let h = ABSOLUTE_POS % n_heads; // query head
    let groups = comptime!(n_heads / n_kv);
    let kv = h / groups;

    let q_off = i * n_heads * head_dim + h * head_dim;

    // Online-softmax running state.
    let mut m = F::from_int(-1000000000); // running max (−inf proxy)
    let mut l = F::from_int(0); // running denom
    let mut acc = Array::<F>::new(head_dim);
    #[unroll]
    for d in 0..head_dim {
        acc[d] = F::from_int(0);
    }

    for j in 0..n_k {
        let masked = mask[i * n_k + j];
        // Skip fully-masked keys (−inf additive) — they contribute nothing.
        if masked > F::from_int(-1000000000) {
            let k_off = j * n_kv * head_dim + kv * head_dim;
            let mut dot = F::from_int(0);
            #[unroll]
            for d in 0..head_dim {
                dot += q[q_off + d] * k[k_off + d];
            }
            let s = dot * F::cast_from(scale) + masked;

            let m_new = Max::max(m, s);
            let alpha = Exp::exp(m - m_new);
            let p = Exp::exp(s - m_new);
            l = l * alpha + p;
            let v_off = j * n_kv * head_dim + kv * head_dim;
            #[unroll]
            for d in 0..head_dim {
                acc[d] = acc[d] * alpha + p * v[v_off + d];
            }
            m = m_new;
        }
    }

    let out_off = i * n_heads * head_dim + h * head_dim;
    let inv_l = recip(l);
    #[unroll]
    for d in 0..head_dim {
        out[out_off + d] = acc[d] * inv_l;
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

    let qs = q.shape();
    let ks = k.shape();
    let (n_q, n_heads, head_dim) = (qs[0], qs[1], qs[2]);
    let (n_k, n_kv) = (ks[0], ks[1]);

    let out = empty_device_dtype(
        q.client.clone(),
        q.device.clone(),
        Shape::new([n_q, n_heads, head_dim]),
        q.dtype,
    );

    let total = (n_q * n_heads) as u32;
    let cube_dim = CubeDim::new(256, 1, 1);
    let cubes = total.div_ceil(256).max(1);
    let cube_count = CubeCount::Static(cubes, 1, 1);

    macro_rules! launch {
        ($f:ty) => {
            flash_decode_kernel::launch::<$f, R>(
                &q.client,
                cube_count,
                cube_dim,
                q.as_tensor_arg(1),
                k.as_tensor_arg(1),
                v.as_tensor_arg(1),
                mask.as_tensor_arg(1),
                out.as_tensor_arg(1),
                ScalarArg::new(n_q as u32),
                ScalarArg::new(n_k as u32),
                n_heads as u32,
                n_kv as u32,
                head_dim as u32,
                scale,
            )
        };
    }
    match q.dtype {
        DType::F32 => launch!(f32),
        DType::F16 => launch!(burn_backend::f16),
        other => panic!("flash_decode_attention: unsupported dtype {other:?}"),
    }
    out
}
