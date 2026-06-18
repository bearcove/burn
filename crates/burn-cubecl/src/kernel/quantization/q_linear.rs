use crate::tensor::CubeTensor;
use crate::{CubeRuntime, kernel::into_contiguous, ops::numeric::empty_device_dtype};
use burn_backend::{DType, Shape, TensorMetadata};

/// Quantized linear `activation @ weightᵀ` (the W4A16 codebook path): the weight
/// stays **packed** and is dequantized **on read** inside cubek's panel matmul —
/// never materialized to dense float. `activation` is float `[m, k]`, `weight` is
/// a quantized `[n, k]` codebook tensor (`PackedU32Dense`, per-16 scales along
/// `k`); the result is float `[m, n]`.
///
/// This mirrors bee's served `tqN_1s_matvec_prerot`: the caller supplies the
/// activation already in prerot space (forward-RHT applied), the weight stays
/// rotated-and-packed, and the kernel dequantizes each weight column once.
pub fn q_linear<R: CubeRuntime>(activation: CubeTensor<R>, weight: CubeTensor<R>) -> CubeTensor<R> {
    // PLAIN matvec: caller supplies the activation already prerot'd (helix `rht_forward`).
    q_linear_inner(activation, weight, cubek::quantization::qa_matmul::RhtSigns(&[]), None, 0.0)
}

/// Like [`q_linear`] but the **forward-RHT (prerot) is applied IN-KERNEL** (bee's
/// `matvec_prerot`): the caller passes the un-rotated activation and the panel
/// rotates each 32-block once in registers. This avoids the separate `rht_forward`
/// op-stream (reshape+mul+matmul+mul_scalar) that does NOT fuse into the matmul
/// (matmul fusion is epilogue-only) — folding the rotation back where it's free.
pub fn q_linear_prerot<R: CubeRuntime>(
    activation: CubeTensor<R>,
    weight: CubeTensor<R>,
) -> CubeTensor<R> {
    q_linear_inner(activation, weight, super::tables::rht_signs(), None, 0.0)
}

/// Like [`q_linear_prerot`] but ALSO folds an RMSNorm (input_ln/post_ln) into the
/// gemv: the caller passes the UN-normed, un-rotated activation `[m,k]` plus the
/// RMSNorm `gamma [k]`; the panel computes `s=rsqrt(mean(h²)+eps)` in-kernel, stages
/// `h⊙gamma`, RHTs, and scales the output by `s`. Decode (m==1) only — removes the
/// separate rms_norm reduce launch. `eps` is the RMSNorm epsilon.
pub fn q_linear_prerot_norm<R: CubeRuntime>(
    activation: CubeTensor<R>,
    weight: CubeTensor<R>,
    gamma: CubeTensor<R>,
    eps: f32,
) -> CubeTensor<R> {
    q_linear_inner(activation, weight, super::tables::rht_signs(), Some(gamma), eps)
}

fn q_linear_inner<R: CubeRuntime>(
    activation: CubeTensor<R>,
    weight: CubeTensor<R>,
    rht: cubek::quantization::qa_matmul::RhtSigns,
    gamma: Option<CubeTensor<R>>,
    eps: f32,
) -> CubeTensor<R> {
    let scheme = match weight.dtype {
        DType::QFloat(scheme) => scheme,
        other => panic!("q_linear weight must be quantized, got {other:?}"),
    };

    let a_shape = activation.shape();
    let w_shape = weight.shape();
    let k = a_shape[a_shape.num_dims() - 1];
    let m = a_shape.num_elements() / k;
    let n = w_shape[0];
    assert_eq!(
        w_shape[w_shape.num_dims() - 1],
        k,
        "q_linear: weight inner dim must match activation inner dim"
    );

    // The kernel reads the activation in its NATIVE float type (f16 for A16, or
    // f32), casting per-element — no separate f32 materialization, and the f16
    // read halves the activation bandwidth (it's re-read once per output column).
    let activation = into_contiguous(activation);
    let output = empty_device_dtype(
        activation.client.clone(),
        activation.device.clone(),
        Shape::new([m, n]),
        DType::F32,
    );
    let (codes, scales) = weight.quantized_handles().unwrap();
    let cb = super::tables::codebook_for(scheme.value);
    // RMSNorm gamma (in-kernel fold): keep the contiguous copy alive for its handle;
    // when absent, pass the activation handle as an unread dummy (do_norm = false).
    let gamma_c = gamma.map(into_contiguous);
    let (gamma_handle, do_norm) = match &gamma_c {
        Some(g) => (g.handle.clone(), true),
        None => (activation.handle.clone(), false),
    };

    macro_rules! launch {
        ($f:ty) => {
            cubek::quantization::qa_matmul::launch_panel::<R, $f>(
                &output.client,
                scheme.value,
                activation.handle.clone(),
                codes.handle.clone(),
                scales.handle.clone(),
                cb,
                rht,
                output.handle.clone(),
                m,
                n,
                k,
                gamma_handle.clone(),
                do_norm,
                eps,
            )
        };
    }
    match activation.dtype {
        DType::F32 => launch!(f32),
        DType::F16 => launch!(burn_backend::f16),
        DType::BF16 => launch!(burn_backend::bf16),
        other => panic!("q_linear: unsupported activation dtype {other:?}"),
    }

    output
}
