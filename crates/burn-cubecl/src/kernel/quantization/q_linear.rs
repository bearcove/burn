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

    // `launch_panel` reads the activation as contiguous f32 `[m, k]`. The served
    // scheme is W4A16, so the activation arrives f16 — cast it up (a fused f16 arm
    // in the kernel is a perf follow-up).
    let activation = if activation.dtype == DType::F32 {
        activation
    } else {
        crate::kernel::cast::cast(activation, DType::F32)
    };
    let activation = into_contiguous(activation);
    let output = empty_device_dtype(
        activation.client.clone(),
        activation.device.clone(),
        Shape::new([m, n]),
        DType::F32,
    );
    let (codes, scales) = weight.quantized_handles().unwrap();

    cubek::quantization::qa_matmul::launch_panel::<R>(
        &output.client,
        scheme.value,
        activation.handle.clone(),
        codes.handle.clone(),
        scales.handle.clone(),
        super::tables::codebook_for(scheme.value),
        // Non-empty signs ⇒ fold bee's forward-RHT (prerot) into the matvec.
        super::tables::rht_signs(),
        output.handle.clone(),
        m,
        n,
        k,
    );

    output
}
