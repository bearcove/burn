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
    // PLAIN matvec: the forward-RHT (prerot) is now applied by the caller as a
    // fusable op on the activation (helix `rht_forward`), so the kernel must NOT
    // rotate again. Empty signs ⇒ plain dequant-on-read matmul (matches the fusion
    // q_matmul path, which is also plain).
    let rht = cubek::quantization::qa_matmul::RhtSigns(&[]);

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
