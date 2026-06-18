use burn_backend::{
    Backend, DType, ExecutionError, TensorData, TensorMetadata,
    ops::QTensorOps,
    quantization::QuantizationParametersPrimitive,
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};
use burn_std::{FloatDType, IntDType, QuantScheme, Shape};

use crate::{
    Autodiff,
    checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
    grads::Gradients,
    ops::{Backward, Ops, OpsKind, unary},
    tensor::AutodiffTensor,
};

impl<B: Backend, C: CheckpointStrategy> QTensorOps<Self> for Autodiff<B, C> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        B::q_from_data(data, device)
    }

    // QAT fake-quant support: the quantized tensor cannot carry an autodiff
    // node (`QuantizedTensorPrimitive = B::QuantizedTensorPrimitive`), so
    // `quantize`/`dequantize` are *value-only* — they run on the inner backend
    // and `dequantize` returns a fresh untracked float. The straight-through
    // gradient is the caller's responsibility, e.g.
    // `x + (x.quantize_dynamic(s).dequantize() - x).detach()`, which is exactly
    // how the fake-quant is used (round-trip value, identity gradient).
    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        B::quantize(
            tensor.primitive,
            scheme,
            QuantizationParametersPrimitive {
                scales: qparams.scales.primitive,
            },
        )
    }

    fn quantize_dynamic(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
    ) -> QuantizedTensor<Self> {
        B::quantize_dynamic(tensor.primitive, scheme)
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        AutodiffTensor::new(B::dequantize(tensor, dtype))
    }

    // W4A16 dequant-on-read matmul, differentiable. The activation is the tracked
    // input; the quantized weight is a FROZEN constant (it carries no autodiff
    // node — `QuantizedTensorPrimitive = B::QuantizedTensorPrimitive`), so it's
    // saved as backward state, not a parent. Forward is the inner fused kernel
    // (`y = x @ Wᵀ`, no dense materialization); backward propagates only the
    // activation gradient `grad_x = grad_y @ dequant(W)` (dequantized transiently
    // for the backward matmul, then freed — so training doesn't hold a dense tower).
    fn q_linear(activation: FloatTensor<Self>, weight: QuantizedTensor<Self>) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct QLinear;

        impl<B: Backend> Backward<B, 1> for QLinear {
            // State carries the activation's float dtype so the backward matmul runs at the
            // SAME precision as the forward (f16/bf16 activation → f16/bf16 backward → tensor
            // cores + half-size dense weight), and grad_x comes out in the activation's dtype
            // (matching the parent node) instead of always f32.
            type State = (QuantizedTensor<B>, DType);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let (weight, act_dtype) = ops.state;
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    // q_linear's forward applies bee's per-32-block forward-RHT (prerot)
                    // to the activation — `R = (1/√32)·H·D_s` (sign, then Walsh-Hadamard)
                    // — then matmuls the RHT-space weight: y = R(x) @ dequant(W_rot)ᵀ.
                    // So grad_x = Rᵀ(grad_y @ dequant(W_rot)), with the adjoint
                    // `Rᵀ = (1/√32)·D_s·H` (Hadamard, then sign). `dequantize` returns the
                    // RHT-space (rotated) weight; the inverse-RHT lives here.
                    //
                    // Dequant the weight + run the matmul in the ACTIVATION's precision (the
                    // big win when that's f16/bf16: tensor cores + the dense weight is half the
                    // bytes — q_linear's output is always f32, so grad arrives f32 and is cast
                    // down). The RHT adjoint stays in f32 (f16 Hadamard rounding is lossy).
                    let fdt: FloatDType = act_dtype.into();
                    let w = B::dequantize(weight, fdt);
                    let g = B::float_cast(grad, fdt);
                    let tmp = B::float_matmul(g, w); // [m, in] in RHT (rotated) space
                    let tmp_f32 = B::float_cast(tmp, FloatDType::F32);
                    B::float_cast(inverse_rht::<B>(tmp_f32), fdt)
                });
            }
        }

        match QLinear
            .prepare::<C>([activation.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => {
                let act_dtype = activation.primitive.dtype();
                let out = B::q_linear(activation.primitive, weight.clone());
                prep.finish((weight, act_dtype), out)
            }
            OpsKind::UnTracked(prep) => prep.finish(B::q_linear(activation.primitive, weight)),
        }
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        B::q_device(tensor)
    }

    fn q_to_device(
        _tensor: QuantizedTensor<Self>,
        _device: &Device<Self>,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        B::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        B::q_into_data(tensor).await
    }

    fn q_swap_dims(
        _tensor: QuantizedTensor<Self>,
        _dim1: usize,
        _dim2: usize,
    ) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_permute(_tensor: QuantizedTensor<Self>, _axes: &[usize]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_flip(_tensor: QuantizedTensor<Self>, _axes: &[usize]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_argmax(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::q_argmax(tensor, dim, out_dtype)
    }

    fn q_argmin(tensor: QuantizedTensor<Self>, dim: usize, out_dtype: IntDType) -> IntTensor<Self> {
        B::q_argmin(tensor, dim, out_dtype)
    }
}

/// Apply bee's inverse-RHT (the adjoint of q_linear's forward prerot) per 32-block
/// along the inner dimension of `tmp` `[m, in]`: `Rᵀ = (1/√32)·D_s·H` — Walsh-
/// Hadamard, then the ±1 sign pattern, then `1/√32`. Used by the `q_linear`
/// backward so the activation gradient is consistent with the served forward.
fn inverse_rht<B: Backend>(tmp: FloatTensor<B>) -> FloatTensor<B> {
    use burn_std::Shape;
    const INV_SQRT32: f32 = 0.176_776_69; // 1/√32
    // bee's 32-wide ±1 RHT sign pattern (mirror of burn-cubecl's RHT_SIGNS_TABLE),
    // pre-scaled by 1/√32 so a single elementwise mul applies `D_s` and the norm.
    const SIGNS: [f32; 32] = [
        1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
        -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
    ];

    let device = B::float_device(&tmp);
    let shape = tmp.shape();
    let (m, k) = (shape[0], shape[1]);
    let nblk = k / 32;

    // Natural-order Walsh-Hadamard H[i,j] = (-1)^popcount(i&j) (matches cubek's
    // radix-2 butterfly). Symmetric, so `WHT(v) = (v · H)` row-wise.
    let mut h = alloc::vec![0.0f32; 32 * 32];
    for i in 0..32usize {
        for j in 0..32usize {
            h[i * 32 + j] = if (i & j).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
        }
    }
    let signs_scaled: alloc::vec::Vec<f32> = SIGNS.iter().map(|&s| s * INV_SQRT32).collect();

    let h32 = B::float_from_data(TensorData::new(h, Shape::new([32, 32])), &device);
    let signs = B::float_from_data(TensorData::new(signs_scaled, Shape::new([1, 32])), &device);

    let t = B::float_reshape(tmp, Shape::new([m * nblk, 32]));
    let wht = B::float_matmul(t, h32); // (v · H) per row = WHT(v)
    let signed = B::float_mul(wht, signs); // ⊙ (sign / √32)  — broadcast [N,32]·[1,32]
    B::float_reshape(signed, Shape::new([m, k]))
}
