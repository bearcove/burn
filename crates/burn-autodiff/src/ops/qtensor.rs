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
                    // The plain `q_linear` forward does NOT apply the forward-RHT (prerot): the
                    // caller pre-rotates the activation with a SEPARATE `rht_forward` (reshape +
                    // mul + matmul, fully differentiable), whose own autodiff supplies the adjoint
                    // Rᵀ. The forward here is just `y = x_rot @ dequant(W_rot)ᵀ`, so the backward
                    // w.r.t. the (already pre-rotated) activation is simply `grad_y @ dequant(W_rot)`.
                    //
                    // BUG FIX: this previously also applied `inverse_rht` here, which DOUBLE-rotated
                    // the gradient (Rᵀ here + Rᵀ again from the caller's `rht_forward`), producing a
                    // gradient ~orthogonal to the truth (cosine ≈ 0) — silently corrupting every
                    // QLoRA training run. Caught by a finite-difference oracle (DISTILL_KV_GRADCHECK).
                    //
                    // Dequant + matmul in the ACTIVATION's precision (f16/bf16 → tensor cores, half-
                    // byte weight; q_linear's output is f32 so grad arrives f32, cast to fdt).
                    let fdt: FloatDType = act_dtype.into();
                    let w = B::dequantize(weight, fdt);
                    let g = B::float_cast(grad, fdt);
                    B::float_matmul(g, w) // [m, in] = grad w.r.t. the pre-rotated activation
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

