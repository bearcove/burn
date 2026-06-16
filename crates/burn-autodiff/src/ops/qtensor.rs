use burn_backend::{
    Backend, ExecutionError, TensorData,
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
            type State = QuantizedTensor<B>;

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                _checkpointer: &mut Checkpointer,
            ) {
                let weight = ops.state;
                unary::<B, _>(ops.parents, ops.node, grads, |grad| {
                    // forward y = x @ Wᵀ with W = dequant(weight) of shape [out, in];
                    // grad_x = grad_y @ W  ([m, out] @ [out, in] = [m, in]).
                    let w = B::dequantize(weight, FloatDType::F32);
                    B::float_matmul(grad, w)
                });
            }
        }

        match QLinear
            .prepare::<C>([activation.node.clone()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                weight.clone(),
                B::q_linear(activation.primitive, weight),
            ),
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
