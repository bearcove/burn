use burn_backend::{
    Backend, ExecutionError, TensorData,
    ops::QTensorOps,
    quantization::QuantizationParametersPrimitive,
    tensor::{Device, FloatTensor, IntTensor, QuantizedTensor},
};
use burn_std::{FloatDType, IntDType, QuantScheme, Shape};

use crate::{Autodiff, checkpoint::strategy::CheckpointStrategy, tensor::AutodiffTensor};

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
