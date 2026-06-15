use crate::kernel::into_contiguous;
use crate::CubeRuntime;
use crate::{ops::empty_qtensor_optimized, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_elem_type;
use burn_backend::quantization::QuantMode;
use burn_backend::{quantization::QuantScheme, TensorMetadata};

/// Convert the tensor to a lower precision data type based on the quantization scheme and parameters.
pub fn quantize<R>(
    tensor: CubeTensor<R>,
    scheme: &QuantScheme,
    scale: CubeTensor<R>,
) -> CubeTensor<R>
where
    R: CubeRuntime,
{
    let output = empty_qtensor_optimized(tensor.shape(), *scheme, &tensor.device);
    let (out_values, out_params) = output.clone().quantized_handles().unwrap();
    let dtype = tensor.dtype;

    if matches!(scheme.mode, QuantMode::Codebook) {
        // TQ codebook activations: cubek's dedicated activation-quant (forward-RHT
        // + RMS + Lloyd refine + dense-pack) writes BOTH codes (out_values) and
        // scales (out_params), computing its own scale — the passed `scale` is
        // ignored. The generic launch_ref panics on dense codebook; this is the
        // wiring that lets `Tensor::quantize_dynamic(Q6F)` stay high-level.
        let tensor = into_contiguous(tensor);
        let shape = tensor.shape();
        let nd = shape.num_dims();
        let k = shape[nd - 1];
        let m = shape.num_elements() / k;
        cubek::quantization::qa_matmul::launch_activation_quant::<R>(
            &output.client,
            tensor.handle.clone(),
            out_values.handle.clone(),
            out_params.handle.clone(),
            m,
            k,
        );
    } else {
        cubek::quantization::quantize::launch_ref(
            &output.client,
            tensor.binding(),
            out_values.binding(),
            scale.binding(),
            out_params.binding(),
            scheme,
            dtype_to_elem_type(dtype),
        )
        .expect("Kernel to never fail");
    }

    output
}
