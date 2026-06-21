use crate::CubeRuntime;
use crate::kernel::into_contiguous;
use crate::{ops::empty_qtensor_optimized, tensor::CubeTensor};
use burn_backend::cubecl::dtype_to_elem_type;
use burn_backend::quantization::QuantMode;
use burn_backend::{TensorMetadata, quantization::QuantScheme};

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
        // The codebook kernels are data-driven: the centroid table and the RHT
        // sign pattern are comptime-injected, chosen by `scheme.value`, so
        // TQ4/TQ6/… all route through the same launch.
        let codebook = super::tables::codebook_for(scheme.value);
        let rht_signs = super::tables::rht_signs();
        cubek::quantization::qa_matmul::launch_activation_quant::<R>(
            &output.client,
            scheme.value,
            tensor.handle.clone(),
            out_values.handle.clone(),
            out_params.handle.clone(),
            codebook,
            rht_signs,
            m,
            k,
        );
    } else if matches!(
        scheme.store,
        burn_backend::quantization::QuantStore::PackedU32Dense(_)
    ) {
        // Symmetric (Q4S/…) dense: cubek's symmetric activation-quant (forward-RHT
        // + per-half-block maxabs scale + round-to-signed + dense pack). The
        // generic launch_ref panics on dense; this is the symmetric analogue of
        // the codebook branch above, so `quantize_dynamic(Q4S)` stays high-level.
        let tensor = into_contiguous(tensor);
        let shape = tensor.shape();
        let nd = shape.num_dims();
        let k = shape[nd - 1];
        let m = shape.num_elements() / k;
        let rht_signs = super::tables::rht_signs();
        cubek::quantization::qa_matmul::launch_symmetric_activation_quant::<R>(
            &output.client,
            scheme.value,
            tensor.handle.clone(),
            out_values.handle.clone(),
            out_params.handle.clone(),
            rht_signs,
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
            super::tables::codebook_for(scheme.value),
            scheme,
            dtype_to_elem_type(dtype),
        )
        .expect("Kernel to never fail");
    }

    output
}
