use burn_backend::{
    Bytes, DType, ExecutionError, Shape, SplitPolicy, TensorData, TensorMetadata, TensorPrimitive,
    get_device_settings,
    ops::QTensorOps,
    quantization::{
        QParamTensor, QuantLevel, QuantMode, QuantParam, QuantPropagation, QuantScheme, QuantValue,
        QuantizationParametersPrimitive, params_shape,
    },
    tensor::{Device, FloatTensor, QuantizedTensor},
};
use burn_std::{FloatDType, Metadata, Strides, strides};
use cubecl::server::{MemoryLayout, MemoryLayoutDescriptor, MemoryLayoutStrategy};
use cubecl::{e2m1x2, quant::scheme::QuantStore};

use crate::{
    CubeBackend, CubeRuntime,
    kernel::{self, matmul::MatmulStrategy},
    tensor::{CubeTensor, QParams},
};

use super::{into_data, permute, swap_dims};

/// Create a quantized tensor with packed values (u32).
fn new_qtensor_optimized<R: CubeRuntime>(
    data: Bytes,
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
) -> CubeTensor<R> {
    new_qtensor(data, shape, scheme, device, MemoryLayoutStrategy::Optimized)
}

/// Create a quantized tensor with packed values (u32).
fn new_qtensor<R: CubeRuntime>(
    data: Bytes,
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
    kind: MemoryLayoutStrategy,
) -> CubeTensor<R> {
    new_quantized(shape, scheme, device, Some(data), kind)
}

/// Create an empty quantized tensor.
pub fn empty_qtensor_optimized<R: CubeRuntime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
) -> CubeTensor<R> {
    empty_qtensor(shape, scheme, device, MemoryLayoutStrategy::Optimized)
}

/// Create an empty quantized tensor.
pub fn empty_qtensor<R: CubeRuntime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
    kind: MemoryLayoutStrategy,
) -> CubeTensor<R> {
    new_quantized(shape, scheme, device, None, kind)
}

fn new_quantized<R: CubeRuntime>(
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
    data: Option<Bytes>,
    alloc_kind: MemoryLayoutStrategy,
) -> CubeTensor<R> {
    let client = R::client(device);
    let shape: Shape = shape.into();
    let mut shape_value: Shape = shape.clone();

    let rank = shape.rank();
    let shape_last = shape[rank - 1];
    let num_quants = scheme.num_quants();

    let data_size = match scheme.store {
        QuantStore::PackedU32(_) => {
            if !shape_last.is_multiple_of(num_quants) {
                panic!("Can't store in u32")
            }
            shape_value[rank - 1] = shape_last.div_ceil(num_quants);
            size_of::<u32>()
        }
        // TQ codebook dense pack: a unit is lcm(value_bits, 32) bits =
        // `num_quants` codes in `size_bits_stored()/32` u32 words (Q6F: 16 codes
        // / 3 words). Uses the scheme's own arithmetic, not a hand-derived guess.
        QuantStore::PackedU32Dense(_) => {
            let words_per_unit = scheme.size_bits_stored() / 32;
            shape_value[rank - 1] = shape_last.div_ceil(num_quants) * words_per_unit;
            size_of::<u32>()
        }
        QuantStore::Native => match scheme.value {
            QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2 => {
                size_of::<i8>()
            }
            QuantValue::Q4F
            | QuantValue::Q4S
            | QuantValue::Q2F
            | QuantValue::Q2S
            | QuantValue::Q6F
            | QuantValue::E2M1 => {
                panic!("Can't store native sub-byte values")
            }
        },
        QuantStore::PackedNative(_) => match scheme.value {
            QuantValue::E2M1 => size_of::<e2m1x2>(),
            other => panic!("{other:?} doesn't support native packing"),
        },
    };

    let scales_dtype = match scheme.param {
        QuantParam::F32 => DType::F32,
        QuantParam::F16 => DType::F16,
        QuantParam::BF16 => DType::BF16,
        // Represented by U8 and reinterpreted in the kernel
        QuantParam::UE8M0 | QuantParam::UE4M3 => DType::U8,
    };

    let scales_shape = params_shape(&shape, scheme.level);
    let data_desc = MemoryLayoutDescriptor::new(alloc_kind, shape_value.clone(), data_size);
    let scales_desc =
        MemoryLayoutDescriptor::new(alloc_kind, scales_shape.clone(), scales_dtype.size());

    let mut tensors = match data {
        Some(data) => {
            let num_bytes = shape_value.num_elements() * data_size;

            match data.split(num_bytes, SplitPolicy::Shared) {
                Ok((bytes_data, bytes_scales)) => client
                    .create_tensors(vec![(data_desc, bytes_data), (scales_desc, bytes_scales)]),
                Err((data, _)) => client.create_tensors_from_slices(vec![
                    (data_desc, &data[..num_bytes]),
                    (scales_desc, &data[num_bytes..]),
                ]),
            }
        }
        None => client.empty_tensors(vec![data_desc, scales_desc]),
    };
    let MemoryLayout {
        memory: scales_handle,
        strides: scales_strides,
    } = tensors.remove(1);
    let MemoryLayout { memory, strides } = tensors.remove(0);

    let scales = QParamTensor {
        offset_start: scales_handle.offset_start.unwrap_or(0) as usize,
        offset_end: scales_handle.offset_end.unwrap_or(0) as usize,
        metadata: Metadata::new(scales_shape, scales_strides),
        dtype: scales_dtype,
    };
    let qparams = QParams { scales };

    CubeTensor::new_quantized(
        client,
        memory,
        shape,
        device.clone(),
        strides,
        DType::QFloat(scheme),
        qparams,
    )
}

/// Contiguous (row-major) strides for `shape`.
fn contiguous_strides(shape: &Shape) -> Strides {
    let ndims = shape.num_dims();
    let mut out = strides![0; ndims];
    let mut current = 1usize;
    for i in (0..ndims).rev() {
        out[i] = current;
        current *= shape[i];
    }
    out
}

/// Build a quantized [`CubeTensor`] whose packed codes AND scales are backed by
/// **external, caller-owned** memory (zero-copy weight load — no `memcpy`).
///
/// `ptr` is the address of the contiguous `[codes | scales]` span (the pack stores
/// `.qs` immediately followed by `.scales`, both page-aligned; verified gap=0), and
/// `data_bytes` / `scales_bytes` are their byte lengths. This wraps that span as a
/// single no-copy buffer with the scales sliced out by offset — identical layout to
/// the copy path ([`new_quantized`]), so [`CubeTensor::scales`] / `quantized_handles`
/// work unchanged.
///
/// # Safety
/// `ptr` must be page-aligned, valid for `data_bytes + scales_bytes`, and the caller
/// MUST keep the backing mapping alive for at least as long as the returned tensor
/// (and any GPU work that reads it).
pub unsafe fn new_qtensor_external<R: CubeRuntime>(
    ptr: u64,
    data_bytes: usize,
    scales_bytes: usize,
    shape: impl Into<Shape>,
    scheme: QuantScheme,
    device: &R::Device,
) -> CubeTensor<R> {
    let client = R::client(device);
    let shape: Shape = shape.into();
    let mut shape_value: Shape = shape.clone();
    let rank = shape.rank();
    let shape_last = shape[rank - 1];
    let num_quants = scheme.num_quants();

    // Packed-data shape + element size — same arithmetic as `new_quantized`.
    let data_size = match scheme.store {
        QuantStore::PackedU32(_) => {
            assert!(shape_last.is_multiple_of(num_quants), "Can't store in u32");
            shape_value[rank - 1] = shape_last.div_ceil(num_quants);
            size_of::<u32>()
        }
        QuantStore::PackedU32Dense(_) => {
            let words_per_unit = scheme.size_bits_stored() / 32;
            shape_value[rank - 1] = shape_last.div_ceil(num_quants) * words_per_unit;
            size_of::<u32>()
        }
        QuantStore::Native => match scheme.value {
            QuantValue::Q8F | QuantValue::Q8S | QuantValue::E4M3 | QuantValue::E5M2 => {
                size_of::<i8>()
            }
            other => panic!("{other:?}: can't store native sub-byte values"),
        },
        QuantStore::PackedNative(_) => match scheme.value {
            QuantValue::E2M1 => size_of::<e2m1x2>(),
            other => panic!("{other:?} doesn't support native packing"),
        },
    };

    let scales_dtype = match scheme.param {
        QuantParam::F32 => DType::F32,
        QuantParam::F16 => DType::F16,
        QuantParam::BF16 => DType::BF16,
        QuantParam::UE8M0 | QuantParam::UE4M3 => DType::U8,
    };
    let scales_shape = params_shape(&shape, scheme.level);

    // The on-disk layout must match what this CubeTensor will read.
    assert_eq!(
        shape_value.num_elements() * data_size,
        data_bytes,
        "external qtensor: packed-codes size mismatch"
    );
    assert_eq!(
        scales_shape.num_elements() * scales_dtype.size(),
        scales_bytes,
        "external qtensor: scales size mismatch"
    );

    // One zero-copy buffer over the contiguous [codes | scales] span. Codes at
    // offset 0; scales sliced out at `offset_start = data_bytes`, ending at the
    // buffer end (`offset_end = 0`) — exactly the packed copy-path layout.
    let memory = unsafe { client.create_external(ptr, data_bytes + scales_bytes) };

    let scales = QParamTensor {
        offset_start: data_bytes,
        offset_end: 0,
        metadata: Metadata::new(scales_shape.clone(), contiguous_strides(&scales_shape)),
        dtype: scales_dtype,
    };

    CubeTensor::new_quantized(
        client,
        memory,
        shape,
        device.clone(),
        contiguous_strides(&shape_value),
        DType::QFloat(scheme),
        QParams { scales },
    )
}

impl<R: CubeRuntime> QTensorOps<Self> for CubeBackend<R> {
    fn q_from_data(data: TensorData, device: &Device<Self>) -> QuantizedTensor<Self> {
        match data.dtype {
            DType::QFloat(scheme) => match scheme {
                QuantScheme {
                    level: QuantLevel::Tensor | QuantLevel::Block(_),
                    mode: QuantMode::Symmetric,
                    value:
                        QuantValue::Q8F
                        | QuantValue::Q8S
                        | QuantValue::Q4F
                        | QuantValue::Q4S
                        | QuantValue::Q2F
                        | QuantValue::Q2S
                        | QuantValue::E4M3
                        | QuantValue::E5M2
                        | QuantValue::Q6F
                        | QuantValue::E2M1,
                    ..
                } => {
                    // TensorData quantized representation is the same, with multiple quantized values
                    // packed into u32 and quantization parameters appended to the bytes
                    new_qtensor_optimized(data.bytes, data.shape.clone(), scheme, device)
                }
                // TQ codebook (Q6F/PackedU32Dense, mode Codebook): the packed
                // bytes + appended params load identically — same construction.
                QuantScheme {
                    mode: QuantMode::Codebook,
                    ..
                } => new_qtensor_optimized(data.bytes, data.shape.clone(), scheme, device),
            },
            _ => panic!(
                "Invalid dtype (expected DType::QFloat, got {:?})",
                data.dtype
            ),
        }
    }

    // TODO: quantize_dynamic (we can compute min-max on the fly and scale, especially when not per-tensor)

    fn quantize(
        tensor: FloatTensor<Self>,
        scheme: &QuantScheme,
        qparams: QuantizationParametersPrimitive<Self>,
    ) -> QuantizedTensor<Self> {
        kernel::quantization::quantize(tensor, scheme, qparams.scales)
    }

    fn dequantize(tensor: QuantizedTensor<Self>, dtype: FloatDType) -> FloatTensor<Self> {
        kernel::quantization::dequantize(tensor, dtype.into())
    }

    fn q_linear(activation: FloatTensor<Self>, weight: QuantizedTensor<Self>) -> FloatTensor<Self> {
        kernel::quantization::q_linear(activation, weight)
    }

    fn q_device(tensor: &QuantizedTensor<Self>) -> Device<Self> {
        tensor.device.clone()
    }

    fn q_to_device(tensor: QuantizedTensor<Self>, device: &Device<Self>) -> QuantizedTensor<Self> {
        super::to_device(tensor, device)
    }

    fn q_reshape(tensor: QuantizedTensor<Self>, shape: Shape) -> QuantizedTensor<Self> {
        super::q_reshape(tensor, shape)
    }

    async fn q_into_data(tensor: QuantizedTensor<Self>) -> Result<TensorData, ExecutionError> {
        if tensor.qparams.is_none() {
            return into_data(tensor).await;
        }

        let (shape, dtype) = (tensor.shape(), tensor.dtype);
        let (values, params) = tensor.quantized_handles().unwrap();

        let mut data_values = into_data(values).await?;
        let data_params = into_data(params).await?;

        data_values.bytes.extend_from_byte_slice(&data_params.bytes);

        Ok(TensorData {
            bytes: data_values.bytes,
            shape,
            dtype,
        })
    }

    fn q_swap_dims(
        tensor: QuantizedTensor<Self>,
        dim1: usize,
        dim2: usize,
    ) -> QuantizedTensor<Self> {
        swap_dims(tensor, dim1, dim2)
    }

    fn q_permute(tensor: QuantizedTensor<Self>, axes: &[usize]) -> QuantizedTensor<Self> {
        permute(tensor, axes)
    }

    fn q_flip(_tensor: QuantizedTensor<Self>, _axes: &[usize]) -> QuantizedTensor<Self> {
        unimplemented!()
    }

    fn q_matmul(lhs: TensorPrimitive<Self>, rhs: TensorPrimitive<Self>) -> TensorPrimitive<Self> {
        let (settings, scheme) = match (&lhs, &rhs) {
            (TensorPrimitive::QFloat(lhs), _) => {
                (get_device_settings::<Self>(&lhs.device), lhs.scheme())
            }
            (_, TensorPrimitive::QFloat(rhs)) => {
                (get_device_settings::<Self>(&rhs.device), rhs.scheme())
            }
            _ => unreachable!(),
        };

        // Inherit precision for mixed inputs, default to `FloatElem` for fully quantized.
        let out_dtype = match (&lhs, &rhs) {
            (TensorPrimitive::Float(lhs), _) => lhs.dtype,
            (_, TensorPrimitive::Float(rhs)) => rhs.dtype,
            _ => settings.float_dtype.into(),
        };

        let (_lhs_dtype, lhs) = match lhs {
            TensorPrimitive::Float(lhs) => (lhs.dtype, lhs),
            TensorPrimitive::QFloat(lhs) => (out_dtype, lhs),
        };
        let (_rhs_dtype, rhs) = match rhs {
            TensorPrimitive::Float(rhs) => (rhs.dtype, rhs),
            TensorPrimitive::QFloat(rhs) => (out_dtype, rhs),
        };

        let out =
            kernel::matmul::matmul(lhs, rhs, None, MatmulStrategy::default(), out_dtype).unwrap();

        match settings.quantization.propagation {
            QuantPropagation::Propagate => {
                TensorPrimitive::QFloat(Self::quantize_dynamic(out, &scheme))
            }
            QuantPropagation::Inhibit => TensorPrimitive::Float(out),
        }
    }
}
