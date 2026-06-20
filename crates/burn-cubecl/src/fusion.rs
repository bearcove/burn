use crate::{CubeBackend, CubeRuntime, kernel, tensor::CubeTensor};
use burn_backend::tensor::{BoolTensor, FloatTensor, IntTensor, QuantizedTensor};
use burn_backend::{DType, Shape};
use burn_cubecl_fusion::optim::reduce::ReduceSettings;
use burn_cubecl_fusion::optim::reduce_broadcasted::ReduceBroadcastedFuser;
use burn_cubecl_fusion::{
    CubeFusionHandle, FallbackOperation,
    optim::{
        CubeOptimization, CubeOptimizationState,
        elemwise::{ElementWiseFuser, ElemwiseOptimization},
        matmul::{MatmulFuser, MatmulOptimization, MatmulPrologueFuser},
        reduce::{ReduceFuser, ReduceOptimization},
        reduce_broadcasted::ReduceBroadcastedOptimization,
    },
};
use burn_fusion::UnfusedOp;
use burn_fusion::{
    FusionBackend, FusionRuntime,
    stream::{Operation, OrderedExecution},
};
use burn_ir::{BackendIr, TensorHandle};
use burn_std::Metadata;
use core::marker::PhantomData;
use std::sync::Arc;

impl<R> burn_fusion::Optimization<FusionCubeRuntime<R>> for CubeOptimization<R>
where
    R: CubeRuntime,
{
    fn execute(
        &mut self,
        context: &mut burn_fusion::stream::Context<
            <FusionCubeRuntime<R> as FusionRuntime>::FusionHandle,
        >,
        execution: &OrderedExecution<FusionCubeRuntime<R>>,
    ) {
        match self {
            Self::ElementWise(op) => op.execute(context),
            Self::Matmul(op) => op.execute(context, |index| {
                let operation = execution.operation_within_optimization(index);
                Box::new(FallbackOperationWrapper::new(operation))
            }),
            Self::Reduce(op) => op.execute(context, |index| {
                let operation = execution.operation_within_optimization(index);
                Box::new(FallbackOperationWrapper::new(operation))
            }),
            Self::ReduceBroadcasted(op) => op.execute(context, |index| {
                let operation = execution.operation_within_optimization(index);
                Box::new(FallbackOperationWrapper::new(operation))
            }),
        }
    }

    fn to_state(&self) -> CubeOptimizationState {
        self.to_opt_state()
    }

    fn from_state(device: &R::Device, state: CubeOptimizationState) -> Self {
        match state {
            CubeOptimizationState::ElementWise(state) => {
                Self::ElementWise(ElemwiseOptimization::from_state(device, state))
            }
            CubeOptimizationState::Matmul(state) => {
                Self::Matmul(MatmulOptimization::from_state(device, state))
            }
            CubeOptimizationState::Reduce(state) => {
                Self::Reduce(ReduceOptimization::from_state(device, state))
            }
            CubeOptimizationState::ReduceBroadcasted(state) => {
                Self::ReduceBroadcasted(ReduceBroadcastedOptimization::from_state(device, state))
            }
        }
    }
}

struct FallbackOperationWrapper<O: Clone> {
    operation: O,
}

impl<O: Clone> FallbackOperationWrapper<O> {
    fn new(op: O) -> Self {
        Self { operation: op }
    }
}

impl<R: CubeRuntime> FallbackOperation<R>
    for FallbackOperationWrapper<Arc<dyn Operation<FusionCubeRuntime<R>>>>
{
    fn run(&self, context: &mut burn_fusion::stream::Context<CubeFusionHandle<R>>) {
        self.operation.as_ref().execute(&mut context.handles);
    }
}

impl<R: CubeRuntime> FallbackOperation<R>
    for FallbackOperationWrapper<UnfusedOp<FusionCubeRuntime<R>>>
{
    fn run(&self, context: &mut burn_fusion::stream::Context<CubeFusionHandle<R>>) {
        self.operation.execute(&mut context.handles);
    }
}

impl<R: CubeRuntime> BackendIr for CubeBackend<R> {
    type Handle = CubeFusionHandle<R>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> FloatTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> IntTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> BoolTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn quantized_tensor(handle: TensorHandle<Self::Handle>) -> QuantizedTensor<Self> {
        into_tensor(handle.handle, handle.shape)
    }

    fn float_tensor_handle(tensor: FloatTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn int_tensor_handle(tensor: IntTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn bool_tensor_handle(tensor: BoolTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn quantized_tensor_handle(tensor: QuantizedTensor<Self>) -> Self::Handle {
        tensor.into()
    }
}

impl<R: CubeRuntime> FusionRuntime for FusionCubeRuntime<R> {
    type OptimizationState = CubeOptimizationState;
    type Optimization = CubeOptimization<R>;
    type FusionHandle = CubeFusionHandle<R>;
    type FusionDevice = R::CubeDevice;

    fn fusers(device: R::Device) -> Vec<Box<dyn burn_fusion::OperationFuser<Self::Optimization>>> {
        let mut fusers: Vec<Box<dyn burn_fusion::OperationFuser<Self::Optimization>>> = vec![
            Box::new(ElementWiseFuser::new(device.clone())),
            Box::new(MatmulFuser::new(device.clone())),
        ];
        // WIP: prologue fusion. Currently the 2-block trace mis-registers the prologue
        // op's output as a materialized input — for some shapes its candidate panics
        // (`args.rs` "Input must be concrete"), and cubecl autotune can't survive a
        // candidate that errors, so the whole matmul dies (seen on the QLoRA train step
        // at batch>=16). Runtime opt-out: BURN_DISABLE_PROLOGUE_FUSION=1 drops the
        // candidate, so the matmul falls to the epilogue/elementwise path. (Was: "comment
        // out this line".) See notes/matmul-prologue-fusion-design.md.
        if std::env::var("BURN_DISABLE_PROLOGUE_FUSION").is_err() {
            fusers.push(Box::new(MatmulPrologueFuser::new(device.clone())));
        }
        fusers.push(Box::new(ReduceFuser::new(device.clone(), ReduceSettings::Always)));
        fusers.push(Box::new(ReduceBroadcastedFuser::new(device.clone())));
        fusers
    }
}

/// Fusion runtime for JIT runtimes.
#[derive(Debug)]
pub struct FusionCubeRuntime<R: CubeRuntime> {
    _b: PhantomData<R>,
}

impl<R: CubeRuntime> FusionBackend for CubeBackend<R> {
    type FusionRuntime = FusionCubeRuntime<R>;

    type FullPrecisionBackend = CubeBackend<R>;

    fn cast_float(tensor: FloatTensor<Self>, dtype: DType) -> Self::Handle {
        kernel::cast(tensor, dtype).into()
    }
}

fn into_tensor<R: CubeRuntime>(handle: CubeFusionHandle<R>, shape: Shape) -> CubeTensor<R> {
    CubeTensor {
        client: handle.client.clone(),
        handle: handle.handle.clone(),
        device: handle.device.clone(),
        meta: Box::new(Metadata::new(shape, handle.strides.clone())),
        dtype: handle.dtype,
        qparams: handle.qparams.clone(),
    }
}

impl<R: CubeRuntime> From<CubeTensor<R>> for CubeFusionHandle<R> {
    fn from(value: CubeTensor<R>) -> Self {
        Self {
            client: value.client.clone(),
            handle: value.handle.clone(),
            device: value.device.clone(),
            strides: value.meta.strides.clone(),
            dtype: value.dtype,
            qparams: value.qparams.clone(),
        }
    }
}
