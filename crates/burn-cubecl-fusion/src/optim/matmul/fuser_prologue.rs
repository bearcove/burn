use super::optimization::{FusedMatmul, MatmulOptimization};
use crate::{
    engine::{
        fuser::TraceOperationFuser,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::CubeOptimization,
    optim::matmul::args::MatmulArg,
};
use crate::engine::trace::RegisterTensor;
use burn_fusion::{FuserStatus, OperationFuser};
use burn_ir::{FloatOperationIr, MatmulOpIr, OperationIr, TensorId};
use burn_std::DType;
use cubecl::Runtime;

#[allow(dead_code)]
fn reg_id(t: &RegisterTensor) -> TensorId {
    match t {
        RegisterTensor::Normal(t, _) => t.id,
        RegisterTensor::QuantValues(t) => t.id,
        RegisterTensor::QuantParams(id) => *id,
    }
}

/// Fuses the elementwise ops *before* a matmul (the prologue: rms_norm / RHT /
/// scale) into the matmul kernel — the mirror of [`MatmulFuser`](super::fuser),
/// which only fuses the epilogue. The decode's matmuls are always preceded by
/// these ops, so they never lead a window for the epilogue-only fuser; this
/// fuser instead **accumulates** the leading ops and **anchors** on the matmul.
///
/// Additive: registered as a competing candidate. When no prologue precedes the
/// matmul it closes and the epilogue `MatmulFuser` / elementwise path handle it.
pub struct MatmulPrologueFuser<R: Runtime> {
    /// The 2-block trace: block 0 accumulates the prologue (read), block 1 the
    /// epilogue (write). `next_block` (on anchor) transitions between them.
    fuser: TraceOperationFuser,
    /// Fallback (unfused) when the fused path can't run: the prologue elementwise
    /// ops run as their own kernel (materializing the rhs)…
    fuser_read_fallback: TraceOperationFuser,
    /// …then the matmul (provided as a fallback op), then the epilogue ops.
    fuser_write_fallback: TraceOperationFuser,
    settings_write: FuseSettings,
    device: R::Device,
    matmul: Option<FusedMatmul>,
}

impl<R: Runtime> Clone for MatmulPrologueFuser<R> {
    fn clone(&self) -> Self {
        Self {
            fuser: self.fuser.clone(),
            fuser_read_fallback: self.fuser_read_fallback.clone(),
            fuser_write_fallback: self.fuser_write_fallback.clone(),
            settings_write: self.settings_write,
            device: self.device.clone(),
            matmul: self.matmul.clone(),
        }
    }
}

impl<R: Runtime> MatmulPrologueFuser<R> {
    pub fn new(device: R::Device) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware.max_bindings;
        // Read (prologue) block: accumulate the leading elementwise ops.
        let settings_read = FuseSettings {
            inplace: true,
            ref_layout: RefLayoutSetting::OnlyContiguous,
            broadcast: false,
            output_shape_updates: true,
            vectorization: VectorizationSetting::Activated,
        };
        // Write (epilogue) block: same shape as today's matmul fusion.
        let settings_write = FuseSettings {
            inplace: false,
            output_shape_updates: false,
            vectorization: VectorizationSetting::SmallerOrEqualThanPreviousBlock { block_pos: 0 },
            broadcast: false,
            ref_layout: RefLayoutSetting::OnlyContiguous,
        };
        let settings_fallback = FuseSettings::default();

        Self {
            fuser: TraceOperationFuser::new(max_bindings, settings_read),
            fuser_read_fallback: TraceOperationFuser::new(max_bindings, settings_fallback),
            fuser_write_fallback: TraceOperationFuser::new(max_bindings, settings_fallback),
            settings_write,
            device,
            matmul: None,
        }
    }

    /// Anchor: the accumulated prologue must produce the matmul's rhs. Transition
    /// to the epilogue block; the rhs becomes the prologue block's output (read
    /// via `fuse_on_read`), the lhs a materialized input, the out the epilogue
    /// block's input.
    fn on_matmul(&mut self, op: &MatmulOpIr) {
        let log = std::env::var("QA_FUSE_LOG").is_ok();
        if log {
            eprintln!(
                "[PrologueFuser] on_matmul: accumulated_out_shape={:?} rhs_shape={:?} lhs_shape={:?} num_ops={}",
                self.fuser.current_output_shape, op.rhs.shape, op.lhs.shape, self.fuser.num_ops,
            );
        }
        // The prologue must produce exactly the rhs (the activation). If nothing
        // was accumulated, or it produced a different shape, this isn't ours —
        // let the epilogue fuser / elementwise path take it.
        if self.fuser.current_output_shape != op.rhs.shape {
            if log {
                eprintln!("[PrologueFuser] close: accumulated shape != rhs (no prologue produces rhs)");
            }
            self.fuser.close();
            self.fuser_read_fallback.close();
            self.fuser_write_fallback.close();
            return;
        }
        if log {
            eprintln!("[PrologueFuser] ANCHOR on matmul (prologue feeds rhs)");
        }

        // rhs crosses from the prologue block into the matmul.
        let [rhs] = self.fuser.next_block([&op.rhs], self.settings_write, false);

        // lhs is a materialized input to the matmul (the packed quant weight for
        // the decode, or a dense tensor) — NOT prologue-fused.
        let lhs = match op.lhs.dtype {
            DType::QFloat(scheme) => {
                let (data, scales) = self.fuser.input_quantized_unhandled(&op.lhs).unwrap();
                MatmulArg::Quantized {
                    data,
                    scales,
                    precision: op.out.dtype.into(),
                    scheme,
                }
            }
            _ => MatmulArg::Normal(self.fuser.input_unhandled(&op.lhs)),
        };

        let out = self.fuser.output_unhandled(&op.out);

        self.matmul = Some(FusedMatmul::new(
            lhs,
            MatmulArg::Normal(rhs),
            out,
            op.clone().into(),
            Default::default(),
        ));

        // The prologue (read) block is complete; its fallback materializes the rhs.
        // The write fallback stays open to collect the epilogue.
        self.fuser_read_fallback.close();
    }

    fn on_elemwise_read(&mut self, operation: &OperationIr) {
        let can_register =
            self.fuser.can_fuse(operation) && self.fuser_read_fallback.can_fuse(operation);
        match can_register {
            true => {
                self.fuser.fuse(operation);
                self.fuser_read_fallback.fuse(operation);
            }
            false => {
                self.fuser.close();
                self.fuser_read_fallback.close();
            }
        };
    }

    fn on_elemwise_write(&mut self, operation: &OperationIr) {
        let can_register =
            self.fuser.can_fuse(operation) && self.fuser_write_fallback.can_fuse(operation);
        match can_register {
            true => {
                self.fuser.fuse(operation);
                self.fuser_write_fallback.fuse(operation);
            }
            false => {
                self.fuser.close();
                self.fuser_write_fallback.close();
            }
        };
    }
}

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for MatmulPrologueFuser<R> {
    fn fuse(&mut self, operation: &OperationIr) {
        if std::env::var("QA_FUSE_LOG").is_ok() {
            let kind = match operation {
                OperationIr::Float(_, FloatOperationIr::Matmul(_)) => "MATMUL",
                OperationIr::NumericFloat(..) => "NumericFloat",
                OperationIr::Float(..) => "Float",
                OperationIr::BaseFloat(..) => "BaseFloat",
                _ => "other",
            };
            eprintln!(
                "[PrologueFuser] fuse op={kind} anchored={} status={:?} cur_shape={:?}",
                self.matmul.is_some(),
                self.fuser.status(),
                self.fuser.current_output_shape,
            );
        }
        if let FuserStatus::Closed = self.fuser.status() {
            return;
        }

        if self.matmul.is_none() {
            // Before the anchor: accumulate prologue ops, anchor on a matmul.
            if let OperationIr::Float(_, FloatOperationIr::Matmul(op)) = operation {
                self.on_matmul(op);
            } else {
                self.on_elemwise_read(operation);
            }
        } else {
            // After the anchor: accumulate the epilogue.
            self.on_elemwise_write(operation);
        }
    }

    fn finish(&mut self) -> CubeOptimization<R> {
        if std::env::var("QA_FUSE_LOG").is_ok() {
            eprintln!(
                "[PrologueFuser] finish: anchored={} fuser.len={}",
                self.matmul.is_some(),
                self.fuser.len()
            );
        }
        let client = R::client(&self.device);
        let trace = self.fuser.finish();
        if std::env::var("QA_FUSE_LOG").is_ok() {
            eprintln!(
                "[PrologueFuser] trace blocks={} inputs={:?} outputs={:?}",
                trace.blocks.len(),
                trace.resources.inputs.iter().map(reg_id).collect::<Vec<_>>(),
                trace.resources.outputs.iter().map(reg_id).collect::<Vec<_>>(),
            );
            for (i, b) in trace.blocks.iter().enumerate() {
                eprintln!(
                    "[PrologueFuser]   block{i}: ops={} reads={:?} writes={:?}",
                    b.ops.len(),
                    b.reads.keys().collect::<Vec<_>>(),
                    b.writes.keys().collect::<Vec<_>>(),
                );
            }
        }
        // Read (prologue) fallback materializes the rhs; write fallback runs the
        // epilogue. `len_read` = prologue op count (the fallback matmul sits after).
        let len_read = self.fuser_read_fallback.len();
        let trace_read_fallback = self.fuser_read_fallback.finish();
        let trace_write_fallback = self.fuser_write_fallback.finish();

        let matmul = MatmulOptimization::new(
            trace,
            trace_write_fallback,
            Some(trace_read_fallback),
            client,
            self.device.clone(),
            self.len(),
            len_read,
            self.matmul.as_ref().unwrap().clone(),
        );

        CubeOptimization::Matmul(matmul)
    }

    fn reset(&mut self) {
        self.fuser.reset();
        self.fuser_read_fallback.reset();
        self.fuser_write_fallback.reset();
        self.matmul = None;
    }

    fn status(&self) -> FuserStatus {
        // Until a matmul is anchored we're just accumulating a prologue; report
        // open so the engine keeps feeding ops until the anchor (or a close).
        self.fuser.status()
    }

    fn properties(&self) -> burn_fusion::FuserProperties {
        let p = self.fuser.properties();
        if std::env::var("QA_FUSE_LOG").is_ok() {
            eprintln!(
                "[PrologueFuser] properties: anchored={} ready={} score={} status={:?}",
                self.matmul.is_some(),
                p.ready,
                p.score,
                self.fuser.status(),
            );
        }
        p
    }

    fn len(&self) -> usize {
        // The matmul op itself isn't registered in the trace.
        self.fuser.len() + 1
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}
