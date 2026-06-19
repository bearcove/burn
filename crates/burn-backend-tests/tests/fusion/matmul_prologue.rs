//! Gate 1 for matmul **prologue** fusion: a leading element-wise op that produces
//! a matmul operand should fold INTO the matmul kernel (read via `fuse_on_read`),
//! not run as a separate element-wise kernel. Asserts both the result (CPU oracle)
//! and that the `MatmulPrologue` fuser owned the block.

use super::*;
use burn_fusion::inspect::FusionInspector;
use burn_tensor::TensorData;

/// `(w @ (x * 2.0)) * 3.0` — a `MulScalar` *prologue* feeding the rhs and a
/// `MulScalar` *epilogue* on the result. The prologue must fold into the matmul's
/// input read (`fuse_on_read`) and the epilogue into its output write — one fused
/// kernel owned by `MatmulPrologue`. Oracle: exactly 6× the plain `w @ x`.
#[test]
fn prologue_and_epilogue_around_matmul() {
    let stream = test_stream();
    stream.executes(|| {
        let device = Default::default();
        let w = TestTensor::<2>::from_data([[1.0, 7.0], [2.0, 3.0], [1.0, 5.0]], &device);
        let x = TestTensor::<2>::from_data([[4.0, 7.0, 5.0], [2.0, 3.0, 5.0]], &device);
        device.sync().unwrap();

        let inspector = FusionInspector::install(stream);
        let out = (w.clone().matmul(x.clone() * 2.0)) * 3.0;
        let data = out.into_data();
        device.sync().unwrap();

        // Oracle: (w @ (x*2)) * 3 = 6 * (w @ x).
        let expected = TensorData::from([
            [108.0, 168.0, 240.0],
            [84.0, 138.0, 150.0],
            [84.0, 132.0, 180.0],
        ]);
        data.assert_eq(&expected, false);

        let reports = inspector.drain();
        let summary: Vec<_> = reports
            .iter()
            .flat_map(|r| {
                r.blocks
                    .iter()
                    .map(|b| (b.fuser_name(), b.operations.len()))
            })
            .collect();
        // The whole `(w @ (x*2)) * 3` must collapse into ONE fused matmul kernel:
        // a "Matmul"-fused block containing the prologue mul, the matmul, and the
        // epilogue mul. (The epilogue-only MatmulFuser can't produce this — the
        // matmul doesn't lead the window — so a multi-op Matmul block with a
        // leading prologue op can only come from MatmulPrologueFuser.)
        let matmul_block = reports
            .iter()
            .flat_map(|r| r.blocks.iter())
            .find(|b| b.fuser_name() == Some("Matmul"))
            .unwrap_or_else(|| panic!("no Matmul-fused block; got {summary:?}\n\n{reports:#?}"));
        assert!(
            matmul_block.operations.len() >= 3,
            "expected prologue+matmul+epilogue fused into one Matmul kernel; got {summary:?}\n\n{reports:#?}"
        );
    });
}
