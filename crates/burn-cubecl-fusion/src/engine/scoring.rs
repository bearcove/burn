use crate::engine::{
    codegen::ir::{FuseArg, FuseOp, UnaryFuseArgs},
    trace::FuseTrace,
};
use burn_ir::OperationIr;

#[derive(Debug, Clone, Default)]
/// Tracks and evaluates the efficiency of operation fusion.
pub struct Scoring {
    num_writes: usize,
    num_reads: usize,
    num_ops: usize,
}

impl Scoring {
    /// Resets the internal O counters.
    pub fn reset(&mut self) {
        self.num_writes = 0;
        self.num_reads = 0;
        self.num_ops = 0;
    }

    /// Registers an unfused operation to the score, counting its total potential I/O.
    pub fn register(&mut self, op: &OperationIr) {
        self.num_writes += op.outputs().count();
        self.num_reads += op.inputs().count();
        self.num_ops += 1;
    }

    /// Credit an *anchored matmul* whose operand is produced by a fused prologue.
    ///
    /// The generic [`register`](Self::register) only sees the elementwise ops; the
    /// matmul itself is the anchor (never registered), so its I/O is missing from the
    /// unfused baseline. For a gemv whose only fusable neighbor is a view (the
    /// q_linear transpose) this makes a real prologue fold score 0: the matmul's
    /// output write lands in `num_fused` (the epilogue block) with no baseline to
    /// cancel it, and the eliminated intermediate read (the prologue→matmul crossing,
    /// now in-register) is never counted as saved. Register exactly those two —
    /// the eliminated `rhs` read and the balancing `out` write — plus the matmul's
    /// own launch. NOT the `lhs` (weight) read: it survives fusion and isn't tracked
    /// in `num_fused`, so crediting it would inflate the score.
    pub fn register_prologue_anchor(&mut self) {
        self.num_reads += 1; // eliminated intermediate (rhs read folds in-register)
        self.num_writes += 1; // matmul output write — balances the epilogue-block write
        self.num_ops += 1; // the matmul launch
    }

    /// Evaluates the efficiency of a fused trace by comparing its actual I/O
    /// against the registered unfused I/O. Returns the number of saved I/O operations.
    pub fn evaluate(&self, trace: &FuseTrace) -> u64 {
        let mut num_reads_fused = 0;
        let mut num_writes_fused = 0;
        let mut num_penalty = 0;

        for b in trace.blocks.iter() {
            // Count reads in block
            for (_, ops) in b.reads.iter() {
                let result = self.count_fused_io(ops, |args| &args.input);
                num_reads_fused += result.0;
                num_penalty += result.1;
            }
            // Count writes in block
            for (_, ops) in b.writes.iter() {
                let result = self.count_fused_io(ops, |args| &args.out);
                num_writes_fused += result.0;
                num_penalty += result.1;
            }
        }

        let score = self.calculate_score(num_reads_fused, num_writes_fused, num_penalty);
        if std::env::var("QA_SCORE_LOG").is_ok() && trace.blocks.len() >= 2 {
            let block_io: Vec<(usize, usize)> = trace
                .blocks
                .iter()
                .map(|b| (b.reads.len(), b.writes.len()))
                .collect();
            eprintln!(
                "[Scoring] blocks={} block_io(reads,writes)={block_io:?} \
                 unfused(r={} w={} ops={}) fused(r={num_reads_fused} w={num_writes_fused}) penalty={num_penalty} → score={score}",
                trace.blocks.len(), self.num_reads, self.num_writes, self.num_ops,
            );
        }
        score
    }

    fn calculate_score(&self, reads_fused: usize, writes_fused: usize, num_penalty: usize) -> u64 {
        // Those could be tweaked eventually.

        const FACTOR_IO: u64 = 100;
        const FACTOR_LAUNCH: u64 = 10;
        const FACTOR_PENALTY: u64 = 50;

        let num_fused = reads_fused + writes_fused;
        let num_unfused = self.num_reads + self.num_writes;

        let score_io = match num_fused >= num_unfused {
            true => 0,
            false => (num_unfused - num_fused) as u64 * FACTOR_IO,
        };

        // We minus 1 since at least one kernel launch is necessary.
        let score_launch = self.num_ops.saturating_sub(1) as u64 * FACTOR_LAUNCH;

        let score_penalty = num_penalty as u64 * FACTOR_PENALTY;

        (score_io + score_launch).saturating_sub(score_penalty)
    }

    fn count_fused_io<F>(&self, ops: &[FuseOp], arg_extractor: F) -> (usize, usize)
    where
        F: Fn(&UnaryFuseArgs) -> &FuseArg,
    {
        let mut num_io = 0;
        let mut penalty = 0;

        for op in ops.iter() {
            let FuseOp::Assign(args) = op else {
                unreachable!()
            };
            let count_normal = matches!(
                arg_extractor(args),
                FuseArg::Input(..) | FuseArg::Output(..)
            ) as usize;
            let count_view = matches!(
                arg_extractor(args),
                FuseArg::InputReshaped { .. } | FuseArg::InputSwapDims { .. }
            ) as usize;
            num_io += count_normal + count_view;
            penalty += count_view;
        }

        (num_io, penalty)
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_scoring_io_savings() {
        let mut scoring = Scoring::default();
        scoring.num_reads = 2;
        scoring.num_writes = 2;
        scoring.num_ops = 2;

        let score = scoring.calculate_score(1, 1, 0);
        assert_eq!(score, 210);
    }

    #[test]
    fn test_scoring_with_penalties() {
        let mut scoring = Scoring::default();
        scoring.num_reads = 2;
        scoring.num_writes = 2;
        scoring.num_ops = 2;

        let score = scoring.calculate_score(1, 1, 1);
        assert_eq!(score, 160);
    }

    #[test]
    fn test_penalty_outweighs_benefit() {
        let mut scoring = Scoring::default();
        scoring.num_reads = 1;
        scoring.num_writes = 1;
        scoring.num_ops = 2;

        let score = scoring.calculate_score(1, 1, 1);
        assert_eq!(score, 0);
    }

    #[test]
    fn test_scoring_no_ops() {
        let scoring = Scoring::default();
        let score = scoring.calculate_score(0, 0, 0);
        assert_eq!(score, 0);
    }

    #[test]
    fn test_reset() {
        let mut scoring = Scoring {
            num_writes: 10,
            num_reads: 10,
            num_ops: 10,
        };
        scoring.reset();
        assert_eq!(scoring.num_writes, 0);
        assert_eq!(scoring.num_reads, 0);
        assert_eq!(scoring.num_ops, 0);
    }
}
