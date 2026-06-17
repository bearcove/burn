use crate::{
    CubeRuntime, CubeTuneId,
    kernel::matmul::{launch_matmul, launch_matmul_naive, utils::init_matmul_output},
    tensor::CubeTensor,
};
use burn_backend::DType;
use burn_backend::cubecl::dtype_to_storage_type;
use cubecl::{
    std::tensor::MatrixBatchLayout,
    tune::{LocalTuner, Tunable, TunableSet, TuneGroup, local_tuner},
};
use cubek::matmul::{
    components::tile::TileMatmulKind,
    definition::MatmulKind,
    routines::{
        BlueprintStrategy, TileSizeSelection,
        batch::{
            double_buffering::DoubleBufferingArgs, ordered_double_buffering::OrderedSelectionArgs,
            simple::SimpleArgs, simple_unit::SimpleUnitSelectionArgs,
        },
        gemm::GemmStrategy,
    },
    strategy::{MatmulAutotuneKey, MatmulGlobalScale, Strategy, should_tune_double_buffering},
};
// Only referenced by the full (desktop/CUDA) unit-matmul sweep; the iOS build
// prunes DoubleUnit to keep the on-device shader-compile count down.
#[cfg(not(target_os = "ios"))]
use cubek::matmul::routines::batch::double_unit::DoubleUnitSelectionArgs;

fn matmul_input_gen<R: CubeRuntime>(
    _key: &MatmulAutotuneKey,
    (lhs, rhs, out): &(CubeTensor<R>, CubeTensor<R>, CubeTensor<R>),
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    (lhs.clone(), rhs.clone(), out.copy())
}

/// Executes autotune on matmul operations
pub fn matmul_autotune<R: CubeRuntime>(
    lhs: CubeTensor<R>,
    rhs: CubeTensor<R>,
    out: Option<CubeTensor<R>>,
    out_dtype: DType,
) -> CubeTensor<R> {
    let output = out.unwrap_or_else(|| init_matmul_output(&lhs, &rhs, out_dtype));

    let client = lhs.client.clone();
    let num_cpu_cores = client.properties().hardware.num_cpu_cores;
    // TMA matmul kernels require sm_90+ (Hopper `Tma::Base`). On devices without
    // TMA (Ampere, Apple/Metal, …) `features.tma` is empty; the cubek TMA strategy
    // hard-faults the compute server at launch there ("matmul_specialized_tma_mma:
    // unknown error" → channel down), so never offer it as an autotune candidate.
    let tma_supported = !client.properties().features.tma.is_empty();

    static TUNER: LocalTuner<MatmulAutotuneKey, CubeTuneId> = local_tuner!();

    let tunables = TUNER.init(move || {
        const PRIORITY_MAX: i8 = 3;
        const PRIORITY_HIGH: i8 = 2;
        const PRIORITY_MEDIUM: i8 = 1;
        const PRIORITY_MIN: i8 = 0;
        const PRIORITY_NEVER: i8 = -1;

        let accelerated = TuneGroup::<MatmulAutotuneKey>::new("accelerated", |key| {
            if matches!(key.analysis.kind, MatmulKind::General) {
                match key.analysis.scale_global {
                    MatmulGlobalScale::Large => PRIORITY_MAX,
                    _ => PRIORITY_HIGH,
                }

            // In some case when a relayout can be fused (no call to into_contiguous) it's better
            // to use accelerated matmul.
            //
            // TODO: Actually implement good gemv with fused relayout.
            } else if matches!(key.analysis.kind, MatmulKind::MatVec | MatmulKind::VecMat) {
                PRIORITY_MAX
            } else {
                PRIORITY_MEDIUM
            }
        });

        let unit = TuneGroup::<MatmulAutotuneKey>::new("unit", |key| {
            if !matches!(key.analysis.kind, MatmulKind::General)
                || matches!(key.analysis.scale_global, MatmulGlobalScale::Small)
            {
                PRIORITY_HIGH
            } else {
                PRIORITY_MEDIUM
            }
        });

        let tma = TuneGroup::<MatmulAutotuneKey>::new("tma", move |key| {
            // Skip TMA entirely on devices that don't support it (else the kernel
            // launch crashes the compute server — see `tma_supported` above).
            if !tma_supported {
                return PRIORITY_NEVER;
            }
            // For large matmul, we set the max priority to TMA kernels, higher than any other
            // matmuls, since they are the best kernels no matter what.
            //
            // But only when all axis are large.
            let max_axis = usize::max(key.definition.m, key.definition.n);
            let max_axis = usize::max(key.definition.k, max_axis);

            let min_axis = usize::min(key.definition.m, key.definition.n);
            let min_axis = usize::min(key.definition.k, min_axis);

            let skewed_factor = max_axis / min_axis;

            let priority_max = if matches!(key.analysis.kind, MatmulKind::General)
                && matches!(key.analysis.scale_global, MatmulGlobalScale::Large)
                && skewed_factor < 4
            {
                PRIORITY_MAX
            } else {
                PRIORITY_HIGH
            };

            if key.definition.lhs_stride_factor >= 4 && key.definition.rhs_stride_factor >= 4 {
                priority_max
            } else {
                PRIORITY_NEVER
            }
        });

        let gemv = TuneGroup::<MatmulAutotuneKey>::new("gemv", move |key| {
            if num_cpu_cores.is_some() {
                return PRIORITY_MAX;
            }

            if matches!(key.analysis.kind, MatmulKind::MatVec) {
                // LHS is the matrix
                match key.definition.matrix_layout_lhs {
                    MatrixBatchLayout::Contiguous => PRIORITY_MAX,
                    MatrixBatchLayout::MildlyPermuted { transposed, .. } => {
                        // We don't yet have algo which are good for col major matvec.
                        if transposed {
                            PRIORITY_HIGH
                        } else {
                            PRIORITY_MAX
                        }
                    }
                    // Every algo will need to relayout, in this case, we should take the optimal
                    // kernel with a gemv.
                    MatrixBatchLayout::HighlyPermuted => PRIORITY_MAX,
                }
            } else if matches!(key.analysis.kind, MatmulKind::VecMat) {
                // RHS is the matrix
                match key.definition.matrix_layout_rhs {
                    // We don't have good algos for row major vecmat.
                    MatrixBatchLayout::Contiguous => PRIORITY_HIGH,
                    MatrixBatchLayout::MildlyPermuted { transposed, .. } => {
                        // Best algo is col major vec mat.
                        if transposed {
                            PRIORITY_MAX
                        } else {
                            PRIORITY_HIGH
                        }
                    }
                    // TODO: Actually do the correct relayout here.
                    //
                    // Every algo will need to relayout, in this case, we should take the optimal
                    // kernel with a gemv.
                    MatrixBatchLayout::HighlyPermuted => PRIORITY_HIGH,
                }
            } else {
                PRIORITY_NEVER
            }
        });

        fn double_buffering_priority(key: &MatmulAutotuneKey, max: i8, min: i8) -> i8 {
            if should_tune_double_buffering(false, key) {
                max
            } else {
                min
            }
        }

        let mut set = TunableSet::new(create_key::<R>, matmul_input_gen::<R>);

        // First entry should always work, since it is considered the fallback.
        set = set.with(
            Tunable::new("matmul_naive", |(lhs, rhs, out)| {
                launch_matmul_naive::<R>(&Strategy::Naive, lhs, rhs, out)
                    .map_err(|err| std::format!("{err:?}"))
            })
            .group(&unit, |key| {
                if matches!(key.analysis.kind, MatmulKind::InnerProduct) {
                    PRIORITY_MAX
                } else if matches!(key.analysis.scale_global, MatmulGlobalScale::Small) {
                    PRIORITY_HIGH
                } else {
                    PRIORITY_MIN
                }
            }),
        );

        // Matrix Vector multiplication kernels. iOS drops the double-buffered
        // variant (DoubleVecMat) for the same shader-compile-count reason as the
        // unit sweep above; SimpleVecMat + Gemm + GemvUnitPerpendicular remain.
        #[cfg(target_os = "ios")]
        let vecmat_strategies = [
            (
                Strategy::SimpleVecMat(BlueprintStrategy::Inferred(().into())),
                false,
            ),
            (
                Strategy::Gemm(BlueprintStrategy::Inferred(Default::default())),
                false,
            ),
            (
                Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(Default::default())),
                false,
            ),
        ];
        #[cfg(not(target_os = "ios"))]
        let vecmat_strategies = [
            (
                Strategy::DoubleVecMat(BlueprintStrategy::Inferred(().into())),
                true,
            ),
            (
                Strategy::SimpleVecMat(BlueprintStrategy::Inferred(().into())),
                false,
            ),
            (
                Strategy::Gemm(BlueprintStrategy::Inferred(Default::default())),
                false,
            ),
            (
                Strategy::GemvUnitPerpendicular(BlueprintStrategy::Inferred(Default::default())),
                false,
            ),
        ];
        for (strategy, double_buf) in vecmat_strategies {
            set = set.with(
                Tunable::new(&strategy.to_string(), move |(lhs, rhs, out)| {
                    launch_matmul::<R>(&strategy, lhs, rhs, out)
                        .map_err(|err| std::format!("{err:?}"))
                })
                .group(&gemv, move |key| match double_buf {
                    false => PRIORITY_MAX,
                    true => double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH),
                }),
            );
        }

        // Unit matmuls.
        //
        // MOBILE (iOS): the on-device Metal shader compiler is ~10× slower than
        // desktop (~600 ms per `matmul_entry` variant), and autotune both compiles
        // AND benchmarks every candidate. On a phone the {MaxTile,MinTile} ×
        // {Simple,Double} sweep = 4 unit kernels/shape dominated the ~30 s cold
        // warmup (75 variants ≈ 8.6 s compile + ~15 s bench). Prune to a single
        // unit candidate (SimpleUnit @ MaxTile) — for the small mobile matmuls the
        // tile/double-buffer variants are near-indistinguishable, and the warm RTF
        // confirms no regression. Desktop/CUDA keep the full sweep.
        #[cfg(target_os = "ios")]
        let unit_tile_sizes = [TileSizeSelection::MaxTileSize];
        #[cfg(not(target_os = "ios"))]
        let unit_tile_sizes = [
            TileSizeSelection::MaxTileSize,
            TileSizeSelection::MinTileSize,
        ];
        for tile_size in unit_tile_sizes {
            #[cfg(target_os = "ios")]
            let unit_strategies = [(
                Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
                    tile_size,
                })),
                false,
            )];
            #[cfg(not(target_os = "ios"))]
            let unit_strategies = [
                (
                    Strategy::SimpleUnit(BlueprintStrategy::Inferred(SimpleUnitSelectionArgs {
                        tile_size,
                    })),
                    false,
                ),
                (
                    Strategy::DoubleUnit(BlueprintStrategy::Inferred(DoubleUnitSelectionArgs {
                        tile_size,
                    })),
                    true,
                ),
            ];
            for (strategy, double_buf) in unit_strategies {
                set = set.with(
                    Tunable::new(&strategy.to_string(), move |(lhs, rhs, out)| {
                        launch_matmul::<R>(&strategy, lhs, rhs, out)
                            .map_err(|err| format!("{err:?}"))
                    })
                    .group(&unit, move |key| match double_buf {
                        false => PRIORITY_MAX,
                        true => double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH),
                    }),
                )
            }
        }

        // Gemm no stage
        // In unit because not accelerated
        let gemm_no_stage_strategy = Strategy::Gemm(BlueprintStrategy::Inferred(GemmStrategy {
            target_num_planes: None,
        }));
        set = set.with(
            Tunable::new(
                &gemm_no_stage_strategy.to_string(),
                move |(lhs, rhs, out)| {
                    launch_matmul::<R>(&gemm_no_stage_strategy, lhs, rhs, out)
                        .map_err(|err| format!("{err:?}"))
                },
            )
            .group(&unit, move |_key| PRIORITY_MAX),
        );

        // Accelerated matmuls
        for (strategy, double_buf, group_extra, tile_group) in [
            (
                Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: false,
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                false,
                None,
                &accelerated,
            ),
            (
                Strategy::SimpleCyclicMma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: false,
                    tile_matmul: TileMatmulKind::Mma,
                })),
                false,
                None,
                &accelerated,
            ),
            (
                Strategy::SimpleCyclicCmma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: true,
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                false,
                None,
                &accelerated,
            ),
            (
                Strategy::SimpleCyclicMma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: true,
                    tile_matmul: TileMatmulKind::Mma,
                })),
                false,
                None,
                &accelerated,
            ),
            (
                Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    partition_k: Some(2),
                    row_count: Some(4),
                    rows_per_plane: Some(2),
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                true,
                None,
                &accelerated,
            ),
            (
                Strategy::OrderedDoubleMma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    partition_k: Some(2),
                    row_count: Some(4),
                    rows_per_plane: Some(2),
                    tile_matmul: TileMatmulKind::Mma,
                })),
                true,
                None,
                &accelerated,
            ),
            (
                Strategy::OrderedDoubleCmma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    partition_k: Some(2),
                    row_count: Some(8),
                    rows_per_plane: Some(2),
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                true,
                None,
                &accelerated,
            ),
            (
                Strategy::OrderedDoubleMma(BlueprintStrategy::Inferred(OrderedSelectionArgs {
                    partition_k: Some(2),
                    row_count: Some(8),
                    rows_per_plane: Some(2),
                    tile_matmul: TileMatmulKind::Mma,
                })),
                true,
                None,
                &accelerated,
            ),
            (
                Strategy::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    specialized: false,
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                true,
                None,
                &accelerated,
            ),
            (
                Strategy::DoubleCyclicMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    specialized: false,
                    tile_matmul: TileMatmulKind::Mma,
                })),
                true,
                None,
                &accelerated,
            ),
            (
                Strategy::DoubleCyclicCmma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    specialized: true,
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                true,
                None,
                &accelerated,
            ),
            (
                Strategy::DoubleCyclicMma(BlueprintStrategy::Inferred(DoubleBufferingArgs {
                    specialized: true,
                    tile_matmul: TileMatmulKind::Mma,
                })),
                true,
                None,
                &accelerated,
            ),
            (
                Strategy::SpecializedCyclicCmma(BlueprintStrategy::Inferred(().into())),
                true,
                None,
                &accelerated,
            ),
            (
                Strategy::SpecializedCyclicMma(BlueprintStrategy::Inferred(().into())),
                true,
                None,
                &accelerated,
            ),
            (
                Strategy::SimpleTmaCmma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: false,
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                false,
                Some(&tma),
                &accelerated,
            ),
            (
                Strategy::SimpleTmaMma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: false,
                    tile_matmul: TileMatmulKind::Mma,
                })),
                false,
                Some(&tma),
                &accelerated,
            ),
            (
                Strategy::SimpleTmaCmma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: true,
                    tile_matmul: TileMatmulKind::Cmma,
                })),
                false,
                Some(&tma),
                &accelerated,
            ),
            (
                Strategy::SimpleTmaMma(BlueprintStrategy::Inferred(SimpleArgs {
                    multi_rows: true,
                    tile_matmul: TileMatmulKind::Mma,
                })),
                false,
                Some(&tma),
                &accelerated,
            ),
            (
                Strategy::SpecializedTmaCmma(BlueprintStrategy::Inferred(().into())),
                true,
                Some(&tma),
                &accelerated,
            ),
            (
                Strategy::SpecializedTmaMma(BlueprintStrategy::Inferred(().into())),
                true,
                Some(&tma),
                &accelerated,
            ),
        ] {
            // Skip TMA strategies on devices without TMA (Ampere / Apple-Metal): in
            // this loop `group_extra` is `Some(&tma)` iff the strategy is a TMA one,
            // and they're ALSO in the `accelerated` tile group — so gating only the
            // `tma` group priority doesn't exclude them. Launching a TMA kernel on a
            // non-TMA device hard-faults the compute server, so drop them outright.
            // (Kept on Hopper, where `tma_supported`.)
            if !tma_supported && group_extra.is_some() {
                continue;
            }
            let priority_within_group = |key: &MatmulAutotuneKey, double_buf: bool| match double_buf
            {
                false => PRIORITY_MAX,
                true => double_buffering_priority(key, PRIORITY_MAX, PRIORITY_HIGH),
            };
            let mut tunable = Tunable::new(&strategy.to_string(), move |(lhs, rhs, out)| {
                launch_matmul::<R>(&strategy, lhs, rhs, out).map_err(|err| format!("{err:?}"))
            });

            // tile group
            tunable = tunable.group(tile_group, move |key| {
                priority_within_group(key, double_buf)
            });

            // extra group
            if let Some(group) = group_extra {
                tunable = tunable.group(group, move |key| priority_within_group(key, double_buf));
            }
            set = set.with(tunable);
        }

        set
    });

    TUNER.execute(
        &CubeTuneId::new(&lhs.client, &lhs.device),
        &client,
        tunables,
        (lhs, rhs, output.clone()),
    );

    output
}

fn create_key<R: CubeRuntime>(
    (lhs, rhs, out): &(CubeTensor<R>, CubeTensor<R>, CubeTensor<R>),
) -> MatmulAutotuneKey {
    let key = MatmulAutotuneKey::generate(
        &lhs.client,
        lhs.meta.shape(),
        rhs.meta.shape(),
        lhs.meta.strides(),
        rhs.meta.strides(),
        dtype_to_storage_type(lhs.dtype),
        dtype_to_storage_type(rhs.dtype),
        dtype_to_storage_type(out.dtype),
        lhs.try_scheme(),
        rhs.try_scheme(),
    );

    // Collision diagnostic (CUBEK_KEY_DEBUG=1): the autotune key buckets m/n/k
    // (anchored) and stores layout as MatrixBatchLayout, but the gemm *variant*
    // is chosen from the raw shapes + MatrixLayout. If two calls print the SAME
    // `key=` line with DIFFERENT `variant=`, the cache will mis-apply a kernel
    // benchmarked for one to the other (the fast→slow + kernel-growth bug).
    if std::env::var("CUBEK_KEY_DEBUG").is_ok() {
        let (lsh, rsh) = (lhs.meta.shape(), rhs.meta.shape());
        let (ls, rs) = (lhs.meta.strides(), rhs.meta.strides());
        eprintln!(
            "CUBEK_KEY variant={} key={key:?} | lhs_shape={lsh:?} lhs_strides={ls:?} \
             rhs_shape={rsh:?} rhs_strides={rs:?}",
            derive_gemm_variant(lsh, rsh, ls, rs),
        );
    }

    key
}

/// Mirror of cubek's `MatmulOperandLayouts::from_problem(..).variant()` so the
/// collision diagnostic shows which kernel variant a given (shape, layout) will
/// route to. Kept in sync with `cubek-matmul/.../batch/gemm/config.rs`.
fn derive_gemm_variant(lsh: &[usize], rsh: &[usize], ls: &[usize], rs: &[usize]) -> &'static str {
    let (nl, nr) = (lsh.len(), rsh.len());
    let (m, n) = (lsh[nl - 2], rsh[nr - 1]);
    // lhs: m==1 → Vector (k-contig); else RowMajor iff k (last) is unit-stride.
    let (lhs_k_contig, lhs_m_contig) = if m == 1 {
        (true, false)
    } else {
        let row = ls[nl - 1] == 1;
        (row, !row)
    };
    // rhs: n==1 → Vector (k-contig); else RowMajor iff n (last) is unit-stride.
    let (rhs_k_contig, rhs_n_contig) = if n == 1 {
        (true, false)
    } else {
        let row = rs[nr - 1] == 1;
        (!row, row)
    };
    match (lhs_k_contig, rhs_k_contig, rhs_n_contig, lhs_m_contig) {
        (true, true, _, _) => "Dot",
        (true, _, true, _) => "OuterNLhsContig",
        (_, _, true, true) => "OuterNLhsStrided",
        (_, true, _, true) => "OuterM",
        _ => "unclassified",
    }
}
