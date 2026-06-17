//! TQ centroid tables and the RHT sign pattern, injected into cubek's codebook
//! kernels as **comptime constants** ([`cubek::quantization::qa_matmul::Codebook`]
//! / [`cubek::quantization::qa_matmul::RhtSigns`]). cubek no longer hardcodes
//! these — the caller (here, burn-cubecl's quant wiring) owns the values and
//! picks the table by [`QuantValue`].

use cubek::quantization::qa_matmul::RhtSigns;

/// ±1 sign pattern for the 32-value RHT (the "prerot" rotation). Shared across TQ formats.
const RHT_SIGNS_TABLE: [f32; 32] = [
    1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
];

/// Centroid table for a table-codebook `value`. The TQ codebooks (`Q4F`/`Q6F`) now
/// live in `cubecl_common::quant::scheme` so burn-cubecl AND burn-cubecl-fusion share
/// ONE table; this just re-exposes the shared lookup under the existing path.
pub use cubek::quantization::scheme::codebook_for;

/// The shared 32-wide RHT sign pattern.
pub fn rht_signs() -> RhtSigns {
    RhtSigns(&RHT_SIGNS_TABLE)
}
