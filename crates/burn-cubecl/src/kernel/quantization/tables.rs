//! TQ centroid tables and the RHT sign pattern, injected into cubek's codebook
//! kernels as **comptime constants** ([`cubek::quantization::qa_matmul::Codebook`]
//! / [`cubek::quantization::qa_matmul::RhtSigns`]). cubek no longer hardcodes
//! these — the caller (here, burn-cubecl's quant wiring) owns the values and
//! picks the table by [`QuantValue`].

use cubek::quantization::qa_matmul::{Codebook, RhtSigns};
use cubek::quantization::scheme::QuantValue;

/// 4-bit (`Q4F`) codebook (TQ4): 16 reconstruction levels for a unit-variance Gaussian.
const Q4F: [f32; 16] = [
    -2.732590, -2.069017, -1.618046, -1.256231, -0.942340, -0.656759, -0.388048, -0.128395,
    0.128395, 0.388048, 0.656759, 0.942340, 1.256231, 1.618046, 2.069017, 2.732590,
];

/// 6-bit (`Q6F`) codebook (TQ6): 64 Lloyd-Max levels for a unit-variance Gaussian.
const Q6F: [f32; 64] = [
    -3.73971331, -3.23553866, -2.91215583, -2.66675206, -2.46556925, -2.29307792, -2.14077946,
    -2.00348979, -1.87780041, -1.76134301, -1.65240050, -1.54968499, -1.45220328, -1.35917132,
    -1.26995767, -1.18404491, -1.10100239, -1.02046671, -0.94212725, -0.86571539, -0.79099622,
    -0.71776211, -0.64582771, -0.57502585, -0.50520434, -0.43622321, -0.36795256, -0.30027058,
    -0.23306199, -0.16621658, -0.09962796, -0.03319237, 0.03319237, 0.09962796, 0.16621658,
    0.23306199, 0.30027058, 0.36795256, 0.43622321, 0.50520434, 0.57502585, 0.64582771, 0.71776211,
    0.79099622, 0.86571539, 0.94212725, 1.02046671, 1.10100239, 1.18404491, 1.26995767, 1.35917132,
    1.45220328, 1.54968499, 1.65240050, 1.76134301, 1.87780041, 2.00348979, 2.14077946, 2.29307792,
    2.46556925, 2.66675206, 2.91215583, 3.23553866, 3.73971331,
];

/// ±1 sign pattern for the 32-value RHT (the "prerot" rotation). Shared across TQ formats.
const RHT_SIGNS_TABLE: [f32; 32] = [
    1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
    -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
];

/// Centroid table for a table-codebook `value`. Linear (`Q8F`) and symmetric
/// values don't read a centroid table, so they get an empty placeholder (the
/// codebook branch is comptime-guarded off for them).
pub fn codebook_for(value: QuantValue) -> Codebook {
    match value {
        QuantValue::Q4F => Codebook(&Q4F),
        QuantValue::Q6F => Codebook(&Q6F),
        _ => Codebook(&[]),
    }
}

/// The shared 32-wide RHT sign pattern.
pub fn rht_signs() -> RhtSigns {
    RhtSigns(&RHT_SIGNS_TABLE)
}
