#![cfg_attr(docsrs, feature(doc_cfg))]
//! Native **Metal 4** backend for Burn — `CubeBackend` over `cubecl-metal4`'s
//! `Metal4Runtime` (no wgpu). Apple-only; an empty crate elsewhere so the
//! workspace still builds on Linux/CUDA boxes.

#[cfg(target_vendor = "apple")]
mod apple {
    use burn_cubecl::CubeBackend;

    pub use cubecl::metal4::{Metal4Device, Metal4Runtime};

    /// The native Metal 4 backend.
    #[cfg(not(feature = "fusion"))]
    pub type Metal4 = CubeBackend<Metal4Runtime>;

    /// The native Metal 4 backend, with operation fusion.
    #[cfg(feature = "fusion")]
    pub type Metal4 = burn_fusion::Fusion<CubeBackend<Metal4Runtime>>;
}

#[cfg(target_vendor = "apple")]
pub use apple::*;
