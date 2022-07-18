#![cfg_attr(not(feature = "std"), no_std)]
#![deny(
    warnings,
    unused,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms
)]
#![forbid(unsafe_code)]

//! This library implements the prime-order curve secp_256k1, specified in Certicom's
//! [SECG](https://www.secg.org/sec2-v2.pdf) in "SEC 2: Recommended Elliptic Curve Domain Parameters".
//! This curve forms a cycle with secq_256k1, i.e. its scalar field and base
//! field respectively are the base field and scalar field of secq_256k1.
//!
//!
//! Curve information:
//! * Base field: q =
//!   115792089237316195423570985008687907853269984665640564039457584007908834671663
//! * Scalar field: r =
//!   115792089237316195423570985008687907852837564279074904382605163141518161494337
//! * Curve equation: y^2 = x^3 + 7
//! * Valuation(q - 1, 2) = 1
//! * Valuation(r - 1, 2) = 6

// #[cfg(feature = "r1cs")]
// pub mod constraints;
// #[cfg(feature = "curve")]
// mod curves;
#[cfg(any(feature = "scalar_field", feature = "base_field"))]
mod fields;

// #[cfg(feature = "curve")]
// pub use curves::*;
#[cfg(any(feature = "scalar_field", feature = "base_field"))]
pub use fields::*;
