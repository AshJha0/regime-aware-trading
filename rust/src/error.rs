//! Error type shared by the whole crate.
//!
//! Bad input is always reported through [`RegimeError`] — the library never
//! panics on invalid arguments (the pinned contract of API_SPEC.md §1.9,
//! §2.3).  EM non-convergence is *not* an error; it is a flag on
//! [`crate::hmm::HmmFitResult`].

use std::fmt;

/// Crate-wide error enum (thiserror-style, hand-rolled to keep deps minimal).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegimeError {
    /// A caller-supplied argument violates the pinned contract
    /// (empty series, NaN/inf, shape mismatch, out-of-range parameter, ...).
    InvalidInput(String),
    /// The bundled data set could not be loaded or parsed.
    Data(String),
}

impl fmt::Display for RegimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegimeError::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            RegimeError::Data(msg) => write!(f, "data error: {msg}"),
        }
    }
}

impl std::error::Error for RegimeError {}

/// Convenience alias used across the crate.
pub type Result<T> = std::result::Result<T, RegimeError>;
