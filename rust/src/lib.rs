//! # regime — regime-aware trading
//!
//! Gaussian HMM regime detection, time-series momentum, FX carry and a
//! pinned no-lookahead backtest engine.  Rust port of the validated Python
//! reference implementation; see `API_SPEC.md` at the project root for
//! every pinned convention.  Everything is deterministic — no RNG anywhere
//! in fitting, strategies or accounting.
//!
//! Modules:
//! * [`hmm`] — scaled forward-backward, Baum-Welch EM, Viterbi, stationary
//!   distribution, AIC/BIC.
//! * [`strategies`] — momentum, FX carry, the walk-forward regime gate.
//! * [`backtest`] — pinned accounting, metrics, crisis-state attribution.
//! * [`pipeline`] — bundled-data loading and the end-to-end study.

pub mod backtest;
pub mod error;
pub mod hmm;
pub mod matrix;
pub mod pipeline;
pub mod strategies;

pub use backtest::{
    compute_metrics, max_drawdown, run_backtest, state_conditional_returns, BacktestResult, Date,
    Metrics, StateStats,
};
pub use error::{RegimeError, Result};
pub use hmm::{
    em_step_status, forward_step, validate_params, EmStep, GaussianHmm, HmmFitResult, HmmParams,
    DEAD_STATE_SUPPORT, MONOTONE_REL_TOL, STOCHASTIC_TOL,
};
pub use matrix::Matrix;
pub use pipeline::{
    carry_rerank_panel, carry_rerank_values, golden_values, load_dataset, run_full_pipeline,
    CarryRerankInputs, Config, Dataset, PipelineOutput,
};
pub use strategies::{
    apply_gate, carry_positions, carry_total_returns, ewma_variance, momentum_positions,
    regime_gate,
};
