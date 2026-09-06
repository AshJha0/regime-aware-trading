"""regime — regime-aware trading: Gaussian HMM, momentum, FX carry, backtest.

Deterministic reference implementation (no RNG anywhere in fitting or
strategy code).  See API_SPEC.md at the project root for every pinned
convention.
"""

from .backtest import (
    BacktestResult,
    compute_metrics,
    max_drawdown,
    run_backtest,
    state_conditional_returns,
)
from .hmm import (
    DEAD_STATE_SUPPORT,
    MONOTONE_REL_TOL,
    EMStep,
    GaussianHMM,
    HMMFitResult,
    HMMParams,
    em_step_status,
    forward_step,
    validate_params,
)
from .pipeline import (
    CARRY_RERANK_INPUTS,
    Dataset,
    carry_rerank_panel,
    carry_rerank_values,
    golden_cases,
    load_dataset,
    run_full_pipeline,
)
from .strategies import (
    apply_gate,
    carry_positions,
    carry_total_returns,
    ewma_variance,
    momentum_positions,
    regime_gate,
)

__all__ = [
    "BacktestResult",
    "CARRY_RERANK_INPUTS",
    "DEAD_STATE_SUPPORT",
    "Dataset",
    "EMStep",
    "GaussianHMM",
    "HMMFitResult",
    "HMMParams",
    "MONOTONE_REL_TOL",
    "apply_gate",
    "carry_positions",
    "carry_rerank_panel",
    "carry_rerank_values",
    "carry_total_returns",
    "compute_metrics",
    "em_step_status",
    "ewma_variance",
    "forward_step",
    "golden_cases",
    "load_dataset",
    "max_drawdown",
    "momentum_positions",
    "regime_gate",
    "run_backtest",
    "run_full_pipeline",
    "state_conditional_returns",
    "validate_params",
]

__version__ = "1.1.0"
