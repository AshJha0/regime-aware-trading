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
from .hmm import GaussianHMM, HMMFitResult, HMMParams, forward_step
from .pipeline import Dataset, golden_cases, load_dataset, run_full_pipeline
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
    "Dataset",
    "GaussianHMM",
    "HMMFitResult",
    "HMMParams",
    "apply_gate",
    "carry_positions",
    "carry_total_returns",
    "compute_metrics",
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
]

__version__ = "1.0.0"
