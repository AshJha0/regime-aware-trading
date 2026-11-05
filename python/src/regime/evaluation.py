"""Out-of-sample evaluation of the regime gate against simple baselines.

The pinned metrics in :mod:`regime.backtest` are full-sample, which is the
right thing for golden values and the wrong thing for judging the gate: the
gate is 1 for the first ``train_min_days - 1`` days, so those days say
nothing about it.  This module measures only the days the gate can affect.

Conventions
-----------
* First gated day: ``first_day = train_min_days - 1`` is the first day the
  gate differs from 1.  ``P[first_day]`` earns the return of the next day,
  so the metrics window is ``t >= train_min_days``.
* Combined book: 50/50 in the momentum and carry net return streams, as in
  :func:`regime.pipeline.run_full_pipeline`.  Gross P&L and turnover do not
  depend on cost, so ``net = gross - cost/1e4 * turnover`` is linear in cost
  and a cost sweep needs one backtest per gate.
* Causality: ``vol_ratio_gate`` and ``ewma_ratio_gate`` at day t read returns
  through t only.  ``constant_gate`` takes its level as an argument; when
  that level is the mean of the HMM gate, the level is a hindsight number and
  the baseline is a matched-exposure control, not a tradable rule.
  ``oracle_gate`` reads the true state of day t+1 and is a ceiling only.
* Randomness: the bootstrap takes a ``numpy.random.Generator`` argument and
  owns no global state.  The rest of the module is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .backtest import TRADING_DAYS, compute_metrics, run_backtest
from .hmm import GaussianHMM
from .pipeline import Dataset
from .strategies import (
    apply_gate,
    carry_positions,
    carry_total_returns,
    ewma_variance,
    momentum_positions,
)

#: Generator labels (data/generate_data.py).
CALM_STATE = 0
CRISIS_STATE = 2


def sharpe(net: np.ndarray, axis: int = -1) -> np.ndarray:
    """Annualized Sharpe with the pinned formula (ddof 0; 0 if the vol is 0)."""
    x = np.asarray(net, dtype=float)
    mu = x.mean(axis=axis)
    sd = x.std(axis=axis, ddof=0)
    safe = np.where(sd > 0.0, sd, 1.0)
    return np.where(sd > 0.0, np.sqrt(TRADING_DAYS) * mu / safe, 0.0)


# ---------------------------------------------------------------------- #
# Combined book
# ---------------------------------------------------------------------- #
@dataclass
class CombinedStream:
    """Cost-free daily series of the 50/50 book under one gate.

    Attributes:
        gross: Combined gross daily return, shape ``(T,)``.
        turnover: Combined daily turnover (half of each sleeve), shape ``(T,)``.
        momentum_turnover: Momentum sleeve turnover, shape ``(T,)``.
        carry_turnover: Carry sleeve turnover, shape ``(T,)``.
    """

    gross: np.ndarray
    turnover: np.ndarray
    momentum_turnover: np.ndarray
    carry_turnover: np.ndarray

    def net(self, cost_bps: float) -> np.ndarray:
        """Daily net return at ``cost_bps`` per unit of turnover."""
        return self.gross - (cost_bps / 1e4) * self.turnover


@dataclass
class GateBook:
    """Ungated positions and returns of both sleeves.

    Attributes:
        momentum_positions / trend_returns: Momentum sleeve, shape ``(T, A)``.
        carry_positions / carry_returns: Carry sleeve, shape ``(T, A)``.
        dates: Trading dates, length T.
    """

    momentum_positions: np.ndarray
    trend_returns: np.ndarray
    carry_positions: np.ndarray
    carry_returns: np.ndarray
    dates: pd.DatetimeIndex

    def stream(self, gate: np.ndarray) -> CombinedStream:
        """Run both sleeves under ``gate`` through the pinned backtest."""
        mom = run_backtest(apply_gate(self.momentum_positions, gate), self.trend_returns, 0.0)
        car = run_backtest(apply_gate(self.carry_positions, gate), self.carry_returns, 0.0)
        return CombinedStream(
            gross=0.5 * mom.gross_returns + 0.5 * car.gross_returns,
            turnover=0.5 * mom.turnover + 0.5 * car.turnover,
            momentum_turnover=mom.turnover,
            carry_turnover=car.turnover,
        )


def build_book(ds: Dataset, config: Optional[Mapping] = None) -> GateBook:
    """Build the ungated book from a dataset with the pipeline's settings."""
    cfg = ds.config if config is None else config
    return GateBook(
        momentum_positions=momentum_positions(
            ds.trend_returns,
            lookback=cfg["momentum_lookback"],
            vol_target=cfg["vol_target"],
            ewma_lambda=cfg["ewma_lambda"],
            leverage_cap=cfg["leverage_cap"],
        ),
        trend_returns=ds.trend_returns,
        carry_positions=carry_positions(
            ds.fx_rate_diffs,
            top_n=cfg["carry_top_n"],
            bottom_n=cfg["carry_bottom_n"],
            rebalance_days=cfg["rebalance_days"],
        ),
        carry_returns=carry_total_returns(ds.fx_spot_returns, ds.fx_rate_diffs),
        dates=ds.dates,
    )


def window_metrics(net: np.ndarray, dates: pd.DatetimeIndex, start: int) -> Dict[str, float]:
    """Pinned metrics on the window ``t >= start``; equity restarts at 1."""
    if not 0 <= start < len(net):
        raise ValueError(f"window start {start} outside series of length {len(net)}")
    return compute_metrics(np.asarray(net)[start:], dates[start:])


def annual_turnover(turnover: np.ndarray, start: int) -> float:
    """Turnover per year on the window ``t >= start``."""
    return float(TRADING_DAYS * np.mean(np.asarray(turnover)[start:]))


# ---------------------------------------------------------------------- #
# Baseline gates
# ---------------------------------------------------------------------- #
def constant_gate(n_days: int, level: float, first_day: int) -> np.ndarray:
    """Gate of 1 before ``first_day`` and ``level`` from then on."""
    if not 0.0 <= level <= 1.0:
        raise ValueError(f"gate level must be in [0, 1], got {level}")
    gate = np.ones(n_days)
    gate[first_day:] = level
    return gate


def _ratio_gate(vol: np.ndarray, first_day: int) -> np.ndarray:
    """``min(1, expanding median of vol / vol)`` from ``first_day`` on."""
    ref = pd.Series(vol).expanding().median().to_numpy()
    gate = np.ones(vol.shape[0])
    live = np.arange(vol.shape[0]) >= first_day
    ok = live & (vol > 0.0)
    gate[ok] = np.minimum(1.0, ref[ok] / vol[ok])
    return gate


def vol_ratio_gate(index_returns: np.ndarray, first_day: int, window: int = 21) -> np.ndarray:
    """Cut exposure when trailing realised vol exceeds its own history.

    ``vol[t]`` is the root mean square of the ``window`` returns ending at t,
    and ``gate[t] = min(1, median(vol[..t]) / vol[t])``.  Gate is 1 before
    ``first_day``.  Causal.
    """
    r = np.asarray(index_returns, dtype=float)
    if window < 1 or first_day < window - 1:
        raise ValueError("need window >= 1 and first_day >= window - 1")
    vol = np.sqrt(pd.Series(r**2).rolling(window).mean().to_numpy())
    return _ratio_gate(vol, first_day)


def ewma_ratio_gate(
    index_returns: np.ndarray, first_day: int, lam: float = 0.94, init_window: int = 21
) -> np.ndarray:
    """As :func:`vol_ratio_gate` with the pinned EWMA vol.  Causal."""
    r = np.asarray(index_returns, dtype=float)
    if first_day < init_window - 1:
        raise ValueError("need first_day >= init_window - 1")
    vol = np.sqrt(ewma_variance(r, lam, init_window)[:, 0])
    return _ratio_gate(vol, first_day)


def oracle_gate(true_states: np.ndarray, first_day: int, crisis_state: int = CRISIS_STATE) -> np.ndarray:
    """Flat on any day whose NEXT day is a true crisis day.

    This reads the future on purpose: ``P[t]`` earns the day t+1 return, and
    the gate at t knows the state of t+1.  It is a ceiling for regime timing
    and cannot be traded.
    """
    s = np.asarray(true_states)
    gate = np.ones(s.shape[0])
    gate[first_day:-1] = np.where(s[first_day + 1:] == crisis_state, 0.0, 1.0)
    return gate


def lag_gate(gate: np.ndarray, lag: int) -> np.ndarray:
    """Delay a gate by ``lag`` days, holding the first value at the start."""
    if lag < 0:
        raise ValueError(f"lag must be >= 0, got {lag}")
    g = np.asarray(gate, dtype=float)
    if lag == 0:
        return g.copy()
    return np.concatenate([np.full(lag, g[0]), g[:-lag]])


# ---------------------------------------------------------------------- #
# Paired stationary block bootstrap
# ---------------------------------------------------------------------- #
@dataclass
class BootstrapResult:
    """Paired bootstrap of ``Sharpe(a) - Sharpe(b)``.

    Attributes:
        diff: Difference on the original sample.
        ci_low / ci_high: Percentile interval of the resampled differences.
        std_error: Standard deviation of the resampled differences.
        prob_positive: Share of resamples with a positive difference.
        n_boot: Number of resamples.
        mean_block: Mean block length in days.
    """

    diff: float
    ci_low: float
    ci_high: float
    std_error: float
    prob_positive: float
    n_boot: int
    mean_block: float


def stationary_bootstrap_indices(
    n: int, n_boot: int, mean_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano index paths, shape ``(n_boot, n)``.

    Each path starts at a uniform day, and each following day either continues
    the block (wrapping at the end of the sample) or, with probability
    ``1 / mean_block``, jumps to a new uniform day.
    """
    if n < 1 or n_boot < 1 or mean_block < 1.0:
        raise ValueError("need n >= 1, n_boot >= 1 and mean_block >= 1")
    new_block = rng.random((n_boot, n)) < 1.0 / mean_block
    new_block[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    pos = np.arange(n)
    block_start = np.maximum.accumulate(np.where(new_block, pos, 0), axis=1)
    return (np.take_along_axis(starts, block_start, axis=1) + pos - block_start) % n


def paired_block_bootstrap(
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    n_boot: int = 10000,
    mean_block: float = 21.0,
    level: float = 0.95,
) -> BootstrapResult:
    """Stationary block bootstrap of a Sharpe difference.

    Both series are resampled with the same index path, so the correlation
    between them is kept and a shared crisis stays shared.

    Args:
        a / b: Daily return series of equal length.
        rng: Generator supplied by the caller.
        n_boot: Number of resamples.
        mean_block: Mean block length in days.
        level: Coverage of the percentile interval.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be non-empty 1-D series of equal length")
    diffs = np.empty(n_boot)
    chunk = 500
    for lo in range(0, n_boot, chunk):
        m = min(chunk, n_boot - lo)
        idx = stationary_bootstrap_indices(a.size, m, mean_block, rng)
        diffs[lo:lo + m] = sharpe(a[idx]) - sharpe(b[idx])
    tail = 100.0 * (1.0 - level) / 2.0
    low, high = np.percentile(diffs, [tail, 100.0 - tail])
    return BootstrapResult(
        diff=float(sharpe(a) - sharpe(b)),
        ci_low=float(low),
        ci_high=float(high),
        std_error=float(diffs.std(ddof=1)) if n_boot > 1 else 0.0,
        prob_positive=float(np.mean(diffs > 0.0)),
        n_boot=n_boot,
        mean_block=float(mean_block),
    )


# ---------------------------------------------------------------------- #
# Lag and cost sweeps, break-even cost
# ---------------------------------------------------------------------- #
def cost_sweep(
    streams: Mapping[str, CombinedStream], costs: Sequence[float], start: int
) -> Dict[str, List[float]]:
    """Window Sharpe of each stream at each cost."""
    return {
        name: [float(sharpe(s.net(c)[start:])) for c in costs]
        for name, s in streams.items()
    }


def lag_sweep(
    book: GateBook,
    gates: Mapping[str, np.ndarray],
    lags: Sequence[int],
    cost_bps: float,
    start: int,
) -> Dict[str, List[float]]:
    """Window Sharpe of each gate when it reaches the book ``lag`` days late."""
    return {
        name: [float(sharpe(book.stream(lag_gate(g, lag)).net(cost_bps)[start:])) for lag in lags]
        for name, g in gates.items()
    }


@dataclass
class BreakEven:
    """Result of :func:`break_even_cost`.

    Attributes:
        status: ``crossed``, ``below_at_zero`` (the gate already loses at the
            low end) or ``above_at_max`` (it still wins at the high end).
        cost_bps: Break-even cost; NaN unless ``crossed``.
        low / high: Final bracket; equal to the search range if not crossed.
    """

    status: str
    cost_bps: float
    low: float
    high: float


def break_even_cost(
    edge: Callable[[float], float], low: float = 0.0, high: float = 100.0, tol: float = 1e-4
) -> BreakEven:
    """Bisect for the cost at which ``edge(cost)`` falls to zero.

    Args:
        edge: Sharpe advantage of the gate at a cost in bps.  Expected
            positive at ``low`` and negative at ``high``.
        low / high: Search range in bps.
        tol: Bracket width at which to stop.
    """
    if not low < high or tol <= 0.0:
        raise ValueError("need low < high and tol > 0")
    if edge(low) <= 0.0:
        return BreakEven("below_at_zero", float("nan"), low, high)
    if edge(high) > 0.0:
        return BreakEven("above_at_max", float("nan"), low, high)
    while high - low > tol:
        mid = 0.5 * (low + high)
        if edge(mid) > 0.0:
            low = mid
        else:
            high = mid
    return BreakEven("crossed", 0.5 * (low + high), low, high)


def gate_break_even(gated: CombinedStream, base: CombinedStream, start: int, high: float = 100.0) -> BreakEven:
    """Cost at which the gated book's window Sharpe falls to the base book's."""

    def edge(cost: float) -> float:
        return float(sharpe(gated.net(cost)[start:]) - sharpe(base.net(cost)[start:]))

    return break_even_cost(edge, 0.0, high)


# ---------------------------------------------------------------------- #
# Out-of-sample classification
# ---------------------------------------------------------------------- #
def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties at half weight).

    Returns NaN when only one class is present.
    """
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("labels and scores must be 1-D and equal length")
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def crisis_episodes(true_states: np.ndarray, start: int, crisis_state: int = CRISIS_STATE) -> List[tuple]:
    """Half-open ``(first, end)`` runs of crisis days that begin at or after ``start``."""
    is_crisis = np.asarray(true_states) == crisis_state
    edges = np.diff(np.concatenate([[0], is_crisis.astype(int), [0]]))
    runs = zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    return [(int(a), int(b)) for a, b in runs if a >= start]


def classification_report(
    gate: np.ndarray,
    true_states: np.ndarray,
    start: int,
    threshold: float = 0.5,
    calm_state: int = CALM_STATE,
    crisis_state: int = CRISIS_STATE,
) -> Dict[str, float]:
    """Score the walk-forward gate against the generator's states.

    The gate is ``P(calm)``, so the two-class task is calm against not calm:
    ``accuracy`` calls a day calm when ``gate >= threshold``, and
    ``majority_accuracy`` is what always naming the commoner class scores.
    ``auc_calm`` ranks calm days by the gate, and ``auc_crisis`` ranks crisis
    days by ``1 - gate`` against all other days.

    Crisis detection lag is the number of days from the first crisis day to
    the first day with ``gate < threshold``, inside the episode.  An episode
    that ends first counts as missed, and lag 0 includes episodes the gate was
    already below threshold for when they began.
    """
    g = np.asarray(gate, dtype=float)
    s = np.asarray(true_states)
    calm = s[start:] == calm_state
    crisis = s[start:] == crisis_state
    window = g[start:]
    lags: List[int] = []
    episodes = crisis_episodes(s, start, crisis_state)
    for first, end in episodes:
        below = np.flatnonzero(g[first:end] < threshold)
        if below.size:
            lags.append(int(below[0]))
    share = float(calm.mean())
    return {
        "n_days": int(calm.size),
        "n_crisis_days": int(crisis.sum()),
        "accuracy": float(np.mean((window >= threshold) == calm)),
        "majority_accuracy": max(share, 1.0 - share),
        "auc_calm": roc_auc(calm, window),
        "auc_crisis": roc_auc(crisis, 1.0 - window),
        "n_episodes": len(episodes),
        "n_missed": len(episodes) - len(lags),
        "median_lag": float(np.median(lags)) if lags else float("nan"),
        "mean_lag": float(np.mean(lags)) if lags else float("nan"),
    }


# ---------------------------------------------------------------------- #
# Model selection
# ---------------------------------------------------------------------- #
def bic_table(
    x: np.ndarray,
    states: Sequence[int] = (1, 2, 3, 4),
    tol: float = 1e-8,
    max_iter: int = 500,
    var_floor: float = 1e-8,
) -> List[Dict[str, float]]:
    """In-sample BIC by state count, with K=1 as the no-regime null.

    ``GaussianHMM`` needs K >= 2, so K=1 is the closed-form i.i.d. Gaussian:
    two parameters, maximum-likelihood mean and variance.
    """
    r = np.asarray(x, dtype=float)
    n = r.size
    rows: List[Dict[str, float]] = []
    for k in states:
        if k == 1:
            var = float(r.var(ddof=0))
            loglik = -0.5 * n * (np.log(2.0 * np.pi * var) + 1.0)
            n_params = 2
        else:
            model = GaussianHMM(k, tol=tol, max_iter=max_iter, var_floor=var_floor)
            loglik = model.fit(r).log_likelihood
            n_params = model.n_parameters(1)
        rows.append(
            {"k": k, "loglik": float(loglik), "n_params": n_params, "bic": float(-2.0 * loglik + n_params * np.log(n))}
        )
    return rows
