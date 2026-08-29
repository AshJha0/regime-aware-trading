"""Backtest engine with pinned no-lookahead accounting and metrics.

Accounting (pinned, shared by every strategy)
---------------------------------------------
Positions ``P[t]`` are decided at the close of day t.  The trade from
``P[t-1]`` to ``P[t]`` executes at that close, so:

    gross[t]    = sum_a P[t-1, a] * r[t, a]          (gross[0] = 0)
    turnover[t] = sum_a |P[t, a] - P[t-1, a]|        (P[-1] = 0)
    net[t]      = gross[t] - (cost_bps / 10^4) * turnover[t]
    equity[t]   = prod_{u<=t} (1 + net[u])

The day-t P&L is earned by yesterday's position — the signal never sees the
return it trades against — while the cost of moving *into* ``P[t]`` is
charged on day t itself.  ``cost_bps = 0`` gives ``net == gross`` exactly.

Metrics (pinned formulas, full-sample, 252 trading days/year)
-------------------------------------------------------------
    ann_return = 252 * mean(net)
    ann_vol    = sqrt(252) * std(net, ddof=0)
    sharpe     = ann_return / ann_vol           (0 if ann_vol == 0)
    max_dd     = min_t (equity[t] / cummax(equity)[t] - 1)   (<= 0)
    calmar     = ann_return / |max_dd|          (0 if max_dd == 0)
    hit_rate   = mean(net > 0)
    worst_month: calendar-month compounded net returns, minimum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    """Daily series and summary metrics of one backtest run.

    Attributes:
        gross_returns: Pre-cost daily portfolio returns, shape ``(T,)``.
        net_returns: Post-cost daily returns, shape ``(T,)``.
        turnover: Daily one-sided turnover ``sum|dP|``, shape ``(T,)``.
        equity: Compounded equity curve starting near 1, shape ``(T,)``.
        metrics: Pinned summary metrics (see module docstring).
    """

    gross_returns: np.ndarray
    net_returns: np.ndarray
    turnover: np.ndarray
    equity: np.ndarray
    metrics: Dict[str, float]


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown of an equity curve, as a non-positive fraction."""
    eq = np.asarray(equity, dtype=float)
    if eq.size == 0:
        raise ValueError("equity curve is empty")
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1.0))


def compute_metrics(net: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, float]:
    """Pinned summary metrics from a daily net-return series.

    Args:
        net: Daily net returns, shape ``(T,)``.
        dates: Optional dates for the worst-calendar-month metric; when
            omitted, ``worst_month`` is computed over consecutive 21-day
            blocks instead.

    Returns:
        Dict with ann_return, ann_vol, sharpe, max_dd, calmar, hit_rate,
        worst_month.
    """
    net = np.asarray(net, dtype=float)
    if net.size == 0:
        raise ValueError("net return series is empty")
    if not np.all(np.isfinite(net)):
        raise ValueError("net returns contain NaN or inf")
    ann_ret = TRADING_DAYS * float(np.mean(net))
    ann_vol = float(np.sqrt(TRADING_DAYS) * np.std(net, ddof=0))
    sharpe = ann_ret / ann_vol if ann_vol > 0.0 else 0.0
    equity = np.cumprod(1.0 + net)
    mdd = max_drawdown(equity)
    calmar = ann_ret / abs(mdd) if mdd < 0.0 else 0.0
    hit = float(np.mean(net > 0.0))
    if dates is not None:
        s = pd.Series(net, index=dates)
        monthly = (1.0 + s).groupby([dates.year, dates.month]).prod() - 1.0
        worst_month = float(monthly.min())
    else:
        nblk = net.size // 21 or 1
        blocks = [net[i * 21:(i + 1) * 21] for i in range(nblk)]
        worst_month = float(min(np.prod(1.0 + b) - 1.0 for b in blocks))
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": mdd,
        "calmar": calmar,
        "hit_rate": hit,
        "worst_month": worst_month,
    }


def run_backtest(
    positions: np.ndarray,
    returns: np.ndarray,
    cost_bps: float = 0.0,
    dates: Optional[pd.DatetimeIndex] = None,
) -> BacktestResult:
    """Run the pinned accounting on a position/return pair.

    Args:
        positions: ``P[t, a]`` decided at close of day t, shape ``(T, A)``
            (or ``(T,)`` for a single asset).
        returns: Asset simple returns ``r[t, a]``, same shape.
        cost_bps: One-way transaction cost in basis points of turnover.
        dates: Optional dates (length T) for calendar-month metrics.

    Returns:
        :class:`BacktestResult`.

    Raises:
        ValueError: On shape mismatch, non-finite input, or negative cost.
    """
    P = np.asarray(positions, dtype=float)
    r = np.asarray(returns, dtype=float)
    if P.ndim == 1:
        P = P[:, None]
    if r.ndim == 1:
        r = r[:, None]
    if P.shape != r.shape:
        raise ValueError(f"positions {P.shape} and returns {r.shape} must have equal shape")
    if P.shape[0] == 0:
        raise ValueError("empty backtest: no days")
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(r))):
        raise ValueError("positions/returns contain NaN or inf")
    if cost_bps < 0.0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    if dates is not None and len(dates) != P.shape[0]:
        raise ValueError("dates length does not match number of days")

    T = P.shape[0]
    gross = np.zeros(T)
    gross[1:] = np.sum(P[:-1] * r[1:], axis=1)
    P_prev = np.vstack([np.zeros((1, P.shape[1])), P[:-1]])
    turnover = np.sum(np.abs(P - P_prev), axis=1)
    net = gross - (cost_bps / 1e4) * turnover
    equity = np.cumprod(1.0 + net)
    return BacktestResult(gross, net, turnover, equity, compute_metrics(net, dates))


def state_conditional_returns(net: np.ndarray, states: np.ndarray, n_states: int) -> Dict[int, Dict[str, float]]:
    """Annualized performance of a return series conditioned on a state path.

    Used for the crisis-period table: day-t net return is attributed to the
    day-t decoded state.

    Args:
        net: Daily net returns, shape ``(T,)``.
        states: Integer state labels per day, shape ``(T,)``.
        n_states: Number of states K.

    Returns:
        Per-state dict with n_days, ann_return, ann_vol, sharpe.
    """
    net = np.asarray(net, dtype=float)
    states = np.asarray(states)
    if net.shape != states.shape:
        raise ValueError("net returns and states must have equal length")
    out: Dict[int, Dict[str, float]] = {}
    for k in range(n_states):
        mask = states == k
        n = int(mask.sum())
        if n == 0:
            out[k] = {"n_days": 0, "ann_return": 0.0, "ann_vol": 0.0, "sharpe": 0.0}
            continue
        mu = TRADING_DAYS * float(np.mean(net[mask]))
        sd = float(np.sqrt(TRADING_DAYS) * np.std(net[mask], ddof=0))
        out[k] = {
            "n_days": n,
            "ann_return": mu,
            "ann_vol": sd,
            "sharpe": mu / sd if sd > 0 else 0.0,
        }
    return out
