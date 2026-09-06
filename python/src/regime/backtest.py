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


def _coerce_dates(dates, T: int) -> pd.DatetimeIndex:
    """Coerce ``dates`` to a ``DatetimeIndex`` of length ``T`` (pinned checks).

    Raises ``ValueError`` if the dates cannot be parsed, the length is not
    ``T``, or they are not strictly increasing (duplicates and out-of-order
    rows are the most common real-feed defects, and every port's
    ``worst_month`` assumes sorted dates — API_SPEC 2.1).
    """
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(dates))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"dates could not be parsed: {exc}") from exc
    if len(idx) != T:
        raise ValueError(f"dates length {len(idx)} does not match number of days {T}")
    if idx.hasnans:
        raise ValueError("dates contain NaT")
    if not (idx.is_monotonic_increasing and idx.is_unique):
        raise ValueError("dates must be strictly increasing")
    return idx


def _validate_net(net) -> np.ndarray:
    """Coerce a daily net-return series; reject empty, non-finite or <= -1."""
    net = np.asarray(net, dtype=float)
    if net.ndim != 1:
        raise ValueError("net return series must be 1-D")
    if net.size == 0:
        raise ValueError("net return series is empty")
    if not np.all(np.isfinite(net)):
        raise ValueError("net returns contain NaN or inf")
    bad = np.flatnonzero(net <= -1.0)
    if bad.size > 0:
        t = int(bad[0])
        raise ValueError(f"equity wiped out on day t={t}: net return {net[t]} <= -100%")
    return net


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum drawdown of an equity curve, as a non-positive fraction.

    Raises:
        ValueError: If the curve is empty, non-finite, or not strictly
            positive (a drawdown from or to a non-positive equity is
            undefined; ``run_backtest`` rejects wipe-outs before this).
    """
    eq = np.asarray(equity, dtype=float)
    if eq.ndim != 1 or eq.size == 0:
        raise ValueError("equity curve is empty")
    if not np.all(np.isfinite(eq)) or np.any(eq <= 0.0):
        raise ValueError("equity curve must be finite and strictly positive")
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1.0))


def compute_metrics(net: np.ndarray, dates: Optional[pd.DatetimeIndex] = None) -> Dict[str, float]:
    """Pinned summary metrics from a daily net-return series.

    Args:
        net: Daily net returns, shape ``(T,)``.
        dates: Optional dates (anything ``pd.to_datetime`` accepts; length
            T, strictly increasing) for the worst-calendar-month metric;
            when omitted, ``worst_month`` is computed over consecutive
            21-day blocks instead (the trailing partial block is dropped —
            pinned, API_SPEC 2.1).

    Returns:
        Dict with ann_return, ann_vol, sharpe, max_dd, calmar, hit_rate,
        worst_month.

    Raises:
        ValueError: Empty/non-finite ``net``, a net return ``<= -1``
            (wipe-out), or dates that are the wrong length, unparseable, or
            not strictly increasing.
    """
    net = _validate_net(net)
    if dates is not None:
        dates = _coerce_dates(dates, net.size)
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
        ValueError: On shape mismatch, empty input (no days or no assets),
            non-finite input, a return ``<= -1``, negative cost, invalid
            dates, or a wipe-out (``net[t] <= -1`` for some day; the
            message names ``t``) — pinned, API_SPEC 2.
    """
    P = np.asarray(positions, dtype=float)
    r = np.asarray(returns, dtype=float)
    if P.ndim == 1:
        P = P[:, None]
    if r.ndim == 1:
        r = r[:, None]
    if P.ndim != 2 or r.ndim != 2 or P.shape != r.shape:
        raise ValueError(f"positions {P.shape} and returns {r.shape} must have equal shape")
    if P.shape[0] == 0:
        raise ValueError("empty backtest: no days")
    if P.shape[1] == 0:
        raise ValueError("empty backtest: no assets")
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(r))):
        raise ValueError("positions/returns contain NaN or inf")
    if np.any(r <= -1.0):
        t, a = np.argwhere(r <= -1.0)[0]
        raise ValueError(f"returns contain a return <= -100% at t={t}, asset={a} ({r[t, a]})")
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}")
    T = P.shape[0]
    if dates is not None:
        dates = _coerce_dates(dates, T)

    gross = np.zeros(T)
    gross[1:] = np.sum(P[:-1] * r[1:], axis=1)
    P_prev = np.vstack([np.zeros((1, P.shape[1])), P[:-1]])
    turnover = np.sum(np.abs(P - P_prev), axis=1)
    net = gross - (cost_bps / 1e4) * turnover
    net = _validate_net(net)  # wipe-out guard: net[t] <= -1 is an error naming t
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

    Raises:
        ValueError: Length mismatch, non-finite ``net``, ``n_states < 1``,
            or a state label outside ``[0, n_states)`` (pinned: labels are
            never silently dropped).
    """
    net = np.asarray(net, dtype=float)
    states = np.asarray(states)
    if net.ndim != 1 or states.ndim != 1 or net.shape != states.shape:
        raise ValueError("net returns and states must have equal length")
    if not np.all(np.isfinite(net)):
        raise ValueError("net returns contain NaN or inf")
    if int(n_states) != n_states or n_states < 1:
        raise ValueError(f"n_states must be an integer >= 1, got {n_states}")
    n_states = int(n_states)
    if states.size > 0 and (
        not np.issubdtype(states.dtype, np.integer) or states.min() < 0 or states.max() >= n_states
    ):
        raise ValueError(f"state labels must be integers in [0, {n_states})")
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
