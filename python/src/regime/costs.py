"""Opt-in cost model with spread and square-root impact.

This module is separate from :func:`regime.backtest.run_backtest` on purpose.
The backtest charges a flat ``cost_bps`` on turnover and its numbers are
pinned across the ports; nothing here changes them.  A flat rate does not grow
with the size of the book, so it cannot say anything about capacity.  This
model can.

For one instrument on one day, with weight change ``dw`` and book size
``AUM``, the traded notional is ``trade = |dw| * AUM`` and

    cost = half_spread * |dw| * AUM
         + Y * sigma_daily * sqrt(trade / ADV) * trade

in dollars, with ``half_spread`` as a fraction (bps / 10^4).

Assumptions, each with its source class
---------------------------------------
* Functional form (general form, not an estimate).  Impact per unit traded
  that grows with the square root of participation is a common rule of thumb.
  Almgren et al. (2005), "Direct estimation of equity market impact", is the
  usual empirical reference for concave impact; they fit an exponent near
  3/5 on US equities, not exactly 1/2.  It is cited for the general concave
  shape only, and no coefficient here comes from that paper.
* ``impact_y`` (illustrative default, not calibrated).  Order one by
  convention.  No trade data was used to choose it.
* ``half_spread_bps`` (illustrative default, not calibrated).  The bundled
  instruments are synthetic, so they have no quoted spread.
* ``adv_usd`` (illustrative default, not calibrated).  The bundled instruments
  are synthetic, so they have no traded volume.
* ``sigma_daily`` (measured from the return series the caller passes in).
* Each day's trade fills in one day at the close.  Nothing is split across
  days, so a large rebalance pays more than a scheduled one would.  Above
  100% participation the formula is outside any range it could describe.
* AUM stays fixed: weights apply to the same dollar book every day.
* Permanent impact, financing, borrow and fees are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

#: Illustrative default, not calibrated: impact coefficient Y.
ILLUSTRATIVE_IMPACT_Y = 1.0
#: Illustrative defaults, not calibrated: trend instruments.
ILLUSTRATIVE_TREND_HALF_SPREAD_BPS = 1.0
ILLUSTRATIVE_TREND_ADV_USD = 5e9
#: Illustrative defaults, not calibrated: currency instruments.
ILLUSTRATIVE_FX_HALF_SPREAD_BPS = 0.5
ILLUSTRATIVE_FX_ADV_USD = 2e10


def _per_instrument(values, name: str, positive: bool) -> np.ndarray:
    """Coerce to a finite non-empty 1-D array; reject negatives or zeros."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contain NaN or inf")
    if np.any(x <= 0.0) if positive else np.any(x < 0.0):
        raise ValueError(f"{name} must be {'> 0' if positive else '>= 0'}")
    return x


def _check_aum(aum: float) -> float:
    if not np.isfinite(aum) or aum <= 0.0:
        raise ValueError(f"aum must be finite and > 0, got {aum}")
    return float(aum)


@dataclass(frozen=True)
class CostAssumptions:
    """Named cost inputs, one entry per instrument.

    Attributes:
        half_spread_bps: Half the bid-offer spread in bps, shape ``(A,)``.
        adv_usd: Average daily volume in dollars, shape ``(A,)``.
        sigma_daily: Daily return volatility as a fraction, shape ``(A,)``.
        impact_y: Dimensionless impact coefficient Y.
    """

    half_spread_bps: np.ndarray
    adv_usd: np.ndarray
    sigma_daily: np.ndarray
    impact_y: float = ILLUSTRATIVE_IMPACT_Y

    def __post_init__(self) -> None:
        spread = _per_instrument(self.half_spread_bps, "half_spread_bps", positive=False)
        adv = _per_instrument(self.adv_usd, "adv_usd", positive=True)
        sigma = _per_instrument(self.sigma_daily, "sigma_daily", positive=False)
        if not spread.shape == adv.shape == sigma.shape:
            raise ValueError("half_spread_bps, adv_usd and sigma_daily must have equal length")
        if not np.isfinite(self.impact_y) or self.impact_y < 0.0:
            raise ValueError(f"impact_y must be finite and >= 0, got {self.impact_y}")
        object.__setattr__(self, "half_spread_bps", spread)
        object.__setattr__(self, "adv_usd", adv)
        object.__setattr__(self, "sigma_daily", sigma)


def _weight_changes(weights: np.ndarray, n_instruments: int) -> np.ndarray:
    """``|W[t] - W[t-1]|`` with ``W[-1] = 0``, the backtest's turnover rule."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] == 0 or w.shape[1] != n_instruments:
        raise ValueError(f"weights must have shape (T, {n_instruments}), got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contain NaN or inf")
    return np.abs(np.diff(w, axis=0, prepend=0.0))


def trade_cost(dw: np.ndarray, aum: float, assumptions: CostAssumptions) -> np.ndarray:
    """Dollar cost of weight changes, same shape as ``dw``.

    Args:
        dw: Weight changes, shape ``(..., A)``; the sign does not matter.
        aum: Book size in dollars.
        assumptions: Cost inputs for the A instruments.
    """
    aum = _check_aum(aum)
    size = np.abs(np.asarray(dw, dtype=float))
    if size.ndim == 0 or size.shape[-1] != assumptions.adv_usd.size:
        raise ValueError(f"dw last axis must have length {assumptions.adv_usd.size}")
    if not np.all(np.isfinite(size)):
        raise ValueError("dw contain NaN or inf")
    trade = size * aum
    spread = assumptions.half_spread_bps / 1e4 * trade
    impact = assumptions.impact_y * assumptions.sigma_daily * np.sqrt(trade / assumptions.adv_usd) * trade
    return spread + impact


def daily_cost_fraction(weights: np.ndarray, aum: float, assumptions: CostAssumptions) -> np.ndarray:
    """Each day's total cost as a fraction of AUM, shape ``(T,)``.

    The move into ``W[t]`` is charged on day t, as in the backtest.
    """
    dw = _weight_changes(weights, assumptions.adv_usd.size)
    return trade_cost(dw, aum, assumptions).sum(axis=1) / aum


def participation(weights: np.ndarray, aum: float, assumptions: CostAssumptions) -> np.ndarray:
    """Trade as a fraction of ADV per instrument and day, shape ``(T, A)``."""
    aum = _check_aum(aum)
    return _weight_changes(weights, assumptions.adv_usd.size) * aum / assumptions.adv_usd


def aum_at_sharpe_fraction(
    sharpe_at: Callable[[float], float],
    low: float,
    high: float,
    fraction: float = 0.5,
    n_grid: int = 41,
    rel_tol: float = 1e-4,
) -> Optional[float]:
    """Smallest AUM in ``[low, high]`` where Sharpe falls to ``fraction`` of its value at ``low``.

    Scans a log grid for the first crossing, then bisects inside that cell, so
    a Sharpe curve that is not monotone still gives the first crossing the
    grid can see.

    Args:
        sharpe_at: Net Sharpe as a function of AUM.
        low / high: AUM range in dollars.
        fraction: Target share of the Sharpe at ``low``, in ``(0, 1)``.
        n_grid: Grid points on the log scale.
        rel_tol: Stop when the bracket's relative width is below this.

    Returns:
        The AUM, or ``None`` if the Sharpe at ``low`` is not positive (the
        fraction has no meaning) or the target is not reached by ``high``.
    """
    if not 0.0 < low < high or not np.isfinite(high):
        raise ValueError(f"need 0 < low < high, got {low}, {high}")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    if n_grid < 2 or rel_tol <= 0.0:
        raise ValueError("need n_grid >= 2 and rel_tol > 0")
    base = float(sharpe_at(low))
    if base <= 0.0:
        return None
    target = fraction * base
    grid = np.geomspace(low, high, n_grid)
    for a, b in zip(grid[:-1], grid[1:]):
        if float(sharpe_at(b)) <= target:
            while b / a - 1.0 > rel_tol:
                mid = float(np.sqrt(a * b))
                if float(sharpe_at(mid)) <= target:
                    b = mid
                else:
                    a = mid
            return float(b)
    return None
