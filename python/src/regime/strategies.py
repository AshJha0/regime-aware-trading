"""Trading strategies: time-series momentum, FX carry, and the HMM regime gate.

No-lookahead convention (shared with the backtest engine)
---------------------------------------------------------
Every position array ``P`` produced here has ``P[t]`` = the position
*decided at the close of day t using information up to and including day t*.
The backtest applies ``P[t-1]`` to the day-``t`` return, so a signal can
never see the return it is traded against.  The regime gate at day ``t``
likewise uses only observations ``x_0..x_t`` (filtered, not smoothed,
probabilities) and models fitted on data through day ``t``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .hmm import GaussianHMM, HMMParams, forward_step

TRADING_DAYS = 252


def _validate_returns(returns: np.ndarray, name: str = "returns", simple_returns: bool = False) -> np.ndarray:
    """Coerce to (T, A) float array; reject empty or non-finite input.

    With ``simple_returns=True`` (pinned, API_SPEC 3): every entry must be
    ``> -1`` — a -100 % (or worse) day is corrupt data (a zero print, an
    unadjusted split) and would make every trailing growth ratio ``0/0``.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim == 1:
        r = r[:, None]
    if r.ndim != 2 or r.shape[0] == 0 or r.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 1-D or 2-D array")
    if not np.all(np.isfinite(r)):
        raise ValueError(f"{name} contain NaN or inf")
    if simple_returns and np.any(r <= -1.0):
        t, a = np.argwhere(r <= -1.0)[0]
        raise ValueError(f"{name} contain a return <= -100% at t={t}, asset={a} ({r[t, a]})")
    return r


def ewma_variance(returns: np.ndarray, lam: float, init_window: int) -> np.ndarray:
    """EWMA variance per asset with a pinned initialization.

    Pinned rule: at ``t = init_window - 1`` the variance is the mean of
    squared returns over the first ``init_window`` days (zero-mean
    convention), and for ``t >= init_window``

        sigma2[t] = lam * sigma2[t-1] + (1 - lam) * r[t]^2.

    Entries before ``init_window - 1`` are NaN (undefined).

    Args:
        returns: Daily returns, shape ``(T, A)``.
        lam: EWMA decay (e.g. 0.94).
        init_window: Seed window length (the momentum lookback).

    Returns:
        Array ``(T, A)`` of daily-frequency variances.
    """
    r = _validate_returns(returns, simple_returns=True)
    T, A = r.shape
    if not 0.0 < lam < 1.0:
        raise ValueError(f"ewma lambda must be in (0, 1), got {lam}")
    if init_window < 1 or init_window > T:
        raise ValueError(f"init_window must be in [1, T], got {init_window}")
    sig2 = np.full((T, A), np.nan)
    sig2[init_window - 1] = np.mean(r[:init_window] ** 2, axis=0)
    for t in range(init_window, T):
        sig2[t] = lam * sig2[t - 1] + (1.0 - lam) * r[t] ** 2
    return sig2


def momentum_positions(
    returns: np.ndarray,
    lookback: int = 252,
    vol_target: float = 0.10,
    ewma_lambda: float = 0.94,
    leverage_cap: float = 4.0,
) -> np.ndarray:
    """Time-series momentum positions with vol-targeted sizing.

    Signal at day t (defined for ``t >= lookback - 1``): the sign of the
    cumulative simple return ``prod(1 + r) - 1`` over the trailing
    ``lookback`` days ending at t (sign in {-1, 0, +1}).  Sizing: per-asset
    leverage ``min(vol_target / ann_vol[t], leverage_cap)`` where
    ``ann_vol = sqrt(252 * sigma2_EWMA)``; a zero EWMA vol gets the cap.
    Positions are divided by the number of assets (equal risk budget), and
    are zero before the first full lookback window.

    Args:
        returns: Daily simple returns, shape ``(T, A)`` (or ``(T,)``).
        lookback: Trailing window length in days.
        vol_target: Annualized per-asset volatility target.
        ewma_lambda: EWMA decay for the vol estimate.
        leverage_cap: Maximum per-asset leverage.

    Returns:
        Positions ``(T, A)``; ``P[t]`` is decided at close of day t.

    Raises:
        ValueError: If the series is not longer than the lookback, any
            parameter is invalid, or a return is ``<= -100%`` (pinned).
    """
    r = _validate_returns(returns, simple_returns=True)
    T, A = r.shape
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if T <= lookback:
        raise ValueError(f"series shorter than momentum lookback: T={T} <= lookback={lookback}")
    if vol_target <= 0 or leverage_cap <= 0:
        raise ValueError("vol_target and leverage_cap must be > 0")

    # Trailing cumulative return via cumulative log(1+r) differences.
    growth = np.cumprod(1.0 + r, axis=0)
    cumret = np.full((T, A), np.nan)
    cumret[lookback - 1] = growth[lookback - 1] - 1.0
    cumret[lookback:] = growth[lookback:] / growth[:-lookback] - 1.0
    signal = np.sign(cumret)

    sig2 = ewma_variance(r, ewma_lambda, lookback)
    ann_vol = np.sqrt(TRADING_DAYS * sig2)
    with np.errstate(divide="ignore"):
        lev = np.where(ann_vol > 0.0, np.minimum(vol_target / ann_vol, leverage_cap), leverage_cap)

    pos = np.zeros((T, A))
    valid = slice(lookback - 1, T)
    pos[valid] = signal[valid] * lev[valid] / A
    return pos


def carry_positions(
    rate_diffs: np.ndarray,
    top_n: int = 3,
    bottom_n: int = 3,
    rebalance_days: int = 21,
) -> np.ndarray:
    """FX carry positions: long high-differential, short low-differential.

    Pinned schedule: rebalance at day indices ``0, 21, 42, ...`` (every
    ``rebalance_days`` trading days); positions are held unchanged between
    rebalances.  At each rebalance the currencies are ranked by that day's
    interest-rate differential, descending; ties are broken by ascending
    asset index (a stable sort on ``(-diff, index)``).  The top ``top_n``
    get ``+1/top_n`` each and the bottom ``bottom_n`` get ``-1/bottom_n``
    each (equal notional, gross leverage 2 when both sides are full).

    Args:
        rate_diffs: Annualized interest differentials, shape ``(T, A)``.
        top_n: Number of longs.
        bottom_n: Number of shorts.
        rebalance_days: Trading days between rebalances.

    Returns:
        Positions ``(T, A)``; ``P[t]`` decided at close of day t.

    Raises:
        ValueError: If ``top_n + bottom_n`` exceeds the asset count or the
            schedule/params are invalid.
    """
    d = _validate_returns(rate_diffs, "rate differentials")
    T, A = d.shape
    if top_n < 1 or bottom_n < 1 or top_n + bottom_n > A:
        raise ValueError(f"need top_n>=1, bottom_n>=1, top_n+bottom_n<=A={A}; got {top_n},{bottom_n}")
    if rebalance_days < 1:
        raise ValueError(f"rebalance_days must be >= 1, got {rebalance_days}")

    pos = np.zeros((T, A))
    current = np.zeros(A)
    for t in range(T):
        if t % rebalance_days == 0:
            # Stable sort on descending differential; index breaks ties.
            order = np.lexsort((np.arange(A), -d[t]))
            current = np.zeros(A)
            current[order[:top_n]] = 1.0 / top_n
            current[order[A - bottom_n:]] = -1.0 / bottom_n
        pos[t] = current
    return pos


def carry_total_returns(spot_returns: np.ndarray, rate_diffs: np.ndarray) -> np.ndarray:
    """Total currency returns: spot move plus accrued carry.

    Pinned accrual: the day-``t`` total return is
    ``spot[t] + diff[t-1] / 252`` — the rate differential is the one set at
    the previous close (known in advance, no lookahead).  Day 0 accrues
    nothing.

    Args:
        spot_returns: Daily spot returns, shape ``(T, A)``.
        rate_diffs: Annualized differentials, shape ``(T, A)``.

    Returns:
        Total returns, shape ``(T, A)``.
    """
    s = _validate_returns(spot_returns, "spot returns", simple_returns=True)
    d = _validate_returns(rate_diffs, "rate differentials")
    if s.shape != d.shape:
        raise ValueError(f"spot returns {s.shape} and differentials {d.shape} must have equal shape")
    total = s.copy()
    total[1:] += d[:-1] / TRADING_DAYS
    return total


def regime_gate(
    index_returns: np.ndarray,
    n_states: int = 3,
    train_min_days: int = 504,
    refit_days: int = 63,
    mode: str = "prob",
    threshold: float = 0.5,
    tol: float = 1e-8,
    max_iter: int = 500,
    var_floor: float = 1e-8,
) -> Tuple[np.ndarray, List[GaussianHMM]]:
    """Walk-forward HMM regime gate on a market proxy.

    Pinned schedule: the first fit happens at day ``t0 = train_min_days - 1``
    on the expanding window ``r[0..t0]``; refits at ``t0 + refit_days * k``
    on ``r[0..t]``.  The first fit starts from the pinned quantile
    initialization; every subsequent refit warm-starts from the previous
    fitted (label-sorted) parameters, which keeps the walk-forward loop
    deterministic and fast.

    The gate at day t (for ``t >= t0``) is based on the *filtered*
    probability ``p_calm(t) = P(s_t = k* | r_0..r_t)`` of the low-vol/bull
    state under the model most recently fitted at or before t, where the
    pinned identification is ``k* = argmin_k variances[k, 0]`` (the
    lowest-variance state; ties broken by lower label).  Identifying the
    benign regime by variance rather than mean is robust to walk-forward
    fits that isolate a small high-mean outlier state.  A full scaled
    forward pass is run at each refit day and single :func:`forward_step`
    updates are used in between.  Before the first fit the gate is 1 (no
    filtering).

    ``mode='prob'`` returns ``gate[t] = p_calm(t)``;
    ``mode='binary'`` returns ``1.0`` if ``p_calm(t) >= threshold`` else ``0.0``.

    Args:
        index_returns: Market proxy daily returns, shape ``(T,)``.
        n_states: HMM state count K.
        train_min_days: Observations required before the first fit.
        refit_days: Trading days between refits.
        mode: 'prob' (scale by probability) or 'binary' (hard gate).
        threshold: Probability threshold for the binary gate.
        tol/max_iter/var_floor: EM settings, passed to :class:`GaussianHMM`.

    Returns:
        ``(gate, models)``: gate values in [0, 1], shape ``(T,)``, and the
        list of fitted models in refit order.

    Raises:
        ValueError: If the series is shorter than ``train_min_days``, the
            mode is unknown, ``threshold`` is outside ``[0, 1]``, the HMM
            settings are invalid (``n_states < 2`` etc.), or a refit fails
            (the message then names the refit day ``t``).
    """
    r = np.asarray(index_returns, dtype=float)
    if r.ndim != 1:
        raise ValueError("index_returns must be 1-D")
    if not np.all(np.isfinite(r)):
        raise ValueError("index_returns contain NaN or inf")
    if mode not in ("prob", "binary"):
        raise ValueError(f"gate mode must be 'prob' or 'binary', got {mode!r}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"gate threshold must be in [0, 1], got {threshold}")
    # Validates n_states >= 2, tol/var_floor > 0, max_iter >= 1 up front.
    GaussianHMM(n_states, tol=tol, max_iter=max_iter, var_floor=var_floor)
    T = r.shape[0]
    if train_min_days <= n_states or refit_days < 1:
        raise ValueError("train_min_days must exceed n_states and refit_days must be >= 1")
    if T < train_min_days:
        raise ValueError(f"series shorter than train_min_days: T={T} < {train_min_days}")

    t0 = train_min_days - 1
    gate = np.ones(T)
    models: List[GaussianHMM] = []
    model: Optional[GaussianHMM] = None
    alpha: Optional[np.ndarray] = None
    calm_state = 0
    for t in range(t0, T):
        if (t - t0) % refit_days == 0:
            model = GaussianHMM(n_states, tol=tol, max_iter=max_iter, var_floor=var_floor)
            warm = models[-1].params.copy() if models else None
            try:
                model.fit(r[: t + 1], init=warm)
                alpha = model.filtered_probabilities(r[: t + 1])[-1]
            except ValueError as exc:
                raise ValueError(f"regime_gate refit at t={t}: {exc}") from exc
            models.append(model)
            calm_state = int(np.argmin(model.params.variances[:, 0]))
        else:
            assert model is not None and model.params is not None and alpha is not None
            try:
                alpha = forward_step(model.params, alpha, r[t])
            except ValueError as exc:
                raise ValueError(f"regime_gate forward step at t={t}: {exc}") from exc
        p_calm = float(alpha[calm_state])
        gate[t] = p_calm if mode == "prob" else (1.0 if p_calm >= threshold else 0.0)
    return gate, models


def apply_gate(positions: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Scale each day's positions by that day's gate value.

    ``P_filtered[t] = gate[t] * P[t]``; both are day-t decisions, so the
    filtered position still only trades against the day t+1 return.
    """
    p = _validate_returns(positions, "positions")
    g = np.asarray(gate, dtype=float)
    if g.ndim != 1 or g.shape[0] != p.shape[0]:
        raise ValueError(f"gate length {g.shape} does not match positions {p.shape}")
    if not np.all(np.isfinite(g)):
        raise ValueError("gate contains NaN or inf")
    return p * g[:, None]
