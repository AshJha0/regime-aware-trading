"""Backtest engine tests: pinned accounting, no-lookahead, metrics."""

import numpy as np
import pandas as pd
import pytest

from regime import compute_metrics, max_drawdown, run_backtest, state_conditional_returns


def test_accounting_identity():
    """net = gross - cost*turnover and equity = cumprod(1+net), exactly."""
    P = np.array([[1.0], [0.5], [-0.5], [-0.5]])
    r = np.array([[0.01], [0.02], [-0.01], [0.03]])
    res = run_backtest(P, r, cost_bps=10.0)
    cost = 10.0 / 1e4
    expected_gross = np.array([0.0, 1.0 * 0.02, 0.5 * -0.01, -0.5 * 0.03])
    expected_to = np.array([1.0, 0.5, 1.0, 0.0])
    assert np.allclose(res.gross_returns, expected_gross, atol=1e-15)
    assert np.allclose(res.turnover, expected_to, atol=1e-15)
    assert np.allclose(res.net_returns, expected_gross - cost * expected_to, atol=1e-15)
    assert np.allclose(res.equity, np.cumprod(1.0 + res.net_returns), atol=1e-15)


def test_no_lookahead_shift():
    """Day-t P&L must come from the day t-1 position; using the same-day
    position (lookahead) gives a measurably different result."""
    rng = np.random.default_rng(3)  # test data only
    r = rng.normal(0.0, 0.01, size=(300, 1))
    P = np.sign(r)  # 'perfect foresight' positions decided at t
    res = run_backtest(P, r, cost_bps=0.0)
    lookahead_pnl = float(np.sum(P * r))  # applying P[t] to r[t]
    honest_pnl = float(np.sum(res.gross_returns))
    assert abs(lookahead_pnl - honest_pnl) > 0.1  # foresight is very profitable
    # honest accounting equals the explicitly shifted computation:
    shifted = float(np.sum(P[:-1] * r[1:]))
    assert honest_pnl == pytest.approx(shifted, abs=1e-15)


def test_zero_cost_path():
    P = np.array([[1.0], [1.0], [-1.0]])
    r = np.array([[0.01], [0.02], [0.03]])
    res = run_backtest(P, r, cost_bps=0.0)
    assert np.array_equal(res.net_returns, res.gross_returns)


def test_turnover_first_day_is_position_build():
    P = np.array([[0.5, -0.5], [0.5, -0.5]])
    r = np.zeros((2, 2))
    res = run_backtest(P, r)
    assert res.turnover[0] == pytest.approx(1.0)
    assert res.turnover[1] == pytest.approx(0.0)


def test_max_drawdown_known_case():
    equity = np.array([1.0, 1.2, 0.6, 0.9, 1.3])
    assert max_drawdown(equity) == pytest.approx(0.6 / 1.2 - 1.0)  # -50%
    assert max_drawdown(np.array([1.0, 1.1, 1.2])) == pytest.approx(0.0)


def test_metrics_formulas():
    net = np.array([0.01, -0.005, 0.02, 0.0, -0.01])
    m = compute_metrics(net)
    assert m["ann_return"] == pytest.approx(252 * net.mean())
    assert m["ann_vol"] == pytest.approx(np.sqrt(252) * net.std(ddof=0))
    assert m["sharpe"] == pytest.approx(m["ann_return"] / m["ann_vol"])
    assert m["hit_rate"] == pytest.approx(2.0 / 5.0)
    assert m["max_dd"] <= 0.0
    # zero-vol series: sharpe defined as 0, no division error
    z = compute_metrics(np.zeros(10))
    assert z["sharpe"] == 0.0 and z["calmar"] == 0.0


def test_worst_month_with_dates():
    dates = pd.bdate_range("2020-01-01", periods=63)
    net = np.zeros(63)
    net[21:42] = -0.01  # one bad calendar stretch
    m = compute_metrics(net, dates)
    monthly = (1 + pd.Series(net, index=dates)).groupby([dates.year, dates.month]).prod() - 1
    assert m["worst_month"] == pytest.approx(float(monthly.min()))
    assert m["worst_month"] < -0.05


def test_state_conditional_returns_partition():
    net = np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.02])
    states = np.array([0, 0, 1, 1, 2, 2])
    out = state_conditional_returns(net, states, 3)
    assert sum(out[k]["n_days"] for k in range(3)) == 6
    assert out[0]["ann_return"] == pytest.approx(252 * np.mean([0.01, -0.02]))
    empty = state_conditional_returns(net, np.zeros(6, dtype=int), 2)
    assert empty[1]["n_days"] == 0 and empty[1]["sharpe"] == 0.0


def test_backtest_validation():
    with pytest.raises(ValueError):
        run_backtest(np.ones((3, 1)), np.ones((4, 1)))  # shape mismatch
    with pytest.raises(ValueError):
        run_backtest(np.ones((3, 1)), np.ones((3, 1)), cost_bps=-1.0)
    with pytest.raises(ValueError):
        run_backtest(np.array([[np.nan]]), np.array([[0.0]]))
    with pytest.raises(ValueError):
        run_backtest(np.ones((0, 1)), np.ones((0, 1)))  # empty
    with pytest.raises(ValueError):
        compute_metrics(np.array([]))


def test_property_costs_never_help():
    """Grid property: for any cost level, higher costs never raise the
    total net return (turnover is non-negative)."""
    rng = np.random.default_rng(5)  # test data only
    P = rng.normal(size=(100, 3))
    r = rng.normal(0.0, 0.01, size=(100, 3))
    totals = []
    for bps in [0.0, 1.0, 5.0, 10.0, 25.0, 50.0]:
        res = run_backtest(P, r, cost_bps=bps)
        totals.append(res.net_returns.sum())
    assert all(a >= b - 1e-12 for a, b in zip(totals, totals[1:]))
