"""Spread and square-root impact model, including its tie to the backtest."""

import numpy as np
import pytest

from regime import (
    CostAssumptions,
    aum_at_sharpe_fraction,
    daily_cost_fraction,
    participation,
    run_backtest,
    trade_cost,
)


def assumptions(y=1.0):
    return CostAssumptions(
        half_spread_bps=[2.0, 1.0], adv_usd=[1e9, 4e9], sigma_daily=[0.01, 0.02], impact_y=y
    )


def test_trade_cost_is_sign_blind_and_grows_as_power_three_halves():
    a = assumptions()
    dw = np.array([0.1, -0.2])
    assert np.array_equal(trade_cost(dw, 1e8, a), trade_cost(-dw, 1e8, a))
    no_spread = CostAssumptions([0.0, 0.0], a.adv_usd, a.sigma_daily)
    ratio = trade_cost(dw, 4e8, no_spread) / trade_cost(dw, 1e8, no_spread)
    assert ratio == pytest.approx([8.0, 8.0])


def test_zero_impact_at_a_flat_spread_matches_the_backtest():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(30, 2))
    r = rng.normal(scale=0.01, size=(30, 2))
    flat = CostAssumptions([5.0, 5.0], [1e9, 1e9], [0.01, 0.01], impact_y=0.0)
    bt = run_backtest(w, r, 5.0)
    assert np.allclose(bt.gross_returns - daily_cost_fraction(w, 1e7, flat), bt.net_returns)


def test_participation_counts_the_entry_from_flat():
    w = np.array([[0.1, 0.0], [0.1, 0.2]])
    p = participation(w, 1e9, assumptions())
    assert p.ravel() == pytest.approx([0.1, 0.0, 0.0, 0.05])


@pytest.mark.parametrize("kwargs", [
    {"half_spread_bps": [-1.0, 1.0]},
    {"adv_usd": [0.0, 1.0]},
    {"sigma_daily": [np.nan, 0.1]},
    {"sigma_daily": [0.1]},
    {"impact_y": -1.0},
])
def test_assumptions_reject_bad_input(kwargs):
    base = {"half_spread_bps": [1.0, 1.0], "adv_usd": [1.0, 1.0], "sigma_daily": [0.1, 0.1], "impact_y": 1.0}
    with pytest.raises(ValueError):
        CostAssumptions(**{**base, **kwargs})


def test_functions_reject_bad_input():
    a = assumptions()
    with pytest.raises(ValueError):
        trade_cost(np.zeros(2), 0.0, a)
    with pytest.raises(ValueError):
        trade_cost(np.zeros(3), 1e6, a)
    with pytest.raises(ValueError):
        daily_cost_fraction(np.zeros((3, 3)), 1e6, a)
    with pytest.raises(ValueError):
        daily_cost_fraction(np.array([[np.inf, 0.0]]), 1e6, a)


def test_halving_aum_finds_the_crossing():
    # Sharpe 1 - aum/1e9 falls to half of about 1 near 5e8.
    found = aum_at_sharpe_fraction(lambda x: 1.0 - x / 1e9, 1.0, 1e10)
    assert found == pytest.approx(5e8, rel=1e-3)


def test_halving_aum_has_no_answer_in_two_cases():
    assert aum_at_sharpe_fraction(lambda x: -1.0, 1.0, 1e3) is None
    assert aum_at_sharpe_fraction(lambda x: 1.0, 1.0, 1e3) is None
    with pytest.raises(ValueError):
        aum_at_sharpe_fraction(lambda x: 1.0, 5.0, 1.0)
