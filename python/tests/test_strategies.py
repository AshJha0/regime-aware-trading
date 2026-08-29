"""Strategy tests: momentum sizing, carry schedule/tie-break, regime gate."""

import numpy as np
import pytest

from regime import (
    apply_gate,
    carry_positions,
    carry_total_returns,
    ewma_variance,
    momentum_positions,
    regime_gate,
)


def test_momentum_rejects_short_series():
    r = np.full((252, 2), 0.001)  # exactly lookback long -> too short
    with pytest.raises(ValueError, match="shorter than momentum lookback"):
        momentum_positions(r, lookback=252)


def test_momentum_sign_correctness():
    """Steady up-trend asset goes long, down-trend short."""
    T = 300
    r = np.column_stack([np.full(T, 0.001), np.full(T, -0.001)])
    pos = momentum_positions(r, lookback=252)
    assert np.all(pos[251:, 0] > 0.0)
    assert np.all(pos[251:, 1] < 0.0)
    assert np.all(pos[:251] == 0.0)  # undefined before first full window


def test_momentum_vol_targeting_caps_leverage():
    """A near-zero-vol series would want huge leverage; the cap binds."""
    T = 300
    r = np.full((T, 1), 1e-6)  # ~0 vol, positive trend
    pos = momentum_positions(r, lookback=252, vol_target=0.10, leverage_cap=4.0)
    assert np.all(pos[251:, 0] == pytest.approx(4.0))  # cap / 1 asset
    # and with meaningful vol the target is hit instead of the cap:
    rng_free = 0.02 * np.sin(np.arange(T))  # deterministic, sd ~1.4%
    pos2 = momentum_positions(rng_free.reshape(-1, 1), lookback=252)
    assert np.all(np.abs(pos2[251:, 0]) < 4.0)


def test_momentum_scaling_matches_formula():
    """Position = sign * min(target/ann_vol, cap) / n_assets exactly."""
    T = 260
    r = np.column_stack([np.full(T, 0.002), np.full(T, 0.003)])
    lb = 252
    pos = momentum_positions(r, lookback=lb)
    sig2 = ewma_variance(r, 0.94, lb)
    t = 255
    for a in range(2):
        ann_vol = np.sqrt(252 * sig2[t, a])
        expect = min(0.10 / ann_vol, 4.0) / 2.0
        assert pos[t, a] == pytest.approx(expect, rel=1e-12)


def test_ewma_variance_recursion():
    r = np.array([0.01, -0.02, 0.005, 0.03]).reshape(-1, 1)
    sig2 = ewma_variance(r, lam=0.9, init_window=2)
    seed = np.mean(r[:2] ** 2)
    assert np.isnan(sig2[0, 0])
    assert sig2[1, 0] == pytest.approx(seed)
    assert sig2[2, 0] == pytest.approx(0.9 * seed + 0.1 * 0.005**2)
    assert sig2[3, 0] == pytest.approx(0.9 * sig2[2, 0] + 0.1 * 0.03**2)


def test_carry_tie_break_by_asset_index():
    """All-equal differentials: longs are assets 0..2, shorts 3..5 (pinned)."""
    d = np.ones((10, 6)) * 0.02
    pos = carry_positions(d, top_n=3, bottom_n=3, rebalance_days=21)
    assert np.allclose(pos[0, :3], 1.0 / 3.0)
    assert np.allclose(pos[0, 3:], -1.0 / 3.0)


def test_carry_ranking_and_weights():
    d = np.tile(np.array([0.01, 0.05, -0.02, 0.03, 0.00, 0.02]), (5, 1))
    pos = carry_positions(d)
    # descending diff order: 1, 3, 5, 0, 4, 2
    assert np.allclose(pos[0], [-1 / 3, 1 / 3, -1 / 3, 1 / 3, -1 / 3, 1 / 3])
    assert pos[0].sum() == pytest.approx(0.0)
    assert np.abs(pos[0]).sum() == pytest.approx(2.0)


def test_carry_rebalance_schedule():
    """Positions change only at t = 0, 21, 42, ... (pinned schedule)."""
    T = 70
    rng = np.random.default_rng(7)  # test data only; strategy code is RNG-free
    d = rng.normal(0.0, 0.02, size=(T, 6))
    pos = carry_positions(d, rebalance_days=21)
    changes = np.where(np.any(np.diff(pos, axis=0) != 0.0, axis=1))[0] + 1
    assert set(changes).issubset({21, 42, 63})
    assert np.array_equal(pos[1], pos[0])
    assert np.array_equal(pos[21], pos[41])


def test_carry_single_rebalance_when_window_exceeds_series():
    """rebalance_days > T: only the day-0 rebalance happens, held throughout."""
    d = np.tile(np.array([0.03, 0.02, 0.01, 0.0, -0.01, -0.02]), (10, 1))
    pos = carry_positions(d, rebalance_days=500)
    assert np.all(pos == pos[0])


def test_carry_validation():
    d = np.ones((10, 4))
    with pytest.raises(ValueError):
        carry_positions(d, top_n=3, bottom_n=3)  # 6 > 4 assets
    with pytest.raises(ValueError):
        carry_positions(d, rebalance_days=0)
    with pytest.raises(ValueError):
        carry_positions(np.ones((0, 4)))  # empty


def test_carry_total_returns_accrual():
    spot = np.array([[0.01, 0.0], [0.005, -0.01]])
    diff = np.array([[0.0252, 0.0504], [0.0252, 0.0504]])
    tot = carry_total_returns(spot, diff)
    assert np.allclose(tot[0], spot[0])  # day 0: no accrual
    assert tot[1, 0] == pytest.approx(0.005 + 0.0252 / 252)
    assert tot[1, 1] == pytest.approx(-0.01 + 0.0504 / 252)


def test_regime_gate_basics(index_returns):
    """Gate is 1 before the first fit, within [0,1] after, deterministic."""
    r = index_returns[:700]
    gate, models = regime_gate(r, train_min_days=400, refit_days=200)
    assert np.all(gate[:399] == 1.0)
    assert np.all((gate >= 0.0) & (gate <= 1.0))
    assert len(models) == 2  # fits at t=399 and t=599
    gate2, _ = regime_gate(r, train_min_days=400, refit_days=200)
    assert np.array_equal(gate, gate2)


def test_regime_gate_binary_mode(index_returns):
    r = index_returns[:600]
    gate, _ = regime_gate(r, train_min_days=400, refit_days=300, mode="binary", threshold=0.5)
    assert set(np.unique(gate)).issubset({0.0, 1.0})


def test_regime_gate_validation(index_returns):
    with pytest.raises(ValueError, match="shorter than train_min_days"):
        regime_gate(index_returns[:100], train_min_days=400)
    with pytest.raises(ValueError, match="mode"):
        regime_gate(index_returns[:600], train_min_days=400, mode="magic")


def test_apply_gate():
    pos = np.ones((5, 2))
    gate = np.array([1.0, 0.5, 0.0, 0.25, 1.0])
    out = apply_gate(pos, gate)
    assert np.allclose(out[:, 0], gate)
    with pytest.raises(ValueError):
        apply_gate(pos, gate[:3])
