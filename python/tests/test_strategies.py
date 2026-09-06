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


# --------------------------------------------------------------------- #
# Robustness: corrupt returns, re-ranking, gate edges
# --------------------------------------------------------------------- #

def test_momentum_rejects_returns_below_minus_one():
    """PT-4: a -100% (or worse) print is corrupt data, not a signal."""
    for bad in (-1.0, -1.5):
        r = np.full((300, 1), 0.001)
        r[10, 0] = bad
        with pytest.raises(ValueError, match="<= -100%"):
            momentum_positions(r, lookback=252)
        with pytest.raises(ValueError, match="<= -100%"):
            ewma_variance(r, 0.94, 252)
        with pytest.raises(ValueError, match="<= -100%"):
            carry_total_returns(r, np.zeros_like(r))
    # -99.9% is legal (extreme, but a valid simple return)
    r = np.full((300, 1), 0.001)
    r[10, 0] = -0.999
    assert np.all(np.isfinite(momentum_positions(r, lookback=252)))
    # rate differentials are not simple returns: -1.5 is accepted there
    d = np.full((10, 4), -1.5)
    assert carry_positions(d, top_n=1, bottom_n=1).shape == (10, 4)


def _rerank_panel() -> np.ndarray:
    d = np.zeros((45, 4))
    d[:21] = [0.04, 0.03, 0.02, 0.01]
    d[21:] = [0.01, 0.02, 0.03, 0.04]
    return d


def test_carry_rerank_after_crossing():
    """PT-6: the book flips exactly at the first rebalance after the ranks
    cross, with turnover 4 (two full round-trips) on that day only."""
    d = _rerank_panel()
    pos = carry_positions(d, top_n=1, bottom_n=1, rebalance_days=21)
    assert pos[20].tolist() == [1.0, 0.0, 0.0, -1.0]
    for t in range(21, 42):
        assert pos[t].tolist() == [-1.0, 0.0, 0.0, 1.0]
    turnover = np.abs(np.diff(pos, axis=0)).sum(axis=1)  # turnover[t-1] is the day-t trade
    assert turnover[20] == 4.0  # trade into P[21]
    assert np.all(turnover[:20] == 0.0) and np.all(turnover[21:] == 0.0)
    # tie at the crossing day -> ascending asset index decides
    tie = d.copy()
    tie[21:] = 0.02
    pos_tie = carry_positions(tie, top_n=1, bottom_n=1, rebalance_days=21)
    assert pos_tie[21].tolist() == [1.0, 0.0, 0.0, -1.0]
    # golden-embedded panel helper reproduces the pinned re-ranking scalars
    from regime import CARRY_RERANK_INPUTS, carry_rerank_panel, carry_rerank_values

    panel = carry_rerank_panel(CARRY_RERANK_INPUTS)
    assert panel.shape == (63, 4)
    assert panel[9].tolist() == [0.04, 0.03, 0.02, 0.01]
    assert panel[10].tolist() == [0.01, 0.02, 0.03, 0.04]
    vals = carry_rerank_values(CARRY_RERANK_INPUTS)
    assert vals["pos_day15_asset0"] == 1.0  # diffs changed at t=10 but no rebalance until 21
    assert vals["pos_day21_asset0"] == -1.0 and vals["pos_day42_asset0"] == 1.0
    assert vals["turnover_day21"] == 4.0 and vals["turnover_day42"] == 4.0
    assert vals["mean_turnover"] == 10.0 / 63.0
    # hand-derived: gross = 20 * 0.03/252 (10 good + 11 bad + 21 good accrual
    # days), costs = 10 * 5e-4; ann_return = 252 * net_total / 63
    assert vals["ann_return"] == pytest.approx(252.0 * (20 * 0.03 / 252.0 - 0.005) / 63.0, abs=1e-15)


def test_binary_gate_threshold_edges(index_returns):
    """PT-12: the '>=' convention at both ends of [0, 1]; outside is an error."""
    r = index_returns[:600]
    prob, _ = regime_gate(r, 3, 400, 300, mode="prob")
    lo, _ = regime_gate(r, 3, 400, 300, mode="binary", threshold=0.0)
    assert np.all(lo[399:] == 1.0)
    hi, _ = regime_gate(r, 3, 400, 300, mode="binary", threshold=1.0)
    assert np.array_equal(hi[399:], np.where(prob[399:] == 1.0, 1.0, 0.0))
    for bad in (1.5, -0.1, np.nan):
        with pytest.raises(ValueError, match="threshold"):
            regime_gate(r, 3, 400, 300, mode="binary", threshold=bad)


def test_gate_edge_windows(index_returns):
    """PT-13: degenerate but legal walk-forward geometries."""
    r = index_returns[:400]
    gate, models = regime_gate(r, 3, train_min_days=400, refit_days=63)
    assert gate.shape == (400,) and len(models) == 1
    assert np.all(gate[:399] == 1.0) and 0.0 <= gate[399] <= 1.0
    gate2, models2 = regime_gate(index_returns[:500], 3, train_min_days=400, refit_days=1000)
    assert len(models2) == 1 and np.all((gate2 >= 0.0) & (gate2 <= 1.0))
    gate3, models3 = regime_gate(index_returns[:4], 3, train_min_days=4, refit_days=1)
    assert len(models3) == 1 and np.all(np.isfinite(gate3))


def test_regime_gate_parameter_validation(index_returns):
    """MAJ-7/MIN-11: schedule and HMM settings are validated before any fit."""
    r = index_returns[:600]
    with pytest.raises(ValueError, match="train_min_days"):
        regime_gate(r, n_states=3, train_min_days=3)
    with pytest.raises(ValueError, match="refit_days"):
        regime_gate(r, train_min_days=400, refit_days=0)
    with pytest.raises(ValueError, match="n_states"):
        regime_gate(r, n_states=1, train_min_days=400)
    with pytest.raises(ValueError):
        regime_gate(r, train_min_days=400, tol=0.0)
    with pytest.raises(ValueError):
        regime_gate(np.column_stack([r, r]), train_min_days=400)  # not 1-D
    with pytest.raises(ValueError, match="NaN"):
        regime_gate(np.append(r, np.nan), train_min_days=400)


def test_regime_gate_reports_refit_day_on_failure(index_returns, monkeypatch):
    """Practitioner gap: a failing refit names the day index."""
    from regime import GaussianHMM

    r = index_returns[:600]
    real_fit = GaussianHMM.fit

    def boom(self, X, init=None):
        if len(X) == 600:
            raise ValueError("synthetic failure")
        return real_fit(self, X, init)

    monkeypatch.setattr(GaussianHMM, "fit", boom)
    with pytest.raises(ValueError, match=r"refit at t=599: synthetic failure"):
        regime_gate(r, train_min_days=400, refit_days=200)


def test_apply_gate_rejects_nan_gate():
    pos = np.ones((5, 2))
    with pytest.raises(ValueError, match="NaN"):
        apply_gate(pos, np.array([1.0, np.nan, 0.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        apply_gate(pos, np.ones((5, 1)))  # not 1-D
