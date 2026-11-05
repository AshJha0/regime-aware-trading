"""Known-answer tests for regime.evaluation."""

import numpy as np
import pytest

from regime import (
    bic_table,
    break_even_cost,
    build_book,
    classification_report,
    constant_gate,
    crisis_episodes,
    ewma_ratio_gate,
    gate_break_even,
    lag_gate,
    oracle_gate,
    paired_block_bootstrap,
    roc_auc,
    sharpe,
    stationary_bootstrap_indices,
    vol_ratio_gate,
    window_metrics,
)


@pytest.fixture(scope="module")
def noise() -> np.ndarray:
    return 0.01 * np.random.default_rng(7).standard_normal(600) + 0.0005


def test_bootstrap_of_identical_series_is_zero(noise):
    res = paired_block_bootstrap(noise, noise, np.random.default_rng(1), n_boot=200)
    assert res.diff == 0.0
    assert res.ci_low == 0.0 and res.ci_high == 0.0 and res.std_error == 0.0


def test_bootstrap_is_reproducible_and_uses_the_supplied_rng(noise):
    other = noise[::-1].copy()
    a = paired_block_bootstrap(noise, other, np.random.default_rng(5), n_boot=200)
    b = paired_block_bootstrap(noise, other, np.random.default_rng(5), n_boot=200)
    c = paired_block_bootstrap(noise, other, np.random.default_rng(6), n_boot=200)
    assert a == b
    assert a.ci_low != c.ci_low


def test_bootstrap_block_length_matches_mean_block():
    idx = stationary_bootstrap_indices(5000, 20, 21.0, np.random.default_rng(2))
    jumps = np.mean(np.diff(idx, axis=1) % 5000 != 1)
    assert jumps == pytest.approx(1.0 / 21.0, rel=0.1)


def test_constant_gate_leaves_sharpe_unchanged_at_zero_cost(pipeline_full):
    book = build_book(pipeline_full["dataset"])
    n, start = pipeline_full["gate"].size, 504
    base = book.stream(np.ones(n)).net(0.0)[start:]
    scaled = book.stream(constant_gate(n, 0.6, start - 1)).net(0.0)[start:]
    assert sharpe(scaled) == pytest.approx(sharpe(base), abs=1e-12)
    assert scaled.mean() == pytest.approx(0.6 * base.mean(), rel=1e-12)


def test_book_matches_pipeline_combined_net(pipeline_full):
    book = build_book(pipeline_full["dataset"])
    cost = pipeline_full["config"]["cost_bps"]
    net = book.stream(pipeline_full["gate"]).net(cost)
    np.testing.assert_allclose(net, pipeline_full["combined"]["filtered"]["net_returns"], atol=1e-15)


def test_window_metrics_skips_the_warm_up(pipeline_full):
    net = pipeline_full["combined"]["unfiltered"]["net_returns"]
    dates = pipeline_full["dataset"].dates
    assert window_metrics(net, dates, 504)["sharpe"] == pytest.approx(float(sharpe(net[504:])))
    with pytest.raises(ValueError):
        window_metrics(net, dates, net.size)


def test_auc_known_answers():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert np.isnan(roc_auc([1, 1], [0.1, 0.2]))


@pytest.mark.parametrize(
    "build",
    [
        lambda r, f: vol_ratio_gate(r, f),
        lambda r, f: ewma_ratio_gate(r, f),
    ],
    ids=["vol", "ewma"],
)
def test_baselines_are_causal(noise, build):
    first_day, cut = 100, 300
    before = build(noise, first_day)
    changed = noise.copy()
    changed[cut + 1:] *= -7.0
    after = build(changed, first_day)
    np.testing.assert_array_equal(before[: cut + 1], after[: cut + 1])
    assert not np.array_equal(before[cut + 1:], after[cut + 1:])
    assert np.all(before[:first_day] == 1.0)
    assert np.all((before > 0.0) & (before <= 1.0))


def test_oracle_reads_exactly_one_day_ahead():
    states = np.array([0, 0, 0, 2, 2, 0, 1, 2])
    gate = oracle_gate(states, first_day=1)
    np.testing.assert_array_equal(gate, [1, 1, 0, 0, 1, 1, 0, 1])


def test_lag_gate_shifts_forward():
    np.testing.assert_array_equal(lag_gate(np.array([1.0, 0.5, 0.2, 0.1]), 2), [1.0, 1.0, 1.0, 0.5])
    with pytest.raises(ValueError):
        lag_gate(np.ones(3), -1)


def test_break_even_brackets_a_known_root():
    res = break_even_cost(lambda c: 2.6 - c, 0.0, 100.0, tol=1e-6)
    assert res.status == "crossed"
    assert res.low <= 2.6 <= res.high
    assert res.high - res.low <= 1e-6
    assert res.cost_bps == pytest.approx(2.6, abs=1e-6)


def test_break_even_reports_a_missing_bracket():
    assert break_even_cost(lambda c: -1.0 - c).status == "below_at_zero"
    assert break_even_cost(lambda c: 1.0 + c).status == "above_at_max"
    assert np.isnan(break_even_cost(lambda c: 1.0).cost_bps)


def test_gate_break_even_edge_changes_sign_there(pipeline_full):
    book = build_book(pipeline_full["dataset"])
    gated = book.stream(pipeline_full["gate"])
    base = book.stream(np.ones(pipeline_full["gate"].size))
    res = gate_break_even(gated, base, 504)
    assert res.status == "crossed"
    assert sharpe(gated.net(res.low)[504:]) > sharpe(base.net(res.low)[504:])
    assert sharpe(gated.net(res.high)[504:]) <= sharpe(base.net(res.high)[504:])


def test_crisis_episodes_skip_runs_begun_before_the_window():
    states = np.array([2, 2, 0, 2, 2, 2, 0, 2])
    assert crisis_episodes(states, start=1) == [(3, 6), (7, 8)]


def test_classification_known_answers():
    states = np.array([0, 0, 0, 1, 2, 2, 2, 0, 2, 2, 0, 0])
    gate = np.array([0.9, 0.9, 0.9, 0.6, 0.8, 0.4, 0.1, 0.9, 0.7, 0.6, 0.9, 0.9])
    rep = classification_report(gate, states, start=0)
    assert rep["n_episodes"] == 2 and rep["n_missed"] == 1
    assert rep["median_lag"] == 1.0
    assert rep["majority_accuracy"] == pytest.approx(0.5)
    # Wrong on days 3, 4, 8 and 9, which are not calm but score >= 0.5.
    assert rep["accuracy"] == pytest.approx(8 / 12)


def test_bic_includes_the_iid_null_and_prefers_it_on_iid_data():
    x = 0.01 * np.random.default_rng(3).standard_normal(800)
    rows = bic_table(x, states=(1, 2))
    var = x.var()
    expected = float(np.sum(-0.5 * np.log(2 * np.pi * var) - 0.5 * (x - x.mean()) ** 2 / var))
    assert rows[0]["k"] == 1 and rows[0]["n_params"] == 2
    assert rows[0]["loglik"] == pytest.approx(expected)
    assert rows[0]["bic"] < rows[1]["bic"]
