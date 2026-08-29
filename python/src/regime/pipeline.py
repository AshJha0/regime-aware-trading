"""End-to-end pipeline on the bundled dataset.

Loads the CSVs + config, fits full-sample HMMs (K=2, K=3) on the market
index, builds the walk-forward regime gate, runs momentum / carry /
combined backtests filtered and unfiltered, and computes the crisis-state
breakdown.  The demo, the golden-value generator, and the golden tests all
call :func:`run_full_pipeline` so they can never drift apart.

Label conventions (pinned): HMM states are sorted by mean ascending, so on
this data state 0 = crisis (lowest mean, highest vol), state K-1 = bull.
The bundled true states use the generator's ordering (0 = calm-bull,
1 = choppy, 2 = crisis); the mapping ``hmm_label = (K-1) - true_state``
aligns them for the Viterbi-accuracy teaching number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .backtest import BacktestResult, compute_metrics, run_backtest, state_conditional_returns
from .hmm import GaussianHMM
from .strategies import apply_gate, carry_positions, carry_total_returns, momentum_positions, regime_gate


@dataclass
class Dataset:
    """Bundled market data.

    Attributes:
        dates: Trading dates, length T.
        index_returns: Market index daily returns, shape ``(T,)``.
        trend_returns: Trend-asset daily returns, shape ``(T, 6)``.
        fx_spot_returns: Currency spot returns, shape ``(T, 6)``.
        fx_rate_diffs: Annualized rate differentials, shape ``(T, 6)``.
        true_states: Generator regime path (teaching only), shape ``(T,)``.
        config: Pinned strategy/backtest configuration.
    """

    dates: pd.DatetimeIndex
    index_returns: np.ndarray
    trend_returns: np.ndarray
    fx_spot_returns: np.ndarray
    fx_rate_diffs: np.ndarray
    true_states: np.ndarray
    config: Dict


def load_dataset(data_dir: str | Path) -> Dataset:
    """Load the bundled CSVs and config from ``data_dir``.

    Raises:
        ValueError: If files are missing or the frames disagree on dates.
    """
    d = Path(data_dir)
    for name in ("market_index.csv", "trend_assets.csv", "fx_carry.csv", "true_states.csv", "config.json"):
        if not (d / name).exists():
            raise ValueError(f"missing bundled data file: {d / name}")
    idx = pd.read_csv(d / "market_index.csv", parse_dates=["date"])
    trend = pd.read_csv(d / "trend_assets.csv", parse_dates=["date"])
    fx = pd.read_csv(d / "fx_carry.csv", parse_dates=["date"])
    true_states = pd.read_csv(d / "true_states.csv", parse_dates=["date"])
    with open(d / "config.json") as f:
        config = json.load(f)
    T = len(idx)
    if not (len(trend) == len(fx) == len(true_states) == T):
        raise ValueError("bundled CSVs have inconsistent lengths")
    spot_cols = [c for c in fx.columns if c.startswith("spot_ret_")]
    diff_cols = [c for c in fx.columns if c.startswith("rate_diff_")]
    trend_cols = [c for c in trend.columns if c.startswith("asset_")]
    return Dataset(
        dates=pd.DatetimeIndex(idx["date"]),
        index_returns=idx["ret"].to_numpy(),
        trend_returns=trend[trend_cols].to_numpy(),
        fx_spot_returns=fx[spot_cols].to_numpy(),
        fx_rate_diffs=fx[diff_cols].to_numpy(),
        true_states=true_states["state"].to_numpy(),
        config=config,
    )


def run_full_pipeline(data_dir: str | Path, refit_days: int | None = None, train_min_days: int | None = None) -> Dict:
    """Run the entire study on the bundled data.

    Args:
        data_dir: Directory holding the bundled CSVs + config.json.
        refit_days / train_min_days: Optional overrides of the pinned
            config (tests may shorten the walk-forward loop; golden values
            always use the config as bundled).

    Returns:
        Dict with fitted models, gate, backtest results (momentum / carry /
        combined, unfiltered and filtered), crisis breakdown, and the
        Viterbi-vs-true accuracy.
    """
    ds = load_dataset(data_dir)
    cfg = dict(ds.config)
    if refit_days is not None:
        cfg["refit_days"] = refit_days
    if train_min_days is not None:
        cfg["train_min_days"] = train_min_days
    K = cfg["n_states"]
    r_idx = ds.index_returns

    # Full-sample HMM fits (model selection + crisis attribution).
    hmm2 = GaussianHMM(2, tol=cfg["em_tol"], max_iter=cfg["em_max_iter"], var_floor=cfg["var_floor"])
    fit2 = hmm2.fit(r_idx)
    hmm3 = GaussianHMM(3, tol=cfg["em_tol"], max_iter=cfg["em_max_iter"], var_floor=cfg["var_floor"])
    fit3 = hmm3.fit(r_idx)
    viterbi3 = hmm3.viterbi(r_idx)
    smoothed3 = hmm3.smoothed_probabilities(r_idx)
    stationary3 = hmm3.stationary_distribution()

    # Teaching comparison: generator labels are 0=calm, 1=choppy, 2=crisis.
    # Pinned mapping to fitted labels goes through the state volatilities
    # (robust even when fitted calm/choppy means order differently than
    # the true means): calm -> argmin std, crisis -> argmax std, choppy ->
    # the remaining label.
    calm_label = int(np.argmin(hmm3.params.variances[:, 0]))
    crisis_label = int(np.argmax(hmm3.params.variances[:, 0]))
    choppy_label = int(3 - calm_label - crisis_label)
    mapped_true = np.array([calm_label, choppy_label, crisis_label])[ds.true_states]
    viterbi_accuracy = float(np.mean(viterbi3 == mapped_true))

    # Walk-forward regime gate on the index (no lookahead).
    gate, gate_models = regime_gate(
        r_idx,
        n_states=K,
        train_min_days=cfg["train_min_days"],
        refit_days=cfg["refit_days"],
        mode=cfg["gate_mode"],
        threshold=cfg["gate_threshold"],
        tol=cfg["em_tol"],
        max_iter=cfg["em_max_iter"],
        var_floor=cfg["var_floor"],
    )

    cost = cfg["cost_bps"]
    dates = ds.dates

    mom_pos = momentum_positions(
        ds.trend_returns,
        lookback=cfg["momentum_lookback"],
        vol_target=cfg["vol_target"],
        ewma_lambda=cfg["ewma_lambda"],
        leverage_cap=cfg["leverage_cap"],
    )
    carry_pos = carry_positions(
        ds.fx_rate_diffs,
        top_n=cfg["carry_top_n"],
        bottom_n=cfg["carry_bottom_n"],
        rebalance_days=cfg["rebalance_days"],
    )
    carry_rets = carry_total_returns(ds.fx_spot_returns, ds.fx_rate_diffs)

    results: Dict[str, BacktestResult] = {}
    results["momentum_unfiltered"] = run_backtest(mom_pos, ds.trend_returns, cost, dates)
    results["momentum_filtered"] = run_backtest(apply_gate(mom_pos, gate), ds.trend_returns, cost, dates)
    results["carry_unfiltered"] = run_backtest(carry_pos, carry_rets, cost, dates)
    results["carry_filtered"] = run_backtest(apply_gate(carry_pos, gate), carry_rets, cost, dates)

    # Combined portfolio: 50/50 in each strategy's net return stream.
    combined: Dict[str, Dict] = {}
    for tag in ("unfiltered", "filtered"):
        net = 0.5 * results[f"momentum_{tag}"].net_returns + 0.5 * results[f"carry_{tag}"].net_returns
        combined[tag] = {"net_returns": net, "metrics": compute_metrics(net, dates)}

    # Crisis table: attribute daily strategy returns to the full-sample
    # K=3 Viterbi state (state 0 = crisis, K-1 = bull).
    crisis: Dict[str, Dict] = {}
    for name, res in results.items():
        crisis[name] = state_conditional_returns(res.net_returns, viterbi3, K)
    for tag in ("unfiltered", "filtered"):
        crisis[f"combined_{tag}"] = state_conditional_returns(combined[tag]["net_returns"], viterbi3, K)

    return {
        "dataset": ds,
        "config": cfg,
        "hmm2": hmm2,
        "fit2": fit2,
        "hmm3": hmm3,
        "fit3": fit3,
        "viterbi3": viterbi3,
        "smoothed3": smoothed3,
        "stationary3": stationary3,
        "viterbi_accuracy": viterbi_accuracy,
        "gate": gate,
        "gate_models": gate_models,
        "results": results,
        "combined": combined,
        "crisis": crisis,
    }


def golden_cases(pipe: Dict) -> List[Dict]:
    """Build the golden-case list from a full-pipeline run.

    Flat scalar keys only (cross-language parse friendliness).  Case names
    and tolerances are pinned in API_SPEC.md.
    """
    hmm3 = pipe["hmm3"]
    p3 = hmm3.params
    cases: List[Dict] = []

    def case(name: str, expect: Dict[str, float], tol: float) -> None:
        cases.append({"name": name, "inputs": {}, "expect": {k: float(v) for k, v in expect.items()}, "tol": tol})

    case("hmm_k2_loglik", {"loglik": pipe["fit2"].log_likelihood}, 1e-6)
    case("hmm_k3_loglik", {"loglik": pipe["fit3"].log_likelihood}, 1e-6)
    case("hmm_k3_means", {f"mean{k}": p3.means[k, 0] for k in range(3)}, 1e-5)
    case("hmm_k3_stds", {f"std{k}": np.sqrt(p3.variances[k, 0]) for k in range(3)}, 1e-5)
    case(
        "hmm_k3_transmat",
        {f"a{i}{j}": p3.transmat[i, j] for i in range(3) for j in range(3)},
        1e-5,
    )
    counts = np.bincount(pipe["viterbi3"], minlength=3)
    case("hmm_k3_viterbi_counts", {f"count{k}": float(counts[k]) for k in range(3)}, 0.0)
    case("hmm_k3_stationary", {f"pi{k}": pipe["stationary3"][k] for k in range(3)}, 1e-8)
    for t in (500, 1500):
        case(
            f"hmm_k3_smoothed_t{t}",
            {f"p{k}": pipe["smoothed3"][t, k] for k in range(3)},
            1e-8,
        )
    case("viterbi_accuracy_vs_true", {"accuracy": pipe["viterbi_accuracy"]}, 1e-8)

    for name in ("momentum_unfiltered", "momentum_filtered", "carry_unfiltered", "carry_filtered"):
        m = pipe["results"][name].metrics
        case(
            name,
            {"ann_return": m["ann_return"], "sharpe": m["sharpe"], "max_dd": m["max_dd"]},
            1e-8,
        )
    case(
        "combined_sharpe",
        {
            "sharpe_unfiltered": pipe["combined"]["unfiltered"]["metrics"]["sharpe"],
            "sharpe_filtered": pipe["combined"]["filtered"]["metrics"]["sharpe"],
        },
        1e-8,
    )
    case(
        "crisis_momentum_return",
        {"ann_return": pipe["crisis"]["momentum_unfiltered"][0]["ann_return"]},
        1e-8,
    )
    case(
        "turnover_momentum",
        {"mean_turnover": float(np.mean(pipe["results"]["momentum_unfiltered"].turnover))},
        1e-8,
    )
    case(
        "turnover_carry",
        {"mean_turnover": float(np.mean(pipe["results"]["carry_unfiltered"].turnover))},
        1e-8,
    )
    return cases
