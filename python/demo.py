"""End-to-end demo: regime detection + regime-aware momentum and carry.

Run:  cd python && PYTHONPATH=src python3 demo.py
Prints the fitted regime table, Viterbi-vs-true accuracy, the strategy
comparison (filtered vs unfiltered) and the crisis-state breakdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from regime import run_full_pipeline  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
LINE = "-" * 78


def main() -> int:
    print("Regime-Aware Trading — HMM + momentum + FX carry (bundled synthetic data)")
    print(LINE)
    pipe = run_full_pipeline(DATA_DIR)
    ds = pipe["dataset"]
    K = pipe["hmm3"].n_states  # the K=3 tables are keyed to the full-sample K=3 fit

    # ---------------- model selection ---------------- #
    r = ds.index_returns
    print(f"Data: {len(r)} trading days, {ds.dates[0].date()} .. {ds.dates[-1].date()}")
    print("\nModel selection on the market index (diagonal Gaussian HMM):")
    print(f"  {'K':>2} {'loglik':>12} {'AIC':>12} {'BIC':>12} {'iters':>6} {'conv':>5}")
    for k, key, fit in ((2, "hmm2", "fit2"), (3, "hmm3", "fit3")):
        model, res = pipe[key], pipe[fit]
        print(
            f"  {k:>2} {res.log_likelihood:>12.4f} {model.aic(r):>12.4f} "
            f"{model.bic(r):>12.4f} {res.n_iter:>6d} {str(res.converged):>5}"
        )

    # ---------------- regime table ---------------- #
    p = pipe["hmm3"].params
    pi = pipe["stationary3"]
    # Interpretation labels via the pinned volatility mapping: crisis =
    # highest-variance state, calm = lowest-variance, choppy = the rest.
    names = ["?"] * K
    names[int(np.argmax(p.variances[:, 0]))] = "crisis"
    names[int(np.argmin(p.variances[:, 0]))] = "calm-bull"
    names[names.index("?")] = "choppy"
    print("\nFitted K=3 regimes (states sorted by mean; 0 = lowest):")
    print(f"  {'state':>5} {'label':>10} {'mean(ann)':>10} {'vol(ann)':>9} {'stationary':>11}")
    for k in range(K):
        print(
            f"  {k:>5} {names[k]:>10} {252 * p.means[k, 0]:>+10.2%} "
            f"{np.sqrt(252 * p.variances[k, 0]):>9.2%} {pi[k]:>11.4f}"
        )
    print("  transition matrix (rows sum to 1):")
    for k in range(K):
        print("    " + "  ".join(f"{p.transmat[k, j]:.4f}" for j in range(K)))

    acc = pipe["viterbi_accuracy"]
    counts = np.bincount(pipe["viterbi3"], minlength=K)
    true_counts = np.bincount(ds.true_states, minlength=K)
    print(f"\nViterbi decoding vs true (generator) states: accuracy = {acc:.2%}")
    print(f"  decoded days per state {counts.tolist()}  (true calm/choppy/crisis {true_counts.tolist()})")

    # ---------------- strategy comparison ---------------- #
    cfg = pipe["config"]
    print(LINE)
    print(
        f"Strategies (cost {cfg['cost_bps']:.0f} bps, gate mode '{cfg['gate_mode']}', "
        f"refit every {cfg['refit_days']}d on expanding window):"
    )
    header = f"  {'strategy':<22} {'ann ret':>8} {'ann vol':>8} {'Sharpe':>7} {'maxDD':>8} {'Calmar':>7} {'hit':>6} {'worst mo':>9}"
    print(header)

    def row(label: str, m: dict) -> None:
        print(
            f"  {label:<22} {m['ann_return']:>+8.2%} {m['ann_vol']:>8.2%} "
            f"{m['sharpe']:>7.2f} {m['max_dd']:>+8.2%} {m['calmar']:>7.2f} "
            f"{m['hit_rate']:>6.1%} {m['worst_month']:>+9.2%}"
        )

    for name in ("momentum_unfiltered", "momentum_filtered", "carry_unfiltered", "carry_filtered"):
        row(name.replace("_", " "), pipe["results"][name].metrics)
    for tag in ("unfiltered", "filtered"):
        row(f"combined {tag}", pipe["combined"][tag]["metrics"])

    # ---------------- crisis breakdown ---------------- #
    print(LINE)
    print("Annualized net return by decoded regime (state 0 = lowest mean):")
    print(f"  {'strategy':<22} {names[0]:>9} {names[1]:>9} {names[2]:>10}")
    for name in (
        "momentum_unfiltered",
        "momentum_filtered",
        "carry_unfiltered",
        "carry_filtered",
        "combined_unfiltered",
        "combined_filtered",
    ):
        c = pipe["crisis"][name]
        print(
            f"  {name.replace('_', ' '):<22} {c[0]['ann_return']:>+9.2%} "
            f"{c[1]['ann_return']:>+9.2%} {c[2]['ann_return']:>+10.2%}"
        )

    mu, mf = pipe["combined"]["unfiltered"]["metrics"], pipe["combined"]["filtered"]["metrics"]
    print(LINE)
    print("Takeaway: momentum and carry both lose money in the high-vol crisis")
    print("regime; scaling by the filtered P(calm) improves the combined Sharpe")
    print(f"({mu['sharpe']:.2f} -> {mf['sharpe']:.2f}) and maxDD ({mu['max_dd']:+.2%} -> {mf['max_dd']:+.2%})")
    print(f"at the cost of total return ({mu['ann_return']:+.2%} -> {mf['ann_return']:+.2%}) — the filter")
    print("buys risk reduction, not free performance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
