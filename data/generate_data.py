"""Regenerate the bundled synthetic dataset and the golden values.

The data come FROM a true 3-state regime model (0 = calm-bull, 1 = choppy,
2 = crisis) so the HMM has something real to find:

  * market_index.csv  — equity-index daily returns with regime-dependent
                        mean/vol (the HMM's observation series);
  * trend_assets.csv  — 6 futures-like assets: persistent AR(1) idiosyncratic
                        drift (fuel for momentum) + regime drift/vol + index
                        beta; crisis drags every asset down, so momentum's
                        mostly-long book loses there;
  * fx_carry.csv      — 6 synthetic currencies: OU rate differentials around
                        distinct bases; spots earn a small carry-proportional
                        drift in calm and violently unwind it in crisis;
  * true_states.csv   — the generator's regime path (teaching only);
  * config.json       — the pinned strategy/backtest configuration.

Everything is seeded (numpy default_rng, SEED = 83); rerunning this script
reproduces the CSVs byte-for-byte.  After writing the data the script
validates the implementation (tiny enumerated forward-backward case and EM
monotonicity on the bundled index) and only then writes
golden/golden.json from a full pinned-config pipeline run.

Run:  python3 data/generate_data.py
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA_DIR.parent / "python" / "src"))

from regime import GaussianHMM, golden_cases, run_full_pipeline  # noqa: E402

SEED = 83
T = 2000
N_TREND = 6
N_FX = 6

# True regime chain: 0 = calm-bull, 1 = choppy, 2 = crisis (persistent).
TRUE_TRANSMAT = np.array(
    [
        [0.985, 0.012, 0.003],
        [0.030, 0.940, 0.030],
        [0.010, 0.045, 0.945],
    ]
)
INDEX_MU = np.array([0.00070, 0.00000, -0.00180])
INDEX_SIGMA = np.array([0.0060, 0.0110, 0.0250])

TREND_MU = np.array([0.00080, 0.00020, -0.00600])   # regime drift, all assets
TREND_VOLMULT = np.array([1.0, 1.6, 2.0])
TREND_BASE_VOL = 0.009
TREND_BETA = 0.30
TREND_AR_PHI = 0.995
TREND_AR_ETA = 0.00006

FX_BASE_DIFF = np.array([0.060, 0.040, 0.025, 0.010, -0.005, -0.020])
FX_OU_SPEED = 0.002
FX_OU_VOL = 0.0004
FX_CARRY_BETA = np.array([0.75, 0.2, -12.0])         # spot drift per unit diff
FX_VOL = np.array([0.0040, 0.0060, 0.0140])

CONFIG = {
    "n_states": 3,
    "em_tol": 1e-8,
    "em_max_iter": 500,
    "var_floor": 1e-8,
    "momentum_lookback": 252,
    "vol_target": 0.10,
    "ewma_lambda": 0.94,
    "leverage_cap": 4.0,
    "carry_top_n": 3,
    "carry_bottom_n": 3,
    "rebalance_days": 21,
    "train_min_days": 504,
    "refit_days": 63,
    "gate_mode": "prob",
    "gate_threshold": 0.5,
    "cost_bps": 5.0,
    "trading_days": 252,
}


def simulate() -> dict:
    """Simulate the full dataset from the true 3-state DGP (seeded)."""
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2015-01-02", periods=T)

    # Regime path.
    states = np.zeros(T, dtype=int)
    for t in range(1, T):
        states[t] = rng.choice(3, p=TRUE_TRANSMAT[states[t - 1]])

    # Market index.
    idx_ret = INDEX_MU[states] + INDEX_SIGMA[states] * rng.standard_normal(T)

    # Trend assets: persistent idiosyncratic drift + regime drift/vol + beta.
    drift = np.zeros((T, N_TREND))
    eta = TREND_AR_ETA * rng.standard_normal((T, N_TREND))
    for t in range(1, T):
        drift[t] = TREND_AR_PHI * drift[t - 1] + eta[t]
    trend_noise = TREND_BASE_VOL * TREND_VOLMULT[states][:, None] * rng.standard_normal((T, N_TREND))
    trend_ret = TREND_MU[states][:, None] + drift + TREND_BETA * idx_ret[:, None] + trend_noise

    # FX: OU rate differentials + regime-dependent carry drift / unwind.
    diffs = np.zeros((T, N_FX))
    diffs[0] = FX_BASE_DIFF
    z = rng.standard_normal((T, N_FX))
    for t in range(1, T):
        diffs[t] = diffs[t - 1] + FX_OU_SPEED * (FX_BASE_DIFF - diffs[t - 1]) + FX_OU_VOL * z[t]
    spot = np.zeros((T, N_FX))
    fx_z = rng.standard_normal((T, N_FX))
    for t in range(1, T):
        rel = diffs[t - 1] - diffs[t - 1].mean()
        s = states[t]
        spot[t] = FX_CARRY_BETA[s] * rel / 252.0 + FX_VOL[s] * fx_z[t]

    return {
        "dates": dates,
        "states": states,
        "index": idx_ret,
        "trend": trend_ret,
        "fx_spot": spot,
        "fx_diff": diffs,
    }


def write_csvs(sim: dict) -> None:
    """Write the four CSVs and config.json (deterministic formatting)."""
    dates = sim["dates"].strftime("%Y-%m-%d")
    fmt = "%.10f"

    pd.DataFrame({"date": dates, "ret": sim["index"]}).to_csv(
        DATA_DIR / "market_index.csv", index=False, float_format=fmt
    )
    trend = pd.DataFrame(sim["trend"], columns=[f"asset_{i}" for i in range(N_TREND)])
    trend.insert(0, "date", dates)
    trend.to_csv(DATA_DIR / "trend_assets.csv", index=False, float_format=fmt)

    fx = pd.DataFrame(sim["fx_spot"], columns=[f"spot_ret_{i}" for i in range(N_FX)])
    for i in range(N_FX):
        fx[f"rate_diff_{i}"] = sim["fx_diff"][:, i]
    fx.insert(0, "date", dates)
    fx.to_csv(DATA_DIR / "fx_carry.csv", index=False, float_format=fmt)

    pd.DataFrame({"date": dates, "state": sim["states"]}).to_csv(
        DATA_DIR / "true_states.csv", index=False
    )
    with open(DATA_DIR / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
        f.write("\n")


def validate_implementation() -> None:
    """Sanity-check the HMM before trusting it to produce golden values.

    1. Tiny enumerated case (K=2, T=3): scaled forward likelihood must
       match brute-force enumeration over all 8 state paths to 1e-12.
    2. EM monotonicity: every Baum-Welch iteration on the bundled index
       must be non-decreasing in log-likelihood.
    """
    # --- tiny enumerated case -----------------------------------------
    from regime.hmm import HMMParams

    hmm = GaussianHMM(2)
    params = HMMParams(
        startprob=np.array([0.6, 0.4]),
        transmat=np.array([[0.7, 0.3], [0.2, 0.8]]),
        means=np.array([[-1.0], [1.0]]),
        variances=np.array([[0.5], [2.0]]),
    )
    hmm.params = params
    x = np.array([-0.5, 0.3, 1.2])

    def norm_pdf(v: float, mu: float, var: float) -> float:
        return float(np.exp(-0.5 * (v - mu) ** 2 / var) / np.sqrt(2 * np.pi * var))

    brute = 0.0
    for path in itertools.product([0, 1], repeat=3):
        p = params.startprob[path[0]] * norm_pdf(x[0], params.means[path[0], 0], params.variances[path[0], 0])
        for t in range(1, 3):
            p *= params.transmat[path[t - 1], path[t]] * norm_pdf(
                x[t], params.means[path[t], 0], params.variances[path[t], 0]
            )
        brute += p
    ll_scaled = hmm.score(x)
    assert abs(np.exp(ll_scaled) - brute) < 1e-12, "scaled forward disagrees with enumeration"
    print(f"  enumerated forward check OK: L={brute:.12e}")

    # --- EM monotonicity on the bundled index -------------------------
    idx = pd.read_csv(DATA_DIR / "market_index.csv")["ret"].to_numpy()
    fitter = GaussianHMM(3, tol=CONFIG["em_tol"], max_iter=CONFIG["em_max_iter"], var_floor=CONFIG["var_floor"])
    res = fitter.fit(idx)
    hist = np.array(res.loglik_history)
    assert np.all(np.diff(hist) >= -1e-9), "EM log-likelihood decreased"
    assert res.converged, "EM failed to converge on bundled index"
    print(f"  EM monotonicity OK: {res.n_iter} iters, ll={res.log_likelihood:.6f}")


def write_golden() -> None:
    """Run the full pinned-config pipeline and write golden/golden.json."""
    pipe = run_full_pipeline(DATA_DIR)
    cases = golden_cases(pipe)
    gold_dir = DATA_DIR / "golden"
    gold_dir.mkdir(exist_ok=True)
    with open(gold_dir / "golden.json", "w") as f:
        json.dump({"cases": cases}, f, indent=2)
        f.write("\n")
    print(f"  wrote {len(cases)} golden cases to {gold_dir / 'golden.json'}")
    crisis_mom = pipe["crisis"]["momentum_unfiltered"][0]["ann_return"]
    print(f"  crisis momentum ann return: {crisis_mom:+.4f} (must be negative)")
    assert crisis_mom < 0.0, "DGP tuning broke the crisis-momentum sign case"
    su = pipe["combined"]["unfiltered"]["metrics"]["sharpe"]
    sf = pipe["combined"]["filtered"]["metrics"]["sharpe"]
    print(f"  combined Sharpe: unfiltered={su:.3f} filtered={sf:.3f}")


if __name__ == "__main__":
    print("simulating dataset...")
    sim = simulate()
    write_csvs(sim)
    counts = np.bincount(sim["states"], minlength=3)
    print(f"  wrote CSVs: T={T}, state days calm/choppy/crisis = {counts.tolist()}")
    print("validating implementation...")
    validate_implementation()
    print("computing golden values (full pinned config)...")
    write_golden()
    print("done.")
