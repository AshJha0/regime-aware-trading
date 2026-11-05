"""Evaluate the HMM regime gate and write docs/GATE_EVALUATION.md.

Two parts.  The first works on the bundled dataset: post-warm-up
metrics, baseline gates, paired bootstrap, lag and cost sweeps, break-even
cost, classification and BIC.  The second re-simulates the DGP for seeds
1..N under four scenarios and repeats the comparison on each path.

Scenarios are built by importing data/generate_data.py and overriding its
SEED and constants in memory.  Nothing is written to data/.

    base         the DGP as shipped
    half_crisis  crisis-state drifts halved (index, trend, FX carry unwind),
                 crisis vols and the regime chain unchanged
    vol_only     every drift replaced by its stationary average, so regimes
                 change volatility only
    fat_tails    every normal shock replaced by unit-variance Student-t(4)

Run:  python tools/evaluate_gate.py --seeds 100
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Sequence, Tuple

# One BLAS thread per worker: the HMM's small matrices gain nothing from more.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.dont_write_bytecode = True  # importing generate_data must not touch data/

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))
sys.path.insert(0, str(ROOT / "data"))

import generate_data as gd  # noqa: E402
from regime import (  # noqa: E402
    Dataset,
    annual_turnover,
    bic_table,
    break_even_cost,
    build_book,
    classification_report,
    constant_gate,
    cost_sweep,
    ewma_ratio_gate,
    gate_break_even,
    lag_sweep,
    load_dataset,
    oracle_gate,
    paired_block_bootstrap,
    regime_gate,
    sharpe,
    vol_ratio_gate,
    window_metrics,
)

BOOTSTRAP_SEED = 83
N_BOOT = 10000
BLOCKS = (10, 21, 63)
COSTS = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
LAGS = (0, 1, 2, 5)
REFERENCE_SEED = gd.SEED
SCENARIOS = ("base", "half_crisis", "vol_only", "fat_tails")
GATE_LABELS = {
    "unfiltered": "Unfiltered",
    "hmm": "HMM gate",
    "constant": "Constant at mean gate",
    "vol": "Trailing 21d vol ratio",
    "ewma": "EWMA vol ratio",
    "oracle": "Next-day oracle (look-ahead)",
}


# ---------------------------------------------------------------------- #
# DGP scenarios
# ---------------------------------------------------------------------- #
class _StudentTGenerator:
    """Generator whose ``standard_normal`` draws unit-variance Student-t(4)."""

    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self.choice = self._rng.choice

    def standard_normal(self, size=None):
        return self._rng.standard_t(4, size) / np.sqrt(2.0)


class _NumpyWithStudentT:
    """Stand-in for ``generate_data.np``: numpy, with the generator swapped."""

    random = SimpleNamespace(default_rng=_StudentTGenerator)

    def __getattr__(self, name: str):
        return getattr(np, name)


def _stationary(transmat: np.ndarray) -> np.ndarray:
    """Stationary distribution of a transition matrix."""
    vals, vecs = np.linalg.eig(transmat.T)
    pi = np.real(vecs[:, np.argmax(np.real(vals))])
    return pi / pi.sum()


def scenario_overrides(scenario: str) -> Dict:
    """Module attributes of generate_data to replace for ``scenario``."""
    if scenario == "base":
        return {}
    if scenario == "half_crisis":
        half = np.array([1.0, 1.0, 0.5])
        return {
            "INDEX_MU": gd.INDEX_MU * half,
            "TREND_MU": gd.TREND_MU * half,
            "FX_CARRY_BETA": gd.FX_CARRY_BETA * half,
        }
    if scenario == "vol_only":
        pi = _stationary(gd.TRUE_TRANSMAT)
        return {
            name: np.full(3, float(pi @ getattr(gd, name)))
            for name in ("INDEX_MU", "TREND_MU", "FX_CARRY_BETA")
        }
    if scenario == "fat_tails":
        return {"np": _NumpyWithStudentT()}
    raise ValueError(f"unknown scenario {scenario!r}")


@contextmanager
def patched_generator(scenario: str, seed: int) -> Iterator[None]:
    """Apply a scenario and seed to generate_data, then put everything back."""
    overrides = dict(scenario_overrides(scenario), SEED=seed)
    saved = {name: getattr(gd, name) for name in overrides}
    for name, value in overrides.items():
        setattr(gd, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(gd, name, value)


def simulate_dataset(scenario: str, seed: int) -> Dataset:
    """Simulate one path in memory."""
    with patched_generator(scenario, seed):
        sim = gd.simulate()
    return Dataset(
        dates=sim["dates"],
        index_returns=sim["index"],
        trend_returns=sim["trend"],
        fx_spot_returns=sim["fx_spot"],
        fx_rate_diffs=sim["fx_diff"],
        true_states=sim["states"],
        config=dict(gd.CONFIG),
    )


# ---------------------------------------------------------------------- #
# Per-path evaluation
# ---------------------------------------------------------------------- #
def build_gates(ds: Dataset) -> Tuple[Dict[str, np.ndarray], int, int]:
    """All gates for one path, with the first gated day and the window start."""
    cfg = ds.config
    start = cfg["train_min_days"]
    first_day = start - 1
    hmm, _ = regime_gate(
        ds.index_returns,
        n_states=cfg["n_states"],
        train_min_days=start,
        refit_days=cfg["refit_days"],
        mode=cfg["gate_mode"],
        threshold=cfg["gate_threshold"],
        tol=cfg["em_tol"],
        max_iter=cfg["em_max_iter"],
        var_floor=cfg["var_floor"],
    )
    n = hmm.size
    # gate[first_day .. n-2] are the values that earn a return in the window.
    level = float(hmm[first_day:-1].mean())
    gates = {
        "unfiltered": np.ones(n),
        "hmm": hmm,
        "constant": constant_gate(n, level, first_day),
        "vol": vol_ratio_gate(ds.index_returns, first_day),
        "ewma": ewma_ratio_gate(ds.index_returns, first_day, lam=cfg["ewma_lambda"]),
        "oracle": oracle_gate(ds.true_states, first_day),
    }
    return gates, first_day, start


def evaluate_path(task: Tuple[str, int]) -> Dict:
    """Evaluate one (scenario, seed).  A failed fit is returned, not dropped."""
    scenario, seed = task
    out: Dict = {"scenario": scenario, "seed": seed}
    try:
        ds = simulate_dataset(scenario, seed)
        gates, first_day, start = build_gates(ds)
        book = build_book(ds)
        streams = {name: book.stream(g) for name, g in gates.items()}
    except ValueError as exc:
        out["error"] = str(exc)
        return out
    cost = ds.config["cost_bps"]
    sr = {name: float(sharpe(s.net(cost)[start:])) for name, s in streams.items()}
    be = gate_break_even(streams["hmm"], streams["unfiltered"], start)
    report = classification_report(gates["hmm"], ds.true_states, start)
    out.update(
        sharpe=sr,
        d_unfiltered=sr["hmm"] - sr["unfiltered"],
        d_vol=sr["hmm"] - sr["vol"],
        d_ewma=sr["hmm"] - sr["ewma"],
        d_oracle=sr["oracle"] - sr["unfiltered"],
        d_zero_cost=float(sharpe(streams["hmm"].net(0.0)[start:]) - sharpe(streams["unfiltered"].net(0.0)[start:])),
        d_full_sample=float(sharpe(streams["hmm"].net(cost)) - sharpe(streams["unfiltered"].net(cost))),
        break_even=be.cost_bps,
        break_even_status=be.status,
        accuracy=report["accuracy"],
        majority=report["majority_accuracy"],
        auc_crisis=report["auc_crisis"],
        n_episodes=report["n_episodes"],
    )
    return out


# ---------------------------------------------------------------------- #
# Reference run on the bundled data
# ---------------------------------------------------------------------- #
def reference_run() -> Dict:
    """Everything measured on the bundled dataset."""
    ds = load_dataset(ROOT / "data")
    cfg = ds.config
    gates, first_day, start = build_gates(ds)
    book = build_book(ds)
    streams = {name: book.stream(g) for name, g in gates.items()}
    cost = cfg["cost_bps"]
    net = {name: s.net(cost) for name, s in streams.items()}
    rows = {}
    for name, s in streams.items():
        rows[name] = dict(
            window_metrics(net[name], ds.dates, start),
            exposure=float(gates[name][first_day:-1].mean()),
            carry_turnover=annual_turnover(s.carry_turnover, start),
            carry_turnover_full=annual_turnover(s.carry_turnover, 0),
            turnover=annual_turnover(s.turnover, start),
            full_sharpe=float(sharpe(net[name])),
        )
    pairs = (("hmm", "unfiltered"), ("hmm", "vol"), ("hmm", "ewma"))
    boot = {
        (a, b, block): paired_block_bootstrap(
            net[a][start:], net[b][start:], np.random.default_rng(BOOTSTRAP_SEED), N_BOOT, block
        )
        for a, b in pairs
        for block in BLOCKS
    }
    fixed_benchmark = rows["unfiltered"]["sharpe"]
    return {
        "cfg": cfg,
        "n_days": len(ds.dates),
        "window_days": len(ds.dates) - start,
        "start": start,
        "rows": rows,
        "boot": boot,
        "cost_sweep": cost_sweep(streams, COSTS, start),
        "lag_sweep": lag_sweep(book, {k: v for k, v in gates.items() if k in ("hmm", "vol", "ewma")}, LAGS, cost, start),
        "break_even": gate_break_even(streams["hmm"], streams["unfiltered"], start),
        "break_even_fixed": break_even_cost(lambda c: float(sharpe(streams["hmm"].net(c)[start:])) - fixed_benchmark),
        "classes": classification_report(gates["hmm"], ds.true_states, start),
        "bic": bic_table(ds.index_returns, tol=cfg["em_tol"], max_iter=cfg["em_max_iter"], var_floor=cfg["var_floor"]),
    }


# ---------------------------------------------------------------------- #
# Report
# ---------------------------------------------------------------------- #
def wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return centre - half, centre + half


def share(values: Sequence[float]) -> str:
    """Share of positive values with its Wilson interval."""
    wins = sum(v > 0.0 for v in values)
    lo, hi = wilson(wins, len(values))
    return f"{wins / len(values):.0%} [{lo:.0%}, {hi:.0%}]"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """Markdown table lines."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines + [""]


def summarise(results: List[Dict], scenario: str) -> Dict:
    """Aggregate one scenario of the seed sweep."""
    all_rows = [r for r in results if r["scenario"] == scenario]
    ok = [r for r in all_rows if "error" not in r]

    def col(key: str) -> np.ndarray:
        return np.array([r[key] for r in ok])

    crossed = col("break_even")[~np.isnan(col("break_even"))]
    statuses = [r["break_even_status"] for r in ok]
    return {
        "n": len(ok),
        "failed": [r for r in all_rows if "error" in r],
        "ok": ok,
        "col": col,
        "crossed": crossed,
        "below_at_zero": statuses.count("below_at_zero"),
        "above_at_max": statuses.count("above_at_max"),
    }


def percentile_of(values: np.ndarray, x: float) -> float:
    """Percent of values at or below ``x``."""
    return 100.0 * float(np.mean(values <= x))


def render(ref: Dict, results: List[Dict], n_seeds: int) -> str:
    """Build the markdown report."""
    rows = ref["rows"]
    unf, hmm = rows["unfiltered"], rows["hmm"]
    cls = ref["classes"]
    cost = ref["cfg"]["cost_bps"]
    years = ref["window_days"] / 252.0
    main_boot = ref["boot"][("hmm", "unfiltered", 21)]
    be, be_fixed = ref["break_even"], ref["break_even_fixed"]
    out: List[str] = [
        "# Gate evaluation",
        "",
        "Generated by `python tools/evaluate_gate.py "
        f"--seeds {n_seeds}`. Do not edit by hand; rerun the tool.",
        "",
        "## What this shows",
        "",
        f"Over the full {ref['n_days']} days at {cost:g} bps the combined Sharpe is "
        f"{unf['full_sharpe']:.2f} unfiltered and {hmm['full_sharpe']:.2f} gated. "
        f"The gate is 1 for the first {ref['start'] - 1} days, so those days do not test it. "
        f"On the {ref['window_days']} days it can affect (t >= {ref['start']}), the Sharpe is "
        f"{unf['sharpe']:.2f} unfiltered and {hmm['sharpe']:.2f} gated, a difference of "
        f"{main_boot.diff:+.3f} with a 95% bootstrap interval of "
        f"[{main_boot.ci_low:.2f}, {main_boot.ci_high:.2f}].",
        "",
        f"Holding a constant {hmm['exposure']:.3f} of the book gives Sharpe "
        f"{rows['constant']['sharpe']:.2f}, return {rows['constant']['ann_return']:.2%} and max drawdown "
        f"{rows['constant']['max_dd']:.2%}, against {hmm['sharpe']:.2f}, {hmm['ann_return']:.2%} and "
        f"{hmm['max_dd']:.2%} for the gate. That level is the gate's own mean, known only in hindsight, "
        "so it is a control and not a rule you could have traded.",
        "",
        "## Caveats",
        "",
        "- All data is synthetic. The DGP has regimes by construction and its constants were tuned "
        "on an earlier seed, 83, so nothing here says the gate works on real markets.",
        f"- Seed {REFERENCE_SEED} was picked by a rule fixed in advance: its unfiltered post-warm-up Sharpe is "
        "the closest to the median of seeds 1 to 100. No gate result entered the choice, so the gate's "
        "result on this path was not selected. It can still be unusual, and the seed sweep below says where it sits.",
        "- Every book carries the same portfolio vol scale (10% target), computed from the unfiltered "
        f"book at {cost:g} bps. A gate multiplies that scaled book. In the cost sweep the cost moves "
        "and the scale stays fixed; in the lag sweep only the gate is delayed.",
        f"- The bundled window is {ref['window_days']} days, about {years:.1f} years, with "
        f"{cls['n_episodes']} crisis episodes ({cls['n_crisis_days']} crisis days) starting inside it.",
        f"- The seed sweep uses {n_seeds} seeds per scenario. With that count a share has a "
        "95% interval of up to about 10 points either side, shown in brackets.",
        "- Windows, baselines and scenarios were chosen after the audit of the bundled result, "
        "not preregistered.",
        "- The vol baselines have one fixed setting each and were not tuned. Their mean exposure "
        f"({rows['vol']['exposure']:.3f} and {rows['ewma']['exposure']:.3f}) differs from the gate's.",
        "",
        f"## Bundled data (seed {REFERENCE_SEED}), t >= {ref['start']}, {cost:g} bps",
        "",
    ]
    out += table(
        ["Gate", "Mean exposure", "Sharpe", "Return", "Vol", "Max DD", "Turnover/yr", "Carry turnover/yr"],
        [
            [
                GATE_LABELS[k],
                f"{r['exposure']:.3f}",
                f"{r['sharpe']:.2f}",
                f"{r['ann_return']:.2%}",
                f"{r['ann_vol']:.2%}",
                f"{r['max_dd']:.2%}",
                f"{r['turnover']:.1f}",
                f"{r['carry_turnover']:.2f}",
            ]
            for k, r in rows.items()
        ],
    )
    out += [
        f"Carry turnover over the full sample, which adds the day-0 entry and the warm-up, is "
        f"{hmm['carry_turnover_full']:.1f}/yr gated and {unf['carry_turnover_full']:.2f}/yr unfiltered. "
        "The portfolio scale moves every day, so carry trades even when its basket does not change; "
        "the gap between the two figures comes from the gate moving.",
        "",
        "### Paired stationary block bootstrap of the Sharpe difference",
        "",
        f"{N_BOOT} resamples, generator seed {BOOTSTRAP_SEED}, percentile intervals.",
        "",
    ]
    out += table(
        ["Comparison", "Mean block", "Difference", "95% interval", "P(diff > 0)"],
        [
            [
                f"{GATE_LABELS[a]} minus {GATE_LABELS[b]}",
                str(block),
                f"{res.diff:+.3f}",
                f"[{res.ci_low:.2f}, {res.ci_high:.2f}]",
                f"{res.prob_positive:.1%}",
            ]
            for (a, b, block), res in ref["boot"].items()
        ],
    )
    out += ["### Cost sweep (Sharpe)", ""]
    out += table(
        ["Gate"] + [f"{c:g} bps" for c in COSTS],
        [[GATE_LABELS[k]] + [f"{v:.2f}" for v in vals] for k, vals in ref["cost_sweep"].items()],
    )
    out += [
        f"Break-even cost, both books paying the same cost: {be.cost_bps:.2f} bps ({be.status}). "
        f"Holding the unfiltered book at its {cost:g} bps Sharpe of {unf['sharpe']:.2f} and charging "
        f"only the gated book: {be_fixed.cost_bps:.2f} bps ({be_fixed.status}).",
        "",
        f"### Execution lag (Sharpe at {cost:g} bps)",
        "",
        "Lag 0 is the pinned convention: the gate is computed at the close and traded at that close.",
        "",
    ]
    out += table(
        ["Gate"] + [f"{lag} d" for lag in LAGS],
        [[GATE_LABELS[k]] + [f"{v:.2f}" for v in vals] for k, vals in ref["lag_sweep"].items()],
    )
    out += [
        "### Classification against the true states",
        "",
        f"Calm against not calm at a 0.5 threshold: accuracy {cls['accuracy']:.1%}, majority class "
        f"{cls['majority_accuracy']:.1%}, AUC {cls['auc_calm']:.3f}. Crisis against the rest: AUC "
        f"{cls['auc_crisis']:.3f}. Of {cls['n_episodes']} crisis episodes, {cls['n_missed']} were missed; "
        f"median detection lag {cls['median_lag']:.1f} days, mean {cls['mean_lag']:.2f}. "
        "Lag 0 includes episodes that began with the gate already below 0.5.",
        "",
        "The classifier and the trading result are separate questions: "
        f"the next-day oracle reaches Sharpe {rows['oracle']['sharpe']:.2f} with exposure "
        f"{rows['oracle']['exposure']:.3f}, while the gate holds {hmm['exposure']:.3f}. "
        "The gate is P(calm), so it also cuts exposure in the choppy state.",
        "",
        "### BIC on the market index (full sample, in-sample)",
        "",
    ]
    out += table(
        ["K", "Log-likelihood", "Parameters", "BIC"],
        [[str(r["k"]), f"{r['loglik']:.2f}", str(r["n_params"]), f"{r['bic']:.2f}"] for r in ref["bic"]],
    )
    out += [
        "K=1 is the i.i.d. Gaussian. Lower BIC is better.",
        "",
        "## Seed sweep",
        "",
        f"Seeds 1 to {n_seeds} per scenario, T = {gd.T} days, window t >= {ref['start']}, {cost:g} bps. "
        "dSharpe is the gate minus the named book on the same path. Brackets are 95% Wilson intervals. "
        "half_crisis halves crisis drifts with crisis vols unchanged; vol_only sets every drift to its "
        "stationary average; fat_tails uses unit-variance Student-t(4) shocks.",
        "",
    ]
    sums = {s: summarise(results, s) for s in SCENARIOS}
    out += table(
        ["Scenario", "Paths ok", "Failed", "P(beats unfiltered)", "P(beats vol)", "P(beats EWMA)",
         "Mean dSharpe vs unfiltered (sd)", "Mean dSharpe vs vol (sd)"],
        [
            [
                s, str(m["n"]), str(len(m["failed"])),
                share(m["col"]("d_unfiltered")), share(m["col"]("d_vol")), share(m["col"]("d_ewma")),
                f"{m['col']('d_unfiltered').mean():+.3f} ({m['col']('d_unfiltered').std(ddof=1):.3f})",
                f"{m['col']('d_vol').mean():+.3f} ({m['col']('d_vol').std(ddof=1):.3f})",
            ]
            for s, m in sums.items()
        ],
    )
    out += table(
        ["Scenario", "Mean Sharpe unfiltered", "Mean Sharpe gate", "Mean oracle gain", "P(beats unfiltered, 0 bps)",
         "P(beats unfiltered, full sample)", "Median break-even bps (crossed)", "Lost at 0 bps", "Mean accuracy / majority",
         "Mean crisis AUC", "Median episodes"],
        [
            [
                s,
                f"{np.mean([r['sharpe']['unfiltered'] for r in m['ok']]):.2f}",
                f"{np.mean([r['sharpe']['hmm'] for r in m['ok']]):.2f}",
                f"{m['col']('d_oracle').mean():+.3f}",
                share(m["col"]("d_zero_cost")),
                share(m["col"]("d_full_sample")),
                f"{np.median(m['crossed']):.2f} (n={m['crossed'].size})" if m["crossed"].size else "none",
                str(m["below_at_zero"]),
                f"{m['col']('accuracy').mean():.1%} / {m['col']('majority').mean():.1%}",
                f"{np.nanmean(m['col']('auc_crisis')):.3f}",
                f"{np.median(m['col']('n_episodes')):.0f}",
            ]
            for s, m in sums.items()
        ],
    )
    base = sums["base"]
    d_base = base["col"]("d_unfiltered")
    out[out.index("## Caveats"):out.index("## Caveats")] = [
        f"Seed {REFERENCE_SEED} is one path, with {cls['n_episodes']} crisis episodes in the window "
        f"against a median of {np.median(base['col']('n_episodes')):.0f} across seeds. Over {base['n']} base-DGP "
        f"seeds the gate beats the unfiltered book on {share(d_base)} of paths, mean dSharpe {d_base.mean():+.3f}, "
        f"and beats the trailing vol ratio on {share(base['col']('d_vol'))}, mean dSharpe "
        f"{base['col']('d_vol').mean():+.3f}. Read the bundled result and the sweep together: the bundled path "
        "alone does not settle whether the gate helps.",
        "",
    ]
    ref_row = next((r for r in base["ok"] if r["seed"] == REFERENCE_SEED), None)
    if ref_row is not None:
        out += [
            f"Seed {REFERENCE_SEED} in the base sweep: post-warm-up dSharpe {ref_row['d_unfiltered']:+.3f} sits at "
            f"percentile {percentile_of(base['col']('d_unfiltered'), ref_row['d_unfiltered']):.0f}, "
            f"and full-sample dSharpe {ref_row['d_full_sample']:+.3f} sits at percentile "
            f"{percentile_of(base['col']('d_full_sample'), ref_row['d_full_sample']):.0f}, of {base['n']} seeds.",
            "",
        ]
    for s, m in sums.items():
        for r in m["failed"]:
            out.append(f"- Failed path {s} seed {r['seed']}: {r['error']}")
    if any(m["failed"] for m in sums.values()):
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=100, help="seeds per scenario (1..N)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "GATE_EVALUATION.md")
    args = parser.parse_args()
    if args.seeds < 2:
        parser.error("--seeds must be at least 2")

    tasks = [(s, seed) for s in SCENARIOS for seed in range(1, args.seeds + 1)]
    results: List[Dict] = []
    with mp.Pool(args.workers) as pool:
        pending = pool.imap_unordered(evaluate_path, tasks, chunksize=1)
        ref = reference_run()
        for done, row in enumerate(pending, 1):
            results.append(row)
            if done % 20 == 0:
                print(f"{done}/{len(tasks)} paths", flush=True)
    results.sort(key=lambda r: (SCENARIOS.index(r["scenario"]), r["seed"]))
    args.out.write_text(render(ref, results, args.seeds), encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
