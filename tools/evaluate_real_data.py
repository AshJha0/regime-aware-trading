#!/usr/bin/env python3
"""Test the regime gate on real US equity data from the Kenneth French library.

The pinned configuration is used unchanged: no parameter is tuned here, and
the sample start (1963-07-01) was fixed before any result was seen.  The raw
files are downloaded to --cache and are not redistributed in this repository.
Writes docs/REAL_DATA_EVALUATION.md.

Two books, each run under every gate at 5 bps:
  * market: long the value-weighted market excess return, weight 1;
  * momentum: time-series momentum on 10 industry excess returns.

There is no real FX carry book: no free daily carry data was used.

Run:  python tools/evaluate_real_data.py --cache <dir>
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))

from regime import (  # noqa: E402
    apply_gate,
    apply_gate_band,
    constant_gate,
    ewma_ratio_gate,
    momentum_positions,
    paired_block_bootstrap,
    regime_gate,
    run_backtest,
    sharpe,
    vol_ratio_gate,
)

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = ("F-F_Research_Data_Factors_daily_CSV.zip", "10_Industry_Portfolios_daily_CSV.zip")
START = "1963-07-01"
BOOT_SEED = 83


def fetch(name: str, cache: Path) -> str:
    """Return the CSV text inside one library zip, downloading it once."""
    path = cache / name
    if not path.exists():
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "regime-aware-trading research"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
    with zipfile.ZipFile(path) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def first_table(text: str) -> pd.DataFrame:
    """Parse the first daily table: rows start with an 8-digit date."""
    lines = text.splitlines()
    first = next(i for i, l in enumerate(lines) if l[:8].isdigit() and len(l.split(",")[0].strip()) == 8)
    header = [c.strip() for c in lines[first - 1].split(",")]
    rows = []
    for line in lines[first:]:
        if not (line[:8].isdigit() and len(line.split(",")[0].strip()) == 8):
            break
        rows.append(line)
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None)
    df.columns = ["date"] + header[1:] if len(header) == df.shape[1] else ["date"] + header
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df = df.set_index("date").astype(float)
    if (df <= -99.0).any().any():
        raise ValueError("missing-value markers in the sample")
    return df / 100.0


def run(gate: np.ndarray, pos: np.ndarray, rets: np.ndarray, cost: float) -> Dict:
    res = run_backtest(apply_gate(pos, gate), rets, cost)
    return {"net": res.net_returns, "turnover": res.turnover}


def stats(x: np.ndarray, turnover: np.ndarray) -> List[str]:
    eq = np.cumprod(1.0 + x)
    dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return [f"{float(sharpe(x)):.2f}", f"{252 * x.mean():.2%}", f"{252 ** 0.5 * x.std():.2%}", f"{dd:.1%}", f"{252 * turnover.mean():.1f}"]


def md(header: List[str], rows: List[List[str]]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "REAL_DATA_EVALUATION.md")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "data" / "config.json").read_text())
    ff = first_table(fetch(FILES[0], args.cache))
    ind = first_table(fetch(FILES[1], args.cache))
    df = ff.join(ind, how="inner").loc[START:]
    dates = df.index
    excess_mkt = df["Mkt-RF"].to_numpy()
    index = excess_mkt + df["RF"].to_numpy()
    industries = [c for c in ind.columns]
    ind_x = df[industries].to_numpy() - df["RF"].to_numpy()[:, None]
    T = len(df)
    start = cfg["train_min_days"]
    first = start - 1
    print(f"T={T} {dates[0].date()}..{dates[-1].date()}", flush=True)

    hmm, models = regime_gate(
        index, n_states=cfg["n_states"], train_min_days=start, refit_days=cfg["refit_days"],
        mode=cfg["gate_mode"], threshold=cfg["gate_threshold"], tol=cfg["em_tol"],
        max_iter=cfg["em_max_iter"], var_floor=cfg["var_floor"],
    )
    level = float(hmm[first:-1].mean())
    gates = {
        "Unfiltered": np.ones(T),
        "HMM gate": hmm,
        "HMM gate, band 0.3": apply_gate_band(hmm, 0.3),
        "Constant at mean gate": constant_gate(T, level, first),
        "Trailing 21d vol ratio": vol_ratio_gate(index, first),
        "EWMA vol ratio": ewma_ratio_gate(index, first, lam=cfg["ewma_lambda"]),
    }
    books = {
        "Market": (np.ones((T, 1)), excess_mkt[:, None]),
        "Industry momentum": (
            momentum_positions(ind_x, lookback=cfg["momentum_lookback"], vol_target=cfg["vol_target"],
                               ewma_lambda=cfg["ewma_lambda"], leverage_cap=cfg["leverage_cap"]),
            ind_x,
        ),
    }
    cost = cfg["cost_bps"]
    fr = [m.fit_result for m in models]
    out = [
        "# Real-data evaluation", "",
        "Generated by `python tools/evaluate_real_data.py`. Do not edit by hand; rerun the tool.", "",
        f"Data: Kenneth R. French Data Library, daily, {dates[0].date()} to {dates[-1].date()}, {T} days. "
        "The HMM observes the market total return; books trade excess returns. "
        f"Pinned configuration, unchanged. Window t >= {start}, {cost:g} bps on turnover.", "",
        "## Caveats", "",
        "- One market, one history: this is a single path, and the bootstrap interval is the honest measure of it.",
        "- These are research portfolios, not traded instruments. Early-sample costs were far above 5 bps, and financing, futures roll and capacity are not modelled.",
        "- There is no FX carry book here, so this does not test the combined book in the rest of the project.",
        "- Band 0.3 came from the bundled synthetic path, so on this data it is a fixed outside choice, not fitted here.",
        "- The default `calm_guard` setting (off) is used, and momentum uses per-asset sizing.", "",
        f"Refits: {len(fr)}, not converged {sum(not f.converged for f in fr)}, dead state {sum(f.dead_state is not None for f in fr)}, non-monotone {sum(not f.monotone for f in fr)}. Mean gate {level:.3f}.", "",
    ]
    rng = np.random.default_rng(BOOT_SEED)
    for book, (pos, rets) in books.items():
        runs = {g: run(gv, pos, rets, cost) for g, gv in gates.items()}
        out += [f"## {book}", ""]
        out += md(["Gate", "Sharpe", "Return", "Vol", "Max DD", "Turnover/yr"],
                  [[g] + stats(r["net"][start:], r["turnover"][start:]) for g, r in runs.items()])
        out += ["", "Paired stationary block bootstrap of the Sharpe difference (mean block 21, 2000 resamples):", ""]
        rows = []
        for a, b in (("HMM gate", "Unfiltered"), ("HMM gate", "Trailing 21d vol ratio"), ("HMM gate", "Constant at mean gate"), ("HMM gate, band 0.3", "Trailing 21d vol ratio")):
            bs = paired_block_bootstrap(runs[a]["net"][start:], runs[b]["net"][start:], rng, n_boot=2000)
            d = float(sharpe(runs[a]["net"][start:]) - sharpe(runs[b]["net"][start:]))
            rows.append([f"{a} minus {b}", f"{d:+.3f}", f"[{bs.ci_low:+.2f}, {bs.ci_high:+.2f}]"])
        out += md(["Comparison", "Difference", "95% interval"], rows)
        out += ["", "Sharpe by period:", ""]
        cuts = [("1965-1984", "1965", "1984"), ("1985-2004", "1985", "2004"), ("2005-end", "2005", "2100")]
        body = []
        for g, r in runs.items():
            s = pd.Series(r["net"], index=dates)
            body.append([g] + [f"{float(sharpe(s.loc[a:b].to_numpy())):.2f}" for _, a, b in cuts])
        out += md(["Gate"] + [c[0] for c in cuts], body) + [""]
    args.out.write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
