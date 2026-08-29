# 11 — Regime-Aware Trading (Gaussian HMM, Momentum, FX Carry)

A four-language reference implementation of a regime-aware trading study:
a **diagonal-covariance Gaussian hidden Markov model** (scaled
forward-backward, deterministic Baum-Welch EM, Viterbi decoding, model
selection) is fitted walk-forward on a market index and used to **gate two
classic strategies** — time-series momentum with vol targeting and an FX
carry basket — through a shared no-lookahead backtest engine. The bundled
data are generated from a true 3-state regime model (calm-bull / choppy /
crisis), so the HMM has something real to find and the honest headline
result can be demonstrated end to end: *the regime filter improves Sharpe
and max drawdown at the cost of total return*.

Everything is pinned in [`API_SPEC.md`](API_SPEC.md) — initialization,
convergence semantics, label ordering, rebalance schedules, accounting —
so all four implementations reproduce identical numbers to the tolerances
in `data/golden/golden.json`. No RNG is used anywhere except in the seeded
data generator.

## Feature matrix

| Feature | Python | C++ | Rust | Java |
|---|---|---|---|---|
| Gaussian HMM (D = 1, 2; K >= 2, diagonal covariance) | yes | yes | yes | yes |
| Scaled forward-backward (per-step normalizers + emission shift) | yes | yes | yes | yes |
| Baum-Welch EM, pinned quantile init, variance floor, RNG-free | yes | yes | yes | yes |
| Viterbi decoding (log-space, pinned tie-break) | yes | yes | yes | yes |
| Filtered / smoothed state probabilities | yes | yes | yes | yes |
| Stationary distribution (pinned power method) | yes | yes | yes | yes |
| Model selection: log-likelihood, AIC, BIC (K = 2 vs 3) | yes | yes | yes | yes |
| Time-series momentum (252d signal, EWMA vol target, 4x cap) | yes | yes | yes | yes |
| FX carry (rank differentials, top/bottom 3, 21d rebalance) | yes | yes | yes | yes |
| Walk-forward regime gate (expanding window, 63d refit, warm start) | yes | yes | yes | yes |
| Backtest engine (pinned accounting, costs on turnover, full metrics) | yes | yes | yes | yes |
| Crisis-state attribution table | yes | yes | yes | yes |
| Golden-value test suite (18 cases from `data/golden/golden.json`) | yes | yes | yes | yes |

## Directory layout

```
11-regime-aware-trading/
  README.md                <- this file
  LEARN.md                 <- theory: HMMs, EM, momentum, carry, regime filtering
  COOKBOOK.md              <- task-oriented recipes in all four languages
  API_SPEC.md              <- the pinned cross-language contract
  docs/
    ARCHITECTURE.md        <- component design, data flow, numerics (Mermaid diagrams)
    diagrams/*.mmd         <- raw Mermaid sources
  data/
    market_index.csv       <- equity-index daily returns (HMM observation series)
    trend_assets.csv       <- 6 futures-like assets for momentum
    fx_carry.csv           <- 6 synthetic currencies: spot returns + rate differentials
    true_states.csv        <- generator regime path (teaching comparison only)
    config.json            <- pinned strategy/backtest configuration
    generate_data.py       <- seeded generator (numpy default_rng(83)); rewrites CSVs + golden
    golden/golden.json     <- 18 cross-language golden cases
  python/
    src/regime/            <- reference implementation (hmm, strategies, backtest, pipeline)
    tests/                 <- pytest suite
    demo.py                <- end-to-end report
  cpp/                     <- C++17 port (namespace regime::), CMake + GoogleTest
  rust/                    <- Rust port (crate regime), cargo
  java/                    <- Java 21 port (com.quant.regime), javac + JUnit 4
```

## Build, run, test

### Python (reference)

```bash
cd python
PYTHONPATH=src python3 demo.py          # end-to-end demo
PYTHONPATH=src pytest -q                # test suite
```

Requires Python 3.11 with numpy and pandas (see `python/requirements.txt`).

### C++

```bash
cd cpp
./build.sh                              # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/demo                            # demo
ctest --test-dir build --output-on-failure   # tests (GoogleTest)
```

### Rust

```bash
cd rust
cargo build --release
cargo run --release --bin demo          # demo
cargo test                              # unit + integration tests
```

### Java

```bash
cd java
./build.sh                              # javac everything into out/
./demo.sh                               # demo main
./test.sh                               # JUnitCore on all *Test classes (JUnit 4)
```

## Demo output (Python reference, truncated)

```
Regime-Aware Trading — HMM + momentum + FX carry (bundled synthetic data)
------------------------------------------------------------------------------
Data: 2000 trading days, 2015-01-02 .. 2022-09-01

Model selection on the market index (diagonal Gaussian HMM):
   K       loglik          AIC          BIC  iters  conv
   2    6831.0255  -13648.0511  -13608.8447     27  True
   3    6886.4498  -13744.8997  -13666.4870     63  True

Fitted K=3 regimes (states sorted by mean; 0 = lowest):
  state      label  mean(ann)  vol(ann)  stationary
      0     crisis    -86.49%    40.96%      0.0566
      1  calm-bull    +16.01%     9.19%      0.6767
      2     choppy    +26.22%    16.80%      0.2667
  transition matrix (rows sum to 1):
    0.9204  0.0000  0.0796
    0.0008  0.9916  0.0076
    0.0148  0.0214  0.9637

Viterbi decoding vs true (generator) states: accuracy = 94.85%
------------------------------------------------------------------------------
Strategies (cost 5 bps, gate mode 'prob', refit every 63d on expanding window):
  strategy                ann ret  ann vol  Sharpe    maxDD  Calmar    hit  worst mo
  momentum unfiltered      +3.38%    4.14%    0.81  -10.48%    0.32  47.1%    -3.61%
  momentum filtered        +3.33%    3.20%    1.04   -7.37%    0.45  44.7%    -2.95%
  carry unfiltered        +10.77%    6.82%    1.58  -10.01%    1.08  54.1%    -3.48%
  carry filtered           +6.43%    4.31%    1.49   -4.95%    1.30  49.6%    -2.07%
  combined unfiltered      +7.07%    3.93%    1.80   -5.23%    1.35  55.8%    -2.27%
  combined filtered        +4.88%    2.62%    1.87   -3.41%    1.43  50.9%    -2.11%
------------------------------------------------------------------------------
Annualized net return by decoded regime (state 0 = lowest mean):
  strategy                  crisis calm-bull     choppy
  momentum unfiltered       -8.77%    +5.08%     +1.24%
  carry unfiltered         -11.09%   +10.06%    +16.70%
  combined filtered         -0.89%    +6.75%     +1.13%
------------------------------------------------------------------------------
Takeaway: momentum and carry both lose money in the high-vol crisis
regime; scaling by the filtered P(calm) improves the combined Sharpe
(1.80 -> 1.87) and maxDD (-5.23% -> -3.41%)
at the cost of total return (+7.07% -> +4.88%) — the filter
buys risk reduction, not free performance.
```

## Golden-value cross-language testing

`data/golden/golden.json` holds 18 named cases — HMM log-likelihoods,
fitted K=3 means/stds/transition matrix, exact Viterbi state counts,
stationary distribution, smoothed probabilities at pinned dates, strategy
ann-return/Sharpe/maxDD filtered vs unfiltered, combined Sharpe, the
crisis-state momentum return (a sign case: it must be negative), and mean
turnovers — each with an absolute tolerance (`tol = 0` means exact
integers). Every value was produced by the Python reference running the
full pinned configuration on the bundled data; the file also bakes in
semantic invariants (filtered Sharpe > unfiltered for the combined book,
filtered momentum has a shallower drawdown but a lower return, state
counts sum to 2000).

Every language's test suite loads this one file and asserts each case
within its tolerance. Because initialization, iteration order, convergence
checks, label sorting, tie-breaks and rebalance schedules are all pinned
in `API_SPEC.md` and no RNG is involved, any cross-language drift is a
real bug, not noise: a port that diverges beyond 1e-6 on the K=3
log-likelihood (or 1e-8 on a Sharpe) has strayed from the contract.
Regenerate data and golden values with `python3 data/generate_data.py`
(seeded; byte-for-byte reproducible).

## License

Educational use. No warranty; not investment advice.
