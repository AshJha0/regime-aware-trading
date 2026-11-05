# Regime-Aware Trading: a case study in evaluating a regime signal

**The result.** On real US equity data from 1963 to 2026, the HMM regime gate
lost. At 5 bps its Sharpe was 0.14 on the market against 0.44 unfiltered and 0.48
for a free 21-day volatility rule, and 0.07 on industry momentum against 0.37
and 0.40. All four 95 % bootstrap intervals on the differences lie below zero:
-0.298 [-0.56, -0.03] and -0.333 [-0.56, -0.08] on the market, -0.295
[-0.55, -0.06] and -0.324 [-0.54, -0.11] on momentum
([`docs/REAL_DATA_EVALUATION.md`](docs/REAL_DATA_EVALUATION.md)). By the rule in
[`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md), tuning has stopped. No
claim is made that the gate adds value, and the code stays as a worked example.

**What this repository is.** A four-language reference implementation of a
regime study, kept for two uses: learning how a **diagonal-covariance Gaussian
hidden Markov model** works (scaled forward-backward, deterministic Baum-Welch
EM, Viterbi decoding, model selection), and learning how to judge a signal
(baselines, paired block bootstrap, seed sweeps, costs, preregistration). The
model is fitted walk-forward on a market index and **gates two strategies**, time
series momentum and an FX carry basket, through a shared no-lookahead backtest
engine. Start with [`LEARN.md`](LEARN.md).

**The bundled data flatter the gate.** They are synthetic, generated from a true
3-state regime model (seed 19, 2000 days). After warm-up the combined Sharpe goes
from 0.54 unfiltered to 0.99 gated, a difference of +0.456 with a 95 % interval
of [-0.13, 1.05], and that gain sits at the 85th percentile of 100 seeds. Across
those seeds the gate beats the unfiltered book on 66 % of paths and the 21-day
vol rule on 49 %, and with Student-t(4) shocks it beats unfiltered on 12 %
([`docs/GATE_EVALUATION.md`](docs/GATE_EVALUATION.md)). The filtered classifier
does well (crisis AUC 0.962); turning detection into P&L is the part that failed.
Seed 19 was chosen by a rule on the unfiltered Sharpe alone, so nobody picked it
for the gate. Any bundled gate number below should be read with this paragraph.

**Costs.** The book carries a 10 % portfolio vol target (API_SPEC 3.5) whose daily
scale adds real turnover. Post-warm-up the combined book turns over 36.8 times a
year unfiltered and 73.9 gated, and the gate's advantage on the bundled path is
gone at 11.54 bps.

| Report | Question it answers | Tool |
|---|---|---|
| [`REAL_DATA_EVALUATION.md`](docs/REAL_DATA_EVALUATION.md) | Does the gate work on real equities? No. | `tools/evaluate_real_data.py` |
| [`GATE_EVALUATION.md`](docs/GATE_EVALUATION.md) | Baselines, bootstrap, costs, seeds on synthetic data | `tools/evaluate_gate.py` |
| [`BAND_EVALUATION.md`](docs/BAND_EVALUATION.md) | Does a no-trade band help across seeds? | `tools/evaluate_band.py` |
| [`CAPACITY_EVALUATION.md`](docs/CAPACITY_EVALUATION.md) | Shape of cost against size, on illustrative inputs | `tools/evaluate_capacity.py` |
| [`PREREGISTRATION.md`](docs/PREREGISTRATION.md) | What a future gate claim must pass | none |

Everything is pinned in [`API_SPEC.md`](API_SPEC.md) — initialization,
convergence semantics, the dead-state guard, label ordering, rebalance
schedules, accounting, validation — so all four implementations reproduce
identical numbers to the tolerances in `data/golden/golden.json`. No RNG
is used in library code except in the seeded data generator (test
fixtures draw their own seeded inputs).

## Feature matrix

| Feature | Python | C++ | Rust | Java |
|---|---|---|---|---|
| Gaussian HMM (D = 1, 2; K >= 2, diagonal covariance) | yes | yes | yes | yes |
| Scaled forward-backward (per-step normalizers + emission shift) | yes | yes | yes | yes |
| Baum-Welch EM, pinned quantile init, variance floor, RNG-free | yes | yes | yes | yes |
| Dead-state guard + monotonicity flag (no NaN ever leaves `fit`) | yes | yes | yes | yes |
| Parameter/data shape + stochasticity validation on every entry point | yes | yes | yes | yes |
| Viterbi decoding (log-space, pinned tie-break) | yes | yes | yes | yes |
| Filtered / smoothed state probabilities | yes | yes | yes | yes |
| Streaming `RegimeFilter` (allocation-free in C++ and Rust) | yes | yes | yes | yes |
| Opt-in `gate_band` and `transition_floor` (off by default) | yes | yes | yes | yes |
| Stationary distribution (pinned power method) | yes | yes | yes | yes |
| Model selection: log-likelihood, AIC, BIC (K = 2 vs 3) | yes | yes | yes | yes |
| Opt-in `calm_guard` with `guard_trips` (off by default) | yes | yes | yes | yes |
| Time-series momentum (252d signal, EWMA vol target, 4x cap) | yes | yes | yes | yes |
| FX carry (rank differentials, top/bottom 3, 21d rebalance) | yes | yes | yes | yes |
| Portfolio vol target (`portfolio_scale`, `scaled_book`) | yes | yes | yes | yes |
| Walk-forward regime gate (expanding window, 63d refit, warm start) | yes | yes | yes | yes |
| Backtest engine (pinned accounting, costs on turnover, full metrics) | yes | yes | yes | yes |
| Wipe-out guard, returns > -1, strictly increasing dates | yes | yes | yes | yes |
| Crisis-state attribution table | yes | yes | yes | yes |
| Gate evaluation: baselines, block bootstrap, seed sweep | yes | no | no | no |
| Real-data, band and capacity evaluations | yes | no | no | no |
| Size-dependent cost model (`regime.costs`) | yes | no | no | no |
| Golden-value test suite (23 cases / 87 scalars from `data/golden/golden.json`) | yes | yes | yes | yes |

## Directory layout

```
regime-aware-trading/
  README.md                <- this file
  LEARN.md                 <- theory, then how to judge a signal and the real-data result
  COOKBOOK.md              <- task-oriented recipes, compiled and run in CI
  API_SPEC.md              <- the pinned cross-language contract
  CHANGELOG.md             <- release notes
  LICENSE                  <- MIT
  .github/workflows/ci.yml <- CI: four languages, sanitizers, cookbook, docs checks
  docs/
    ARCHITECTURE.md        <- component design, data flow, numerics, desk usage (Mermaid)
    REAL_DATA_EVALUATION.md <- generated: the gate on 1963-2026 equity data
    GATE_EVALUATION.md     <- generated: baselines, bootstrap, seed sweep on synthetic data
    BAND_EVALUATION.md     <- generated: no-trade band across seeds
    CAPACITY_EVALUATION.md <- generated: cost against size, illustrative inputs
    PREREGISTRATION.md     <- the rule any future gate claim must pass
    GITHUB_PAGES.md        <- publishing guide for the docs site: https://ashjha0.github.io/regime-aware-trading/
    diagrams/*.mmd         <- raw Mermaid sources
  data/
    market_index.csv       <- equity-index daily returns (HMM observation series)
    trend_assets.csv       <- 6 futures-like assets for momentum
    fx_carry.csv           <- 6 synthetic currencies: spot returns + rate differentials
    true_states.csv        <- generator regime path (teaching comparison only)
    config.json            <- pinned strategy/backtest configuration
    generate_data.py       <- seeded generator (numpy default_rng(19)); rewrites CSVs + golden
    golden/golden.json     <- 23 cross-language golden cases (87 scalars)
  python/
    src/regime/            <- reference implementation (hmm, strategies, backtest, pipeline, evaluation, costs)
    tests/                 <- pytest suite
    demo.py                <- end-to-end report
  cpp/                     <- C++17 port (namespace regime::), CMake + GoogleTest
  rust/                    <- Rust port (crate regime), cargo
  java/                    <- Java 21 port (com.quant.regime), javac + JUnit 4
  tools/check_cookbook.py  <- compiles and runs every COOKBOOK snippet against the built libraries
  tools/check_docs.py      <- fails CI if documented counts drift from the repository
  tools/evaluate_gate.py   <- regenerates docs/GATE_EVALUATION.md (--seeds N)
  tools/evaluate_band.py   <- regenerates docs/BAND_EVALUATION.md (--seeds N)
  tools/evaluate_real_data.py <- regenerates docs/REAL_DATA_EVALUATION.md (downloads French data)
  tools/evaluate_capacity.py  <- regenerates docs/CAPACITY_EVALUATION.md
```

## Build, run, test

Prerequisites on a fresh Linux clone: Python 3.11 with `numpy >= 1.24`,
`pandas >= 2.0` and `pytest`; CMake >= 3.20, a C++17 compiler and
GoogleTest (`find_package(GTest)` — e.g. `libgtest-dev` with the static
library built, or a system package that ships `GTestConfig.cmake`);
cargo (Rust 2021 edition; crates.io access for `serde`/`serde_json`, and
`rand`/`rand_distr` for tests); Java 21 with JUnit 4 at
`/usr/share/java/junit4.jar` and Hamcrest at `/usr/share/java/hamcrest.jar`
(edit the two paths at the top of `java/build.sh` and `java/test.sh` for
other layouts). The shell scripts are invoked with `bash` so the
executable bit does not matter.

### Python (reference)

```bash
cd python
PYTHONPATH=src python3 demo.py          # end-to-end demo
PYTHONPATH=src python3 -m pytest -q     # test suite (150 tests)
```

### C++

```bash
cd cpp
bash build.sh                           # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/demo                            # demo
ctest --test-dir build --output-on-failure   # tests (GoogleTest, 92 tests in one binary)
```

### Rust

```bash
cd rust
cargo build --release                   # zero warnings
cargo run --release --bin demo          # demo
cargo test --release                    # 93 integration tests
```

### Java

```bash
cd java
bash build.sh                           # javac -Xlint:all -Werror into out/
bash demo.sh                            # demo main
bash test.sh                            # JUnitCore on all *Test classes (92 tests)
```

### Cookbook snippets

```bash
python3 tools/check_cookbook.py         # after the three native builds; 75 snippets
```

## Demo output (Python reference)

```
Regime-Aware Trading — HMM + momentum + FX carry (bundled synthetic data)
------------------------------------------------------------------------------
Data: 2000 trading days, 2015-01-02 .. 2022-09-01

Model selection on the market index (diagonal Gaussian HMM):
   K       loglik          AIC          BIC  iters  conv
   2    6612.4532  -13210.9064  -13171.7001     23  True
   3    6643.0767  -13258.1533  -13179.7407     71  True

Fitted K=3 regimes (states sorted by mean; 0 = lowest):
  state      label  mean(ann)  vol(ann)  stationary
      0     choppy     -3.98%    16.05%      0.2455
      1  calm-bull    +13.51%     9.62%      0.6204
      2     crisis    +66.89%    38.69%      0.1341
  transition matrix (rows sum to 1):
    0.9327  0.0319  0.0354
    0.0191  0.9809  0.0000
    0.0347  0.0301  0.9352

Viterbi decoding vs true (generator) states: accuracy = 90.45%
  decoded days per state [377, 1397, 226]  (true calm/choppy/crisis [1304, 470, 226])
------------------------------------------------------------------------------
Strategies (cost 5 bps, gate mode 'prob', refit every 63d on expanding window):
  strategy                ann ret  ann vol  Sharpe    maxDD  Calmar    hit  worst mo
  momentum unfiltered      +9.23%   11.04%    0.84  -25.22%    0.37  46.1%    -8.52%
  momentum filtered        +6.06%    8.76%    0.69  -21.95%    0.28  43.2%    -7.63%
  carry unfiltered         +6.16%   17.06%    0.36  -61.13%    0.10  51.4%   -16.62%
  carry filtered          +11.37%   12.07%    0.94  -17.43%    0.65  48.6%   -10.05%
  combined unfiltered      +7.70%    9.86%    0.78  -36.08%    0.21  52.6%    -8.71%
  combined filtered        +8.72%    7.19%    1.21  -15.20%    0.57  49.5%    -4.97%
------------------------------------------------------------------------------
Annualized net return by decoded regime (state 0 = lowest mean):
  strategy                  choppy calm-bull     crisis
  momentum unfiltered       +7.30%    +8.95%    +14.21%
  momentum filtered         +6.23%    +5.92%     +6.63%
  carry unfiltered          -6.28%   +21.78%    -69.62%
  carry filtered            +1.19%   +16.91%     -5.90%
  combined unfiltered       +0.51%   +15.36%    -27.70%
  combined filtered         +3.71%   +11.42%     +0.36%
------------------------------------------------------------------------------
Takeaway: the figures below are full-sample and include 503 warm-up days
on which the gate is 1, so they are not evidence for the gate:
Sharpe (0.78 -> 1.21), maxDD (-36.08% -> -15.20%)
return (+7.70% -> +8.72%).
Seed 19 is favourable to the gate: its post-warm-up Sharpe gain (0.54 -> 0.99)
sits at the 85th percentile of 100 seeds, and its 95% interval includes zero.
On real US equity data, 1963-2026, the gate lost to the unfiltered book and
to a 21-day vol rule, with bootstrap intervals below zero. Tuning has stopped.
See docs/REAL_DATA_EVALUATION.md and docs/GATE_EVALUATION.md.
```

The demo reports full-sample metrics on the bundled seed, so every gate figure in
it comes with the warnings it prints. The sample includes 251 days on which
momentum holds no position and 503 days on which the gate is 1; LEARN section 10
shows how such a warm-up can reverse a conclusion. States are sorted by mean, so
crisis is the highest-variance state, here state 2, and its sampled mean is
positive on this path. The bundled carry basket changes on only 2 days, and the
re-ranking machinery is pinned by the synthetic `carry_rerank_*` golden cases
instead (API_SPEC 4). Sharpe ratios are raw (no risk-free rate subtracted).

## Golden-value cross-language testing

`data/golden/golden.json` holds 23 named cases (87 scalars): HMM
log-likelihoods, fitted K=3 means/stds/transition matrix, exact Viterbi
state counts, stationary distribution, smoothed probabilities at pinned
dates, all seven backtest metrics (ann-return, ann-vol, Sharpe, maxDD,
Calmar, hit rate, worst month) of momentum and carry filtered vs
unfiltered, combined Sharpe and maxDD, the momentum return in the
highest-variance state, mean turnovers, the portfolio scale, the band and
guard cases, and two cases on a small synthetic carry panel embedded in the
file that actually re-ranks the basket (positions exact, backtest metrics at
1e-8). Each has an absolute tolerance (`tol = 0` means exact). Every value was
produced by the Python reference running the full pinned configuration. The
goldens pin behaviour, not merit: nothing in them says the gate helps. The tests
also check semantic invariants (shallower drawdown for the combined book,
filtered momentum has a shallower drawdown but a lower return, state counts sum
to 2000, the re-rank panel flips the book at day 21).

Every language's test suite loads this one file and asserts every scalar
within its tolerance. Because initialization, iteration order, convergence
checks, the dead-state guard, label sorting, tie-breaks and rebalance
schedules are all pinned in `API_SPEC.md` and no RNG is involved, any
cross-language drift is a real bug, not noise: a port that diverges beyond
1e-6 on the K=3 log-likelihood (or 1e-8 on a Sharpe) has strayed from the
contract. Regenerate data and golden values with
`python3 data/generate_data.py` (seeded; CSVs reproduce byte for byte, golden values to about 1e-11 across numpy versions).

Test counts: Python 150 (pytest), C++ 92 (GoogleTest), Rust 93
(cargo integration tests), Java 92 (JUnit 4); plus 75 compiled-and-run
COOKBOOK snippets and the four demos.

## Real-world usage notes

What a practitioner gets, precisely:

* **Units and conventions.** Inputs are *simple daily returns* (not log
  returns), rate differentials are *annualized simple* rates accrued as
  `diff[t-1] / 252` with no compounding and no forward points, costs are
  *one-way basis points of notional turnover* charged on the trade date,
  and every annualization uses 252 days (`config.trading_days` is
  informational; the constant is compiled in). Sharpe ratios are raw
  (`ann_return / ann_vol`, population volatility, no risk-free or
  benchmark subtraction); drawdowns and worst months are compounded.
* **Timing.** `P[t]` is decided at the close of day t from data through
  t and earns `r[t+1]`; the walk-forward gate uses filtered (causal)
  probabilities under a model fitted on `r[0..t]`; full-sample fits are
  used only for model selection and attribution. There is no calendar,
  day-count, settlement (T+2), holiday or intraday logic — a "day" is a
  row, and dates are used only to define calendar months and are required
  to be strictly increasing.
* **What is validated at every public entry point** (same rules, same
  error family in all four languages — `ValueError`,
  `std::invalid_argument`, `Err(RegimeError::InvalidInput)`,
  `IllegalArgumentException`): empty inputs, NaN/inf, shape mismatches
  (data vs parameters on K and D, positions vs returns, dates vs days),
  non-stochastic parameter sets, returns `<= -1`, negative costs,
  `K < 2`, `T <= K`, invalid EM/gate settings, thresholds outside
  `[0, 1]`, state labels outside `[0, K)`, duplicated or unsorted dates,
  and a wipe-out (`net[t] <= -1`). A zero forward normalizer (data with
  zero likelihood under every reachable state) is an error, never a NaN.
  Rust never panics on library input; Java never throws
  `ArrayIndexOutOfBoundsException` from a public method.
* **What is reported, not thrown.** EM non-convergence at `max_iter`, a
  dead state (a state that lost all posterior support) and a
  non-monotone EM step are flags on the fit result (`converged`,
  `dead_state`, `monotone`); the returned parameters are always finite
  and the returned log-likelihood is always their exact score. Walk-forward
  refits keep going with a flagged model; inspect
  `models[i].fit_result` to audit a gate.
* **Parameter bounds.** `n_states >= 2` (no upper bound is enforced —
  K = 20 on 504 points will fit and over-fit), `tol > 0`, `max_iter >= 1`,
  `var_floor > 0`, `0 < ewma_lambda < 1`, `vol_target > 0`,
  `portfolio_vol_target > 0`, `leverage_cap > 0`, `1 <= top_n, bottom_n` with `top_n + bottom_n <= A`,
  `rebalance_days >= 1`, `train_min_days > n_states`, `refit_days >= 1`,
  `cost_bps >= 0` (no upper sanity bound), `0 <= gate_threshold <= 1`.
* **Known weakness, with an opt-in guard.** One corrupt print can capture
  the lowest-variance state: injecting a single -30 % or +50 % return at t=1000
  took the mean gate from 0.624 to 0.145 with every fit still reporting
  `converged`, and this is still the default behaviour. Setting
  `calm_guard = 0.01` detects the collapse, refits on a clipped window and
  records the day in `guard_trips`; the same injections then give 0.657 and
  0.630 (API_SPEC 3.3). The mean stays slightly high and days near the shock
  still differ, so clean outliers before fitting all the same.
* **Deliberately out of scope.** Missing-data handling (a NaN is
  rejected, not filled), market impact or bid/ask asymmetry in the pinned
  backtest (the opt-in `regime.costs` model adds spread and square-root impact,
  on illustrative inputs), position limits, financing, FX quote conventions,
  non-Gaussian emissions,
  multiple random restarts (the initialization is pinned for
  reproducibility, so EM reaches one deterministic local optimum), and a
  label-switching warning between refits (the calm state is
  re-identified by lowest variance at every refit instead).
* **Skipped by rule.** Student-t emissions, a P(crisis) gate and real FX carry
  were conditional experiments, skipped when tuning stopped. There is therefore
  no real-data result for the combined book.
* **What is verified.** Enumerated ground truth for the forward pass,
  smoothing and Viterbi; EM monotonicity; cross-language golden agreement
  to 1e-6..1e-8 on 87 scalars; the no-lookahead shift test; the
  robustness cases listed in `docs/ARCHITECTURE.md` section 6. That is
  evidence the code does what the spec says. The bundled data are synthetic and
  generated *from* a 3-state model, so the 90.45 % decoding accuracy is a best
  case for the method, and the real-data report is the evidence on the strategy.

## References

Works the code implements or the documentation relies on:

* L. E. Baum, T. Petrie, G. Soules and N. Weiss (1970), "A Maximization
  Technique Occurring in the Statistical Analysis of Probabilistic
  Functions of Markov Chains", *Annals of Mathematical Statistics* 41(1),
  164–171. https://doi.org/10.1214/aoms/1177697196
* L. R. Rabiner (1989), "A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition", *Proceedings of the IEEE* 77(2),
  257–286. https://doi.org/10.1109/5.18626 — scaled forward-backward
  (§V.A), Baum-Welch re-estimation, the zero-mass/parameter-floor
  discussion behind the dead-state guard (§V.B), Viterbi (§III.B).
* J. A. Bilmes (1998), "A Gentle Tutorial of the EM Algorithm and its
  Application to Parameter Estimation for Gaussian Mixture and Hidden
  Markov Models", International Computer Science Institute, TR-97-021.
  https://www.icsi.berkeley.edu/ftp/global/pub/techreports/1997/tr-97-021.pdf
  — Gaussian-emission M-step with the `t <= T-1` transition denominator.
* A. J. Viterbi (1967), "Error Bounds for Convolutional Codes and an
  Asymptotically Optimum Decoding Algorithm", *IEEE Transactions on
  Information Theory* 13(2), 260–269. https://doi.org/10.1109/TIT.1967.1054010
* J. D. Hamilton (1989), "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle", *Econometrica*
  57(2), 357–384. https://doi.org/10.2307/1912559 — filtered regime
  probabilities as the causal signal.
* H. Akaike (1974), "A New Look at the Statistical Model Identification",
  *IEEE Transactions on Automatic Control* 19(6), 716–723.
  https://doi.org/10.1109/TAC.1974.1100705
* G. Schwarz (1978), "Estimating the Dimension of a Model", *Annals of
  Statistics* 6(2), 461–464. https://doi.org/10.1214/aos/1176344136
* R. J. Hyndman and Y. Fan (1996), "Sample Quantiles in Statistical
  Packages", *The American Statistician* 50(4), 361–365.
  https://doi.org/10.1080/00031305.1996.10473566 — definition 7 (linear
  interpolation), the pinned quantile rule of the initialization.
* T. J. Moskowitz, Y. H. Ooi and L. H. Pedersen (2012), "Time Series
  Momentum", *Journal of Financial Economics* 104(2), 228–250.
  https://doi.org/10.1016/j.jfineco.2011.11.003
* B. Hurst, Y. H. Ooi and L. H. Pedersen (2017), "A Century of Evidence
  on Trend-Following Investing", *Journal of Portfolio Management* 44(1),
  15–29. https://doi.org/10.3905/jpm.2017.44.1.015
* J.P. Morgan/Reuters (1996), *RiskMetrics — Technical Document*, 4th
  edition, New York — the λ = 0.94 daily EWMA volatility convention.
  https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a
* E. F. Fama (1984), "Forward and Spot Exchange Rates", *Journal of
  Monetary Economics* 14(3), 319–338.
  https://doi.org/10.1016/0304-3932(84)90046-1
* M. K. Brunnermeier, S. Nagel and L. H. Pedersen (2008), "Carry Trades
  and Currency Crashes", *NBER Macroeconomics Annual* 23, 313–347.
  https://doi.org/10.1086/593088
* H. Lustig, N. Roussanov and A. Verdelhan (2011), "Common Risk Factors in
  Currency Markets", *Review of Financial Studies* 24(11), 3731–3777.
  https://doi.org/10.1093/rfs/hhr068 — rank-sorted carry baskets.
* W. F. Sharpe (1994), "The Sharpe Ratio", *Journal of Portfolio
  Management* 21(1), 49–58. https://doi.org/10.3905/jpm.1994.409501 —
  the ratio here omits the risk-free/benchmark term (raw Sharpe).
* D. N. Politis and J. P. Romano (1994), "The Stationary Bootstrap", *Journal
  of the American Statistical Association* 89(428), 1303–1313.
  https://doi.org/10.1080/01621459.1994.10476870 — the block bootstrap of
  `paired_block_bootstrap`.
* K. R. French, Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
  — the daily data of the real-data evaluation.
* R. Almgren, C. Thum, E. Hauptmann and H. Li (2005), "Direct Estimation of
  Equity Market Impact", *Risk* 18(7) — cited in `regime.costs` for the general
  concave shape only; no coefficient comes from it.
* C. M. Bishop (2006), *Pattern Recognition and Machine Learning*,
  Springer, chapter 13; J. D. Hamilton (1994), *Time Series Analysis*,
  Princeton University Press, chapter 22; A. Ang and G. Bekaert (2002),
  "International Asset Allocation with Regime Shifts", *Review of
  Financial Studies* 15(4), 1137–1187, https://doi.org/10.1093/rfs/15.4.1137;
  M. López de Prado (2018), *Advances in Financial Machine Learning*,
  Wiley — background reading cited in LEARN.md.

## License

MIT (see [`LICENSE`](LICENSE)). No warranty; not investment advice.
