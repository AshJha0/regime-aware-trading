# Regime-Aware Trading (Gaussian HMM, Momentum, FX Carry)

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
| Stationary distribution (pinned power method) | yes | yes | yes | yes |
| Model selection: log-likelihood, AIC, BIC (K = 2 vs 3) | yes | yes | yes | yes |
| Time-series momentum (252d signal, EWMA vol target, 4x cap) | yes | yes | yes | yes |
| FX carry (rank differentials, top/bottom 3, 21d rebalance) | yes | yes | yes | yes |
| Walk-forward regime gate (expanding window, 63d refit, warm start) | yes | yes | yes | yes |
| Backtest engine (pinned accounting, costs on turnover, full metrics) | yes | yes | yes | yes |
| Wipe-out guard, returns > -1, strictly increasing dates | yes | yes | yes | yes |
| Crisis-state attribution table | yes | yes | yes | yes |
| Golden-value test suite (20 cases / 77 scalars from `data/golden/golden.json`) | yes | yes | yes | yes |

## Directory layout

```
regime-aware-trading/
  README.md                <- this file
  LEARN.md                 <- theory: HMMs, EM, momentum, carry, regime filtering
  COOKBOOK.md              <- task-oriented recipes in all four languages (all compiled + run)
  API_SPEC.md              <- the pinned cross-language contract
  LICENSE                  <- MIT
  docs/
    ARCHITECTURE.md        <- component design, data flow, numerics, desk usage (Mermaid)
    GITHUB_PAGES.md        <- publishing guide for the docs site: https://ashjha0.github.io/regime-aware-trading/
    diagrams/*.mmd         <- raw Mermaid sources
  data/
    market_index.csv       <- equity-index daily returns (HMM observation series)
    trend_assets.csv       <- 6 futures-like assets for momentum
    fx_carry.csv           <- 6 synthetic currencies: spot returns + rate differentials
    true_states.csv        <- generator regime path (teaching comparison only)
    config.json            <- pinned strategy/backtest configuration
    generate_data.py       <- seeded generator (numpy default_rng(83)); rewrites CSVs + golden
    golden/golden.json     <- 20 cross-language golden cases (77 scalars)
  python/
    src/regime/            <- reference implementation (hmm, strategies, backtest, pipeline)
    tests/                 <- pytest suite
    demo.py                <- end-to-end report
  cpp/                     <- C++17 port (namespace regime::), CMake + GoogleTest
  rust/                    <- Rust port (crate regime), cargo
  java/                    <- Java 21 port (com.quant.regime), javac + JUnit 4
  tools/check_cookbook.py  <- compiles and runs every COOKBOOK snippet against the built libraries
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
PYTHONPATH=src python3 -m pytest -q     # test suite (74 tests)
```

### C++

```bash
cd cpp
bash build.sh                           # cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2
./build/demo                            # demo
ctest --test-dir build --output-on-failure   # tests (GoogleTest, 69 tests in one binary)
```

### Rust

```bash
cd rust
cargo build --release                   # zero warnings
cargo run --release --bin demo          # demo
cargo test --release                    # 70 integration tests
```

### Java

```bash
cd java
bash build.sh                           # javac -Xlint:all -Werror into out/
bash demo.sh                            # demo main
bash test.sh                            # JUnitCore on all *Test classes (70 tests)
```

### Cookbook snippets

```bash
python3 tools/check_cookbook.py         # after the three native builds; 56 snippets
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

The demo above reports full-sample metrics. Note that the sample includes
251 days on which momentum holds no position (before its first full
lookback window) and 503 days on which the gate is 1 (before the first
walk-forward fit); a from-first-trade window would raise the momentum
Sharpe by roughly `1/sqrt(1 - 251/2000)`, about 6.5 % relative. The
bundled carry differentials never change rank order, so the carry basket
built on day 0 is held for all 2000 days (see API_SPEC section 4); the
re-ranking machinery is pinned by the synthetic `carry_rerank_*` golden
cases instead. Sharpe ratios are raw (no risk-free rate subtracted).

## Golden-value cross-language testing

`data/golden/golden.json` holds 20 named cases (77 scalars) — HMM
log-likelihoods, fitted K=3 means/stds/transition matrix, exact Viterbi
state counts, stationary distribution, smoothed probabilities at pinned
dates, all seven backtest metrics (ann-return, ann-vol, Sharpe, maxDD,
Calmar, hit rate, worst month) of momentum and carry filtered vs
unfiltered, combined Sharpe and maxDD, the crisis-state momentum return (a
sign case: it must be negative), mean turnovers, and two cases on a small
synthetic carry panel embedded in the file that actually re-ranks the
basket (positions exact, backtest metrics at 1e-8) — each with an
absolute tolerance (`tol = 0` means exact). Every value was produced by
the Python reference running the full pinned configuration; the file also
bakes in semantic invariants (filtered Sharpe > unfiltered and shallower
drawdown for the combined book, filtered momentum has a shallower
drawdown but a lower return, state counts sum to 2000, the re-rank panel
flips the book at day 21).

Every language's test suite loads this one file and asserts every scalar
within its tolerance. Because initialization, iteration order, convergence
checks, the dead-state guard, label sorting, tie-breaks and rebalance
schedules are all pinned in `API_SPEC.md` and no RNG is involved, any
cross-language drift is a real bug, not noise: a port that diverges beyond
1e-6 on the K=3 log-likelihood (or 1e-8 on a Sharpe) has strayed from the
contract. Regenerate data and golden values with
`python3 data/generate_data.py` (seeded; byte-for-byte reproducible).

Test counts (all green): Python 74 (pytest), C++ 69 (GoogleTest), Rust 70
(cargo integration tests), Java 70 (JUnit 4); plus 56 compiled-and-run
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
  `leverage_cap > 0`, `1 <= top_n, bottom_n` with `top_n + bottom_n <= A`,
  `rebalance_days >= 1`, `train_min_days > n_states`, `refit_days >= 1`,
  `cost_bps >= 0` (no upper sanity bound), `0 <= gate_threshold <= 1`.
* **Deliberately out of scope.** Missing-data handling (a NaN is
  rejected, not filled), market impact or bid/ask asymmetry, position
  limits, financing, FX quote conventions, non-Gaussian emissions,
  multiple random restarts (the initialization is pinned for
  reproducibility, so EM reaches one deterministic local optimum), and a
  label-switching warning between refits (the calm state is
  re-identified by lowest variance at every refit instead).
* **What is verified.** Enumerated ground truth for the forward pass,
  smoothing and Viterbi; EM monotonicity; cross-language golden agreement
  to 1e-6..1e-8 on 77 scalars; the no-lookahead shift test; the
  robustness cases listed in `docs/ARCHITECTURE.md` section 6. The
  bundled data are synthetic and generated *from* a 3-state model, so the
  94.85 % decoding accuracy and the Sharpe improvements are a best case
  for the method, not an expectation for real markets.

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
* C. M. Bishop (2006), *Pattern Recognition and Machine Learning*,
  Springer, chapter 13; J. D. Hamilton (1994), *Time Series Analysis*,
  Princeton University Press, chapter 22; A. Ang and G. Bekaert (2002),
  "International Asset Allocation with Regime Shifts", *Review of
  Financial Studies* 15(4), 1137–1187, https://doi.org/10.1093/rfs/15.4.1137;
  M. López de Prado (2018), *Advances in Financial Machine Learning*,
  Wiley — background reading cited in LEARN.md.

## License

MIT (see [`LICENSE`](LICENSE)). No warranty; not investment advice.
