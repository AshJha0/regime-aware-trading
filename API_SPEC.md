# regime-aware-trading — API contract (pinned, all languages)

This file is the normative contract for every implementation (Python
reference `python/src/regime`, C++ `regime::`, Rust crate `regime`, Java
`com.quant.regime`). Anything pinned here must agree across languages to
the golden tolerances in `data/golden/golden.json`. All computations are
double precision, deterministic, and **RNG-free** (the only RNG in library
code is the seeded data generator `data/generate_data.py`, SEED = 83; test
fixtures may draw their own seeded random inputs).

---

## 1. Gaussian HMM (diagonal covariance, K >= 2, D in {1, 2})

Observations `x_t` are vectors of dimension D (a 1-D series is treated as
D = 1). Emissions are independent per dimension:

```
log b_t(i) = sum_{d=1..D} log N(x_td ; mu_id, var_id)
           = -0.5 * sum_d [ log(2*pi) + log(var_id) + (x_td - mu_id)^2 / var_id ]
```

### 1.1 Scaled forward-backward (pinned: scaled, NOT log-space)

Per-step emission shift guards underflow of the densities themselves:
`m_t = max_i log b_t(i)`, `bt_t(i) = exp(log b_t(i) - m_t)` (all <= 1).

Forward (with normalizers `d_t`):

```
a~_1(i) = pi_i * bt_1(i)                       d_1 = sum_i a~_1(i)
a^_1(i) = a~_1(i) / d_1
a~_t(j) = [ sum_i a^_{t-1}(i) * A_ij ] * bt_t(j)
d_t     = sum_j a~_t(j)
a^_t(j) = a~_t(j) / d_t
```

The true normalization constant is `c_t = d_t * exp(m_t)` and the
**log-likelihood is stored as the sum of the log normalizers**:

```
log L = sum_{t=1..T} ( log d_t + m_t )        # == sum_t log c_t
```

Backward, sharing the same `d_t`:

```
b^_T(i) = 1
b^_t(i) = [ sum_j A_ij * bt_{t+1}(j) * b^_{t+1}(j) ] / d_{t+1}
```

Posteriors (no further normalization needed):

```
gamma_t(i) = a^_t(i) * b^_t(i)                          # smoothed P(s_t=i | x_1..T)
xi_t(i,j)  = a^_t(i) * A_ij * bt_{t+1}(j) * b^_{t+1}(j) / d_{t+1}
```

Filtered probabilities are the `a^_t` rows: `P(s_t = i | x_1..t)`.

### 1.2 Baum-Welch EM updates (pinned)

```
pi_i    <- gamma_1(i)
A_ij    <- sum_{t=1..T-1} xi_t(i,j) / sum_{t=1..T-1} gamma_t(i)
mu_id   <- sum_t gamma_t(i) x_td   / sum_t gamma_t(i)
var_id  <- sum_t gamma_t(i) (x_td - mu_id_new)^2 / sum_t gamma_t(i)
var_id  <- max(var_id, var_floor)              # variance floor, every iteration
```

`var_floor = 1e-8` (config `var_floor`). The floor is also applied to the
initial variances. Variance update uses the **new** means.

### 1.3 Pinned initialization (no RNG)

For K states on data of T observations (require `T > K`, else error):

* **Means**: per dimension d, `mu_kd = quantile(x_.d, q_k)` at
  `q_k = (2k + 1) / (2K)`, k = 0..K-1, using **linear-interpolation**
  quantiles (numpy default: index `h = q*(n-1)` on the sorted sample,
  `v = x[floor(h)] + (h - floor(h)) * (x[floor(h)+1] - x[floor(h)])`).
* **Variances**: every state gets the full-sample variance per dimension,
  `var(x_.d)` with **ddof = 0**, floored at `var_floor`.
* **Transitions**: `A_ii = 0.9`, `A_ij = 0.1 / (K - 1)` for i != j.
* **Initial distribution**: uniform `pi_i = 1/K`.

Walk-forward refits (section 3.3) **warm-start** from the previous fitted
(already label-sorted) parameter set instead of the pinned init.

### 1.4 Convergence, monotonicity and dead-state semantics (pinned)

`tol = 1e-8` on the log-likelihood, `max_iter = 500`. Loop:

1. E-step scores the current parameters -> `ll_k` (append to history).
   Every forward normalizer must satisfy `d_t > 0`; a zero normalizer
   (the observation has zero likelihood under every reachable state) is
   an **error** (1.9), never a NaN, in `fit`, every inference call and
   `forward_step`.
2. If `k >= 2`, classify the step (`em_step_status(ll_k, ll_{k-1}, tol)`):
   * `ll_k - ll_{k-1} < -MONOTONE_REL_TOL * max(1, |ll_{k-1}|)` with
     `MONOTONE_REL_TOL = 1e-6` (or either score non-finite) is
     **NON_MONOTONE**: EM provably never decreases the log-likelihood, so
     this is a numerical fault. Stop; `converged = false`,
     `monotone = false`; the returned parameters are the ones just scored.
   * else `ll_k - ll_{k-1} < tol` is **CONVERGED**: stop; the returned
     parameters are the ones just scored (no further M-step).
   * else **CONTINUE**.
3. Backward pass, posteriors `gamma`. **Dead-state guard**: compute the
   transition-row support `support_i = sum_{t=1..T-1} gamma_t(i)` (the
   denominator of the `A` update, and a lower bound on the `mu`/`var`
   denominators). If any `support_i < DEAD_STATE_SUPPORT = 1e-12` the
   state has lost posterior mass (Rabiner 1989 §V.B; Bilmes 1998 §4) and
   the M-step would produce `0/0`. Stop **before** the M-step:
   `converged = false`, `dead_state = i` (the lowest such `i`; `None`/`-1`
   otherwise), the returned parameters are the ones just scored. This is
   a flag, never an exception, and no NaN ever leaves `fit`.
4. Otherwise M-step, continue.

If `max_iter` E-steps run without any of the stops above:
**converged = false** (a flag, never an exception), and the returned
log-likelihood is one extra E-step score of the final parameters so it
always matches the returned parameter set. `n_iter` = number of E-steps
that appended to the history (the consistency re-score is **not**
counted: with `max_iter = 3`, `n_iter == 3` and `max_iter + 1` forward
passes ran). `log_likelihood` is always finite and always the score of
exactly the returned parameter set. The history is non-decreasing up to
the monotonicity tolerance (tested on the bundled data with slack 1e-9).

Walk-forward refits (3.3) do **not** stop on a dead state or a
non-monotone step: the flagged model (finite parameters) is used and kept
in the returned model list, where `fit_result` exposes the flags.

### 1.5 State label ordering (pinned)

After EM finishes (once, not during iterations), states are permuted so
`means[:, 0]` is **ascending** (stable sort; state 0 = lowest mean on
dimension 0). `pi`, rows+columns of `A`, means and variances are permuted
together. All golden HMM values use these labels.

### 1.6 Viterbi

Log-space dynamic programming with `log A`, `log pi`, `log b`. Ties broken
toward the **lower state index** (first argmax). Returns the state path.

### 1.7 Stationary distribution (pinned: power method)

Start `pi = (1/K, ..., 1/K)`; iterate `pi <- pi @ A`, renormalize by the
L1 sum each step; stop when `max_i |pi_new_i - pi_i| < tol` with the
pinned defaults `tol = 1e-13`, `max_iter = 10000` (both exposed as
arguments in every language; `tol <= 0` or `max_iter < 1` is an error).
Result satisfies `pi P = pi` to 1e-10 (tested). If the cap is reached
without convergence (a periodic or reducible `A`) the call is an
**error** — an unconverged vector is never returned as stationary.

### 1.8 Model selection

Free parameter count for K states, D dims, diagonal covariance:
`p = 2*K*D + K*(K-1) + (K-1)`.
`AIC = -2 logL + 2p`, `BIC = -2 logL + p ln T`. Lower is better.

### 1.9 HMM error behavior

Errors (Python `ValueError`; C++ `std::invalid_argument`; Rust
`Err(RegimeError::InvalidInput)` — never a panic; Java
`IllegalArgumentException` — never an `ArrayIndexOutOfBoundsException`)
for: `K < 2` (K = 1 is degenerate), empty observations, `D > 2`, NaN/inf
observations, `T <= K`, non-positive `tol`/`var_floor`, `max_iter < 1`,
inference calls before fitting, a zero forward normalizer (1.4), and
**parameter/data shape or stochasticity mismatch**:

* `validate_params(params, K, D)` (public in every language) requires
  `startprob.shape == (K,)`, `transmat.shape == (K, K)`,
  `means.shape == variances.shape == (K, D)`, all entries finite,
  `variances > 0`, `startprob >= 0` with `|sum - 1| <= 1e-9`, every row
  of `transmat >= 0` with `|row sum - 1| <= 1e-9` (`STOCHASTIC_TOL`).
* It runs at the top of `score`, `filtered_probabilities`,
  `smoothed_probabilities`, `viterbi`, `aic`, `bic`,
  `stationary_distribution` (with `K = n_states`, `D = X.shape[1]`), on
  the warm start in `fit(X, init)`, in `set_params` (C++/Rust/Java), and
  in `forward_step` (with `K = len(startprob)`, `D = means.shape[1]`).
* `forward_step` additionally requires `len(alpha_prev) == K` with
  `alpha_prev` finite, non-negative and summing to 1 (within 1e-9), and
  `len(x_t) == D` finite.

EM non-convergence, a dead state and a non-monotone step are NOT errors
(1.4). K larger than the distinct-value support of the data must not
crash: the variance floor engages (tested).

---

## 2. No-lookahead accounting (backtest engine, pinned)

Positions `P[t, a]` are **decided at the close of day t** with information
up to and including day t. For daily simple returns `r[t, a]`:

```
gross[t]    = sum_a P[t-1, a] * r[t, a]         for t >= 1;  gross[0] = 0
turnover[t] = sum_a |P[t, a] - P[t-1, a]|       with P[-1, a] = 0
net[t]      = gross[t] - (cost_bps / 10000) * turnover[t]
equity[t]   = prod_{u <= t} (1 + net[u])
```

The trade into `P[t]` executes at the close of day t and its cost is
charged on day t (so `net[0] = -cost * sum|P[0]|`). `cost_bps = 0` gives
`net == gross` exactly. Shifting signals by one day must change the
result (tested).

**Wipe-out guard (pinned)**: the accounting is linear in returns, so a
leveraged book can lose more than 100 % in a day. If any `net[t] <= -1`
the backtest is an **error** whose message names the day (`"equity wiped
out on day t=..."`); no equity curve, drawdown or Sharpe is ever reported
for a book that no longer exists. Consequently every accepted result has
`equity > 0` and `max_dd >= -1`. `compute_metrics` applies the same check
to its input series, and `max_drawdown` requires a finite, strictly
positive equity curve.

### 2.1 Metrics (pinned formulas; full sample; 252 days/year)

```
ann_return  = 252 * mean(net)
ann_vol     = sqrt(252) * std(net, ddof=0)
sharpe      = ann_return / ann_vol              (0 if ann_vol == 0)
max_dd      = min_t ( equity[t] / cummax(equity)[t] - 1 )     # <= 0
calmar      = ann_return / |max_dd|             (0 if max_dd == 0)
hit_rate    = mean( net > 0 )
worst_month = min over calendar months of ( prod(1 + net_in_month) - 1 )
              (fallback without dates: consecutive 21-day blocks)
```

`sharpe` is a **raw** ratio: no risk-free rate or benchmark is
subtracted, and `ann_vol` is the population (`ddof = 0`) volatility.

**Dates (pinned)**: when `dates` are given they must have length `T`,
be well-formed calendar dates (Python: anything `pandas.to_datetime`
parses; C++: `YYYY-MM-DD` strings; Rust: `Date` with month 1..12, day
1..31; Java: non-null `LocalDate`) and be **strictly increasing**;
duplicates or out-of-order rows are an error. A calendar month is
therefore a contiguous run of equal `(year, month)`; all four languages
compound those runs. Without dates the fallback uses `floor(T / 21)`
consecutive 21-day blocks and **drops the trailing partial block**
(`T < 21` gives a single block covering everything).

### 2.2 Crisis-state attribution

Day-t net return is attributed to the day-t **full-sample K=3 Viterbi
state** of the market index. Per state: `n_days`,
`ann_return = 252 * mean`, `ann_vol = sqrt(252) * std(ddof=0)`, `sharpe`.
Crisis = state 0 (lowest mean; also the highest-variance state on the
bundled data), calm-bull = the lowest-variance state. The table always
has 3 rows (the K of that fit), whatever `n_states` the walk-forward gate
uses. `state_conditional_returns(net, states, K)` requires `K >= 1`,
finite `net`, and every label in `[0, K)` — labels are never silently
dropped.

### 2.3 Backtest error behavior

Errors (same type family as 1.9) for: shape mismatch between positions
and returns, empty input (no days **or no assets**), NaN/inf, a return
`<= -1` (3), negative or non-finite `cost_bps`, invalid dates (2.1), and a
wipe-out (2).

---

## 3. Strategies (all decide `P[t]` from data through day t)

**Simple-return domain (pinned)**: every *return* input (`momentum_positions`,
`ewma_variance`, `carry_total_returns` spot returns, `run_backtest`
returns) must be finite **and `> -1`**; a -100 % (or worse) print is
corrupt data (a zero price, an unadjusted split) that would turn every
trailing growth ratio into `0/0` and is rejected with the day and asset
index in the message. Rate differentials, positions and gate values are
not simple returns and carry no such bound.

### 3.1 Time-series momentum

Config: `momentum_lookback = 252`, `vol_target = 0.10` (annualized),
`ewma_lambda = 0.94`, `leverage_cap = 4.0`. Require `T > lookback`
(else error).

* **Signal** (defined for `t >= lookback - 1`): sign of the trailing
  cumulative simple return `prod_{u=t-lookback+1..t} (1 + r[u]) - 1`,
  in {-1, 0, +1}.
* **EWMA variance** (pinned init): at `t = lookback - 1`,
  `sig2 = mean(r[0..lookback-1]^2)` (zero-mean convention); for
  `t >= lookback`: `sig2[t] = lambda * sig2[t-1] + (1 - lambda) * r[t]^2`.
* **Sizing**: `ann_vol = sqrt(252 * sig2)`;
  `lev = min(vol_target / ann_vol, leverage_cap)`; if `ann_vol == 0`
  then `lev = leverage_cap`.
* **Position**: `P[t, a] = signal[t, a] * lev[t, a] / A` (A = number of
  assets); `P[t, a] = 0` for `t < lookback - 1`.

### 3.2 FX carry

Config: `carry_top_n = 3`, `carry_bottom_n = 3`, `rebalance_days = 21`.
Require `1 <= top_n`, `1 <= bottom_n`, `top_n + bottom_n <= A`,
`rebalance_days >= 1`.

* **Schedule (pinned)**: rebalance at day indices `0, 21, 42, ...`
  (`t % rebalance_days == 0`); positions held unchanged in between. A
  `rebalance_days >= T` simply keeps the day-0 book for the whole sample
  (no error).
* **Ranking**: sort currencies by that day's annualized rate differential,
  **descending**; **ties broken by ascending asset index** (equivalently:
  stable sort on the key `(-diff, index)`). Top `top_n` get `+1/top_n`
  each; bottom `bottom_n` get `-1/bottom_n` each; the rest 0.
* **Traded return (pinned accrual)**: currency total return on day t is
  `spot_ret[t] + rate_diff[t-1] / 252` (differential set at the previous
  close; day 0 accrues nothing).

### 3.3 Regime filter (walk-forward HMM gate)

Config: `n_states = 3`, `train_min_days = 504`, `refit_days = 63`,
`gate_mode = "prob"` (alternative `"binary"`), `gate_threshold = 0.5`.
Fitted on the market-index return series. Require
`T >= train_min_days > n_states`, `refit_days >= 1`,
`0 <= gate_threshold <= 1`, and valid HMM settings (`n_states >= 2`,
`tol > 0`, `max_iter >= 1`, `var_floor > 0`) — all checked **before** the
first fit; unknown mode is an error. A refit or forward step that fails
re-raises the same error type with the day index prepended
(`"regime_gate refit at t=...: ..."`).

* **Refit schedule (pinned)**: first fit at `t0 = train_min_days - 1` on
  the expanding window `r[0..t0]`; refits at `t = t0 + refit_days * k`
  on `r[0..t]`. First fit uses the pinned init (1.3); each later refit
  **warm-starts from the previous fitted, label-sorted parameters**.
* **Gate state (pinned)**: the low-vol/bull state is
  `k* = argmin_k variances[k, 0]` of the current model (lowest variance;
  ties -> lower label). Variance, not mean, identifies the benign regime
  robustly.
* **Gate value**: `p_calm(t) = P(s_t = k* | r_0..r_t)` — the **filtered**
  probability under the current model. At a refit day it comes from a
  full scaled forward pass over `r[0..t]`; between refits it is advanced
  by single forward steps (identical math, no recomputation, no
  lookahead). For `t < t0`: `gate[t] = 1`.
* `mode "prob"`: `gate[t] = p_calm(t)`;
  `mode "binary"`: `gate[t] = 1 if p_calm(t) >= gate_threshold else 0`.
* **Application**: `P_filtered[t, a] = gate[t] * P[t, a]` (both are day-t
  decisions; the accounting of section 2 then applies them to day t+1).

### 3.4 Combined portfolio

`net_combined[t] = 0.5 * net_momentum[t] + 0.5 * net_carry[t]` (each leg
already net of its own costs), same for the filtered variants. Metrics per
section 2.1.

---

## 4. Bundled data (`data/`)

Generated by `data/generate_data.py` (numpy `default_rng(83)`, T = 2000
business days from 2015-01-02) from a true 3-state chain
(0 = calm-bull, 1 = choppy, 2 = crisis; see the generator for the exact
DGP constants). Files, all with a leading `date` column (YYYY-MM-DD),
floats printed with `%.10f`:

| file | columns |
|---|---|
| `market_index.csv` | `ret` — index daily simple returns |
| `trend_assets.csv` | `asset_0 .. asset_5` — daily returns |
| `fx_carry.csv` | `spot_ret_0..5`, `rate_diff_0..5` (annualized) |
| `true_states.csv` | `state` in {0,1,2} — teaching comparison only |
| `config.json` | all pinned parameters (keys as quoted in this spec) |

`config.json` keys: `n_states`, `em_tol`, `em_max_iter`, `var_floor`,
`momentum_lookback`, `vol_target`, `ewma_lambda`, `leverage_cap`,
`carry_top_n`, `carry_bottom_n`, `rebalance_days`, `train_min_days`,
`refit_days`, `gate_mode`, `gate_threshold`, `cost_bps`, `trading_days`.
Integer keys must hold integral values (`3.0` is rejected by every
loader; `trading_days` is informational — the annualization constant 252
is compiled into every port). `true_states.csv` labels must be integers
in `[0, 2]` (checked at load).

On the bundled data the six rate differentials **never change rank
order** (OU noise 0.0004 against base gaps >= 1.5 %), so the carry basket
built on day 0 is held for all 2000 days: `turnover_carry` is exactly
`2 / 2000`. The rebalance/ranking machinery is therefore pinned by the
synthetic `carry_rerank_*` cases below, not by the bundled panel.

**Viterbi-vs-true mapping (pinned, teaching only)**: true labels map to
fitted labels via volatility — calm-bull -> `argmin_k variances[k,0]`,
crisis -> `argmax_k variances[k,0]`, choppy -> the remaining label.
Accuracy = fraction of days where the full-sample K=3 Viterbi path equals
the mapped true state.

---

## 5. Golden cases (`data/golden/golden.json`)

Schema: `{"cases": [{"name", "inputs": {}, "expect": {flat scalar keys},
"tol"}]}`. All values computed by the Python reference on the bundled
data with the bundled `config.json` (full pinned config — test suites in
other languages may shorten walk-forward loops for speed elsewhere, but
golden comparisons must run the pinned config). Every language's test
suite loads this file and asserts each `expect` key within `tol`
(absolute); `tol = 0` means exact (integers).

| case | expect keys | tol |
|---|---|---|
| `hmm_k2_loglik` | `loglik` | 1e-6 |
| `hmm_k3_loglik` | `loglik` | 1e-6 |
| `hmm_k3_means` | `mean0..mean2` | 1e-5 |
| `hmm_k3_stds` | `std0..std2` (sqrt of variances) | 1e-5 |
| `hmm_k3_transmat` | `a00..a22` (row-major) | 1e-5 |
| `hmm_k3_viterbi_counts` | `count0..count2` (days per state) | 0 (exact) |
| `hmm_k3_stationary` | `pi0..pi2` | 1e-8 |
| `hmm_k3_smoothed_t500` | `p0..p2` at day index 500 | 1e-8 |
| `hmm_k3_smoothed_t1500` | `p0..p2` at day index 1500 | 1e-8 |
| `viterbi_accuracy_vs_true` | `accuracy` (fraction) | 1e-8 |
| `momentum_unfiltered` | `ann_return`, `ann_vol`, `sharpe`, `max_dd`, `calmar`, `hit_rate`, `worst_month` | 1e-8 |
| `momentum_filtered` | same seven metrics | 1e-8 |
| `carry_unfiltered` | same seven metrics | 1e-8 |
| `carry_filtered` | same seven metrics | 1e-8 |
| `combined_sharpe` | `sharpe_unfiltered`, `sharpe_filtered`, `max_dd_unfiltered`, `max_dd_filtered` | 1e-8 |
| `crisis_momentum_return` | `ann_return` (must be **negative**) | 1e-8 |
| `turnover_momentum` | `mean_turnover` (mean daily, unfiltered) | 1e-8 |
| `turnover_carry` | `mean_turnover` (mean daily, unfiltered) | 1e-8 |
| `carry_rerank_positions` | `mean_turnover`, `turnover_day21`, `turnover_day42`, `pos_day{15,21,42}_asset{0,3}` | 0 (exact) |
| `carry_rerank_backtest` | `ann_return`, `sharpe`, `max_dd` | 1e-8 |

20 cases, 77 scalars; every language's golden test compares all 77.

**`carry_rerank_*` (synthetic panel in `inputs`)**: the only two cases
whose `inputs` are non-empty. `inputs = {n_days: 63, n_assets: 4,
switch_days: [0, 10, 42], diffs: [[0.04, 0.03, 0.02, 0.01],
[0.01, 0.02, 0.03, 0.04], [0.02, 0.02, 0.02, 0.02]], top_n: 1,
bottom_n: 1, rebalance_days: 21, cost_bps: 5.0}`. The panel is
`diff[t] = diffs[k]` with `k` the last index whose `switch_days[k] <= t`;
spot returns are zero. Every port rebuilds the panel from `inputs`, runs
`carry_positions` -> `carry_total_returns` -> `run_backtest` and compares.
The ranks cross at day 10 but the pinned schedule re-ranks only at day 21
(`pos_day15_asset0 = +1`, `pos_day21_asset0 = -1`, `turnover_day21 = 4`),
and the all-tied row at day 42 exercises the index tie-break
(`pos_day42_asset0 = +1`). Hand-derivable: gross accrual
`20 * 0.03 / 252`, costs `10 * 5e-4`, `ann_return = 252 * net / 63`.

Semantic invariants baked into the golden set (tested): crisis momentum
return < 0; `sharpe_filtered > sharpe_unfiltered` and
`max_dd_filtered > max_dd_unfiltered` for the combined portfolio;
filtered momentum has a smaller (less negative) max_dd but a lower
ann_return than unfiltered; `pi0+pi1+pi2 = 1`;
`count0+count1+count2 = 2000`; the re-rank panel really flips the book at
day 21.

---

## 6. Determinism requirements

* No RNG anywhere in fitting, strategies, or accounting.
* Identical inputs -> bitwise-identical Python results (tested); across
  languages, agreement to the golden tolerances above.
* Summation order inside a pass is left-to-right over time; vectorized
  reductions may differ in rounding, which the tolerances absorb.
