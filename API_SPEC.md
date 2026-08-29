# P11 Regime-Aware Trading — API contract (pinned, all languages)

This file is the normative contract for every implementation (Python
reference `python/src/regime`, C++ `regime::`, Rust crate `regime`, Java
`com.quant.regime`). Anything pinned here must agree across languages to
the golden tolerances in `data/golden/golden.json`. All computations are
double precision, deterministic, and **RNG-free** (the only RNG in the
project is the seeded data generator `data/generate_data.py`, SEED = 83).

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

### 1.4 Convergence semantics (pinned)

`tol = 1e-8` on the log-likelihood, `max_iter = 500`. Loop:

1. E-step scores the current parameters -> `ll_k` (append to history).
2. If `k >= 2` and `ll_k - ll_{k-1} < tol`: **converged**, stop; the
   returned parameters are the ones just scored (no further M-step).
3. Otherwise M-step, continue.

If `max_iter` E-steps run without convergence: **converged = false**
(a flag on the result — never an exception), and the returned
log-likelihood is one extra E-step score of the final parameters so it
always matches the returned parameter set. `n_iter` = number of E-steps.
The history is non-decreasing (EM monotonicity — tested).

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
L1 sum each step; stop when `max_i |pi_new_i - pi_i| < 1e-13` or after
10000 iterations. Result satisfies `pi P = pi` to 1e-10 (tested).

### 1.8 Model selection

Free parameter count for K states, D dims, diagonal covariance:
`p = 2*K*D + K*(K-1) + (K-1)`.
`AIC = -2 logL + 2p`, `BIC = -2 logL + p ln T`. Lower is better.

### 1.9 HMM error behavior

Errors (Python `ValueError`; C++ `std::invalid_argument`; Rust
`Err(RegimeError)`; Java `IllegalArgumentException`) for: `K < 2`
(K = 1 is degenerate), empty observations, `D > 2`, NaN/inf observations,
`T <= K`, non-positive `tol`/`var_floor`, `max_iter < 1`, and inference
calls before fitting. EM non-convergence is NOT an error (1.4). K larger
than the distinct-value support of the data must not crash: the variance
floor engages (tested).

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

### 2.2 Crisis-state attribution

Day-t net return is attributed to the day-t **full-sample K=3 Viterbi
state** of the market index. Per state: `n_days`,
`ann_return = 252 * mean`, `ann_vol = sqrt(252) * std(ddof=0)`, `sharpe`.
Crisis = state 0 (lowest mean; also the highest-variance state on the
bundled data), calm-bull = the lowest-variance state.

### 2.3 Backtest error behavior

Errors for: shape mismatch between positions and returns, empty input,
NaN/inf, negative `cost_bps`, dates length mismatch.

---

## 3. Strategies (all decide `P[t]` from data through day t)

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
`T >= train_min_days > n_states` (else error); unknown mode is an error.

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
| `momentum_unfiltered` | `ann_return`, `sharpe`, `max_dd` | 1e-8 |
| `momentum_filtered` | `ann_return`, `sharpe`, `max_dd` | 1e-8 |
| `carry_unfiltered` | `ann_return`, `sharpe`, `max_dd` | 1e-8 |
| `carry_filtered` | `ann_return`, `sharpe`, `max_dd` | 1e-8 |
| `combined_sharpe` | `sharpe_unfiltered`, `sharpe_filtered` | 1e-8 |
| `crisis_momentum_return` | `ann_return` (must be **negative**) | 1e-8 |
| `turnover_momentum` | `mean_turnover` (mean daily, unfiltered) | 1e-8 |
| `turnover_carry` | `mean_turnover` (mean daily, unfiltered) | 1e-8 |

Semantic invariants baked into the golden set (tested): crisis momentum
return < 0; `sharpe_filtered > sharpe_unfiltered` for the combined
portfolio; filtered momentum has a smaller (less negative) max_dd but a
lower ann_return than unfiltered; `pi0+pi1+pi2 = 1`;
`count0+count1+count2 = 2000`.

---

## 6. Determinism requirements

* No RNG anywhere in fitting, strategies, or accounting.
* Identical inputs -> bitwise-identical Python results (tested); across
  languages, agreement to the golden tolerances above.
* Summation order inside a pass is left-to-right over time; vectorized
  reductions may differ in rounding, which the tolerances absorb.
