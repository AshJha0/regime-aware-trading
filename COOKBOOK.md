# COOKBOOK — regime recipes

Task-oriented recipes for the `regime` toolkit. Recipes 1 to 18 are in all
four languages; 19 to 21 use the Python-only evaluation and cost code. Every
snippet below is a complete program body that compiles
and runs against the real library — `tools/check_cookbook.py` extracts
each fenced block, builds it against the built library and runs it (see
the end of this file). Snippets locate the bundled data as `../data`, i.e.
they run from the `python/`, `cpp/`, `rust/` or `java/` directory.

* **Python**: `cd python && PYTHONPATH=src python3 snippet.py`.
* **C++**: namespace `regime`, headers `<regime/*.hpp>`; series are
  `std::vector<double>`, panels are the row-major `regime::Matrix`
  (`m(t, a)`, `m.rows`, `m.cols`; `regime::to_matrix(vec)` wraps a series
  as a `T x 1` matrix). Errors are `std::invalid_argument`.
* **Rust**: crate `regime`; panels are `regime::Matrix` (`m.get(t, a)`,
  `m.rows()`, `m.cols()`, `Matrix::column(&series)`); fallible functions
  return `regime::Result<_>` — the snippets are the body of
  `fn main() -> regime::Result<()>`.
* **Java**: package `com.quant.regime`; series are `double[]`, panels are
  `double[][]`; errors are `IllegalArgumentException`. The dataset is the
  nested record `Pipeline.Dataset`, backtest results are `Backtest.Result`.

Numbers quoted in comments come from the bundled data: golden values
(`data/golden/golden.json`) or the output of the snippet. Recipes 7 to 13 run
the strategies at their raw sizes, before the portfolio scale of API_SPEC 3.5,
so their Sharpe ratios differ from the demo.

**Read this before reusing a recipe.** The code works; the idea it serves did
not. On real US equity data from 1963 to 2026 the HMM gate lost to the
unfiltered book and to a 21-day vol rule, with bootstrap intervals below zero
(`docs/REAL_DATA_EVALUATION.md`), so tuning has stopped
(`docs/PREREGISTRATION.md`). The bundled seed 19 flatters the gate: post-warm-up
Sharpe 0.54 to 0.99 sits at the 85th percentile of 100 seeds and the interval
includes zero (`docs/GATE_EVALUATION.md`). Gate figures below show how the API
behaves. They are not evidence that the gate works.

---

## 1. How do I load the bundled dataset?

**Python**

```python
from regime import load_dataset

ds = load_dataset("../data")
print(len(ds.index_returns), "days,", ds.trend_returns.shape[1], "trend assets")
print("config:", ds.config["n_states"], "states,", ds.config["cost_bps"], "bps costs")
# 2000 days, 6 trend assets
```

**C++**

```cpp
#include <iostream>
#include <regime/pipeline.hpp>

regime::Dataset ds = regime::load_dataset("../data");
std::cout << ds.index_returns.size() << " days, " << ds.trend_returns.cols
          << " trend assets\n";                                   // 2000 days, 6 trend assets
std::cout << "config: " << ds.config.n_states << " states, " << ds.config.cost_bps
          << " bps costs\n";
```

**Rust**

```rust
use regime::load_dataset;

let ds = load_dataset("../data")?;
println!("{} days, {} trend assets", ds.index_returns.len(), ds.trend_returns.cols());
println!("config: {} states, {} bps costs", ds.config.n_states, ds.config.cost_bps);
```

**Java**

```java
import com.quant.regime.Pipeline;
import java.nio.file.Path;

Pipeline.Dataset ds = Pipeline.loadDataset(Path.of("../data"));
System.out.println(ds.indexReturns().length + " days, "
        + ds.trendReturns()[0].length + " trend assets");        // 2000 days, 6 trend assets
System.out.println("config: " + ds.config().nStates() + " states, "
        + ds.config().costBps() + " bps costs");
```

---

## 2. How do I fit a Gaussian HMM and read off the regimes?

States come back sorted by mean ascending (pinned), so a label says nothing
about risk. On this data state 0 is the choppy regime and state 2, the
highest-variance state, is the crisis regime, with a positive sampled mean.
Identify crisis by `argmax` of the variance, and calm by `argmin`.

**Python**

```python
import numpy as np
from regime import GaussianHMM, load_dataset

r = load_dataset("../data").index_returns
hmm = GaussianHMM(n_states=3, tol=1e-8, max_iter=500, var_floor=1e-8)
res = hmm.fit(r)
print(f"loglik {res.log_likelihood:.6f} in {res.n_iter} iters, converged={res.converged}")
for k in range(3):                       # golden: means -0.000158, 0.000536, 0.002654
    mu, sd = hmm.params.means[k, 0], np.sqrt(hmm.params.variances[k, 0])
    print(f"state {k}: ann mean {252*mu:+.1%}  ann vol {np.sqrt(252)*sd:.1%}")
# loglik 6643.076654 (golden hmm_k3_loglik)
```

**C++**

```cpp
#include <cmath>
#include <iostream>
#include <regime/hmm.hpp>
#include <regime/pipeline.hpp>

const regime::Matrix r = regime::to_matrix(regime::load_dataset("../data").index_returns);
regime::GaussianHMM hmm(3, /*tol=*/1e-8, /*max_iter=*/500, /*var_floor=*/1e-8);
const regime::HMMFitResult res = hmm.fit(r);
std::cout << "loglik " << res.log_likelihood << " in " << res.n_iter
          << " iters, converged=" << res.converged << "\n";       // 6643.08, 71, 1
for (std::size_t k = 0; k < 3; ++k)
    std::cout << "state " << k << ": mean " << hmm.params().means(k, 0)
              << " std " << std::sqrt(hmm.params().variances(k, 0)) << "\n";
```

**Rust**

```rust
use regime::{load_dataset, GaussianHmm, Matrix};

let r = Matrix::column(&load_dataset("../data")?.index_returns);
let mut hmm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?;
let res = hmm.fit(&r, None)?;
println!("loglik {:.6} in {} iters, converged={}", res.log_likelihood, res.n_iter, res.converged);
let p = hmm.params().expect("fitted");
for k in 0..3 {
    println!("state {k}: mean {:+.6} std {:.6}", p.means.get(k, 0), p.variances.get(k, 0).sqrt());
}
```

**Java**

```java
import com.quant.regime.GaussianHmm;
import com.quant.regime.HmmFitResult;
import com.quant.regime.Pipeline;
import java.nio.file.Path;

double[] r = Pipeline.loadDataset(Path.of("../data")).indexReturns();
GaussianHmm hmm = new GaussianHmm(3, 1e-8, 500, 1e-8);
HmmFitResult res = hmm.fit(r);
System.out.printf("loglik %.6f in %d iters, converged=%b%n",
        res.logLikelihood(), res.nIter(), res.converged());
for (int k = 0; k < 3; k++)
    System.out.printf("state %d: mean %+.6f std %.6f%n",
            k, hmm.params().means[k][0], Math.sqrt(hmm.params().variances[k][0]));
```

---

## 3. How do I choose between K=2 and K=3 (AIC/BIC)?

**Python**

```python
from regime import GaussianHMM, load_dataset

r = load_dataset("../data").index_returns
for k in (2, 3):
    hmm = GaussianHMM(k)
    hmm.fit(r)
    print(f"K={k}  loglik {hmm.fit_result.log_likelihood:12.4f}"
          f"  AIC {hmm.aic(r):12.4f}  BIC {hmm.bic(r):12.4f}")
# K=2 loglik 6612.4532  |  K=3 loglik 6643.0767 — both AIC and BIC prefer K=3
```

**C++**

```cpp
#include <iostream>
#include <regime/hmm.hpp>
#include <regime/pipeline.hpp>

const regime::Matrix r = regime::to_matrix(regime::load_dataset("../data").index_returns);
for (int k : {2, 3}) {
    regime::GaussianHMM hmm(k);
    hmm.fit(r);
    std::cout << "K=" << k << " AIC " << hmm.aic(r) << " BIC " << hmm.bic(r) << "\n";
}
```

**Rust**

```rust
use regime::{load_dataset, GaussianHmm, Matrix};

let r = Matrix::column(&load_dataset("../data")?.index_returns);
for k in [2usize, 3] {
    let mut hmm = GaussianHmm::new(k, 1e-8, 500, 1e-8)?;
    hmm.fit(&r, None)?;
    println!("K={k}  AIC {:.4}  BIC {:.4}", hmm.aic(&r)?, hmm.bic(&r)?);
}
```

**Java**

```java
import com.quant.regime.GaussianHmm;
import com.quant.regime.Pipeline;
import java.nio.file.Path;

double[] r = Pipeline.loadDataset(Path.of("../data")).indexReturns();
for (int k : new int[] {2, 3}) {
    GaussianHmm hmm = new GaussianHmm(k, 1e-8, 500, 1e-8);
    hmm.fit(r);
    System.out.printf("K=%d  AIC %.4f  BIC %.4f%n", k, hmm.aic(r), hmm.bic(r));
}
```

---

## 4. How do I decode the regime path with Viterbi?

**Python**

```python
import numpy as np
from regime import GaussianHMM, load_dataset

r = load_dataset("../data").index_returns
hmm = GaussianHMM(3)
hmm.fit(r)
path = hmm.viterbi(r)
print("days per state:", np.bincount(path, minlength=3).tolist())
# [99, 1372, 529]  (golden hmm_k3_viterbi_counts, exact)
```

**C++**

```cpp
#include <array>
#include <iostream>
#include <regime/hmm.hpp>
#include <regime/pipeline.hpp>

const regime::Matrix r = regime::to_matrix(regime::load_dataset("../data").index_returns);
regime::GaussianHMM hmm(3);
hmm.fit(r);
const std::vector<int> path = hmm.viterbi(r);
std::array<int, 3> counts{};
for (int s : path) ++counts[static_cast<std::size_t>(s)];
std::cout << counts[0] << " " << counts[1] << " " << counts[2] << "\n";  // 99 1372 529
```

**Rust**

```rust
use regime::{load_dataset, GaussianHmm, Matrix};

let r = Matrix::column(&load_dataset("../data")?.index_returns);
let mut hmm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?;
hmm.fit(&r, None)?;
let path = hmm.viterbi(&r)?;
let mut counts = [0usize; 3];
for &s in &path {
    counts[s] += 1;
}
println!("{counts:?}"); // [99, 1372, 529]
```

**Java**

```java
import com.quant.regime.GaussianHmm;
import com.quant.regime.Pipeline;
import java.nio.file.Path;

double[] r = Pipeline.loadDataset(Path.of("../data")).indexReturns();
GaussianHmm hmm = new GaussianHmm(3, 1e-8, 500, 1e-8);
hmm.fit(r);
int[] path = hmm.viterbi(r);
int[] counts = new int[3];
for (int s : path) counts[s]++;
System.out.println(counts[0] + " " + counts[1] + " " + counts[2]); // 99 1372 529
```

---

## 5. How do I get filtered vs smoothed probabilities — and which may I trade on?

Filtered = `P(state | data up to today)` (causal, tradable). Smoothed =
`P(state | full sample)` (uses the future; analysis only — see LEARN.md
section 9.3).

**Python**

```python
from regime import GaussianHMM, load_dataset

r = load_dataset("../data").index_returns
hmm = GaussianHMM(3)
hmm.fit(r)
filt = hmm.filtered_probabilities(r)     # (T, 3) — causal
smth = hmm.smoothed_probabilities(r)     # (T, 3) — non-causal
print("day 500 smoothed:", smth[500])    # golden: [0.15071, 0.84761, 0.00168]
print("day 500 filtered:", filt[500])
```

**C++**

```cpp
#include <iostream>
#include <regime/hmm.hpp>
#include <regime/pipeline.hpp>

const regime::Matrix r = regime::to_matrix(regime::load_dataset("../data").index_returns);
regime::GaussianHMM hmm(3);
hmm.fit(r);
const regime::Matrix filt = hmm.filtered_probabilities(r);   // T x 3, causal
const regime::Matrix smth = hmm.smoothed_probabilities(r);   // T x 3, non-causal
std::cout << "day 500 smoothed p1 = " << smth(500, 1)          // 0.8476064
          << ", filtered p1 = " << filt(500, 1) << "\n";
```

**Rust**

```rust
use regime::{load_dataset, GaussianHmm, Matrix};

let r = Matrix::column(&load_dataset("../data")?.index_returns);
let mut hmm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?;
hmm.fit(&r, None)?;
let filt = hmm.filtered_probabilities(&r)?;
let smth = hmm.smoothed_probabilities(&r)?;
println!("day 500 smoothed {:?}, filtered {:?}", smth.row(500), filt.row(500));
```

**Java**

```java
import com.quant.regime.GaussianHmm;
import com.quant.regime.Pipeline;
import java.nio.file.Path;

double[] r = Pipeline.loadDataset(Path.of("../data")).indexReturns();
GaussianHmm hmm = new GaussianHmm(3, 1e-8, 500, 1e-8);
hmm.fit(r);
double[][] smth = hmm.smoothedProbabilities(r);
double[][] filt = hmm.filteredProbabilities(r);
System.out.printf("day 500 smoothed p1 = %.7f, filtered p1 = %.7f%n",
        smth[500][1], filt[500][1]);                              // 0.8476064, ...
```

---

## 6. How do I compute the stationary distribution and expected regime durations?

**Python**

```python
from regime import GaussianHMM, load_dataset

r = load_dataset("../data").index_returns
hmm = GaussianHMM(3)
hmm.fit(r)
pi = hmm.stationary_distribution()       # golden: [0.24551, 0.62042, 0.13407]
for k in range(3):
    dur = 1.0 / (1.0 - hmm.params.transmat[k, k])
    print(f"state {k}: long-run share {pi[k]:.4f}, expected duration {dur:.1f} days")
```

**C++**

```cpp
#include <iostream>
#include <regime/hmm.hpp>
#include <regime/pipeline.hpp>

const regime::Matrix r = regime::to_matrix(regime::load_dataset("../data").index_returns);
regime::GaussianHMM hmm(3);
hmm.fit(r);
const std::vector<double> pi = hmm.stationary_distribution();  // throws if it cannot converge
for (std::size_t k = 0; k < 3; ++k)
    std::cout << "state " << k << ": share " << pi[k]
              << " duration " << 1.0 / (1.0 - hmm.params().transmat(k, k)) << "d\n";
```

**Rust**

```rust
use regime::{load_dataset, GaussianHmm, Matrix};

let r = Matrix::column(&load_dataset("../data")?.index_returns);
let mut hmm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?;
hmm.fit(&r, None)?;
let pi = hmm.stationary_distribution()?; // Err if the power method cannot converge
let a = &hmm.params().expect("fitted").transmat;
for k in 0..3 {
    println!("state {k}: share {:.4}, duration {:.1}d", pi[k], 1.0 / (1.0 - a.get(k, k)));
}
```

**Java**

```java
import com.quant.regime.GaussianHmm;
import com.quant.regime.Pipeline;
import java.nio.file.Path;

double[] r = Pipeline.loadDataset(Path.of("../data")).indexReturns();
GaussianHmm hmm = new GaussianHmm(3, 1e-8, 500, 1e-8);
hmm.fit(r);
double[] pi = hmm.stationaryDistribution();   // throws if it cannot converge
for (int k = 0; k < 3; k++)
    System.out.printf("state %d: share %.4f, duration %.1fd%n",
            k, pi[k], 1.0 / (1.0 - hmm.params().transmat[k][k]));
```

---

## 7. How do I build vol-targeted momentum positions?

Returns must be simple daily returns strictly greater than -1; a -100 %
print is rejected as corrupt data.

**Python**

```python
from regime import load_dataset, momentum_positions

ds = load_dataset("../data")
pos = momentum_positions(ds.trend_returns, lookback=252,
                         vol_target=0.10, ewma_lambda=0.94, leverage_cap=4.0)
print(pos.shape)                # (2000, 6); rows before day 251 are zero
print(pos[300])                 # day-300 book, decided at that close
```

**C++**

```cpp
#include <iostream>
#include <regime/pipeline.hpp>
#include <regime/strategies.hpp>

const regime::Dataset ds = regime::load_dataset("../data");
const regime::Matrix pos = regime::momentum_positions(
    ds.trend_returns, /*lookback=*/252, /*vol_target=*/0.10, /*ewma_lambda=*/0.94,
    /*leverage_cap=*/4.0);
std::cout << pos.rows << " x " << pos.cols << "\n";             // 2000 x 6
std::cout << "day-300 book, asset 0: " << pos(300, 0) << "\n";
```

**Rust**

```rust
use regime::{load_dataset, momentum_positions};

let ds = load_dataset("../data")?;
let pos = momentum_positions(&ds.trend_returns, 252, 0.10, 0.94, 4.0)?;
println!("{} x {}", pos.rows(), pos.cols()); // 2000 x 6
println!("day-300 book: {:?}", pos.row(300));
```

**Java**

```java
import com.quant.regime.Pipeline;
import com.quant.regime.Strategies;
import java.nio.file.Path;

Pipeline.Dataset ds = Pipeline.loadDataset(Path.of("../data"));
double[][] pos = Strategies.momentumPositions(ds.trendReturns(), 252, 0.10, 0.94, 4.0);
System.out.println(pos.length + " x " + pos[0].length); // 2000 x 6
System.out.println("day-300 book, asset 0: " + pos[300][0]);
```

---

## 8. How do I build the FX carry basket?

Rank by rate differential at day indices 0, 21, 42, ...; the traded
return adds the previous day's accrual (`spot[t] + diff[t-1]/252`). On
the bundled data the basket changes on only 2 days (798 and 819); the
re-ranking logic is exercised directly by the synthetic `carry_rerank_*` golden
cases instead.

**Python**

```python
from regime import carry_positions, carry_total_returns, load_dataset

ds = load_dataset("../data")
pos = carry_positions(ds.fx_rate_diffs, top_n=3, bottom_n=3, rebalance_days=21)
rets = carry_total_returns(ds.fx_spot_returns, ds.fx_rate_diffs)
print(pos[0])   # day-0 book: three +1/3 longs, three -1/3 shorts
```

**C++**

```cpp
#include <iostream>
#include <regime/pipeline.hpp>
#include <regime/strategies.hpp>

const regime::Dataset ds = regime::load_dataset("../data");
const regime::Matrix pos = regime::carry_positions(ds.fx_rate_diffs, /*top_n=*/3,
                                                   /*bottom_n=*/3, /*rebalance_days=*/21);
const regime::Matrix rets = regime::carry_total_returns(ds.fx_spot_returns, ds.fx_rate_diffs);
for (std::size_t a = 0; a < pos.cols; ++a) std::cout << pos(0, a) << " ";   // day-0 book
std::cout << "| total return day 1, asset 0: " << rets(1, 0) << "\n";
```

**Rust**

```rust
use regime::{carry_positions, carry_total_returns, load_dataset};

let ds = load_dataset("../data")?;
let pos = carry_positions(&ds.fx_rate_diffs, 3, 3, 21)?;
let rets = carry_total_returns(&ds.fx_spot_returns, &ds.fx_rate_diffs)?;
println!("day-0 book {:?} | total return day 1, asset 0: {}", pos.row(0), rets.get(1, 0));
```

**Java**

```java
import com.quant.regime.Pipeline;
import com.quant.regime.Strategies;
import java.nio.file.Path;
import java.util.Arrays;

Pipeline.Dataset ds = Pipeline.loadDataset(Path.of("../data"));
double[][] pos = Strategies.carryPositions(ds.fxRateDiffs(), 3, 3, 21);
double[][] rets = Strategies.carryTotalReturns(ds.fxSpotReturns(), ds.fxRateDiffs());
System.out.println("day-0 book " + Arrays.toString(pos[0])
        + " | total return day 1, asset 0: " + rets[1][0]);
```

---

## 9. How do I run a backtest with costs and read the metrics?

Dates are optional; when given they must be strictly increasing and
enable the calendar-month `worst_month`. A day with `net <= -1`
(wipe-out) is an error naming the day, never a negative equity curve.

**Python**

```python
from regime import load_dataset, momentum_positions, run_backtest

ds = load_dataset("../data")
pos = momentum_positions(ds.trend_returns)
res = run_backtest(pos, ds.trend_returns, cost_bps=5.0, dates=ds.dates)
m = res.metrics
print(f"ann ret {m['ann_return']:+.2%}  Sharpe {m['sharpe']:.4f}  maxDD {m['max_dd']:+.2%}")
# ann ret +4.06%  Sharpe 0.9720  maxDD -9.26%  (before the portfolio scale)
```

**C++**

```cpp
#include <iostream>
#include <regime/backtest.hpp>
#include <regime/pipeline.hpp>
#include <regime/strategies.hpp>

const regime::Dataset ds = regime::load_dataset("../data");
const regime::Matrix pos = regime::momentum_positions(ds.trend_returns);
const regime::BacktestResult res =
    regime::run_backtest(pos, ds.trend_returns, /*cost_bps=*/5.0, &ds.dates);
std::cout << "ann ret " << res.metrics.ann_return << "  Sharpe " << res.metrics.sharpe
          << "  maxDD " << res.metrics.max_dd << "\n";           // 0.0406252 0.9720241 -0.0926142
```

**Rust**

```rust
use regime::{load_dataset, momentum_positions, run_backtest};

let ds = load_dataset("../data")?;
let pos = momentum_positions(&ds.trend_returns, 252, 0.10, 0.94, 4.0)?;
let res = run_backtest(&pos, &ds.trend_returns, 5.0, Some(&ds.dates))?;
let m = &res.metrics;
println!("ann ret {:+.4}  Sharpe {:.7}  maxDD {:+.4}", m.ann_return, m.sharpe, m.max_dd);
// Sharpe 0.9720241 (before the portfolio scale)
```

**Java**

```java
import com.quant.regime.Backtest;
import com.quant.regime.Pipeline;
import com.quant.regime.Strategies;
import java.nio.file.Path;

Pipeline.Dataset ds = Pipeline.loadDataset(Path.of("../data"));
double[][] pos = Strategies.momentumPositions(ds.trendReturns(), 252, 0.10, 0.94, 4.0);
Backtest.Result res = Backtest.run(pos, ds.trendReturns(), 5.0, ds.dates());
Backtest.Metrics m = res.metrics();
System.out.printf("ann ret %+.4f  Sharpe %.7f  maxDD %+.4f%n",
        m.annReturn(), m.sharpe(), m.maxDd());                    // Sharpe 0.9720241
```

---

## 10. How do I build the walk-forward regime gate and filter a strategy?

The gate is causal by construction: expanding-window refits every 63 days,
filtered probability of the lowest-variance state in between. The gate
validates its schedule, threshold and HMM settings before the first fit;
a failing refit reports the day index.

**Python**

```python
from regime import (apply_gate, load_dataset, momentum_positions,
                    regime_gate, run_backtest)

ds = load_dataset("../data")
gate, models = regime_gate(ds.index_returns, n_states=3,
                           train_min_days=504, refit_days=63, mode="prob")
pos = momentum_positions(ds.trend_returns)
res = run_backtest(apply_gate(pos, gate), ds.trend_returns, 5.0, ds.dates)
print(f"filtered Sharpe {res.metrics['sharpe']:.4f}")   # 0.8066 vs 0.9720 unfiltered
print(f"{len(models)} walk-forward fits")
```

**C++**

```cpp
#include <iostream>
#include <regime/backtest.hpp>
#include <regime/pipeline.hpp>
#include <regime/strategies.hpp>

const regime::Dataset ds = regime::load_dataset("../data");
const regime::RegimeGateResult g = regime::regime_gate(
    ds.index_returns, /*n_states=*/3, /*train_min_days=*/504, /*refit_days=*/63,
    /*mode=*/"prob", /*threshold=*/0.5);
const regime::Matrix pos = regime::apply_gate(regime::momentum_positions(ds.trend_returns), g.gate);
const regime::BacktestResult res = regime::run_backtest(pos, ds.trend_returns, 5.0, &ds.dates);
std::cout << "filtered Sharpe " << res.metrics.sharpe << " from " << g.models.size()
          << " walk-forward fits\n";                              // 0.806598, 24
```

**Rust**

```rust
use regime::{apply_gate, load_dataset, momentum_positions, regime_gate, run_backtest};

let ds = load_dataset("../data")?;
let (gate, models) = regime_gate(&ds.index_returns, 3, 504, 63, "prob", 0.5, 1e-8, 500, 1e-8)?;
let pos = apply_gate(&momentum_positions(&ds.trend_returns, 252, 0.10, 0.94, 4.0)?, &gate)?;
let res = run_backtest(&pos, &ds.trend_returns, 5.0, Some(&ds.dates))?;
println!("filtered Sharpe {:.7} from {} walk-forward fits", res.metrics.sharpe, models.len());
// 0.8065977, 24
```

**Java**

```java
import com.quant.regime.Backtest;
import com.quant.regime.Pipeline;
import com.quant.regime.Strategies;
import java.nio.file.Path;

Pipeline.Dataset ds = Pipeline.loadDataset(Path.of("../data"));
Strategies.GateResult g = Strategies.regimeGate(ds.indexReturns(), 3, 504, 63, "prob", 0.5,
        1e-8, 500, 1e-8);
double[][] pos = Strategies.applyGate(
        Strategies.momentumPositions(ds.trendReturns(), 252, 0.10, 0.94, 4.0), g.gate());
Backtest.Result res = Backtest.run(pos, ds.trendReturns(), 5.0, ds.dates());
System.out.printf("filtered Sharpe %.7f from %d walk-forward fits%n",
        res.metrics().sharpe(), g.models().size());               // 0.8065977, 24
```

---

## 11. How do I attribute strategy returns to regimes (crisis table)?

Attribute each day's net return to the day's full-sample K=3 Viterbi
state. (Full-sample decoding is fine *here* — attribution is analysis,
not a trading decision.) Every label must lie in `[0, K)`.

**Python**

```python
from regime import (GaussianHMM, load_dataset, momentum_positions,
                    run_backtest, state_conditional_returns)

ds = load_dataset("../data")
hmm = GaussianHMM(3)
hmm.fit(ds.index_returns)
states = hmm.viterbi(ds.index_returns)
res = run_backtest(momentum_positions(ds.trend_returns), ds.trend_returns, 5.0)
table = state_conditional_returns(res.net_returns, states, 3)
for k, row in table.items():
    print(f"state {k}: {row['n_days']:4d} days  ann ret {row['ann_return']:+.2%}")
# state 2 (crisis, highest variance): ann ret +7.37%  (before the portfolio scale)
```

**C++**

```cpp
#include <iostream>
#include <regime/backtest.hpp>
#include <regime/hmm.hpp>
#include <regime/pipeline.hpp>
#include <regime/strategies.hpp>

const regime::Dataset ds = regime::load_dataset("../data");
const regime::Matrix r = regime::to_matrix(ds.index_returns);
regime::GaussianHMM hmm(3);
hmm.fit(r);
const std::vector<int> states = hmm.viterbi(r);
const regime::BacktestResult res =
    regime::run_backtest(regime::momentum_positions(ds.trend_returns), ds.trend_returns, 5.0);
const std::vector<regime::StateStats> table =
    regime::state_conditional_returns(res.net_returns, states, 3);
std::cout << "crisis: " << table[2].n_days << " days, ann ret " << table[2].ann_return
          << "\n";                                                // 226 days, 0.0736696
```

**Rust**

```rust
use regime::{
    load_dataset, momentum_positions, run_backtest, state_conditional_returns, GaussianHmm, Matrix,
};

let ds = load_dataset("../data")?;
let r = Matrix::column(&ds.index_returns);
let mut hmm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?;
hmm.fit(&r, None)?;
let states = hmm.viterbi(&r)?;
let pos = momentum_positions(&ds.trend_returns, 252, 0.10, 0.94, 4.0)?;
let res = run_backtest(&pos, &ds.trend_returns, 5.0, None)?;
let table = state_conditional_returns(&res.net_returns, &states, 3)?;
println!("crisis: {} days, ann ret {:+.5}", table[2].n_days, table[2].ann_return); // 226, +0.07367
```

**Java**

```java
import com.quant.regime.Backtest;
import com.quant.regime.GaussianHmm;
import com.quant.regime.Pipeline;
import com.quant.regime.Strategies;
import java.nio.file.Path;

Pipeline.Dataset ds = Pipeline.loadDataset(Path.of("../data"));
GaussianHmm hmm = new GaussianHmm(3, 1e-8, 500, 1e-8);
hmm.fit(ds.indexReturns());
int[] states = hmm.viterbi(ds.indexReturns());
Backtest.Result res = Backtest.run(
        Strategies.momentumPositions(ds.trendReturns(), 252, 0.10, 0.94, 4.0),
        ds.trendReturns(), 5.0, null);
Backtest.StateStats[] table = Backtest.stateConditionalReturns(res.netReturns(), states, 3);
System.out.printf("crisis: %d days, ann ret %+.5f%n",
        table[2].nDays(), table[2].annReturn());                  // 226, +0.07367
```

---

## 12. How do I verify my build against the golden values?

Every language ships a golden test suite that compares all 23 cases /
87 scalars (`python -m pytest tests/test_golden.py`, the `Golden.*`
GoogleTest cases, `cargo test --test golden_tests`, `GoldenTest`). This is
the manual equivalent for one pinned number: the full-sample K=3
log-likelihood `6643.076654247344` (tolerance 1e-6). Python reads the
file; the ports compare against the pinned constant (their JSON readers
are test-tree helpers, not library API).

**Python**

```python
import json
from regime import run_full_pipeline

pipe = run_full_pipeline("../data")                 # full pinned config
with open("../data/golden/golden.json") as f:
    golden = {c["name"]: c for c in json.load(f)["cases"]}
got = pipe["fit3"].log_likelihood
want = golden["hmm_k3_loglik"]["expect"]["loglik"]
assert abs(got - want) <= golden["hmm_k3_loglik"]["tol"], (got, want)
print(f"K=3 loglik OK: {got:.6f}")
```

**C++**

```cpp
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <regime/pipeline.hpp>

const regime::PipelineResult pipe = regime::run_full_pipeline("../data");
const double want = 6643.076654247344;              // golden hmm_k3_loglik, tol 1e-6
if (std::abs(pipe.fit3.log_likelihood - want) > 1e-6)
    throw std::runtime_error("golden mismatch");
std::cout << "K=3 loglik OK: " << pipe.fit3.log_likelihood << "\n";
```

**Rust**

```rust
use regime::{golden_values, run_full_pipeline};

let pipe = run_full_pipeline("../data", None, None)?;
let got = golden_values(&pipe)["hmm_k3_loglik"]["loglik"];
let want = 6643.076654247344; // golden hmm_k3_loglik, tol 1e-6
assert!((got - want).abs() <= 1e-6, "golden mismatch: {got} vs {want}");
println!("K=3 loglik OK: {got:.6}");
```

**Java**

```java
import com.quant.regime.Pipeline;
import java.nio.file.Path;

Pipeline.Result pipe = Pipeline.runFullPipeline(Path.of("../data"));
double want = 6643.076654247344;                    // golden hmm_k3_loglik, tol 1e-6
if (Math.abs(pipe.fit3().logLikelihood() - want) > 1e-6)
    throw new IllegalStateException("golden mismatch");
System.out.printf("K=3 loglik OK: %.6f%n", pipe.fit3().logLikelihood());
```

---

## 13. How do I convince myself there is no lookahead?

Shift the positions forward one day; if the engine were leaking (applying
day-t decisions to day-t returns), the shift would change nothing.

**Python**

```python
import numpy as np
from regime import load_dataset, momentum_positions, run_backtest

ds = load_dataset("../data")
pos = momentum_positions(ds.trend_returns)
lagged = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])   # decide a day later
a = run_backtest(pos, ds.trend_returns).metrics["sharpe"]
b = run_backtest(lagged, ds.trend_returns).metrics["sharpe"]
print(f"on-time {a:.4f} vs lagged {b:.4f}")   # must differ — and they do
assert a != b
```

**C++**

```cpp
#include <iostream>
#include <stdexcept>
#include <regime/backtest.hpp>
#include <regime/pipeline.hpp>
#include <regime/strategies.hpp>

const regime::Dataset ds = regime::load_dataset("../data");
const regime::Matrix pos = regime::momentum_positions(ds.trend_returns);
regime::Matrix lagged(pos.rows, pos.cols, 0.0);      // row 0 zero, row t = pos row t-1
for (std::size_t t = 1; t < pos.rows; ++t)
    for (std::size_t a = 0; a < pos.cols; ++a) lagged(t, a) = pos(t - 1, a);
const double on_time = regime::run_backtest(pos, ds.trend_returns, 0.0).metrics.sharpe;
const double late = regime::run_backtest(lagged, ds.trend_returns, 0.0).metrics.sharpe;
std::cout << "on-time " << on_time << " vs lagged " << late << "\n";
if (on_time == late) throw std::runtime_error("lookahead: the shift changed nothing");
```

**Rust**

```rust
use regime::{load_dataset, momentum_positions, run_backtest, Matrix};

let ds = load_dataset("../data")?;
let pos = momentum_positions(&ds.trend_returns, 252, 0.10, 0.94, 4.0)?;
let mut lagged = Matrix::zeros(pos.rows(), pos.cols()); // row 0 zero, row t = pos row t-1
for t in 1..pos.rows() {
    for a in 0..pos.cols() {
        lagged.set(t, a, pos.get(t - 1, a));
    }
}
let on_time = run_backtest(&pos, &ds.trend_returns, 0.0, None)?.metrics.sharpe;
let late = run_backtest(&lagged, &ds.trend_returns, 0.0, None)?.metrics.sharpe;
println!("on-time {on_time:.4} vs lagged {late:.4}");
assert!(on_time != late, "lookahead: the shift changed nothing");
```

**Java**

```java
import com.quant.regime.Backtest;
import com.quant.regime.Pipeline;
import com.quant.regime.Strategies;
import java.nio.file.Path;

Pipeline.Dataset ds = Pipeline.loadDataset(Path.of("../data"));
double[][] pos = Strategies.momentumPositions(ds.trendReturns(), 252, 0.10, 0.94, 4.0);
double[][] lagged = new double[pos.length][pos[0].length];   // row 0 zero, row t = pos row t-1
for (int t = 1; t < pos.length; t++) lagged[t] = pos[t - 1].clone();
double onTime = Backtest.run(pos, ds.trendReturns(), 0.0, null).metrics().sharpe();
double late = Backtest.run(lagged, ds.trendReturns(), 0.0, null).metrics().sharpe();
System.out.printf("on-time %.4f vs lagged %.4f%n", onTime, late);
if (onTime == late) throw new IllegalStateException("lookahead: the shift changed nothing");
```

---

## 14. How do I read the fit diagnostics (dead states, monotonicity)?

A warm start can leave a state with no posterior support; EM then stops
with finite parameters and flags it (API_SPEC 1.4). Nothing in the fit
result is ever NaN, and a numerical fault that *decreased* the
log-likelihood is reported as `monotone = false`, never as convergence.
The example below deliberately parks a state 5.0 away from low-vol data.

**Python**

```python
import numpy as np
from regime import GaussianHMM

x = 0.01 * np.sin(np.arange(300))
hmm = GaussianHMM(3)
init = hmm.pinned_init(x)
init.means[2, 0], init.variances[2, 0] = 5.0, 1e-8      # state 2 can never fire
res = hmm.fit(x, init=init)
print(f"converged={res.converged} monotone={res.monotone} dead_state={res.dead_state}")
# converged=False monotone=True dead_state=2 — parameters stay finite:
assert np.all(np.isfinite(hmm.params.means)) and np.isfinite(res.log_likelihood)
```

**C++**

```cpp
#include <cmath>
#include <iostream>
#include <regime/hmm.hpp>

regime::Matrix x(300, 1);
for (std::size_t t = 0; t < 300; ++t) x(t, 0) = 0.01 * std::sin(static_cast<double>(t));
regime::GaussianHMM hmm(3);
regime::HMMParams init = hmm.pinned_init(x);
init.means(2, 0) = 5.0;                              // state 2 can never fire
init.variances(2, 0) = 1e-8;
const regime::HMMFitResult res = hmm.fit(x, init);
std::cout << "converged=" << res.converged << " monotone=" << res.monotone
          << " dead_state=" << res.dead_state << "\n";   // 0 1 2 (-1 means none)
```

**Rust**

```rust
use regime::{GaussianHmm, Matrix};

let x: Vec<f64> = (0..300).map(|t| 0.01 * (t as f64).sin()).collect();
let x = Matrix::column(&x);
let mut hmm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?;
let mut init = hmm.pinned_init(&x)?;
init.means.set(2, 0, 5.0);                           // state 2 can never fire
init.variances.set(2, 0, 1e-8);
let res = hmm.fit(&x, Some(&init))?;
println!("converged={} monotone={} dead_state={:?}", res.converged, res.monotone, res.dead_state);
// converged=false monotone=true dead_state=Some(2)
```

**Java**

```java
import com.quant.regime.GaussianHmm;
import com.quant.regime.HmmFitResult;
import com.quant.regime.HmmParams;

double[] x = new double[300];
for (int t = 0; t < 300; t++) x[t] = 0.01 * Math.sin(t);
GaussianHmm hmm = new GaussianHmm(3);
HmmParams init = hmm.pinnedInit(GaussianHmm.column(x));
init.means[2][0] = 5.0;                              // state 2 can never fire
init.variances[2][0] = 1e-8;
HmmFitResult res = hmm.fit(x, init);
System.out.printf("converged=%b monotone=%b deadState=%d%n",
        res.converged(), res.monotone(), res.deadState());   // false true 2 (-1 means none)
```

---

## 15. How do I run the regime filter online, one observation at a time?

Fit once, build a `RegimeFilter`, then call `update` per observation. It
validates the parameters once, and in C++ and Rust `update` allocates
nothing. The result equals the batch filtered probabilities.

**Python**

```python
from regime import GaussianHMM, RegimeFilter, load_dataset

r = load_dataset("../data").index_returns[:600]
hmm = GaussianHMM(3)
hmm.fit(r)
live = RegimeFilter(hmm.params)
for x in r:
    p = live.update(x)
assert abs(p[0] - hmm.filtered_probabilities(r)[-1, 0]) < 1e-12
print("P(state 0) now:", round(float(p[0]), 6))
```

**C++**

```cpp
#include <iostream>
#include <vector>
#include <regime/hmm.hpp>
#include <regime/pipeline.hpp>

std::vector<double> r = regime::load_dataset("../data").index_returns;
r.resize(600);
regime::GaussianHMM hmm(3, /*tol=*/1e-8, /*max_iter=*/500, /*var_floor=*/1e-8);
hmm.fit(regime::to_matrix(r));
regime::RegimeFilter live(hmm.params());
for (double x : r) live.update(x);
std::cout << "P(state 0) now: " << live.alpha()[0] << "\n";
```

**Rust**

```rust
use regime::{load_dataset, GaussianHmm, Matrix, RegimeFilter};

let r: Vec<f64> = load_dataset("../data")?.index_returns[..600].to_vec();
let mut hmm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?;
hmm.fit(&Matrix::column(&r), None)?;
let mut live = RegimeFilter::new(hmm.params().expect("fitted"), None)?;
for x in &r {
    live.update(&[*x])?;
}
println!("P(state 0) now: {:.6}", live.alpha()[0]);
```

**Java**

```java
import com.quant.regime.GaussianHmm;
import com.quant.regime.Pipeline;
import com.quant.regime.RegimeFilter;
import java.nio.file.Path;
import java.util.Arrays;

double[] r = Arrays.copyOf(Pipeline.loadDataset(Path.of("../data")).indexReturns(), 600);
GaussianHmm hmm = new GaussianHmm(3, 1e-8, 500, 1e-8);
hmm.fit(r);
RegimeFilter live = new RegimeFilter(hmm.params());
for (double x : r) live.update(x);
System.out.printf("P(state 0) now: %.6f%n", live.alpha()[0]);
```

---

## 16. How do I stop the gate trading small moves (`gate_band`)?

`gate_band` holds the applied gate until the raw gate has moved by more than
the band: `o[t] = g[t]` if `|g[t] - o[t-1]| > band`, else `o[t-1]`. It is off
by default. On real data a band of 0.3 narrowed the gate's loss but did not
close it, and five bands tried on synthetic seeds are a selection, so this
recipe shows the mechanism and not a recommended value
(`docs/BAND_EVALUATION.md`). The snippets use the first 1200 days to stay
fast.

**Python**

```python
import numpy as np
from regime import apply_gate_band, load_dataset, regime_gate

r = load_dataset("../data").index_returns[:1200]
gate, _ = regime_gate(r, n_states=3, train_min_days=504, refit_days=63, mode="prob")
banded = apply_gate_band(gate, 0.3)
raw_travel = np.abs(np.diff(gate)).sum()
band_travel = np.abs(np.diff(banded)).sum()
print(f"gate travel {raw_travel:.2f} -> {band_travel:.2f}")   # 69.84 -> 30.75
assert band_travel < raw_travel
```

**C++**

```cpp
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <regime/pipeline.hpp>
#include <regime/strategies.hpp>

std::vector<double> r = regime::load_dataset("../data").index_returns;
r.resize(1200);
const std::vector<double> gate = regime::regime_gate(r).gate;
const std::vector<double> banded = regime::apply_gate_band(gate, 0.3);
double raw_travel = 0.0, band_travel = 0.0;
for (std::size_t t = 1; t < gate.size(); ++t) {
    raw_travel += std::abs(gate[t] - gate[t - 1]);
    band_travel += std::abs(banded[t] - banded[t - 1]);
}
std::cout << "gate travel " << raw_travel << " -> " << band_travel << "\n";
if (band_travel >= raw_travel) throw std::runtime_error("the band did not cut travel");
```

**Rust**

```rust
use regime::{apply_gate_band, load_dataset, regime_gate};

let r: Vec<f64> = load_dataset("../data")?.index_returns[..1200].to_vec();
let (gate, _) = regime_gate(&r, 3, 504, 63, "prob", 0.5, 1e-8, 500, 1e-8)?;
let banded = apply_gate_band(&gate, 0.3)?;
let travel = |g: &[f64]| g.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>();
println!("gate travel {:.2} -> {:.2}", travel(&gate), travel(&banded));
assert!(travel(&banded) < travel(&gate));
```

**Java**

```java
import com.quant.regime.Pipeline;
import com.quant.regime.Strategies;
import java.nio.file.Path;
import java.util.Arrays;

double[] r = Arrays.copyOf(Pipeline.loadDataset(Path.of("../data")).indexReturns(), 1200);
double[] gate = Strategies.regimeGate(r, 3, 504, 63, "prob", 0.5, 1e-8, 500, 1e-8).gate();
double[] banded = Strategies.applyGateBand(gate, 0.3);
double rawTravel = 0.0, bandTravel = 0.0;
for (int t = 1; t < gate.length; t++) {
    rawTravel += Math.abs(gate[t] - gate[t - 1]);
    bandTravel += Math.abs(banded[t] - banded[t - 1]);
}
System.out.printf("gate travel %.2f -> %.2f%n", rawTravel, bandTravel);
if (bandTravel >= rawTravel) throw new IllegalStateException("the band did not cut travel");
```

---

## 17. How do I let a warm start recover a zero transition (`transition_floor`)?

EM multiplies the old `A_ij` into the new one, so a transition that reached
zero stays at zero across warm-started refits. A floor `f > 0` replaces `A`
by `(A_ij + f) / sum_j (A_ij + f)` before a warm start. It only restarts the
entry, and EM may drive it back down. A cold start ignores it (API_SPEC 1.3).
The snippets show the arithmetic on a fitted matrix.

**Python**

```python
from regime import GaussianHMM, apply_transition_floor, load_dataset

r = load_dataset("../data").index_returns[:800]
hmm = GaussianHMM(3)
hmm.fit(r)
floored = apply_transition_floor(hmm.params.transmat, 1e-3)
print(f"smallest entry {hmm.params.transmat.min():.3e} -> {floored.min():.3e}")   # 1.942e-36 -> 9.970e-04
assert floored.min() >= hmm.params.transmat.min()
warm = GaussianHMM(3, transition_floor=1e-3)     # applied to every warm start
res = warm.fit(r, init=hmm.params)
print(f"warm refit: {res.n_iter} iterations, converged={res.converged}")
```

**C++**

```cpp
#include <algorithm>
#include <iostream>
#include <vector>
#include <regime/hmm.hpp>
#include <regime/pipeline.hpp>

std::vector<double> series = regime::load_dataset("../data").index_returns;
series.resize(800);
const regime::Matrix r = regime::to_matrix(series);
regime::GaussianHMM hmm(3);
hmm.fit(r);
const regime::Matrix floored = regime::apply_transition_floor(hmm.params().transmat, 1e-3);
double before = 1.0, after = 1.0;
for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) {
        before = std::min(before, hmm.params().transmat(i, j));
        after = std::min(after, floored(i, j));
    }
std::cout << "smallest entry " << before << " -> " << after << "\n";
regime::GaussianHMM warm(3, 1e-8, 500, 1e-8, /*transition_floor=*/1e-3);
const regime::HMMFitResult res = warm.fit(r, hmm.params());
std::cout << "warm refit: " << res.n_iter << " iterations\n";
```

**Rust**

```rust
use regime::{apply_transition_floor, load_dataset, GaussianHmm, Matrix};

let series: Vec<f64> = load_dataset("../data")?.index_returns[..800].to_vec();
let r = Matrix::column(&series);
let mut hmm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?;
hmm.fit(&r, None)?;
let params = hmm.params().expect("fitted").clone();
let floored = apply_transition_floor(&params.transmat, 1e-3)?;
let (mut before, mut after) = (1.0_f64, 1.0_f64);
for i in 0..3 {
    for j in 0..3 {
        before = before.min(params.transmat.get(i, j));
        after = after.min(floored.get(i, j));
    }
}
println!("smallest entry {before:.3e} -> {after:.3e}");
let mut warm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?.with_transition_floor(1e-3)?;
let res = warm.fit(&r, Some(&params))?;
println!("warm refit: {} iterations", res.n_iter);
```

**Java**

```java
import com.quant.regime.GaussianHmm;
import com.quant.regime.HmmFitResult;
import com.quant.regime.Pipeline;
import java.nio.file.Path;
import java.util.Arrays;

double[] r = Arrays.copyOf(Pipeline.loadDataset(Path.of("../data")).indexReturns(), 800);
GaussianHmm hmm = new GaussianHmm(3, 1e-8, 500, 1e-8);
hmm.fit(r);
double[][] floored = GaussianHmm.applyTransitionFloor(hmm.params().transmat, 1e-3);
double before = 1.0, after = 1.0;
for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
        before = Math.min(before, hmm.params().transmat[i][j]);
        after = Math.min(after, floored[i][j]);
    }
System.out.printf("smallest entry %.3e -> %.3e%n", before, after);
GaussianHmm warm = new GaussianHmm(3, 1e-8, 500, 1e-8, 1e-3);
HmmFitResult res = warm.fit(r, hmm.params());
System.out.printf("warm refit: %d iterations%n", res.nIter());
```

---

## 18. How do I guard the gate against one extreme print (`calm_guard`)?

One extreme return can take a state for itself: its variance sits at the
floor, it wins the lowest-variance test, and the gate reads a state the chain
almost never visits, with every fit reporting `converged`. With
`calm_guard = m > 0`, a refit whose lowest-variance state has stationary
weight below `m` is recorded in `guard_trips` and repeated on a clipped window
(API_SPEC 3.3). LEARN section 14 has the decision flow and the measured
effect. The guard narrows the damage and does not remove it, so clean outliers
before fitting all the same. The snippets inject -30 % at day 700.

**Python**

```python
from regime import load_dataset, regime_gate_report

r = load_dataset("../data").index_returns[:1200].copy()
r[700] = -0.30
plain = regime_gate_report(r, calm_guard=0.0)
guarded = regime_gate_report(r, calm_guard=0.01)
print(f"mean gate {plain.gate[503:].mean():.3f} unguarded, {guarded.gate[503:].mean():.3f} guarded")
print(f"guard tripped on {len(guarded.guard_trips)} refits, first at t={guarded.guard_trips[0]}")
assert not plain.guard_trips and guarded.guard_trips
```

**C++**

```cpp
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <regime/pipeline.hpp>
#include <regime/strategies.hpp>

std::vector<double> r = regime::load_dataset("../data").index_returns;
r.resize(1200);
r[700] = -0.30;
const regime::RegimeGateResult guarded = regime::regime_gate(
    r, 3, 504, 63, "prob", 0.5, 1e-8, 500, 1e-8, /*gate_band=*/0.0,
    /*transition_floor=*/0.0, /*calm_guard=*/0.01);
const double mean = std::accumulate(guarded.gate.begin() + 503, guarded.gate.end(), 0.0)
                    / static_cast<double>(guarded.gate.size() - 503);
std::cout << "mean gate " << mean << ", " << guarded.guard_trips.size() << " guard trips\n";
if (guarded.guard_trips.empty()) throw std::runtime_error("the guard did not trip");
```

**Rust**

```rust
use regime::{load_dataset, regime_gate_report, GateOptions};

let mut r: Vec<f64> = load_dataset("../data")?.index_returns[..1200].to_vec();
r[700] = -0.30;
let options = GateOptions { calm_guard: 0.01, ..GateOptions::default() };
let guarded = regime_gate_report(&r, 3, 504, 63, "prob", 0.5, 1e-8, 500, 1e-8, options)?;
let tail = &guarded.gate[503..];
let mean = tail.iter().sum::<f64>() / tail.len() as f64;
println!("mean gate {mean:.3}, {} guard trips", guarded.guard_trips.len());
assert!(!guarded.guard_trips.is_empty());
```

**Java**

```java
import com.quant.regime.Pipeline;
import com.quant.regime.Strategies;
import java.nio.file.Path;
import java.util.Arrays;

double[] r = Arrays.copyOf(Pipeline.loadDataset(Path.of("../data")).indexReturns(), 1200);
r[700] = -0.30;
Strategies.GateResult guarded = Strategies.regimeGate(r, 3, 504, 63, "prob", 0.5, 1e-8, 500, 1e-8,
        0.0, 0.0, 0.01);
double mean = Arrays.stream(guarded.gate(), 503, guarded.gate().length).average().orElseThrow();
System.out.printf("mean gate %.3f, %d guard trips%n", mean, guarded.guardTrips().length);
if (guarded.guardTrips().length == 0) throw new IllegalStateException("the guard did not trip");
```

---

## 19. How do I compare a gate with baselines? (Python)

`regime.evaluation` has no ports: research code stays in the reference. A
gate multiplies the same scaled book as its baselines, and only days it can
affect count, so the window starts at `t = 504`. Any gate you test should beat
at least the unfiltered book and the 21-day vol rule; the constant gate at the
same mean exposure tells you how much of a result is only lower exposure. The
bundled seed is favourable to the HMM gate (85th percentile of 100 seeds), and
the gate failed on real data, so read the output as a demonstration of the
method.

**Python**

```python
import numpy as np
from regime import (build_book, constant_gate, load_dataset, regime_gate,
                    vol_ratio_gate, window_metrics)

ds = load_dataset("../data")
book = build_book(ds)
start = 504
first_day = start - 1                    # the gate on day t earns day t+1
n = len(ds.index_returns)
hmm_gate, _ = regime_gate(ds.index_returns)
gates = {
    "unfiltered": np.ones(n),
    "HMM gate": hmm_gate,
    "constant": constant_gate(n, float(hmm_gate[first_day:-1].mean()), first_day),
    "21d vol rule": vol_ratio_gate(ds.index_returns, first_day),
}
for name, gate in gates.items():
    m = window_metrics(book.stream(gate).net(5.0), ds.dates, start)
    print(f"{name:<13} Sharpe {m['sharpe']:.2f}  maxDD {m['max_dd']:+.2%}")
# Matches docs/GATE_EVALUATION.md: 0.54, 0.99, 0.54, 0.81
```

---

## 20. How do I bootstrap a Sharpe difference? (Python)

Two series that share a crisis must be resampled together, in blocks, or the
interval comes out too narrow. `paired_block_bootstrap` draws one index path
for both series. The report uses 10000 resamples; 500 keep this snippet fast
and are too few for a published interval.

**Python**

```python
import numpy as np
from regime import build_book, load_dataset, paired_block_bootstrap, regime_gate, vol_ratio_gate

ds = load_dataset("../data")
book = build_book(ds)
start = 504
gate, _ = regime_gate(ds.index_returns)
gated = book.stream(gate).net(5.0)[start:]
vol = book.stream(vol_ratio_gate(ds.index_returns, start - 1)).net(5.0)[start:]
res = paired_block_bootstrap(gated, vol, np.random.default_rng(7), n_boot=500, mean_block=21.0)
print(f"diff {res.diff:+.3f}  95% [{res.ci_low:+.2f}, {res.ci_high:+.2f}]  P(>0) {res.prob_positive:.0%}")
# diff +0.186, as in docs/GATE_EVALUATION.md; the interval includes zero
assert res.ci_low < res.diff < res.ci_high
```

---

## 21. How do I apply the size-dependent cost model? (Python)

`run_backtest` charges a flat rate, which cannot grow with the book.
`regime.costs` adds a spread and a square-root impact term and leaves every
pinned number alone. The spreads, ADV and Y below are the module's illustrative
defaults, not calibrated values: the bundled instruments have no real volume
or quotes (`docs/CAPACITY_EVALUATION.md`). No gate is involved here, so the
snippet is quick.

**Python**

```python
import numpy as np
from regime import CostAssumptions, build_book, daily_cost_fraction, load_dataset, participation, sharpe
from regime.costs import (ILLUSTRATIVE_FX_ADV_USD, ILLUSTRATIVE_FX_HALF_SPREAD_BPS,
                          ILLUSTRATIVE_IMPACT_Y, ILLUSTRATIVE_TREND_ADV_USD,
                          ILLUSTRATIVE_TREND_HALF_SPREAD_BPS)

ds = load_dataset("../data")
book = build_book(ds)
start = 504
weights = 0.5 * np.hstack([book.momentum_positions, book.carry_positions])
returns = np.hstack([book.trend_returns, book.carry_returns])
n_trend, n_fx = book.trend_returns.shape[1], book.carry_returns.shape[1]
costs = CostAssumptions(
    half_spread_bps=np.r_[[ILLUSTRATIVE_TREND_HALF_SPREAD_BPS] * n_trend,
                          [ILLUSTRATIVE_FX_HALF_SPREAD_BPS] * n_fx],
    adv_usd=np.r_[[ILLUSTRATIVE_TREND_ADV_USD] * n_trend, [ILLUSTRATIVE_FX_ADV_USD] * n_fx],
    sigma_daily=returns[start:].std(axis=0),
    impact_y=ILLUSTRATIVE_IMPACT_Y,
)
gross = book.stream(np.ones(len(ds.index_returns))).gross
for aum in (1e6, 1e9):
    net = (gross - daily_cost_fraction(weights, aum, costs))[start:]
    peak = participation(weights, aum, costs)[start:].max()
    print(f"${aum:,.0f}: net Sharpe {float(sharpe(net)):.2f}, peak participation {peak:.2%}")
# Matches the Unfiltered table in docs/CAPACITY_EVALUATION.md: 0.67 at $1m, 0.35 at $1bn
```

---

## Checking the snippets

`python3 tools/check_cookbook.py` runs in CI (from the repository root, after
`bash cpp/build.sh`, `cargo build --release` in `rust/` and
`bash java/build.sh`) extracts every fenced block above into a scratch
directory, wraps the C++/Rust/Java bodies in a `main`, compiles them
against the built libraries and runs each one from its language
directory. It exits non-zero if any snippet fails to compile or run.
