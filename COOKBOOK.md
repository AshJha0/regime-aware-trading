# COOKBOOK — regime recipes

Task-oriented recipes for the `regime` toolkit, each in all four
languages. Every snippet below is a complete program body that compiles
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

Numbers quoted in comments are golden values for the bundled data
(`data/golden/golden.json`).

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

States come back sorted by mean ascending (pinned): state 0 is the
lowest-mean state — on this data, the crisis regime.

**Python**

```python
import numpy as np
from regime import GaussianHMM, load_dataset

r = load_dataset("../data").index_returns
hmm = GaussianHMM(n_states=3, tol=1e-8, max_iter=500, var_floor=1e-8)
res = hmm.fit(r)
print(f"loglik {res.log_likelihood:.6f} in {res.n_iter} iters, converged={res.converged}")
for k in range(3):                       # golden: means -0.003432, 0.000635, 0.001040
    mu, sd = hmm.params.means[k, 0], np.sqrt(hmm.params.variances[k, 0])
    print(f"state {k}: ann mean {252*mu:+.1%}  ann vol {np.sqrt(252)*sd:.1%}")
# loglik 6886.449837 (golden hmm_k3_loglik)
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
          << " iters, converged=" << res.converged << "\n";       // 6886.45, 63, 1
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
# K=2 loglik 6831.0255  |  K=3 loglik 6886.4498 — both AIC and BIC prefer K=3
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
print("day 500 smoothed:", smth[500])    # golden: [3.86e-06, 0.99866, 0.00134]
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
std::cout << "day 500 smoothed p1 = " << smth(500, 1)          // 0.9986576
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
        smth[500][1], filt[500][1]);                              // 0.9986576, ...
```

---

## 6. How do I compute the stationary distribution and expected regime durations?

**Python**

```python
from regime import GaussianHMM, load_dataset

r = load_dataset("../data").index_returns
hmm = GaussianHMM(3)
hmm.fit(r)
pi = hmm.stationary_distribution()       # golden: [0.05658, 0.67674, 0.26667]
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
the bundled data the six differentials never change rank order, so the
day-0 book is held for the whole sample (turnover `2/2000`); the
re-ranking logic is exercised by the synthetic `carry_rerank_*` golden
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
# ann ret +3.38%  Sharpe 0.8149  maxDD -10.48%  (golden momentum_unfiltered)
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
          << "  maxDD " << res.metrics.max_dd << "\n";           // 0.0337758 0.8148843 -0.1048329
```

**Rust**

```rust
use regime::{load_dataset, momentum_positions, run_backtest};

let ds = load_dataset("../data")?;
let pos = momentum_positions(&ds.trend_returns, 252, 0.10, 0.94, 4.0)?;
let res = run_backtest(&pos, &ds.trend_returns, 5.0, Some(&ds.dates))?;
let m = &res.metrics;
println!("ann ret {:+.4}  Sharpe {:.7}  maxDD {:+.4}", m.ann_return, m.sharpe, m.max_dd);
// Sharpe 0.8148843 (golden momentum_unfiltered)
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
        m.annReturn(), m.sharpe(), m.maxDd());                    // Sharpe 0.8148843
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
print(f"filtered Sharpe {res.metrics['sharpe']:.4f}")   # 1.0402 vs 0.8149 unfiltered
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
          << " walk-forward fits\n";                              // 1.040234, 24
```

**Rust**

```rust
use regime::{apply_gate, load_dataset, momentum_positions, regime_gate, run_backtest};

let ds = load_dataset("../data")?;
let (gate, models) = regime_gate(&ds.index_returns, 3, 504, 63, "prob", 0.5, 1e-8, 500, 1e-8)?;
let pos = apply_gate(&momentum_positions(&ds.trend_returns, 252, 0.10, 0.94, 4.0)?, &gate)?;
let res = run_backtest(&pos, &ds.trend_returns, 5.0, Some(&ds.dates))?;
println!("filtered Sharpe {:.7} from {} walk-forward fits", res.metrics.sharpe, models.len());
// 1.0402340, 24
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
        res.metrics().sharpe(), g.models().size());               // 1.0402340, 24
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
# state 0 (crisis): ann ret -8.77%  (golden crisis_momentum_return, sign case)
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
std::cout << "crisis: " << table[0].n_days << " days, ann ret " << table[0].ann_return
          << "\n";                                                // 99 days, -0.0877060
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
println!("crisis: {} days, ann ret {:+.5}", table[0].n_days, table[0].ann_return); // 99, -0.08771
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
        table[0].nDays(), table[0].annReturn());                  // 99, -0.08771
```

---

## 12. How do I verify my build against the golden values?

Every language ships a golden test suite that compares all 20 cases /
77 scalars (`python -m pytest tests/test_golden.py`, the `Golden.*`
GoogleTest cases, `cargo test --test golden_tests`, `GoldenTest`). This is
the manual equivalent for one pinned number: the full-sample K=3
log-likelihood `6886.449836599692` (tolerance 1e-6). Python reads the
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
const double want = 6886.449836599692;              // golden hmm_k3_loglik, tol 1e-6
if (std::abs(pipe.fit3.log_likelihood - want) > 1e-6)
    throw std::runtime_error("golden mismatch");
std::cout << "K=3 loglik OK: " << pipe.fit3.log_likelihood << "\n";
```

**Rust**

```rust
use regime::{golden_values, run_full_pipeline};

let pipe = run_full_pipeline("../data", None, None)?;
let got = golden_values(&pipe)["hmm_k3_loglik"]["loglik"];
let want = 6886.449836599692; // golden hmm_k3_loglik, tol 1e-6
assert!((got - want).abs() <= 1e-6, "golden mismatch: {got} vs {want}");
println!("K=3 loglik OK: {got:.6}");
```

**Java**

```java
import com.quant.regime.Pipeline;
import java.nio.file.Path;

Pipeline.Result pipe = Pipeline.runFullPipeline(Path.of("../data"));
double want = 6886.449836599692;                    // golden hmm_k3_loglik, tol 1e-6
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

## Checking the snippets

`python3 tools/check_cookbook.py` (from the repository root, after
`bash cpp/build.sh`, `cargo build --release` in `rust/` and
`bash java/build.sh`) extracts every fenced block above into a scratch
directory, wraps the C++/Rust/Java bodies in a `main`, compiles them
against the built libraries and runs each one from its language
directory. It exits non-zero if any snippet fails to compile or run.
