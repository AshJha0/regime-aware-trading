# COOKBOOK — regime recipes

Task-oriented recipes for the `regime` toolkit, each in all four
languages. Python snippets run as-is from the `python/` directory with
`PYTHONPATH=src python3 snippet.py` (they locate the bundled data
relative to the project root). C++/Rust/Java snippets follow the pinned
contract in [`API_SPEC.md`](API_SPEC.md): C++ namespace `regime` (headers
`<regime/*.hpp>`, series as `std::vector<double>`, panels as row-major
`std::vector<std::vector<double>>`), Rust crate `regime` (fallible
functions return `Result<_, RegimeError>`), Java package
`com.quant.regime` (series as `double[]`, panels as `double[][]`, errors
as `IllegalArgumentException`).

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
#include <regime/pipeline.hpp>

regime::Dataset ds = regime::load_dataset("../data");
std::cout << ds.index_returns.size() << " days, "
          << ds.trend_returns[0].size() << " trend assets\n";
```

**Rust**

```rust
use regime::load_dataset;

let ds = load_dataset("../data")?;
println!("{} days, {} trend assets", ds.index_returns.len(), ds.trend_returns[0].len());
```

**Java**

```java
import com.quant.regime.Dataset;
import com.quant.regime.Pipeline;

Dataset ds = Pipeline.loadDataset("../data");
System.out.println(ds.indexReturns().length + " days, "
        + ds.trendReturns()[0].length + " trend assets");
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
#include <regime/hmm.hpp>
#include <regime/pipeline.hpp>

auto r = regime::load_dataset("../data").index_returns;
regime::GaussianHMM hmm(3, /*tol=*/1e-8, /*max_iter=*/500, /*var_floor=*/1e-8);
regime::HMMFitResult res = hmm.fit(r);
std::cout << "loglik " << res.log_likelihood << " converged=" << res.converged << "\n";
for (int k = 0; k < 3; ++k)
    std::cout << "state " << k << ": mean " << hmm.params().means[k][0]
              << " var " << hmm.params().variances[k][0] << "\n";
```

**Rust**

```rust
use regime::{load_dataset, GaussianHmm};

let r = load_dataset("../data")?.index_returns;
let mut hmm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?;
let res = hmm.fit(&r, None)?;
println!("loglik {:.6} in {} iters, converged={}", res.log_likelihood, res.n_iter, res.converged);
for k in 0..3 {
    println!("state {k}: mean {:+.6} var {:.6e}",
             hmm.params().means[k][0], hmm.params().variances[k][0]);
}
```

**Java**

```java
import com.quant.regime.GaussianHmm;
import com.quant.regime.HmmFitResult;
import com.quant.regime.Pipeline;

double[] r = Pipeline.loadDataset("../data").indexReturns();
GaussianHmm hmm = new GaussianHmm(3, 1e-8, 500, 1e-8);
HmmFitResult res = hmm.fit(r);
System.out.printf("loglik %.6f in %d iters, converged=%b%n",
        res.logLikelihood(), res.nIter(), res.converged());
for (int k = 0; k < 3; k++)
    System.out.printf("state %d: mean %+.6f var %.3e%n",
            k, hmm.params().means()[k][0], hmm.params().variances()[k][0]);
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
#include <regime/hmm.hpp>
#include <regime/pipeline.hpp>

auto r = regime::load_dataset("../data").index_returns;
for (int k : {2, 3}) {
    regime::GaussianHMM hmm(k);
    hmm.fit(r);
    std::cout << "K=" << k << " AIC " << hmm.aic(r) << " BIC " << hmm.bic(r) << "\n";
}
```

**Rust**

```rust
use regime::{load_dataset, GaussianHmm};

let r = load_dataset("../data")?.index_returns;
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

double[] r = Pipeline.loadDataset("../data").indexReturns();
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
#include <regime/hmm.hpp>
#include <regime/pipeline.hpp>

auto r = regime::load_dataset("../data").index_returns;
regime::GaussianHMM hmm(3);
hmm.fit(r);
std::vector<int> path = hmm.viterbi(r);
std::array<int, 3> counts{};
for (int s : path) ++counts[s];
std::cout << counts[0] << " " << counts[1] << " " << counts[2] << "\n";  // 99 1372 529
```

**Rust**

```rust
use regime::{load_dataset, GaussianHmm};

let r = load_dataset("../data")?.index_returns;
let mut hmm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?;
hmm.fit(&r, None)?;
let path = hmm.viterbi(&r)?;
let mut counts = [0usize; 3];
for &s in &path { counts[s] += 1; }
println!("{counts:?}"); // [99, 1372, 529]
```

**Java**

```java
import com.quant.regime.GaussianHmm;
import com.quant.regime.Pipeline;

double[] r = Pipeline.loadDataset("../data").indexReturns();
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
#include <regime/hmm.hpp>
#include <regime/pipeline.hpp>

auto r = regime::load_dataset("../data").index_returns;
regime::GaussianHMM hmm(3);
hmm.fit(r);
auto filt = hmm.filtered_probabilities(r);   // vector<array-like rows>, T x 3
auto smth = hmm.smoothed_probabilities(r);
std::cout << "day 500 smoothed p1 = " << smth[500][1] << "\n";  // 0.9986576
```

**Rust**

```rust
use regime::{load_dataset, GaussianHmm};

let r = load_dataset("../data")?.index_returns;
let mut hmm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?;
hmm.fit(&r, None)?;
let filt = hmm.filtered_probabilities(&r)?;
let smth = hmm.smoothed_probabilities(&r)?;
println!("day 500 smoothed {:?}, filtered {:?}", smth[500], filt[500]);
```

**Java**

```java
import com.quant.regime.GaussianHmm;
import com.quant.regime.Pipeline;

double[] r = Pipeline.loadDataset("../data").indexReturns();
GaussianHmm hmm = new GaussianHmm(3, 1e-8, 500, 1e-8);
hmm.fit(r);
double[][] smth = hmm.smoothedProbabilities(r);
double[][] filt = hmm.filteredProbabilities(r);
System.out.printf("day 500 smoothed p1 = %.7f%n", smth[500][1]); // 0.9986576
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
#include <regime/hmm.hpp>
#include <regime/pipeline.hpp>

auto r = regime::load_dataset("../data").index_returns;
regime::GaussianHMM hmm(3);
hmm.fit(r);
std::vector<double> pi = hmm.stationary_distribution();
for (int k = 0; k < 3; ++k)
    std::cout << "state " << k << ": share " << pi[k]
              << " duration " << 1.0 / (1.0 - hmm.params().transmat[k][k]) << "d\n";
```

**Rust**

```rust
use regime::{load_dataset, GaussianHmm};

let r = load_dataset("../data")?.index_returns;
let mut hmm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?;
hmm.fit(&r, None)?;
let pi = hmm.stationary_distribution()?;
for k in 0..3 {
    let dur = 1.0 / (1.0 - hmm.params().transmat[k][k]);
    println!("state {k}: share {:.4}, duration {:.1}d", pi[k], dur);
}
```

**Java**

```java
import com.quant.regime.GaussianHmm;
import com.quant.regime.Pipeline;

double[] r = Pipeline.loadDataset("../data").indexReturns();
GaussianHmm hmm = new GaussianHmm(3, 1e-8, 500, 1e-8);
hmm.fit(r);
double[] pi = hmm.stationaryDistribution();
for (int k = 0; k < 3; k++)
    System.out.printf("state %d: share %.4f, duration %.1fd%n",
            k, pi[k], 1.0 / (1.0 - hmm.params().transmat()[k][k]));
```

---

## 7. How do I build vol-targeted momentum positions?

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
#include <regime/strategies.hpp>
#include <regime/pipeline.hpp>

auto ds = regime::load_dataset("../data");
auto pos = regime::momentum_positions(ds.trend_returns, /*lookback=*/252,
                                      /*vol_target=*/0.10, /*ewma_lambda=*/0.94,
                                      /*leverage_cap=*/4.0);
std::cout << pos.size() << " x " << pos[0].size() << "\n";  // 2000 x 6
```

**Rust**

```rust
use regime::{load_dataset, momentum_positions};

let ds = load_dataset("../data")?;
let pos = momentum_positions(&ds.trend_returns, 252, 0.10, 0.94, 4.0)?;
println!("{} x {}", pos.len(), pos[0].len()); // 2000 x 6
```

**Java**

```java
import com.quant.regime.Pipeline;
import com.quant.regime.Strategies;

var ds = Pipeline.loadDataset("../data");
double[][] pos = Strategies.momentumPositions(ds.trendReturns(), 252, 0.10, 0.94, 4.0);
System.out.println(pos.length + " x " + pos[0].length); // 2000 x 6
```

---

## 8. How do I build the FX carry basket?

Rank by rate differential every 21 days; the traded return adds the
previous day's accrual (`spot[t] + diff[t-1]/252`).

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
#include <regime/strategies.hpp>
#include <regime/pipeline.hpp>

auto ds = regime::load_dataset("../data");
auto pos  = regime::carry_positions(ds.fx_rate_diffs, /*top_n=*/3, /*bottom_n=*/3,
                                    /*rebalance_days=*/21);
auto rets = regime::carry_total_returns(ds.fx_spot_returns, ds.fx_rate_diffs);
```

**Rust**

```rust
use regime::{carry_positions, carry_total_returns, load_dataset};

let ds = load_dataset("../data")?;
let pos = carry_positions(&ds.fx_rate_diffs, 3, 3, 21)?;
let rets = carry_total_returns(&ds.fx_spot_returns, &ds.fx_rate_diffs)?;
println!("{:?}", pos[0]);
```

**Java**

```java
import com.quant.regime.Pipeline;
import com.quant.regime.Strategies;

var ds = Pipeline.loadDataset("../data");
double[][] pos = Strategies.carryPositions(ds.fxRateDiffs(), 3, 3, 21);
double[][] rets = Strategies.carryTotalReturns(ds.fxSpotReturns(), ds.fxRateDiffs());
```

---

## 9. How do I run a backtest with costs and read the metrics?

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
#include <regime/backtest.hpp>
#include <regime/strategies.hpp>
#include <regime/pipeline.hpp>

auto ds = regime::load_dataset("../data");
auto pos = regime::momentum_positions(ds.trend_returns);
regime::BacktestResult res = regime::run_backtest(pos, ds.trend_returns, /*cost_bps=*/5.0);
std::cout << "Sharpe " << res.metrics.at("sharpe") << "\n";  // 0.8148843
```

**Rust**

```rust
use regime::{load_dataset, momentum_positions, run_backtest};

let ds = load_dataset("../data")?;
let pos = momentum_positions(&ds.trend_returns, 252, 0.10, 0.94, 4.0)?;
let res = run_backtest(&pos, &ds.trend_returns, 5.0)?;
println!("Sharpe {:.7}", res.metrics.sharpe); // 0.8148843
```

**Java**

```java
import com.quant.regime.Backtest;
import com.quant.regime.BacktestResult;
import com.quant.regime.Pipeline;
import com.quant.regime.Strategies;

var ds = Pipeline.loadDataset("../data");
double[][] pos = Strategies.momentumPositions(ds.trendReturns(), 252, 0.10, 0.94, 4.0);
BacktestResult res = Backtest.run(pos, ds.trendReturns(), 5.0);
System.out.printf("Sharpe %.7f%n", res.metrics().get("sharpe")); // 0.8148843
```

---

## 10. How do I build the walk-forward regime gate and filter a strategy?

The gate is causal by construction: expanding-window refits every 63 days,
filtered probability of the lowest-variance state in between.

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
#include <regime/strategies.hpp>
#include <regime/backtest.hpp>
#include <regime/pipeline.hpp>

auto ds = regime::load_dataset("../data");
regime::GateResult g = regime::regime_gate(ds.index_returns, /*n_states=*/3,
                                           /*train_min_days=*/504, /*refit_days=*/63,
                                           regime::GateMode::Prob, /*threshold=*/0.5);
auto pos = regime::apply_gate(regime::momentum_positions(ds.trend_returns), g.gate);
auto res = regime::run_backtest(pos, ds.trend_returns, 5.0);
std::cout << "filtered Sharpe " << res.metrics.at("sharpe") << "\n";  // 1.0402340
```

**Rust**

```rust
use regime::{apply_gate, load_dataset, momentum_positions, regime_gate, run_backtest, GateMode};

let ds = load_dataset("../data")?;
let g = regime_gate(&ds.index_returns, 3, 504, 63, GateMode::Prob, 0.5)?;
let pos = apply_gate(&momentum_positions(&ds.trend_returns, 252, 0.10, 0.94, 4.0)?, &g.gate)?;
let res = run_backtest(&pos, &ds.trend_returns, 5.0)?;
println!("filtered Sharpe {:.7}", res.metrics.sharpe); // 1.0402340
```

**Java**

```java
import com.quant.regime.Backtest;
import com.quant.regime.Pipeline;
import com.quant.regime.Strategies;

var ds = Pipeline.loadDataset("../data");
var g = Strategies.regimeGate(ds.indexReturns(), 3, 504, 63, "prob", 0.5);
double[][] pos = Strategies.applyGate(
        Strategies.momentumPositions(ds.trendReturns(), 252, 0.10, 0.94, 4.0), g.gate());
var res = Backtest.run(pos, ds.trendReturns(), 5.0);
System.out.printf("filtered Sharpe %.7f%n", res.metrics().get("sharpe")); // 1.0402340
```

---

## 11. How do I attribute strategy returns to regimes (crisis table)?

Attribute each day's net return to the day's full-sample K=3 Viterbi
state. (Full-sample decoding is fine *here* — attribution is analysis,
not a trading decision.)

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
#include <regime/backtest.hpp>
#include <regime/hmm.hpp>
#include <regime/strategies.hpp>
#include <regime/pipeline.hpp>

auto ds = regime::load_dataset("../data");
regime::GaussianHMM hmm(3);
hmm.fit(ds.index_returns);
auto states = hmm.viterbi(ds.index_returns);
auto res = regime::run_backtest(regime::momentum_positions(ds.trend_returns),
                                ds.trend_returns, 5.0);
auto table = regime::state_conditional_returns(res.net_returns, states, 3);
std::cout << "crisis ann ret " << table[0].ann_return << "\n";  // -0.0877060
```

**Rust**

```rust
use regime::{load_dataset, momentum_positions, run_backtest,
             state_conditional_returns, GaussianHmm};

let ds = load_dataset("../data")?;
let mut hmm = GaussianHmm::new(3, 1e-8, 500, 1e-8)?;
hmm.fit(&ds.index_returns, None)?;
let states = hmm.viterbi(&ds.index_returns)?;
let res = run_backtest(&momentum_positions(&ds.trend_returns, 252, 0.10, 0.94, 4.0)?,
                       &ds.trend_returns, 5.0)?;
let table = state_conditional_returns(&res.net_returns, &states, 3)?;
println!("crisis ann ret {:+.5}", table[0].ann_return); // -0.08771
```

**Java**

```java
import com.quant.regime.Backtest;
import com.quant.regime.GaussianHmm;
import com.quant.regime.Pipeline;
import com.quant.regime.Strategies;

var ds = Pipeline.loadDataset("../data");
GaussianHmm hmm = new GaussianHmm(3, 1e-8, 500, 1e-8);
hmm.fit(ds.indexReturns());
int[] states = hmm.viterbi(ds.indexReturns());
var res = Backtest.run(Strategies.momentumPositions(ds.trendReturns(), 252, 0.10, 0.94, 4.0),
        ds.trendReturns(), 5.0);
var table = Backtest.stateConditionalReturns(res.netReturns(), states, 3);
System.out.printf("crisis ann ret %+.5f%n", table.get(0).annReturn()); // -0.08771
```

---

## 12. How do I verify my build against the golden values?

Every language ships a golden test suite; this is the manual equivalent.

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
#include <regime/golden.hpp>   // minimal flat-schema JSON reader
#include <regime/pipeline.hpp>

auto cases = regime::load_golden("../data/golden/golden.json");
auto pipe  = regime::run_full_pipeline("../data");
const auto& c = cases.at("hmm_k3_loglik");
assert(std::abs(pipe.fit3.log_likelihood - c.expect.at("loglik")) <= c.tol);
```

**Rust**

```rust
use regime::{load_golden, run_full_pipeline};

let cases = load_golden("../data/golden/golden.json")?;
let pipe = run_full_pipeline("../data")?;
let c = &cases["hmm_k3_loglik"];
assert!((pipe.fit3.log_likelihood - c.expect["loglik"]).abs() <= c.tol);
println!("K=3 loglik OK");
```

**Java**

```java
import com.quant.regime.Golden;
import com.quant.regime.Pipeline;

var cases = Golden.load("../data/golden/golden.json");
var pipe = Pipeline.runFullPipeline("../data");
var c = cases.get("hmm_k3_loglik");
assert Math.abs(pipe.fit3().logLikelihood() - c.expect().get("loglik")) <= c.tol();
System.out.println("K=3 loglik OK");
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
auto pos = regime::momentum_positions(ds.trend_returns);
auto lagged = pos;                     // shift rows down one, zero row 0
lagged.insert(lagged.begin(), std::vector<double>(pos[0].size(), 0.0));
lagged.pop_back();
double a = regime::run_backtest(pos,    ds.trend_returns, 0.0).metrics.at("sharpe");
double b = regime::run_backtest(lagged, ds.trend_returns, 0.0).metrics.at("sharpe");
assert(a != b);
```

**Rust**

```rust
let pos = momentum_positions(&ds.trend_returns, 252, 0.10, 0.94, 4.0)?;
let mut lagged = vec![vec![0.0; pos[0].len()]];
lagged.extend_from_slice(&pos[..pos.len() - 1]);
let a = run_backtest(&pos, &ds.trend_returns, 0.0)?.metrics.sharpe;
let b = run_backtest(&lagged, &ds.trend_returns, 0.0)?.metrics.sharpe;
assert!(a != b);
```

**Java**

```java
double[][] pos = Strategies.momentumPositions(ds.trendReturns(), 252, 0.10, 0.94, 4.0);
double[][] lagged = new double[pos.length][pos[0].length];
for (int t = 1; t < pos.length; t++) lagged[t] = pos[t - 1];
double a = Backtest.run(pos, ds.trendReturns(), 0.0).metrics().get("sharpe");
double b = Backtest.run(lagged, ds.trendReturns(), 0.0).metrics().get("sharpe");
assert a != b;
```
