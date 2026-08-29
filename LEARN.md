# LEARN — Market Regimes, Hidden Markov Models, and Regime-Aware Trading

This document teaches the theory behind the `regime` toolkit: why markets
have regimes, how a hidden Markov model finds them, why the algorithms are
written the way they are, and what happens when you use the detected
regimes to gate real strategies. Notation follows
[`API_SPEC.md`](API_SPEC.md); every worked number is reproduced by the
code and pinned in `data/golden/golden.json`.

---

## 1. Market regimes: why they exist

Look at any long history of daily equity returns and two things jump out.
First, **volatility clusters**: large moves follow large moves. The
autocorrelation of returns themselves is near zero (markets are hard to
predict), but the autocorrelation of *squared* or *absolute* returns is
strongly positive for months. Second, the *character* of the market shifts
in episodes: long stretches of gentle drift upward (bull markets), choppy
sideways periods, and short, violent drawdowns in which volatility triples
and correlations lurch toward one (crises: 1987, 1998, 2008, March 2020).

Why should regimes exist at all?

* **The economy itself is regime-like.** Expansions and recessions are
  persistent states; monetary policy switches between easing and
  tightening cycles; funding liquidity is either abundant or suddenly
  scarce.
* **Risk appetite is reflexive.** After losses, leveraged investors are
  forced to de-risk, which causes more losses — a feedback loop that makes
  high-volatility states self-sustaining while they last.
* **Volatility clustering** is exactly what a mixture of persistent states
  produces: if today's return is drawn from a "crisis" distribution with
  25% annualized-equivalent daily vol, tomorrow's probably is too, because
  the state is sticky.

A two- or three-state description — *calm-bull*, *choppy*, *crisis* — is a
crude but remarkably effective summary. The empirical signature that any
model must reproduce: crisis states have **negative mean, high variance,
and short duration**; calm states have **positive mean, low variance, and
long duration**. That asymmetry ("stairs up, elevator down") is precisely
what the bundled data generator builds in and what the fitted HMM
recovers.

### Why not just threshold realized volatility?

You could label "high vol" whenever a 20-day realized vol exceeds some
cutoff. That works, but it is ad hoc (which window? which cutoff?), it
produces jittery labels near the threshold, and it gives you no
probabilistic machinery — no likelihood to compare models, no smoothed
probabilities, no forecast of regime persistence. The HMM gives all of
that from one coherent generative model.

---

## 2. Hidden Markov models from first principles

### 2.1 Markov chains

A **Markov chain** on states $\{1,\dots,K\}$ is a random sequence
$s_1, s_2, \dots$ in which the future depends on the past only through the
present:

$$
P(s_{t+1} = j \mid s_t = i, s_{t-1}, \dots) = P(s_{t+1}=j \mid s_t = i) = A_{ij}.
$$

The matrix $A$ is **row-stochastic** ($\sum_j A_{ij} = 1$), and
$\pi_i = P(s_1 = i)$ is the initial distribution. Persistence lives on the
diagonal: with self-transition probability $A_{ii}$, the expected sojourn
time in state $i$ is geometric,

$$
E[\text{duration}] = \frac{1}{1 - A_{ii}}.
$$

The fitted crisis state on the bundled data has $A_{00} \approx 0.9204$,
an expected duration of about $1/(1-0.9204) \approx 12.6$ trading days —
short and violent, exactly as it should be. The calm state's
$A_{11} \approx 0.9916$ implies a ~118-day expected stay.

A chain that can reach every state from every state has a unique
**stationary distribution** $\pi^\*$ solving $\pi^\* A = \pi^\*$ — the
long-run fraction of time spent in each state. The toolkit computes it by
the pinned power method (start uniform, iterate $\pi \leftarrow \pi A$,
renormalize by the L1 sum, stop when the max change is below $10^{-13}$).
On the bundled data: $\pi^\* \approx (0.0566,\ 0.6767,\ 0.2667)$ — the
market spends ~5.7% of its life in crisis.

### 2.2 Hiding the chain: emissions

In markets we never observe the regime; we observe returns. A **hidden
Markov model** keeps the Markov chain but makes it latent, and attaches to
each state an **emission distribution** for what we actually see:

$$
x_t \mid s_t = i \;\sim\; \mathcal N(\mu_i, \Sigma_i), \qquad
\Sigma_i = \mathrm{diag}(\sigma^2_{i1},\dots,\sigma^2_{iD}).
$$

This toolkit uses diagonal Gaussians with $D \in \{1,2\}$ (a return
series, optionally paired with a second feature). The full parameter set
is $\theta = (\pi, A, \{\mu_i\}, \{\Sigma_i\})$. Two conditional
independence assumptions define the model:

1. $s_t$ depends on the past only through $s_{t-1}$ (Markov chain);
2. $x_t$ depends on everything else only through $s_t$ (emission).

The joint likelihood of a path and the data factorizes as

$$
p(x_{1:T}, s_{1:T}) = \pi_{s_1} b_{s_1}(x_1) \prod_{t=2}^{T} A_{s_{t-1} s_t}\, b_{s_t}(x_t),
\qquad b_i(x) = \mathcal N(x; \mu_i, \Sigma_i).
$$

Marginally, each $x_t$ is a **Gaussian mixture** — which is why an HMM
fitted to returns naturally produces fat tails and volatility clustering
even though each state is thin-tailed Gaussian.

### 2.3 The three classic problems (Rabiner)

1. **Evaluation** — given $\theta$, compute the likelihood
   $p(x_{1:T} \mid \theta)$. Naively this is a sum over $K^T$ paths;
   the **forward algorithm** does it in $O(TK^2)$.
2. **Decoding** — given $\theta$, find the most probable state path
   $\arg\max_{s_{1:T}} p(s_{1:T} \mid x_{1:T})$. Solved by **Viterbi**
   dynamic programming, also $O(TK^2)$.
3. **Learning** — find $\theta$ maximizing $p(x_{1:T} \mid \theta)$.
   Solved (to a local optimum) by **Baum-Welch**, the EM algorithm for
   HMMs, which repeatedly runs forward-backward (the E-step) and
   re-estimates parameters in closed form (the M-step).

---

## 3. Forward-backward and the scaling problem

### 3.1 The forward recursion

Define the forward variable $\alpha_t(i) = p(x_{1:t}, s_t = i)$. It obeys

$$
\alpha_1(i) = \pi_i\, b_i(x_1), \qquad
\alpha_{t}(j) = \Big[\sum_i \alpha_{t-1}(i)\, A_{ij}\Big] b_j(x_t),
\qquad p(x_{1:T}) = \sum_i \alpha_T(i).
$$

### 3.2 Why underflow happens

Each step multiplies by a transition probability ($<1$) and an emission
density. A typical daily-return density value is $O(10)$–$O(100)$ (the
density of a $\mathcal N(0, 0.006^2)$ at its mode is ~66), but the
*product* over 2000 days is $e^{\log L}$ with $\log L \approx 6886$ on the
bundled data — fine — while for a poorly matched state the per-step factor
can be astronomically small: a $-10\%$ return under a calm state with
$\sigma = 0.006$ has density $e^{-139}$. Raw $\alpha_t(i)$ therefore
under- or overflows IEEE doubles (min normal $\approx 10^{-308}$) within a
few hundred steps. Two standard fixes:

* **Log-space**: propagate $\log\alpha$ with log-sum-exp at every sum.
  Robust but slow (a `log`/`exp` pair per state pair per step) and
  awkward for the $\xi$ statistics in EM.
* **Scaling (Rabiner)** — the pinned choice: renormalize $\alpha$ each
  step and remember the normalizers.

### 3.3 The pinned scaled algorithm

Two-level protection. First, the emission densities themselves are
shifted by the per-step maximum log-emission $m_t = \max_i \log b_i(x_t)$,
giving $\tilde b_t(i) = e^{\log b_i(x_t) - m_t} \le 1$ (guards against the
$e^{-139}$-type underflow *inside one step*). Then the Rabiner
normalization handles the accumulation across steps:

$$
\hat\alpha_1 \propto \pi_i \tilde b_1(i),\qquad
\hat\alpha_t(j) \propto \Big[\sum_i \hat\alpha_{t-1}(i) A_{ij}\Big]\tilde b_t(j),
$$

each step divided by its sum $d_t$. The true normalizer is
$c_t = d_t e^{m_t} = p(x_t \mid x_{1:t-1})$ — the one-step predictive
density — and the log-likelihood falls out for free:

$$
\log p(x_{1:T}) = \sum_{t=1}^{T} \log c_t = \sum_{t=1}^{T} (\log d_t + m_t).
$$

The scaled $\hat\alpha_t(i)$ are exactly the **filtered probabilities**
$P(s_t = i \mid x_{1:t})$ — the real-time regime estimate the trading gate
uses. The backward pass reuses the same $d_t$, after which the
**smoothed** posteriors and pair marginals need *no further
normalization*:

$$
\gamma_t(i) = P(s_t = i \mid x_{1:T}) = \hat\alpha_t(i)\,\hat\beta_t(i),
\qquad
\xi_t(i,j) = \hat\alpha_t(i)\, A_{ij}\, \tilde b_{t+1}(j)\, \hat\beta_{t+1}(j) / d_{t+1}.
$$

**Filtered vs smoothed matters enormously for trading.** Smoothed
probabilities use the whole sample — including the future — and are for
teaching and post-hoc analysis only. Any live decision must use filtered
probabilities. (Section 9.)

### 3.4 Worked example (matches `test_forward_likelihood_matches_enumeration`)

Take $K=2$, $T=3$: $\pi = (0.6, 0.4)$,
$A = \begin{pmatrix}0.7 & 0.3\\ 0.2 & 0.8\end{pmatrix}$,
$\mu = (-1, 1)$, $\sigma^2 = (0.5, 2.0)$, data $x = (-0.5, 0.3, 1.2)$.

Emission densities $b_t(i)$:

| $t$ | $x_t$ | $b_t(0)$ | $b_t(1)$ |
|---|---|---|---|
| 1 | $-0.5$ | 0.439391 | 0.160733 |
| 2 | $0.3$ | 0.104104 | 0.249571 |
| 3 | $1.2$ | 0.004461 | 0.279288 |

Forward pass with plain Rabiner scaling (the $m_t$ shift changes nothing
here because the densities are moderate):

* $t=1$: $\tilde a = (0.6 \cdot 0.4394,\ 0.4 \cdot 0.1607) = (0.263635, 0.064293)$,
  $c_1 = 0.327928$, $\hat\alpha_1 = (0.803941, 0.196059)$.
* $t=2$: $\tilde a = (0.062668, 0.099337)$, $c_2 = 0.162004$,
  $\hat\alpha_2 = (0.386827, 0.613173)$.
* $t=3$: $\tilde a = (0.001755, 0.169412)$, $c_3 = 0.171167$,
  $\hat\alpha_3 = (0.010253, 0.989747)$.

Likelihood $L = c_1 c_2 c_3 = 9.09337754 \times 10^{-3}$, so
$\log L = -4.700208872784$. Brute-force enumeration over all $2^3 = 8$
paths gives the same number to $10^{-12}$ — the test that anchors every
port's forward pass. Note how the filter flips: after seeing $-0.5$ the
model is 80% sure of state 0; after $1.2$ it is 99% sure of state 1.

---

## 4. Baum-Welch EM

### 4.1 The updates

EM alternates: **E-step** — run forward-backward under the current
$\theta$ to get $\gamma_t(i)$ and $\xi_t(i,j)$; **M-step** — re-estimate
in closed form:

$$
\pi_i \leftarrow \gamma_1(i), \qquad
A_{ij} \leftarrow \frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)},
$$

$$
\mu_{id} \leftarrow \frac{\sum_t \gamma_t(i)\, x_{td}}{\sum_t \gamma_t(i)}, \qquad
\sigma^2_{id} \leftarrow \frac{\sum_t \gamma_t(i)\,(x_{td} - \mu_{id}^{\text{new}})^2}{\sum_t \gamma_t(i)},
$$

i.e., soft-count versions of the obvious frequency/mean/variance
estimators, with each observation weighted by the posterior probability of
having come from state $i$. The variance update uses the **new** means
(pinned), and every variance is floored at $10^{-8}$ each iteration.

Each iteration provably does not decrease the log-likelihood (Jensen's
inequality on the EM lower bound); `test_em_monotonic_on_bundled_data`
asserts it numerically. Convergence is declared when the improvement drops
below $10^{-8}$; on the bundled index, K=2 converges in 27 E-steps and K=3
in 63.

### 4.2 Local optima and initialization

The likelihood surface of an HMM is riddled with local optima and is
invariant under state relabeling ($K!$ equivalent maxima). Two design
choices in this toolkit deal with that:

* **Pinned deterministic initialization** (no RNG anywhere): state $k$'s
  mean is set to the empirical quantile at $q_k = (2k+1)/2K$ (for $K=3$:
  the 1/6, 1/2, 5/6 quantiles — spread across the data), every state
  starts at the full-sample variance, transitions start at 0.9
  self-transition, $\pi$ uniform. Every language starts EM from the
  *identical* point, so every language walks the *identical* path to the
  *identical* local optimum — this is what makes 1e-6 cross-language
  golden agreement possible at all. Random restarts would find slightly
  better optima occasionally, at the price of irreproducibility.
* **Post-fit label sorting**: after EM finishes (once, never during
  iterations), states are permuted so means are ascending on dimension 0.
  State 0 = lowest mean, which on the bundled data is the crisis state.

Failure modes to know: a state can collapse onto a handful of points
(variance $\to 0$, likelihood $\to \infty$ — the classic Gaussian-mixture
degeneracy), which the variance floor prevents; and with more states than
the data supports, EM parks surplus states on near-duplicates (again the
floor engages — tested via `test_variance_floor_engages`). Non-convergence
after 500 iterations is reported as a flag on the result, never an
exception: a nearly converged model is still usable, and walk-forward
loops must not die mid-backtest.

### 4.3 What EM found on the bundled data (golden values)

Full-sample K=3 fit on `market_index.csv` (2000 days), daily units:

| state | label | mean | std | annualized mean | annualized vol |
|---|---|---|---|---|---|
| 0 | crisis | $-0.003432$ | $0.025804$ | $-86.5\%$ | $41.0\%$ |
| 1 | calm-bull | $+0.000635$ | $0.005792$ | $+16.0\%$ | $9.2\%$ |
| 2 | choppy | $+0.001040$ | $0.010580$ | $+26.2\%$ | $16.8\%$ |

with $\log L = 6886.4498$ and transition matrix (rows sum to 1)

$$
A \approx \begin{pmatrix}
0.9204 & 0.0000 & 0.0796\\
0.0008 & 0.9916 & 0.0076\\
0.0148 & 0.0214 & 0.9637
\end{pmatrix}.
$$

Note that the generator's truth was $\mu = (0.0007, 0.0, -0.0018)$ and
$\sigma = (0.006, 0.011, 0.025)$ for calm/choppy/crisis: EM recovered the
volatilities almost exactly and the means roughly (means of noisy daily
returns are intrinsically hard to estimate — a theme that returns in
Section 10). Also note $A_{01} \approx 10^{-9}$: the fitted crisis state
never exits directly into deep calm; it decompresses through the choppy
state first. Real markets do the same.

---

## 5. Viterbi decoding

Smoothed probabilities give per-day marginals; Viterbi answers a different
question — the single **jointly** most probable path. Define
$\delta_t(j) = \max_{s_{1:t-1}} \log p(x_{1:t}, s_{1:t-1}, s_t = j)$:

$$
\delta_1(j) = \log\pi_j + \log b_j(x_1), \qquad
\delta_t(j) = \max_i \big[\delta_{t-1}(i) + \log A_{ij}\big] + \log b_j(x_t),
$$

recording the argmax in $\psi_t(j)$ and backtracking from
$\arg\max_j \delta_T(j)$. Everything is done in log-space (no underflow
possible; $\log 0 = -\infty$ is harmless in a max). Ties break toward the
lower state index (pinned, so all languages produce the identical path).

On the worked example above, the best path is $(0, 1, 1)$ with path
probability $4.410 \times 10^{-3}$ — verified by enumerating all 8 paths.
On the bundled data the full-sample K=3 Viterbi path spends
$(99, 1372, 529)$ days in states $(0, 1, 2)$ (exact golden integers) and
agrees with the (volatility-mapped) true generator states on **94.85%** of
days — a teaching number showing that with well-separated states, regime
recovery works well *in hindsight*.

The subtlety: the Viterbi path is not the sequence of per-day argmaxes of
$\gamma_t$. The path respects transition costs; the marginals do not, and
the per-day argmax sequence can even contain transitions with probability
zero.

---

## 6. Model selection: choosing K

More states always fit better in-sample, so raw likelihood cannot choose
$K$. Penalize by the free parameter count — for $K$ diagonal-Gaussian
states in $D$ dimensions:

$$
p = \underbrace{2KD}_{\text{means+vars}} + \underbrace{K(K-1)}_{A\ \text{rows}} + \underbrace{K-1}_{\pi},
\qquad
\mathrm{AIC} = -2\log L + 2p, \quad
\mathrm{BIC} = -2\log L + p\ln T .
$$

Lower is better; BIC penalizes harder for $T = 2000$ ($\ln T \approx 7.6$
vs 2). On the bundled index:

| K | $\log L$ | p | AIC | BIC |
|---|---|---|---|---|
| 2 | 6831.0255 | 7 | $-13648.05$ | $-13608.84$ |
| 3 | 6886.4498 | 14 | $-13744.90$ | $-13666.49$ |

Both criteria prefer $K = 3$ — correctly, since the generator truly has
three states. In real data the choice is murkier: information criteria for
HMMs are heuristics (the regularity conditions behind AIC/BIC don't
strictly hold for mixtures), K=4+ states often just split one regime into
twins, and the honest approach is to fix a small K on economic grounds
(2 or 3) and check robustness. $K=1$ is rejected outright by the toolkit —
it is not an HMM, just a single Gaussian.

---

## 7. Time-series momentum

### 7.1 The effect and the evidence

Time-series momentum (TSM, "trend following"): buy assets whose own
trailing return is positive, short those where it is negative. Moskowitz,
Ooi & Pedersen (2012) documented significant 12-month TSM across 58
futures markets; the effect underlies the entire managed-futures/CTA
industry, with a track record over a century of data (Hurst, Ooi &
Pedersen, 2017). Behavioral stories: initial underreaction to news
(anchoring), later herding, and non-profit-seeking flows (hedgers, central
banks). Its famous property is **crisis alpha in slow bear markets**
(short positions have time to build) but poor performance in sharp
V-shaped reversals — a 252-day signal is still long after a two-week
crash, which is exactly what the crisis-state attribution table shows.

### 7.2 The pinned implementation

Signal at day $t$ (needs $t \ge \text{lookback}-1$): the sign of the
trailing 252-day cumulative return $\prod_{u=t-251}^{t}(1+r_u) - 1 \in \{-1,0,+1\}$.

**Vol targeting** sizes each position inversely to risk. The EWMA variance
(RiskMetrics $\lambda = 0.94$, zero-mean convention) is seeded at
$t = 251$ with the mean squared return over the first window, then

$$
\sigma^2_t = \lambda \sigma^2_{t-1} + (1-\lambda) r_t^2, \qquad
\text{lev}_t = \min\!\Big(\frac{10\%}{\sqrt{252\,\sigma_t^2}},\ 4\Big),
\qquad
P_{t,a} = \text{sign}_{t,a} \cdot \text{lev}_{t,a} / A .
$$

Why vol target? (i) A constant-notional trend book is dominated by its
most volatile asset and its most volatile months; scaling by
$1/\hat\sigma$ equalizes risk across assets and time. (ii) Because
volatility is persistent and negatively related to returns, deleveraging
into rising vol is itself a (mild) source of Sharpe improvement. The 4x
cap prevents grotesque leverage when the vol estimate goes tiny. Zero
estimated vol gets the cap (pinned edge case).

On the bundled data, unfiltered momentum earns +3.38% ann. at 4.14% vol
(Sharpe 0.81, maxDD $-10.5\%$) — but $-8.77\%$ annualized inside the
decoded crisis state (a golden sign case).

---

## 8. FX carry

### 8.1 Interest differentials and uncovered interest parity

A currency position earns two things: the **spot move** and the **interest
differential** (deposit rate at home minus abroad — in practice harvested
through forward points, since covered interest parity ties the
forward-spot basis to the rate differential). Uncovered interest parity
(UIP) says high-rate currencies should *depreciate* by exactly their rate
advantage, leaving no profit. Empirically UIP fails on average — the
"forward premium puzzle" (Fama, 1984): high-rate currencies have
historically depreciated less than the differential, so borrowing low-rate
currencies (JPY, CHF) to lend high-rate ones (AUD, NZD, EM) has been
profitable *on average*.

### 8.2 Crash risk: pennies before a steamroller

The catch is the *distribution* of those profits: carry returns are
strongly **negatively skewed**. Carry trades are crowded and funded with
leverage; when volatility spikes, everyone unwinds at once, and the
high-rate currencies crash against the funding currencies precisely when
everything else is also going wrong (AUD/JPY in 2008: years of accrued
carry gone in weeks). Hence Brunnermeier, Nagel & Pedersen's summary:
**"picking up pennies in front of a steamroller"** — a steady stream of
small gains (the daily accrual, worth $\text{diff}/252$ per day) punctuated
by rare, large, correlated losses. Carry is, in effect, short a
volatility/liquidity option; its Sharpe overstates its safety because
Sharpe is blind to skew.

The bundled generator builds this in literally: spot drift per unit of
relative differential is $+0.75$ in calm, $+0.2$ in choppy, and $-12$ in
crisis (the unwind). The result in the crisis-attribution table: carry
earns +10.1%/+16.7% annualized in calm/choppy states and **loses 11.1%**
annualized in crisis.

### 8.3 The pinned implementation

Every 21 trading days (day indices 0, 21, 42, ...), rank the six synthetic
currencies by that day's annualized differential, descending; ties break
by ascending asset index (pinned so all languages agree). Long the top 3
at $+1/3$ each, short the bottom 3 at $-1/3$ each; hold in between. Traded
return on day $t$ is $\text{spot}_t + \text{diff}_{t-1}/252$ — the
differential accrued is the one *set at the previous close* (known in
advance; day 0 accrues nothing). Result: +10.77% ann., Sharpe 1.58, maxDD
$-10.0\%$ unfiltered.

**Market-convention note (equity vs FX).** Equity regimes are usually
detected on index returns (as here); FX carry practitioners watch rate
differentials, cross-currency basis, and vol risk-reversals. Real G10
carry must also handle quote conventions (EURUSD vs USDJPY direction),
transaction costs via forward spreads rather than bps on notional, and
T+2 settlement. The synthetic setup keeps the economics (accrual +
spot + regime-dependent unwind) without the conventions.

---

## 9. Regime filtering, and the danger of lookahead

### 9.1 The gate

The filter multiplies each day's positions by
$p_{\text{calm}}(t) = P(s_t = k^\* \mid r_{1:t})$ — the **filtered**
probability of the benign state under a model fitted **only on data
through day $t$**:

* First fit at day 503 (after `train_min_days` = 504 observations),
  refits every 63 days on an expanding window, warm-started from the
  previous fitted parameters (faster, and keeps the label identity
  stable).
* Between refits the probability advances by single scaled-forward steps —
  identical math, no recomputation, no peeking.
* $k^\*$ is the **lowest-variance** state, not the highest-mean one:
  walk-forward fits occasionally isolate a tiny high-mean state, while
  the low-vol state is robustly "the calm one."
* Before the first fit, the gate is 1 (no filtering).

`gate_mode` is `"prob"` (position scaled continuously by the probability)
or `"binary"` (all-in/all-out at a 0.5 threshold). Probability scaling
trades more smoothly and avoids threshold-flapping turnover.

### 9.2 What filtering does — and what it costs

Golden results on the bundled data (5 bps costs):

| strategy | ann ret | Sharpe | maxDD |
|---|---|---|---|
| momentum unfiltered | +3.38% | 0.815 | $-10.48\%$ |
| momentum filtered | +3.33% | 1.040 | $-7.37\%$ |
| carry unfiltered | +10.77% | 1.579 | $-10.01\%$ |
| carry filtered | +6.43% | 1.493 | $-4.95\%$ |
| combined unfiltered | +7.07% | 1.800 | $-5.23\%$ |
| combined filtered | +4.88% | 1.866 | $-3.41\%$ |

The mechanism: both strategies make their money in calm/choppy states and
lose in crisis, so scaling exposure by $P(\text{calm})$ cuts most of the
crisis losses (crisis-state combined return improves from $-9.9\%$ to
$-0.9\%$ annualized) while also cutting *some* good exposure — the gate
is below 1 plenty of the time, and it reacts with a lag. Net effect,
honestly reported: **total return falls** (7.07% → 4.88%), volatility and
drawdown fall *faster*, so **risk-adjusted metrics improve** (Sharpe
1.80 → 1.87, maxDD $-5.2\%$ → $-3.4\%$, Calmar 1.35 → 1.43). A regime
filter buys risk reduction; it is not free performance. If your leverage
is constrained by drawdown (as every real fund's is), risk-adjusted
improvement can be re-levered into higher return — that is the actual
economic argument for filtering.

Carry shows the trade-off starkest: its unfiltered Sharpe (1.58) barely
changes filtered (1.49) but its drawdown halves — the filter converts a
negatively skewed return stream into a much less scary one at the price of
40% of the return.

### 9.3 Lookahead bias

Lookahead is the deadliest and most common backtest sin: using information
at decision time that did not exist yet. Regime studies are *especially*
prone to it, in three flavors:

1. **Smoothed probabilities.** $\gamma_t = P(s_t \mid x_{1:T})$
   conditions on the entire future. Gating on smoothed (or full-sample
   Viterbi) states produces spectacular fake Sharpe — the model "knew" the
   crash was coming because it saw the data after it. Only filtered
   probabilities $P(s_t \mid x_{1:t})$ are legal at decision time.
2. **Full-sample fitting.** Even with filtered probabilities, parameters
   fitted on all 2000 days leak the future through $\hat\theta$ (the model
   already "knows" a crisis state with exactly 2020-vintage volatility
   exists). Hence the walk-forward refit schedule on expanding windows.
3. **Same-day execution.** A position "decided at the close of day $t$"
   must earn day $t+1$'s return, never day $t$'s. The pinned accounting is
   $\text{gross}_t = \sum_a P_{t-1,a}\, r_{t,a}$; the test suite verifies
   that shifting signals by one day changes the results
   (`test_no_lookahead_shift`) — if it didn't, the engine would be
   leaking.

The same discipline shows up in small pinned details: the carry accrual
uses $\text{diff}_{t-1}$, the EWMA vol at day $t$ uses returns through
$t$, and the gate between refits is advanced by forward steps that never
re-read the past under future parameters.

---

## 10. Honest limitations

* **Detection lag.** A filtered probability can only react *after*
  crisis-scale returns arrive; the first few bad days are always taken at
  full exposure, and re-entry lags recoveries symmetrically. Regime
  filters shave tails, they do not remove them.
* **Regimes are a modeling fiction.** Markets do not literally switch
  between three Gaussians. The HMM is a useful compression of vol
  clustering; its "states" are statistical, and their boundaries move
  with every refit.
* **Mean estimates are weak.** Vol separates regimes well; means barely
  do ($0.07\%$ daily drift against $0.6\%$ daily noise needs decades to
  estimate). Any regime logic keyed on fitted *means* is fragile — this is
  why the pinned gate keys on variance.
* **Local optima and label instability.** Warm-starting keeps walk-forward
  fits coherent, but a refit can still reorganize states after unusual
  data; the variance-based $k^\*$ identification is a mitigation, not a
  cure.
* **Overfitting the meta-strategy.** K, refit frequency, gate mode,
  thresholds — each is a knob, and tuning knobs on the same history that
  evaluates them is in-sample selection in disguise. The bundled data are
  *generated from* a 3-state model; real data will never reward the HMM
  this cleanly (94.85% decoding accuracy is a best case, not an
  expectation).
* **Costs and capacity.** The filter trades every day (the gate wiggles);
  in `"prob"` mode this is modest, but with expensive instruments the
  extra turnover eats the benefit. Golden turnover values exist precisely
  so ports can't quietly diverge on this.
* **Gaussian tails.** Within-state Gaussian emissions understate extreme
  days even in the crisis state; Student-t emissions are the standard
  upgrade.

---

## 11. Common pitfalls and numerical issues

* **Underflow in forward-backward** — the whole point of scaling; see
  Section 3.2. Never implement the textbook unscaled recursion for
  $T > \sim 100$.
* **Dividing by $\sum_t \gamma_t(i) \approx 0$** for a dead state — the
  variance floor plus quantile initialization keep states alive here; in
  general, guard the M-step denominators.
* **$\log 0$ in Viterbi** when $A_{ij} = 0$: fine in log-space max
  ($-\infty$ never wins), fatal if you exponentiate.
* **Comparing likelihoods across different scalings.** $\log L$ must be
  $\sum_t (\log d_t + m_t)$, including the emission shifts $m_t$;
  forgetting the shifts silently corrupts AIC/BIC and EM convergence
  checks.
* **Sorting states mid-EM.** Relabeling during iterations breaks the
  monotonicity guarantee and cross-language agreement; the pin is to sort
  once, after convergence.
* **`ddof` mismatches.** All variances here are population (`ddof=0`) —
  a `ddof=1` port fails golden values at the fourth decimal.
* **Tie-breaks.** Carry ranks ties by asset index; Viterbi argmax ties
  break low. Undefined tie-breaks are the classic source of
  cross-language flakiness.
* **Calendar vs block months.** `worst_month` uses calendar months when
  dates exist and 21-day blocks otherwise — mixing the two changes the
  metric.

---

## 12. Interview-style Q&A

**Q1. Why does the raw forward algorithm underflow, and what are the two
standard fixes?**
Each $\alpha_t(i)$ multiplies $t$ probabilities and densities; over
hundreds of steps the product leaves the range of doubles (below
$\sim 10^{-308}$). Fixes: log-space with log-sum-exp, or Rabiner scaling
(normalize each step, keep $\log L = \sum_t \log c_t$). Scaling is used
here because the normalizers give the likelihood and the filtered
probabilities for free, and the E-step statistics need no re-normalization.

**Q2. What is the difference between filtered, smoothed, and Viterbi
state estimates, and which may a trading rule use?**
Filtered: $P(s_t \mid x_{1:t})$ — real-time, causal. Smoothed:
$P(s_t \mid x_{1:T})$ — uses the future; analysis only. Viterbi: the
single jointly most probable path given all data — also non-causal.
A live trading rule may only use filtered probabilities (and parameters
fitted on past data).

**Q3. Why is EM guaranteed not to decrease the likelihood, and does it
find the global optimum?**
Each M-step maximizes the expected complete-data log-likelihood
$Q(\theta \mid \theta^{(k)})$, a Jensen lower bound touching $\log L$ at
$\theta^{(k)}$; raising the bound raises the likelihood. It converges only
to a stationary point — HMM surfaces have many local optima plus $K!$
relabeled copies of each, so initialization determines the answer. This
toolkit pins a deterministic quantile-based init precisely so every
language reaches the same optimum.

**Q4. How would you pick the number of states?**
Fit K=2,3,... and compare AIC/BIC (penalized likelihood, $p = 2KD +
K(K-1) + (K-1)$ free parameters); prefer BIC for long samples. But treat
them as heuristics: check that added states are economically distinct (not
twins), confirm stability across subsamples, and default to small K on
economic priors. Here both criteria pick K=3, which is the true DGP.

**Q5. Expected duration of a regime with self-transition 0.98?**
Geometric: $1/(1-0.98) = 50$ days. Persistence lives on the diagonal of
$A$; the off-diagonal pattern tells you *where* a regime exits to (the
fitted crisis here exits to choppy, essentially never straight to calm).

**Q6. Why does vol targeting tend to improve momentum's Sharpe?**
It equalizes risk contributions across assets and time, so no single
volatile episode dominates the sample; and because vol is persistent and
rises in drawdowns, scaling by $1/\hat\sigma$ systematically trims
exposure exactly when tail risk is elevated. The cap (4x) stops the
inverse scaling from exploding when estimated vol is tiny.

**Q7. Why is FX carry called "picking up pennies in front of a
steamroller"?**
The daily accrual $\text{diff}/252$ is small and steady (pennies), while
the loss mode — a crowded, levered unwind where high-rate currencies crash
against funding currencies during vol spikes — is rare, large, and
correlated with everything else (the steamroller). Carry is short a
volatility/liquidity option; its arithmetic Sharpe hides negative skew.

**Q8. Your regime filter improved Sharpe but lowered total return. Is it
worth running?**
Depends on the binding constraint. If capital is constrained and leverage
free, higher Sharpe can be re-levered to dominate outright. If leverage is
capped, you are explicitly paying return for smaller drawdowns — often
still rational (drawdowns trigger redemptions and forced deleveraging),
but it must be stated as a trade, not a free lunch. Also check the
filter's extra turnover survives realistic costs.

**Q9. Name three ways lookahead bias can sneak into a regime-switching
backtest.**
(1) Gating on smoothed probabilities or full-sample Viterbi states;
(2) using HMM parameters fitted on the full sample even with filtered
probabilities; (3) applying day-$t$ decisions to day-$t$ returns (or
accruing day-$t$ carry at the day-$t$ differential). The defenses:
walk-forward refits on expanding windows, filtered-only gates, strict
$P_{t-1} \cdot r_t$ accounting, and a test asserting that a one-day signal
shift changes the results.

**Q10. Why floor the variances in Baum-Welch?**
Without a floor, a state can collapse onto one or a few nearly identical
points: its variance $\to 0$ and the likelihood diverges (the Gaussian
mixture degeneracy), which is a spurious "optimum". The floor ($10^{-8}$)
keeps every state's density proper, and it is what lets K exceed the
data's distinct-value support without crashing.

---

## 13. Further reading

* **L. Rabiner (1989)**, "A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition," *Proc. IEEE* — the canonical HMM
  tutorial; the scaling scheme here is his.
* **C. Bishop (2006)**, *Pattern Recognition and Machine Learning*,
  ch. 13 — HMMs and EM with modern notation.
* **J. Hamilton (1989)**, "A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle," *Econometrica* —
  regime-switching models in economics; and Hamilton (1994), *Time Series
  Analysis*, ch. 22.
* **A. Ang & G. Bekaert (2002)**, "International Asset Allocation with
  Regime Shifts," *RFS* — regime-aware portfolio choice.
* **T. Moskowitz, Y. H. Ooi & L. Pedersen (2012)**, "Time Series
  Momentum," *JFE*; **B. Hurst, Y. H. Ooi & L. Pedersen (2017)**, "A
  Century of Evidence on Trend-Following Investing," *JPM*.
* **M. Brunnermeier, S. Nagel & L. Pedersen (2008)**, "Carry Trades and
  Currency Crashes," *NBER Macro Annual* — carry crash risk.
* **E. Fama (1984)**, "Forward and Spot Exchange Rates," *JME* — the
  forward premium puzzle.
* **RiskMetrics Technical Document (1996)** — the $\lambda = 0.94$ EWMA
  volatility convention.
* **M. López de Prado (2018)**, *Advances in Financial Machine Learning* —
  backtest hygiene, lookahead and selection bias.
