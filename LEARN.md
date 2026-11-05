# LEARN — Market Regimes, Hidden Markov Models, and Regime-Aware Trading

**The result first.** On real US equity data from 1963 to 2026 the HMM gate
lost to the unfiltered book and to a 21-day vol rule, with bootstrap intervals
below zero (section 15, `docs/REAL_DATA_EVALUATION.md`). By the rule in section
16, tuning has stopped. This guide is now a case study in evaluating a regime
signal: sections 1 to 9 teach the machinery, which works as specified, and
sections 10 to 16 teach how to tell whether it earned anything.

The guide is written for two readers: a student meeting these ideas for the
first time, and a practitioner who wants the formal statement and the evidence.
Each new idea comes in that order: plain idea, formal statement, worked example.
Notation follows [`API_SPEC.md`](API_SPEC.md). Worked numbers on the bundled data
are reproduced by the code and pinned in `data/golden/golden.json`.

**A warning that applies to every bundled gate figure.** The bundled data are
seed 19 of a synthetic generator, and that seed flatters the gate: its
post-warm-up Sharpe goes from 0.54 to 0.99, which sits at the 85th percentile of
100 seeds, and the 95 % interval includes zero. Bundled figures show how the code
behaves. They are not evidence for the gate.

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
long duration**. That asymmetry ("stairs up, elevator down") is what the
bundled generator builds in. The fitted HMM recovers the variances well. It
does not reliably recover the means: on seed 19 the crisis state's sampled mean
is positive (section 4.3).

### Why not just threshold realized volatility?

You could label "high vol" whenever a 20-day realized vol exceeds some
cutoff. It is ad hoc (which window? which cutoff?), it produces jittery labels
near the threshold, and it gives you no likelihood to compare models, no
smoothed probabilities and no forecast of regime persistence. The HMM gives all
of that from one generative model.

That is an argument about description, and it does not carry over to trading. On
real data a 21-day vol rule beat the HMM gate (section 15). Always keep the
simple rule in the comparison.

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

The fitted crisis state on the bundled data (state 2, the highest-variance
state) has $A_{22} \approx 0.9352$, an expected duration of about
$1/(1-0.9352) \approx 15.4$ trading days. The calm state's
$A_{11} \approx 0.9809$ implies a stay of about 52 days.

A chain that can reach every state from every state has a unique
**stationary distribution** $\pi^\*$ solving $\pi^\* A = \pi^\*$ — the
long-run fraction of time spent in each state. The toolkit computes it by
the pinned power method (start uniform, iterate $\pi \leftarrow \pi A$,
renormalize by the L1 sum, stop when the max change is below $10^{-13}$).
On the bundled data: $\pi^\* \approx (0.2455,\ 0.6204,\ 0.1341)$ for states
(choppy, calm, crisis), so the fitted market spends about 13 % of its life in
crisis.

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
*product* over 2000 days is $e^{\log L}$ with $\log L \approx 6643$ on the
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
below $10^{-8}$; on the bundled index, K=2 converges in 23 E-steps and K=3
in 71. Because the guarantee is a theorem, the implementation treats a
*decrease* larger than $10^{-6}\max(1, |\log L|)$ as a numerical fault:
EM stops and reports `converged = False, monotone = False` rather than
mistaking the tiny "improvement" for convergence (pinned in API_SPEC 1.4).

**Dead states.** Every M-step denominator is a posterior mass
$\sum_t \gamma_t(i)$. A state whose emission underflows to zero at every
$t$ — e.g. a warm start that parks a state at mean 5.0 with variance
$10^{-8}$ over data of size $10^{-2}$ — has zero mass, and the textbook
update returns $0/0$. Rabiner (1989, §V.B) discusses the collapse and
recommends floors; Bilmes (1998, §4) writes the update assuming positive
mass. The pinned rule here is detection rather than re-seeding: after
each E-step the transition-row support $\sum_{t<T}\gamma_t(i)$ is
compared with $10^{-12}$, and if any state is below it EM stops *before*
the M-step with the just-scored (finite) parameters, `converged = False`
and `dead_state = i`. Nothing is ever NaN, the reported log-likelihood is
exactly the score of the returned parameters, and a walk-forward refit
keeps going with the flagged model (COOKBOOK recipe 14 shows it).

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
  State 0 = lowest mean. A label therefore says nothing about risk: on the
  bundled data state 0 is the choppy regime, and crisis is whichever state has
  the highest variance, here state 2.

Failure modes to know: a state can collapse onto a handful of points
(variance $\to 0$, likelihood $\to \infty$ — the classic Gaussian-mixture
degeneracy), which the variance floor prevents; with more states than
the data supports, EM parks surplus states on near-duplicates (again the
floor engages — tested via `test_variance_floor_engages`); and a state can
lose all posterior support (the dead state above), which is detected and
flagged. Non-convergence after 500 iterations, a dead state and a
non-monotone step are all reported as flags on the result, never as
exceptions: a nearly converged model is still usable, and walk-forward
loops must not die mid-backtest. Genuinely impossible input — a
parameter set whose shapes disagree with the data, or an observation
with zero likelihood under every reachable state — *is* an error.

### 4.3 What EM found on the bundled data (golden values)

Full-sample K=3 fit on `market_index.csv` (2000 days), daily units:

| state | label | mean | std | annualized mean | annualized vol |
|---|---|---|---|---|---|
| 0 | choppy | $-0.000158$ | $0.010114$ | $-4.0\%$ | $16.1\%$ |
| 1 | calm-bull | $+0.000536$ | $0.006059$ | $+13.5\%$ | $9.6\%$ |
| 2 | crisis | $+0.002654$ | $0.024373$ | $+66.9\%$ | $38.7\%$ |

with $\log L = 6643.0767$ and transition matrix (rows sum to 1)

$$
A \approx \begin{pmatrix}
0.9327 & 0.0319 & 0.0354\\
0.0191 & 0.9809 & 0.0000\\
0.0347 & 0.0301 & 0.9352
\end{pmatrix}.
$$

The generator's truth was $\mu = (0.0007, 0.0, -0.0018)$ and
$\sigma = (0.006, 0.011, 0.025)$ for calm/choppy/crisis. EM recovered the
volatilities closely. It missed the means, and the crisis mean has the wrong
sign: the true drift is negative, while the fitted mean on this path is
$+66.9\%$ annualized. Means of noisy daily returns are intrinsically hard to
estimate, a theme that returns in Section 17. This is why crisis is identified
by $\arg\max$ of the variance and never by the mean. Note also
$A_{12} \approx 4 \times 10^{-12}$: the fitted calm state never jumps straight
into crisis, and goes through the choppy state first.

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
$(377, 1397, 226)$ days in states $(0, 1, 2)$ (exact golden integers) and
agrees with the (volatility-mapped) true generator states on **90.45%** of
days. That is a teaching number: it is in-sample and uses hindsight, on data
generated from the model being fitted.

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
| 2 | 6612.4532 | 7 | $-13210.91$ | $-13171.70$ |
| 3 | 6643.0767 | 14 | $-13258.15$ | $-13179.74$ |

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
V-shaped reversals: a 252-day signal is still long after a two-week crash.

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
estimated vol gets the cap (pinned edge case). Returns must be simple
daily returns above $-100\%$: a $-1$ print would zero the running growth
product and turn every later trailing ratio into $0/0$, so it is rejected
as corrupt data rather than traded.

**Where this deliberately departs from Moskowitz, Ooi & Pedersen
(2012).** MOP use a 12-month (~252-day) return signal, a 40% annualized
vol target, an EWMA with a 60-day centre of mass and no leverage cap, and
size each market independently. The pinned configuration here keeps the
252-day signal but targets 10% per asset, uses the RiskMetrics
$\lambda = 0.94$ EWMA (about 16 days of memory) seeded with the first
window's mean squared return, caps leverage at 4x, and divides by the
asset count so the *book* is equal-risk-budgeted. These are configuration
choices made for a readable teaching study, not a replication of the
paper's numbers.

On the bundled data, unfiltered momentum at the portfolio scale earns +9.23%
ann. at 11.04% vol (Sharpe 0.84, maxDD $-25.22\%$), and +14.21% annualized inside
the decoded crisis state. That sign is a property of this path, which is why no
golden case asserts it any longer. All such numbers are full-sample: the first
251 days carry no momentum position at all, which understates the
from-first-trade Sharpe by roughly $1 - \sqrt{1 - 251/2000} \approx 6.5\%$
relative.

### 7.3 The portfolio vol target

**The plain idea.** Sizing each asset to 10 % does not make the book 10 %,
because the assets diversify. So one more scale sets the level of the whole book.

**Formally** (API_SPEC 3.5). Let $u_t$ be the net return of the unscaled, ungated
combined book. With a zero-mean EWMA variance $\sigma^2_t$ of $u$, seeded by the
21 returns from day 252 on,
$\text{scale}_t = \min(0.10 / \sqrt{252\,\sigma^2_t},\ 4)$, and the traded book is
$\text{gate}_t \cdot \text{scale}_t \cdot P_t$. The scale reads $u$ through day
$t$ only and is always the one from the *unfiltered* book. A scale measured on
the gated book would see lower volatility and lever back up, undoing the gate.

**Worked example.** Unscaled, the combined book on seed 19 runs at 4.67 % a year.
The scale averages 2.32, reaches its cap of 4, and takes the unfiltered book to
9.86 %. The scale moves every day, so it adds real turnover (section 13.1).

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
crisis (the unwind). The result in the crisis-attribution table on seed 19:
unfiltered carry earns +21.78% annualized in the calm state, loses 6.28% in the
choppy state and **loses 69.62%** annualized in the crisis state.

### 8.3 The pinned implementation

Every 21 trading days (day indices 0, 21, 42, ...), rank the six synthetic
currencies by that day's annualized differential, descending; ties break
by ascending asset index (pinned so all languages agree). Long the top 3
at $+1/3$ each, short the bottom 3 at $-1/3$ each; hold in between. Traded
return on day $t$ is $\text{spot}_t + \text{diff}_{t-1}/252$ — the
differential accrued is the one *set at the previous close* (known in
advance; day 0 accrues nothing) — a simple-interest approximation with no
forward points and no compounding. Result at the portfolio scale: +6.16% ann.,
Sharpe 0.36, maxDD $-61.13\%$ unfiltered.

**Be honest about what the bundled data exercise.** The generator's
differentials are Ornstein-Uhlenbeck around well-separated bases, so on this
path they change rank order on only 36 days and the basket changes on only 2.
The 21-day rebalance and the ranking tie-break are real and pinned, but they
barely move the book here. They are exercised and golden-pinned instead by the
synthetic `carry_rerank_*` panel (API_SPEC 5), whose ranks cross at day 10,
re-rank at day 21 and tie at day 42. Carry turnover on the bundled data comes
almost entirely from the portfolio scale.

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
  the low-vol state is robustly "the calm one." The exception is a state
  that holds a single extreme return: its variance collapses to the floor
  and it wins the argmin. The opt-in `calm_guard` catches this by checking
  how much stationary weight the state has (API_SPEC 3.3).
* Before the first fit, the gate is 1 (no filtering).

`gate_mode` is `"prob"` (position scaled continuously by the probability)
or `"binary"` (all-in/all-out at a 0.5 threshold). Probability scaling
trades more smoothly and avoids threshold-flapping turnover.

### 9.2 What filtering does — and what it costs

Golden results on the bundled data (5 bps costs; full sample of 2000 days, which
includes 251 flat momentum days and 503 days with the gate at 1; Sharpe is raw,
with no risk-free rate subtracted and population volatility). **These figures
are full-sample, seed 19 is favourable to the gate, and on real data the gate
failed**, so read this table as an output of the code:

| strategy | ann ret | Sharpe | maxDD |
|---|---|---|---|
| momentum unfiltered | +9.23% | 0.84 | $-25.22\%$ |
| momentum filtered | +6.06% | 0.69 | $-21.95\%$ |
| carry unfiltered | +6.16% | 0.36 | $-61.13\%$ |
| carry filtered | +11.37% | 0.94 | $-17.43\%$ |
| combined unfiltered | +7.70% | 0.78 | $-36.08\%$ |
| combined filtered | +8.72% | 1.21 | $-15.20\%$ |

The mechanism on this path: carry is where the crisis losses are, so scaling by
$P(\text{calm})$ moves crisis-state combined return from $-27.70\%$ to $+0.36\%$
annualized. Momentum made money in the crisis state here, so the gate cost it
return and Sharpe. The gate lags, and it is below 1 in the choppy state as well.

The right window is the one the gate can affect. There, Sharpe goes from 0.54 to
0.99 with a 95 % interval of [-0.13, 1.05], which sits at the 85th percentile of
seeds. Across 100 seeds the mean gain is +0.063, and the gate beats the 21-day
vol rule on 49 % of paths. On real data it lost (section 15).

A common argument for filtering is that lower drawdown can be re-levered into
return. That needs a risk-adjusted gain to re-lever, and the evidence above does
not show one.

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

## 10. Why full-sample results mislead

**The plain idea.** A gate that is switched off for part of the sample cannot
win or lose on those days, yet they still count in a full-sample Sharpe. The
gated and unfiltered books are the same book on those days, and that shared
stretch can move the comparison in either direction.

**Formally.** Let $W$ be the days a gate can affect. Here the first fit is at
$t_0 = 503$, the gate is 1 before it, and the gate on day $t$ earns day $t+1$, so
$W = \{t \ge 504\}$. A Sharpe ratio is not additive across periods: it is a mean
over a standard deviation, and both pool the warm-up with $W$. The gated book
has lower volatility inside $W$ and the same volatility outside it, so the
pooled figure weights the two stretches differently for the two books. The
only comparison that tests the gate is the one restricted to $W$.

**Worked example, using historical figures.** These numbers come from the
earlier bundled data, seed 83, before the portfolio scale existed. They are
kept because the effect is clearest there, and they describe no current file.

| Seed 83 (historical), 5 bps | Unfiltered | Gated |
|---|---|---|
| Full sample, 2000 days | 1.80 | 1.87 |
| Days the gate can affect, 1496 days | 1.49 | 1.35 |

The full-sample figures show a gain. On the days the gate could affect it lost
0.14 of Sharpe, with a 95 % bootstrap interval of [-0.85, 0.58]. The sign flipped
because of the 503 warm-up days on which the gate was 1. Version 1 quoted the
first row as its headline, which is why this section exists.

The same split on the current seed 19 (`docs/GATE_EVALUATION.md`) gives 0.78 to
1.21 on the full sample and 0.54 to 0.99 after warm-up. The sign agrees this
time, and the gate's result is still flattered, for the separate reason in
section 11.3. Agreement on one path says nothing about whether the artefact is
present on the next one, so report the restricted window every time.

Momentum has a smaller version of the same effect: its first 251 days hold no
position, so a full-sample momentum Sharpe understates the from-first-trade
figure by about $1 - \sqrt{1 - 251/2000}$ in relative terms.

---

## 11. Judging a signal

Three questions, in order. Does it beat something simple? Is the gap larger
than noise? Does it survive on other paths?

### 11.1 Baselines

**The plain idea.** A gate that holds 60 % of a book on average will have lower
drawdown than the whole book, whatever it knows. So compare it with rules that
need no model.

**Formally.** All gates multiply the same scaled book (API_SPEC 3.5) and are
measured on the same window at the same cost:

* unfiltered: gate 1;
* constant at mean gate: the HMM gate's own average, known only in hindsight, so
  it is a control and not a tradable rule;
* 21-day vol ratio and EWMA vol ratio: free rules from trailing volatility;
* next-day oracle: knows tomorrow's true state, so it gives a ceiling and uses
  look-ahead on purpose.

**Worked example** (seed 19, $t \ge 504$, 5 bps, `docs/GATE_EVALUATION.md`;
seed 19 is favourable to the gate, see 11.3):

| Gate | Mean exposure | Sharpe | Return | Max DD |
|---|---|---|---|---|
| Unfiltered | 1.000 | 0.54 | 5.64% | -36.08% |
| HMM gate | 0.625 | 0.99 | 7.02% | -15.20% |
| Constant at mean gate | 0.625 | 0.54 | 3.52% | -24.15% |
| Trailing 21d vol ratio | 0.881 | 0.81 | 7.62% | -26.27% |
| Next-day oracle (look-ahead) | 0.886 | 1.23 | 11.01% | -23.33% |

A constant gate has the same Sharpe as the unfiltered book up to costs, because
scaling a return stream changes mean and volatility alike. Its drawdown row
shows how much of the gate's lower drawdown is exposure: more than half the way from
-36.08% to -15.20% is available with no signal.

### 11.2 Paired block bootstrap

**The plain idea.** Two books that share a crisis are not independent, and daily
returns cluster in volatility. Resampling days one at a time, or each book
separately, destroys both facts and gives an interval that is too narrow.

**Formally.** The stationary bootstrap of Politis and Romano (1994) draws blocks
with geometric lengths of mean $b$, wrapping at the end of the sample. Paired
means the two series use the same index path $\iota$ on each resample, and the
statistic is
$\widehat{\Delta}^{*} = \mathrm{Sharpe}(a_\iota) - \mathrm{Sharpe}(b_\iota)$. The
interval is the percentile interval of $\widehat{\Delta}^{*}$.
`paired_block_bootstrap` implements this (COOKBOOK recipe 20).

**Worked example.** The gate minus unfiltered on seed 19 is +0.456, and with
mean block 21 and 10000 resamples the 95 % interval is [-0.13, 1.05], with 93.6 %
of resamples positive. Blocks of 10 and 63 days give [-0.08, 1.02] and
[-0.20, 1.10]. A difference of nearly half a Sharpe point on 5.9 years cannot be
told from zero. Against the 21-day vol rule it is +0.186, [-0.21, 0.59].

### 11.3 Seed sweeps

**The plain idea.** A bootstrap resamples the one path you have. It cannot tell
you the path is unusual. On simulated data you can draw new paths, so do.

**Formally.** Rerun the study over seeds 1 to 100 per scenario and report the
share of paths on which the gate wins, with a Wilson interval, plus the mean
difference and its standard deviation. Change the generator too, because a
gate can depend on one feature of it: `vol_only` removes regime drifts,
`half_crisis` halves crisis drifts, `fat_tails` uses Student-t(4) shocks.

**Worked example** (`docs/GATE_EVALUATION.md`, window $t \ge 504$, 5 bps):

| Scenario | P(beats unfiltered) | P(beats 21d vol) | Mean dSharpe vs unfiltered | Mean dSharpe vs vol |
|---|---|---|---|---|
| base | 66% [56%, 75%] | 49% [39%, 59%] | +0.063 | -0.110 |
| half_crisis | 70% [60%, 78%] | 46% [37%, 56%] | +0.034 | -0.162 |
| vol_only | 19% [13%, 28%] | 7% [3%, 14%] | -0.225 | -0.234 |
| fat_tails | 12% [7%, 20%] | 4% [2%, 10%] | -0.385 | -0.498 |

Seed 19's post-warm-up gain of +0.456 sits at the 85th percentile of the base
sweep, against a mean of +0.063. Seed 19 was chosen by a rule on the *unfiltered*
Sharpe alone (API_SPEC 4), so nobody selected it for the gate, and it still
happens to flatter the gate. That is why every bundled gate figure in this
project carries a warning.

---

## 12. Detection is not profit

**The plain idea.** A detector can be right about the regime and still lose
money, because what pays is not the label. It pays to be out on the days
returns are bad, and only if being out costs less than it saves.

**Formally.** Classification scores compare $p_t$ with the true state $s_t$.
P&L depends on $\sum_t g_{t-1}\, r_t$ minus costs. The two agree only if returns
differ by state by more than the lag, the false alarms and the turnover take
away.

**Worked example.** On seed 19 the filtered gate classifies calm against not
calm with accuracy 88.4 % against a majority class of 67.5 %, AUC 0.929, and
crisis against the rest with AUC 0.962. It missed 0 of 13 crisis episodes
(`docs/GATE_EVALUATION.md`). The next-day oracle reaches Sharpe 1.23 at exposure
0.886 while the gate holds 0.625: the gate is $P(\text{calm})$, so it also cuts
exposure in the choppy state, which is not where the losses are.

The seed sweep isolates this. In `vol_only`, where regimes change volatility
but not drift, mean crisis AUC is still 0.926 and the gate beats the unfiltered
book on only 19 % of paths. Detection held, and the profit went with the drift.
The generator gave crises a large negative drift, so the base scenario rewards
detection by construction. Whether real crises do the same is an empirical
question, and section 15 has the answer for this gate.

---

## 13. Turnover, break-even and the band

### 13.1 Where the turnover comes from

**The plain idea.** A probability moves every day, so a book multiplied by it
trades every day. The portfolio scale also moves every day, so even the
unfiltered book now trades.

**Formally.** With $P_t = g_t\, s_t\, Q_t$, for gate $g$, scale $s$ and raw
positions $Q$, turnover is $\sum_a |P_{t,a} - P_{t-1,a}|$ and the net return is
$\text{gross}_t - c\,\text{turnover}_t$ for $c$ in return units per unit traded.

**Worked example.** Raw carry on the bundled data trades 0.588 times its
notional per year, because the basket changes on only 2 days. With the portfolio
scale the same book trades 34.38 times a year over the full sample, and with the
gate 83.2 times. On the post-warm-up window the combined book turns over 36.8
times unfiltered and 73.9 times gated, so at 5 bps the gate adds about 1.9 % a
year of cost ($0.0005 \times (73.9 - 36.8)$). The scale is real turnover and it
is charged: it is not free risk management.

### 13.2 Break-even cost

**Formally.** The break-even cost is the $c$ at which the gated Sharpe equals the
comparison. Since net returns are linear in $c$ this is a root search
(`break_even_cost`, `gate_break_even`).

**Worked example.** On seed 19 the gate's Sharpe advantage disappears at 11.54 bps
if both books pay the same cost, and at 9.35 bps if the unfiltered book is held
at its 5 bps Sharpe. The seed sweep's median is 10.44 bps, over the 83 of 100
paths where the lines cross; on 17 paths the gate had already lost at 0 bps. With
fat tails the median falls to 2.49 bps. The gated Sharpe drops from 1.52 at 0 bps
to 0.99 at 5 bps and -0.57 at 20 bps, while the unfiltered book goes from 0.71 to
0.54 to 0.01.

Flat basis points do not grow with size. `regime.costs` adds a spread and a
square-root impact term, and `docs/CAPACITY_EVALUATION.md` shows the shape of the
curve. Its inputs are illustrative, not calibrated, so its dollar figures are
not a capacity estimate.

### 13.3 The no-trade band

**The plain idea.** Ignore gate moves too small to be worth trading.

**Formally** (API_SPEC 3.3):
$o_t = g_t$ if $|g_t - o_{t-1}| > \text{band}$, else $o_t = o_{t-1}$. The rule is
causal, and a band of 0 returns the gate unchanged.

**Worked example.** On the bundled data a band of 0.3 takes full-sample gated
carry turnover from 83.2 to 45.7 a year. It cannot go lower, because the
portfolio scale keeps trading. Over 50 base seeds (`docs/BAND_EVALUATION.md`):

| Band | Mean Sharpe | Turnover/yr | P(beats banded vol) |
|---|---|---|---|
| 0.0 | 0.67 | 52.6 | 52% [39%, 65%] |
| 0.3 | 0.77 | 33.1 | 64% [50%, 76%] |
| 0.5 | 0.80 | 23.2 | 70% [56%, 81%] |

Each row is a separate band, so reading off the best is selection: five bands
were tried and none was tested out of sample. On real data a band of 0.3
narrowed the loss to the vol rule and did not close it (section 15). `gate_band`
stays off by default.

---

## 14. Fat tails, outlier capture and `calm_guard`

### 14.1 Fat tails

**The plain idea.** A Gaussian state has almost no room for a large return, so
the filter reads one fat-tailed day as a change of regime, and the gate jumps.

**Formally.** The filter update weights state $i$ by $b_i(x_t)$, and
$\log b_i(x)$ falls with the square of $x$. Under a Student-t(4) shock a large
$|x|$ is ordinary inside a single regime, but the Gaussian likelihood ratio
between a low- and a high-variance state grows without bound in $x^2$.

**Worked example.** In the `fat_tails` scenario mean Sharpe falls to 0.10 for the
gate against 0.49 unfiltered, and the gate beats the unfiltered book on 12 % of
paths. Turnover is the visible mechanism: 111.2 a year at band 0 over 50 seeds,
against 52.6 in the base scenario. Mean crisis AUC stays at 0.901. Student-t
emissions are the standard response. That experiment was conditional and was
skipped when work stopped under the rule in section 16.

### 14.2 Outlier capture

**The plain idea.** One corrupt or extreme print can claim a whole state for
itself. That state then has the smallest variance, so it is picked as "calm",
and the gate reads the probability of a state the chain almost never visits.
Every fit still reports `converged`.

**Formally.** With `calm_guard = m > 0` (API_SPEC 3.3), a refit whose
lowest-variance state has stationary weight $\pi_{k_0} < m$ trips the guard: the
day goes into `guard_trips`, the window is clipped to
$\text{median} \pm 8 \times 1.4826 \times \text{MAD}$ for the fit and that refit's
filter pass, and $k^\*$ is the lowest-variance state with $\pi_k \ge m$. Backtests
and forward steps still use true returns. The flow is drawn in
`docs/diagrams/calm_guard_flow.mmd`.

**Worked example** (bundled data, $m = 0.01$, mean gate from $t_0$ on; clean run
0.624 with no trips):

| Injected return | No guard | Guard on | Trips of 24 |
|---|---|---|---|
| -30 % at t=1000 | 0.145 | 0.657 | 5 |
| +50 % at t=1000 | 0.145 | 0.630 | 16 |
| -30 % at t=700 | 0.093 | 0.672 | 8 |

The guard restores the level, and its mean stays high by 0.01 to 0.05, with
single days still far from clean (API_SPEC 3.3). It narrows the damage and does
not remove it, and $m = 0.01$ was checked on two samples of one generator, so it
is not a tuned setting. Clean outliers before fitting all the same. The guard is
off by default, and the real-data report ran with it off.

---

## 15. The real-data result

**What was run.** The pinned configuration, unchanged, on Kenneth French daily
data from 1963-07-01 to 2026-09-30, 15918 days. The HMM observes the market total
return and the books trade excess returns, at 5 bps, on the window $t \ge 504$.
There were 245 refits, 1 not converged, none with a dead state, and the mean gate
was 0.255 (`docs/REAL_DATA_EVALUATION.md`).

| Book | Unfiltered | HMM gate | 21-day vol rule | Gate minus unfiltered | Gate minus vol rule |
|---|---|---|---|---|---|
| Market | 0.44 | 0.14 | 0.48 | -0.298 [-0.56, -0.03] | -0.333 [-0.56, -0.08] |
| Industry momentum | 0.37 | 0.07 | 0.40 | -0.295 [-0.55, -0.06] | -0.324 [-0.54, -0.11] |

Brackets are 95 % paired block bootstrap intervals, mean block 21, 2000 resamples.
All four lie below zero. The gate also lost in each of the three periods
reported, on both books, against the unfiltered book.

**What it means.** A constant gate at the same mean exposure keeps the
unfiltered Sharpe of 0.44, so the HMM gate lost 0.298 against simply holding less
of the market. Its timing subtracted value on this history. A band of 0.3 helped
and did not close the gap to the vol rule: -0.293 [-0.51, -0.06] on the market
and -0.204 [-0.43, +0.01] on industry momentum, against the unbanded rule.

**What it does not mean.** This is one market and one history. These are research
portfolios, with early-sample costs far above 5 bps, and no financing or capacity
model. There is no FX carry book, so the combined book has no real-data result.
It tests this gate in this configuration, not every regime model.

---

## 16. Preregistration

**The plain idea.** If you decide what counts as success after seeing the
results, you can always find a way to succeed. So write the test down first.

**Formally** (`docs/PREREGISTRATION.md`). The hypothesis: a regime gate beats the
same-banded 21-day vol rule on Sharpe at 5 bps out of sample. To pass, one
configuration is committed before the holdout is read, the paired bootstrap uses
mean block 21 and at least 2000 resamples, the lower end of the 95 % interval is
above zero, every registered book passes, and nothing is tuned afterwards. The
result is published whatever it shows.

**Worked example.** The 1963-2026 history has now been seen, so no part of it can
serve as a holdout. A future test needs data after 2026-09-30, or a market no gate
here has touched. Until one passes, the owner's rule applies: no new gate
variants, no tuning and no claim that the gate adds value. This is why the
Student-t, P(crisis) gate and real FX carry experiments were skipped. Each would
have been another try on data that had already given its answer.

The rule binds the rest of this guide. Sections 1 to 9 teach machinery that works
as specified. The conclusion about using it to gate these strategies is in
sections 10 to 15.

---

## 17. Honest limitations

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
  *generated from* a 3-state model, so the 90.45% decoding accuracy is a best
  case, not an expectation.
* **Measured evidence.** The gate failed on real data (section 15). On simulated
  seeds it does not beat a 21-day vol rule on average, and it loses to the
  unfiltered book under fat tails (section 11.3). Treat bundled figures in this
  guide as worked examples, not evidence of an edge.
* **No-trade band.** `gate_band` cuts turnover (section 13.3). It is off by
  default, because five bands tried on synthetic seeds are a selection, and on
  real data the band did not close the gap.
* **Costs and capacity.** The gate and the portfolio scale both trade every day.
  Flat basis points ignore size, and the opt-in `regime.costs` model has
  illustrative inputs, so no capacity claim is made. Golden turnover values exist
  so ports cannot quietly diverge on this.
* **Gaussian tails.** Within-state Gaussian emissions understate extreme
  days even in the crisis state (section 14). Student-t emissions are the
  standard upgrade, and that experiment was skipped by the stop rule.

---

## 18. Common pitfalls and numerical issues

* **Underflow in forward-backward** — the whole point of scaling; see
  Section 3.2. Never implement the textbook unscaled recursion for
  $T > \sim 100$.
* **Dividing by $\sum_t \gamma_t(i) \approx 0$** for a dead state — the
  quantile initialization keeps states alive from a cold start, but a
  warm start can kill one; the pinned guard (Section 4.1) checks the
  support after every E-step and stops with a flag instead of producing
  NaN parameters that only surface days later as a NaN gate.
* **Leveraged accounting past $-100\%$.** `equity = prod(1 + net)` is
  linear in returns, so a 4x book on a $-30\%$ day produces a $-120\%$
  net return, negative equity and a "drawdown" below $-100\%$. The engine
  refuses such a day (an error naming $t$) rather than compounding a
  book that no longer exists; likewise a raw return $\le -1$ is rejected
  as corrupt input.
* **Shape drift between a model and its data.** Scoring $D=2$ data with a
  $D=1$ model, or warm-starting $K=3$ from $K=2$ parameters, must be an
  error — a vectorized reference can otherwise broadcast its way to a
  finite, wrong number. Every entry point validates $K$, $D$, finiteness
  and stochasticity (API_SPEC 1.9).
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
  dates exist and 21-day blocks (trailing partial block dropped) otherwise
  — mixing the two changes the metric. Dates must be strictly increasing;
  a duplicated or out-of-order row is rejected because the four ports
  would otherwise silently disagree on which days share a month.

---

## 19. Knowledge check Q&A

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
fitted calm state here exits to choppy, essentially never straight to crisis).

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
worth running?** (Sharpe here is the raw ratio; with a risk-free rate the
comparison shifts slightly but the ordering does not.)
First check that the improvement is real: restrict to the days the filter can
affect, compare with a vol rule, and bootstrap the difference. If it survives,
it depends on the binding constraint. If capital is constrained and leverage
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

**Q11. A gate lifts full-sample Sharpe. Name the first check.**
Remove the days it could not affect. A warm-up with the gate at 1 is shared by
both books, and on the old seed 83 it turned a loss of 0.14 into an apparent gain
(section 10).

**Q12. Why must the bootstrap be paired and in blocks?**
Paired, because the two books share crises and the question is about their
difference. In blocks, because volatility clusters. Resampling days singly or
books separately gives an interval that is too narrow (section 11.2).

**Q13. A detector has crisis AUC 0.96. Why might it still lose money?**
Profit depends on returns by state, lag, false alarms and turnover, not on the
label. In the `vol_only` scenario AUC stays high and the gate loses, because
regimes there carry no drift to avoid (section 12).

**Q14. Why does a bootstrap interval on one path not replace a seed sweep?**
The bootstrap resamples the path you have, so it cannot show that the path is
unusual. Seed 19's gain has an interval and still sits at the 85th percentile of
seeds (section 11.3).

**Q15. The holdout failed by a small margin. Can you adjust the band and rerun?**
No. A changed configuration is a new hypothesis and needs data nobody has seen
(section 16).

---

## 20. Further reading

* **L. Rabiner (1989)**, "A Tutorial on Hidden Markov Models and Selected
  Applications in Speech Recognition," *Proc. IEEE* — the canonical HMM
  tutorial; the scaling scheme here is his, and §V.B is the source of the
  zero-mass discussion behind the dead-state guard.
* **J. Bilmes (1998)**, "A Gentle Tutorial of the EM Algorithm and its
  Application to Parameter Estimation for Gaussian Mixture and Hidden
  Markov Models," ICSI TR-97-021 — the Gaussian-emission M-step in the
  exact form implemented here.
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
* **D. Politis & J. Romano (1994)**, "The Stationary Bootstrap," *JASA* — the
  resampling scheme of section 11.2.
* **M. López de Prado (2018)**, *Advances in Financial Machine Learning* —
  backtest hygiene, lookahead and selection bias.

Full citations with DOIs are collected in the README's References
section.
