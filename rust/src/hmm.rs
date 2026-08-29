//! Gaussian hidden Markov model with scaled forward-backward and Baum-Welch EM.
//!
//! Everything here is deterministic: initialization is pinned (quantile-based
//! means, 0.9 self-transition prior, sample variance) and no random number
//! generator is used anywhere (API_SPEC.md §6).
//!
//! # Scaled forward-backward (pinned: scaled, not log-space)
//!
//! Let `b_t(i) = p(x_t | s_t = i)` be the Gaussian emission density, `A` the
//! transition matrix and `pi` the initial distribution.  The unscaled forward
//! variable underflows for long series, so each step is normalized (Rabiner
//! scaling).  To guard the emission densities themselves, the per-step maximum
//! log-emission `m_t = max_i log b_t(i)` is factored out and the recursion
//! runs on `bt_t(i) = exp(log b_t(i) - m_t) <= 1`:
//!
//! ```text
//! a~_1(i) = pi_i * bt_1(i)                      d_1 = sum_i a~_1(i)
//! a^_1(i) = a~_1(i) / d_1
//! a~_t(j) = [sum_i a^_{t-1}(i) A_ij] * bt_t(j)  d_t = sum_j a~_t(j)
//! a^_t(j) = a~_t(j) / d_t
//! ```
//!
//! The true per-step normalizer is `c_t = d_t * exp(m_t)` and the
//! log-likelihood is stored as the sum of the log normalizers:
//! `log L = sum_t (log d_t + m_t)`.  The backward pass shares the same `d_t`,
//! after which the posteriors need no further normalization:
//! `gamma_t(i) = a^_t(i) * b^_t(i)` and
//! `xi_t(i,j) = a^_t(i) A_ij bt_{t+1}(j) b^_{t+1}(j) / d_{t+1}`.

use crate::error::{RegimeError, Result};
use crate::matrix::Matrix;

const LOG_2PI: f64 = 1.8378770664093453; // ln(2*pi)

/// Parameter set of a diagonal-covariance Gaussian HMM.
#[derive(Debug, Clone, PartialEq)]
pub struct HmmParams {
    /// Initial state distribution, length `K`.
    pub startprob: Vec<f64>,
    /// Row-stochastic transition matrix, shape `(K, K)`.
    pub transmat: Matrix,
    /// State means, shape `(K, D)`.
    pub means: Matrix,
    /// Per-dimension state variances, shape `(K, D)`.
    pub variances: Matrix,
}

/// Outcome of a Baum-Welch fit.
#[derive(Debug, Clone)]
pub struct HmmFitResult {
    /// Log-likelihood of the returned parameter set.
    pub log_likelihood: f64,
    /// Number of E-steps performed.
    pub n_iter: usize,
    /// True iff the log-likelihood improvement fell below `tol` before
    /// `max_iter` was reached.  Non-convergence is reported through this
    /// flag, never as an error (API_SPEC.md §1.4).
    pub converged: bool,
    /// Log-likelihood at each E-step (non-decreasing, EM monotonicity).
    pub loglik_history: Vec<f64>,
}

/// Validate observations of shape `(T, D)` with `D` in {1, 2}.
fn validate_obs(x: &Matrix) -> Result<()> {
    if x.rows() == 0 {
        return Err(RegimeError::InvalidInput("observation sequence is empty".into()));
    }
    if x.cols() == 0 || x.cols() > 2 {
        return Err(RegimeError::InvalidInput(format!(
            "only univariate or 2-D observations supported, got D={}",
            x.cols()
        )));
    }
    if !x.all_finite() {
        return Err(RegimeError::InvalidInput("observations contain NaN or inf".into()));
    }
    Ok(())
}

/// Linear-interpolation quantile on a pre-sorted sample (numpy default rule):
/// `h = q*(n-1)`, `v = x[floor(h)] + (h - floor(h)) * (x[floor(h)+1] - x[floor(h)])`.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let h = q * (n - 1) as f64;
    let lo = h.floor() as usize;
    if lo + 1 >= n {
        return sorted[n - 1];
    }
    sorted[lo] + (h - lo as f64) * (sorted[lo + 1] - sorted[lo])
}

/// Diagonal-covariance Gaussian HMM with deterministic Baum-Welch EM.
///
/// `K = 1` is a degenerate model and is rejected at construction.  The
/// variance floor guards zero-variance states when `K` exceeds the distinct
/// support of the data.
#[derive(Debug, Clone)]
pub struct GaussianHmm {
    /// Number of hidden states `K` (>= 2).
    pub n_states: usize,
    /// Convergence tolerance on the log-likelihood improvement.
    pub tol: f64,
    /// Maximum number of EM iterations (E-steps).
    pub max_iter: usize,
    /// Lower bound applied to every variance after each update.
    pub var_floor: f64,
    params: Option<HmmParams>,
    fit_result: Option<HmmFitResult>,
}

impl GaussianHmm {
    /// Create a model with explicit EM settings.
    ///
    /// # Errors
    /// `K < 2`, non-positive `tol`/`var_floor`, or `max_iter < 1`.
    pub fn new(n_states: usize, tol: f64, max_iter: usize, var_floor: f64) -> Result<Self> {
        if n_states < 2 {
            return Err(RegimeError::InvalidInput(format!(
                "n_states must be >= 2 (K=1 is degenerate), got {n_states}"
            )));
        }
        if !(tol > 0.0) || !(var_floor > 0.0) || max_iter < 1 {
            return Err(RegimeError::InvalidInput(
                "tol and var_floor must be > 0 and max_iter >= 1".into(),
            ));
        }
        Ok(GaussianHmm { n_states, tol, max_iter, var_floor, params: None, fit_result: None })
    }

    /// Create a model with the pinned defaults `tol = 1e-8`, `max_iter = 500`,
    /// `var_floor = 1e-8`.
    pub fn with_defaults(n_states: usize) -> Result<Self> {
        Self::new(n_states, 1e-8, 500, 1e-8)
    }

    /// Fitted parameters, if `fit` has run.
    pub fn params(&self) -> Option<&HmmParams> {
        self.params.as_ref()
    }

    /// Result of the last `fit` call, if any.
    pub fn fit_result(&self) -> Option<&HmmFitResult> {
        self.fit_result.as_ref()
    }

    /// Install an externally supplied parameter set (used by tests and the
    /// tiny enumerated ground-truth cases).
    pub fn set_params(&mut self, params: HmmParams) {
        self.params = Some(params);
    }

    fn require_fitted(&self) -> Result<&HmmParams> {
        self.params
            .as_ref()
            .ok_or_else(|| RegimeError::InvalidInput("model is not fitted; call fit() first".into()))
    }

    // ------------------------------------------------------------------ //
    // Initialization (pinned, RNG-free)
    // ------------------------------------------------------------------ //

    /// Deterministic EM starting point (API_SPEC.md §1.3).
    ///
    /// State `k` gets per-dimension means at the empirical quantile
    /// `q_k = (2k + 1) / (2K)` (linear interpolation); every state gets the
    /// full-sample variance (`ddof = 0`, floored); the transition matrix has
    /// 0.9 self-transitions with the rest spread uniformly; the initial
    /// distribution is uniform.
    pub fn pinned_init(&self, x: &Matrix) -> Result<HmmParams> {
        validate_obs(x)?;
        let (k, d) = (self.n_states, x.cols());
        let t_len = x.rows();

        let mut means = Matrix::zeros(k, d);
        let mut variances = Matrix::zeros(k, d);
        for dim in 0..d {
            let mut col = x.col(dim);
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // Full-sample variance with ddof=0, floored.
            let mean: f64 = x.col(dim).iter().sum::<f64>() / t_len as f64;
            let var: f64 = x
                .col(dim)
                .iter()
                .map(|v| (v - mean) * (v - mean))
                .sum::<f64>()
                / t_len as f64;
            let var = var.max(self.var_floor);
            for s in 0..k {
                let q = (2.0 * s as f64 + 1.0) / (2.0 * k as f64);
                means.set(s, dim, quantile_sorted(&col, q));
                variances.set(s, dim, var);
            }
        }

        let mut transmat = Matrix::filled(k, k, 0.1 / (k - 1) as f64);
        for s in 0..k {
            transmat.set(s, s, 0.9);
        }
        let startprob = vec![1.0 / k as f64; k];
        Ok(HmmParams { startprob, transmat, means, variances })
    }

    // ------------------------------------------------------------------ //
    // Emission densities
    // ------------------------------------------------------------------ //

    /// Log emission densities, shape `(T, K)`:
    /// `log b_t(i) = -0.5 * sum_d [log 2pi + log var_id + (x_td - mu_id)^2 / var_id]`.
    fn log_emissions(x: &Matrix, params: &HmmParams) -> Matrix {
        let (t_len, d) = (x.rows(), x.cols());
        let k = params.startprob.len();
        let mut logb = Matrix::zeros(t_len, k);
        for t in 0..t_len {
            for i in 0..k {
                let mut acc = 0.0;
                for dim in 0..d {
                    let v = params.variances.get(i, dim);
                    let diff = x.get(t, dim) - params.means.get(i, dim);
                    acc += LOG_2PI + v.ln() + diff * diff / v;
                }
                logb.set(t, i, -0.5 * acc);
            }
        }
        logb
    }

    // ------------------------------------------------------------------ //
    // Scaled forward / backward
    // ------------------------------------------------------------------ //

    /// Scaled forward pass; returns `(alpha_hat, d, m)` with the scaled
    /// forward variables `(T, K)`, per-step normalizers `d` and log-emission
    /// shifts `m`.  The log-likelihood is `sum_t (ln d_t + m_t)`.
    fn forward(logb: &Matrix, params: &HmmParams) -> (Matrix, Vec<f64>, Vec<f64>) {
        let (t_len, k) = (logb.rows(), logb.cols());
        let mut m = vec![0.0; t_len];
        let mut bt = Matrix::zeros(t_len, k);
        for t in 0..t_len {
            let mx = logb.row(t).iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            m[t] = mx;
            for i in 0..k {
                bt.set(t, i, (logb.get(t, i) - mx).exp());
            }
        }
        let mut alpha = Matrix::zeros(t_len, k);
        let mut d = vec![0.0; t_len];
        let mut a = vec![0.0; k];
        for i in 0..k {
            a[i] = params.startprob[i] * bt.get(0, i);
        }
        d[0] = a.iter().sum();
        for i in 0..k {
            alpha.set(0, i, a[i] / d[0]);
        }
        for t in 1..t_len {
            for (j, aj) in a.iter_mut().enumerate() {
                let mut s = 0.0;
                for i in 0..k {
                    s += alpha.get(t - 1, i) * params.transmat.get(i, j);
                }
                *aj = s * bt.get(t, j);
            }
            d[t] = a.iter().sum();
            for j in 0..k {
                alpha.set(t, j, a[j] / d[t]);
            }
        }
        (alpha, d, m)
    }

    /// Scaled backward pass sharing the forward normalizers `d`.
    fn backward(logb: &Matrix, params: &HmmParams, d: &[f64], m: &[f64]) -> Matrix {
        let (t_len, k) = (logb.rows(), logb.cols());
        let mut beta = Matrix::zeros(t_len, k);
        for i in 0..k {
            beta.set(t_len - 1, i, 1.0);
        }
        for t in (0..t_len - 1).rev() {
            for i in 0..k {
                let mut s = 0.0;
                for j in 0..k {
                    s += params.transmat.get(i, j)
                        * (logb.get(t + 1, j) - m[t + 1]).exp()
                        * beta.get(t + 1, j);
                }
                beta.set(t, i, s / d[t + 1]);
            }
        }
        beta
    }

    // ------------------------------------------------------------------ //
    // Public inference API
    // ------------------------------------------------------------------ //

    /// Log-likelihood of `x` under the fitted parameters.
    pub fn score(&self, x: &Matrix) -> Result<f64> {
        let params = self.require_fitted()?;
        validate_obs(x)?;
        let logb = Self::log_emissions(x, params);
        let (_, d, m) = Self::forward(&logb, params);
        Ok(d.iter().zip(&m).map(|(dv, mv)| dv.ln() + mv).sum())
    }

    /// Filtered posteriors `P(s_t = i | x_1..t)`, shape `(T, K)`.
    pub fn filtered_probabilities(&self, x: &Matrix) -> Result<Matrix> {
        let params = self.require_fitted()?;
        validate_obs(x)?;
        let logb = Self::log_emissions(x, params);
        let (alpha, _, _) = Self::forward(&logb, params);
        Ok(alpha)
    }

    /// Smoothed posteriors `P(s_t = i | x_1..T)`, shape `(T, K)`.
    pub fn smoothed_probabilities(&self, x: &Matrix) -> Result<Matrix> {
        let params = self.require_fitted()?;
        validate_obs(x)?;
        let logb = Self::log_emissions(x, params);
        let (alpha, d, m) = Self::forward(&logb, params);
        let beta = Self::backward(&logb, params, &d, &m);
        let (t_len, k) = (alpha.rows(), alpha.cols());
        let mut gamma = Matrix::zeros(t_len, k);
        for t in 0..t_len {
            for i in 0..k {
                gamma.set(t, i, alpha.get(t, i) * beta.get(t, i));
            }
        }
        Ok(gamma)
    }

    /// Most likely state path (log-space Viterbi), length `T`.
    ///
    /// Ties are broken toward the lower state index (first argmax).
    pub fn viterbi(&self, x: &Matrix) -> Result<Vec<usize>> {
        let params = self.require_fitted()?;
        validate_obs(x)?;
        let logb = Self::log_emissions(x, params);
        let (t_len, k) = (logb.rows(), logb.cols());
        let log_a: Vec<f64> = (0..k * k)
            .map(|idx| params.transmat.get(idx / k, idx % k).ln())
            .collect();
        let mut delta: Vec<f64> =
            (0..k).map(|i| params.startprob[i].ln() + logb.get(0, i)).collect();
        let mut psi = vec![0usize; t_len * k];
        let mut next = vec![0.0; k];
        for t in 1..t_len {
            for j in 0..k {
                let mut best_i = 0;
                let mut best = delta[0] + log_a[j];
                for i in 1..k {
                    let cand = delta[i] + log_a[i * k + j];
                    if cand > best {
                        best = cand;
                        best_i = i;
                    }
                }
                psi[t * k + j] = best_i;
                next[j] = best + logb.get(t, j);
            }
            delta.copy_from_slice(&next);
        }
        let mut states = vec![0usize; t_len];
        let mut best_i = 0;
        for i in 1..k {
            if delta[i] > delta[best_i] {
                best_i = i;
            }
        }
        states[t_len - 1] = best_i;
        for t in (0..t_len - 1).rev() {
            states[t] = psi[(t + 1) * k + states[t + 1]];
        }
        Ok(states)
    }

    /// Stationary distribution of the fitted transition matrix.
    ///
    /// Pinned power method (API_SPEC.md §1.7): start from the uniform vector
    /// and iterate `pi <- pi A` with L1 renormalization until
    /// `max_i |pi_new_i - pi_i| < 1e-13` or 10000 iterations.
    pub fn stationary_distribution(&self) -> Result<Vec<f64>> {
        let params = self.require_fitted()?;
        let k = self.n_states;
        let mut pi = vec![1.0 / k as f64; k];
        let mut nxt = vec![0.0; k];
        for _ in 0..10_000 {
            for (j, nx) in nxt.iter_mut().enumerate() {
                let mut s = 0.0;
                for (i, p) in pi.iter().enumerate() {
                    s += p * params.transmat.get(i, j);
                }
                *nx = s;
            }
            let total: f64 = nxt.iter().sum();
            for v in nxt.iter_mut() {
                *v /= total;
            }
            let diff = pi
                .iter()
                .zip(&nxt)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            pi.copy_from_slice(&nxt);
            if diff < 1e-13 {
                return Ok(pi);
            }
        }
        Ok(pi)
    }

    /// Free parameter count for a diagonal-covariance model:
    /// `2*K*D` (means + variances) `+ K*(K-1)` (transitions) `+ (K-1)` (initial).
    pub fn n_parameters(&self, n_dims: usize) -> usize {
        let k = self.n_states;
        2 * k * n_dims + k * (k - 1) + (k - 1)
    }

    /// Akaike information criterion `-2 LL + 2p` (lower is better).
    pub fn aic(&self, x: &Matrix) -> Result<f64> {
        Ok(-2.0 * self.score(x)? + 2.0 * self.n_parameters(x.cols()) as f64)
    }

    /// Bayesian information criterion `-2 LL + p ln T` (lower is better).
    pub fn bic(&self, x: &Matrix) -> Result<f64> {
        Ok(-2.0 * self.score(x)? + self.n_parameters(x.cols()) as f64 * (x.rows() as f64).ln())
    }

    // ------------------------------------------------------------------ //
    // Baum-Welch EM
    // ------------------------------------------------------------------ //

    /// Fit by Baum-Welch EM from a deterministic starting point.
    ///
    /// `init` supplies warm-start parameters (used by the walk-forward refit
    /// schedule); without it the pinned initialization is used.  The E-step
    /// scores the *current* parameters, so on convergence the reported
    /// log-likelihood matches the returned parameter set; on `max_iter`
    /// exhaustion one extra E-step scores the final parameters and
    /// `converged` is `false` (API_SPEC.md §1.4).  After EM, states are
    /// permuted once so `means[:, 0]` is ascending (§1.5).
    ///
    /// # Errors
    /// Invalid observations or `T <= n_states`.
    pub fn fit(&mut self, x: &Matrix, init: Option<&HmmParams>) -> Result<HmmFitResult> {
        validate_obs(x)?;
        let t_len = x.rows();
        let (k, d_dim) = (self.n_states, x.cols());
        if t_len <= k {
            return Err(RegimeError::InvalidInput(format!(
                "need more observations than states: T={t_len}, K={k}"
            )));
        }
        let mut params = match init {
            Some(p) => p.clone(),
            None => self.pinned_init(x)?,
        };

        let mut history: Vec<f64> = Vec::new();
        let mut ll_prev = f64::NEG_INFINITY;
        let mut converged = false;
        let mut n_iter = 0usize;
        let mut ll = f64::NAN;

        for _ in 0..self.max_iter {
            // E-step: scores the CURRENT parameters.
            let logb = Self::log_emissions(x, &params);
            let (alpha, d, m) = Self::forward(&logb, &params);
            ll = d.iter().zip(&m).map(|(dv, mv)| dv.ln() + mv).sum();
            history.push(ll);
            n_iter += 1;
            if n_iter > 1 && ll - ll_prev < self.tol {
                converged = true;
                break;
            }
            ll_prev = ll;
            let beta = Self::backward(&logb, &params, &d, &m);

            // gamma_t(i) = alpha_t(i) * beta_t(i); rows already sum to 1.
            let mut gamma = Matrix::zeros(t_len, k);
            for t in 0..t_len {
                for i in 0..k {
                    gamma.set(t, i, alpha.get(t, i) * beta.get(t, i));
                }
            }

            // xi summed over t: xi_sum[i][j] = A_ij * sum_t alpha_t(i) *
            // bt_{t+1}(j) * beta_{t+1}(j) / d_{t+1}   (left-to-right in t).
            let mut xi_sum = Matrix::zeros(k, k);
            for t in 0..t_len - 1 {
                for j in 0..k {
                    let w = (logb.get(t + 1, j) - m[t + 1]).exp() * beta.get(t + 1, j) / d[t + 1];
                    for i in 0..k {
                        let v = xi_sum.get(i, j) + alpha.get(t, i) * w;
                        xi_sum.set(i, j, v);
                    }
                }
            }

            // M-step.
            let mut gsum = vec![0.0; k];
            let mut gsum_trans = vec![0.0; k];
            for t in 0..t_len {
                for i in 0..k {
                    gsum[i] += gamma.get(t, i);
                    if t < t_len - 1 {
                        gsum_trans[i] += gamma.get(t, i);
                    }
                }
            }
            for i in 0..k {
                params.startprob[i] = gamma.get(0, i);
                for j in 0..k {
                    params.transmat.set(i, j, xi_sum.get(i, j) * params.transmat.get(i, j) / gsum_trans[i]);
                }
            }
            for i in 0..k {
                for dim in 0..d_dim {
                    let mut num = 0.0;
                    for t in 0..t_len {
                        num += gamma.get(t, i) * x.get(t, dim);
                    }
                    params.means.set(i, dim, num / gsum[i]);
                }
            }
            for i in 0..k {
                for dim in 0..d_dim {
                    let mu = params.means.get(i, dim);
                    let mut num = 0.0;
                    for t in 0..t_len {
                        let diff = x.get(t, dim) - mu;
                        num += gamma.get(t, i) * diff * diff;
                    }
                    params.variances.set(i, dim, (num / gsum[i]).max(self.var_floor));
                }
            }
        }

        if !converged {
            // max_iter exhausted: params had one more M-step than the last
            // recorded E-step, so score them once for a consistent report.
            let logb = Self::log_emissions(x, &params);
            let (_, d, m) = Self::forward(&logb, &params);
            ll = d.iter().zip(&m).map(|(dv, mv)| dv.ln() + mv).sum();
        }

        // Pinned label convention: sort states by means[:, 0] ascending
        // (stable; state 0 = lowest mean).  Applied once, after EM finishes.
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| {
            params
                .means
                .get(a, 0)
                .partial_cmp(&params.means.get(b, 0))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut sorted = HmmParams {
            startprob: vec![0.0; k],
            transmat: Matrix::zeros(k, k),
            means: Matrix::zeros(k, d_dim),
            variances: Matrix::zeros(k, d_dim),
        };
        for (new_i, &old_i) in order.iter().enumerate() {
            sorted.startprob[new_i] = params.startprob[old_i];
            for (new_j, &old_j) in order.iter().enumerate() {
                sorted.transmat.set(new_i, new_j, params.transmat.get(old_i, old_j));
            }
            for dim in 0..d_dim {
                sorted.means.set(new_i, dim, params.means.get(old_i, dim));
                sorted.variances.set(new_i, dim, params.variances.get(old_i, dim));
            }
        }
        self.params = Some(sorted);
        let result = HmmFitResult { log_likelihood: ll, n_iter, converged, loglik_history: history };
        self.fit_result = Some(result.clone());
        Ok(result)
    }
}

/// One incremental scaled-forward update for online filtering.
///
/// Given the filtered posterior `alpha_prev = P(s_{t-1} | x_1..t-1)` and a
/// new observation, returns `P(s_t | x_1..t)`.  Identical math to one step
/// of the batch forward pass; used by the regime gate between refits so the
/// filter never re-reads the past.
pub fn forward_step(params: &HmmParams, alpha_prev: &[f64], x_t: &[f64]) -> Vec<f64> {
    let k = params.startprob.len();
    let d_dim = params.means.cols();
    let mut logb = vec![0.0; k];
    for (i, lb) in logb.iter_mut().enumerate() {
        let mut acc = 0.0;
        for dim in 0..d_dim {
            let v = params.variances.get(i, dim);
            let diff = x_t[dim] - params.means.get(i, dim);
            acc += LOG_2PI + v.ln() + diff * diff / v;
        }
        *lb = -0.5 * acc;
    }
    let mx = logb.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut a = vec![0.0; k];
    for (j, aj) in a.iter_mut().enumerate() {
        let mut s = 0.0;
        for i in 0..k {
            s += alpha_prev[i] * params.transmat.get(i, j);
        }
        *aj = s * (logb[j] - mx).exp();
    }
    let total: f64 = a.iter().sum();
    for v in a.iter_mut() {
        *v /= total;
    }
    a
}
