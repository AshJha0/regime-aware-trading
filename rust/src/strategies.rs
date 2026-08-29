//! Trading strategies: time-series momentum, FX carry, and the HMM regime gate.
//!
//! # No-lookahead convention (shared with the backtest engine)
//!
//! Every position matrix `P` produced here has `P[t]` = the position
//! *decided at the close of day t using information up to and including day
//! t*.  The backtest applies `P[t-1]` to the day-t return, so a signal can
//! never see the return it is traded against.  The regime gate at day t
//! likewise uses only observations `x_0..x_t` (filtered, not smoothed,
//! probabilities) and models fitted on data through day t.

use crate::error::{RegimeError, Result};
use crate::hmm::{forward_step, GaussianHmm};
use crate::matrix::Matrix;

/// Trading days per year (pinned).
pub const TRADING_DAYS: f64 = 252.0;

fn validate_returns(r: &Matrix, name: &str) -> Result<()> {
    if r.rows() == 0 || r.cols() == 0 {
        return Err(RegimeError::InvalidInput(format!("{name} must be a non-empty matrix")));
    }
    if !r.all_finite() {
        return Err(RegimeError::InvalidInput(format!("{name} contain NaN or inf")));
    }
    Ok(())
}

/// EWMA variance per asset with a pinned initialization.
///
/// At `t = init_window - 1` the variance is the mean of squared returns over
/// the first `init_window` days (zero-mean convention); for
/// `t >= init_window`:
///
/// ```text
/// sigma2[t] = lam * sigma2[t-1] + (1 - lam) * r[t]^2
/// ```
///
/// Entries before `init_window - 1` are NaN (undefined).
///
/// # Errors
/// Invalid returns, `lam` outside `(0, 1)`, or `init_window` outside `[1, T]`.
pub fn ewma_variance(returns: &Matrix, lam: f64, init_window: usize) -> Result<Matrix> {
    validate_returns(returns, "returns")?;
    let (t_len, n_assets) = (returns.rows(), returns.cols());
    if !(lam > 0.0 && lam < 1.0) {
        return Err(RegimeError::InvalidInput(format!("ewma lambda must be in (0, 1), got {lam}")));
    }
    if init_window < 1 || init_window > t_len {
        return Err(RegimeError::InvalidInput(format!(
            "init_window must be in [1, T], got {init_window}"
        )));
    }
    let mut sig2 = Matrix::filled(t_len, n_assets, f64::NAN);
    for a in 0..n_assets {
        let mut acc = 0.0;
        for t in 0..init_window {
            let v = returns.get(t, a);
            acc += v * v;
        }
        sig2.set(init_window - 1, a, acc / init_window as f64);
        for t in init_window..t_len {
            let r = returns.get(t, a);
            let prev = sig2.get(t - 1, a);
            sig2.set(t, a, lam * prev + (1.0 - lam) * r * r);
        }
    }
    Ok(sig2)
}

/// Time-series momentum positions with vol-targeted sizing (API_SPEC.md §3.1).
///
/// The signal at day t (defined for `t >= lookback - 1`) is the sign of the
/// trailing cumulative simple return `prod(1 + r) - 1` over the `lookback`
/// days ending at t.  Sizing: per-asset leverage
/// `min(vol_target / ann_vol[t], leverage_cap)` with
/// `ann_vol = sqrt(252 * sigma2_EWMA)`; zero EWMA vol gets the cap.
/// Positions are divided by the asset count and are zero before the first
/// full lookback window.
///
/// # Errors
/// `T <= lookback`, non-positive `vol_target`/`leverage_cap`, or invalid
/// returns/lambda.
pub fn momentum_positions(
    returns: &Matrix,
    lookback: usize,
    vol_target: f64,
    ewma_lambda: f64,
    leverage_cap: f64,
) -> Result<Matrix> {
    validate_returns(returns, "returns")?;
    let (t_len, n_assets) = (returns.rows(), returns.cols());
    if lookback < 1 {
        return Err(RegimeError::InvalidInput(format!("lookback must be >= 1, got {lookback}")));
    }
    if t_len <= lookback {
        return Err(RegimeError::InvalidInput(format!(
            "series shorter than momentum lookback: T={t_len} <= lookback={lookback}"
        )));
    }
    if vol_target <= 0.0 || leverage_cap <= 0.0 {
        return Err(RegimeError::InvalidInput("vol_target and leverage_cap must be > 0".into()));
    }

    // Trailing cumulative return via cumulative growth ratios (matches the
    // reference: growth[t] / growth[t - lookback] - 1).
    let mut growth = Matrix::zeros(t_len, n_assets);
    for a in 0..n_assets {
        let mut acc = 1.0;
        for t in 0..t_len {
            acc *= 1.0 + returns.get(t, a);
            growth.set(t, a, acc);
        }
    }
    let sig2 = ewma_variance(returns, ewma_lambda, lookback)?;

    let mut pos = Matrix::zeros(t_len, n_assets);
    for t in (lookback - 1)..t_len {
        for a in 0..n_assets {
            let cumret = if t == lookback - 1 {
                growth.get(t, a) - 1.0
            } else {
                growth.get(t, a) / growth.get(t - lookback, a) - 1.0
            };
            let signal = if cumret > 0.0 {
                1.0
            } else if cumret < 0.0 {
                -1.0
            } else {
                0.0
            };
            let ann_vol = (TRADING_DAYS * sig2.get(t, a)).sqrt();
            let lev = if ann_vol > 0.0 {
                (vol_target / ann_vol).min(leverage_cap)
            } else {
                leverage_cap
            };
            pos.set(t, a, signal * lev / n_assets as f64);
        }
    }
    Ok(pos)
}

/// FX carry positions: long high-differential, short low-differential (§3.2).
///
/// Pinned schedule: rebalance at day indices `0, rebalance_days, ...`;
/// positions held unchanged in between.  At each rebalance, currencies are
/// ranked by that day's differential, descending, with ties broken by
/// ascending asset index (stable sort).  The top `top_n` get `+1/top_n`
/// each, the bottom `bottom_n` get `-1/bottom_n` each, the rest 0.
///
/// # Errors
/// `top_n + bottom_n > A`, zero `top_n`/`bottom_n`/`rebalance_days`, or
/// invalid differentials.
pub fn carry_positions(
    rate_diffs: &Matrix,
    top_n: usize,
    bottom_n: usize,
    rebalance_days: usize,
) -> Result<Matrix> {
    validate_returns(rate_diffs, "rate differentials")?;
    let (t_len, n_assets) = (rate_diffs.rows(), rate_diffs.cols());
    if top_n < 1 || bottom_n < 1 || top_n + bottom_n > n_assets {
        return Err(RegimeError::InvalidInput(format!(
            "need top_n>=1, bottom_n>=1, top_n+bottom_n<=A={n_assets}; got {top_n},{bottom_n}"
        )));
    }
    if rebalance_days < 1 {
        return Err(RegimeError::InvalidInput(format!(
            "rebalance_days must be >= 1, got {rebalance_days}"
        )));
    }

    let mut pos = Matrix::zeros(t_len, n_assets);
    let mut current = vec![0.0; n_assets];
    for t in 0..t_len {
        if t % rebalance_days == 0 {
            // Stable sort, descending differential; index breaks ties.
            let mut order: Vec<usize> = (0..n_assets).collect();
            order.sort_by(|&a, &b| {
                rate_diffs
                    .get(t, b)
                    .partial_cmp(&rate_diffs.get(t, a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            current = vec![0.0; n_assets];
            for &i in order.iter().take(top_n) {
                current[i] = 1.0 / top_n as f64;
            }
            for &i in order.iter().skip(n_assets - bottom_n) {
                current[i] = -1.0 / bottom_n as f64;
            }
        }
        for a in 0..n_assets {
            pos.set(t, a, current[a]);
        }
    }
    Ok(pos)
}

/// Total currency returns: spot move plus accrued carry (§3.2).
///
/// Pinned accrual: the day-t total return is `spot[t] + diff[t-1] / 252` —
/// the differential set at the previous close (known in advance, no
/// lookahead).  Day 0 accrues nothing.
///
/// # Errors
/// Shape mismatch or invalid input.
pub fn carry_total_returns(spot_returns: &Matrix, rate_diffs: &Matrix) -> Result<Matrix> {
    validate_returns(spot_returns, "spot returns")?;
    validate_returns(rate_diffs, "rate differentials")?;
    if spot_returns.rows() != rate_diffs.rows() || spot_returns.cols() != rate_diffs.cols() {
        return Err(RegimeError::InvalidInput(
            "spot returns and differentials must have equal shape".into(),
        ));
    }
    let mut total = spot_returns.clone();
    for t in 1..total.rows() {
        for a in 0..total.cols() {
            let v = total.get(t, a) + rate_diffs.get(t - 1, a) / TRADING_DAYS;
            total.set(t, a, v);
        }
    }
    Ok(total)
}

/// Walk-forward HMM regime gate on a market proxy (API_SPEC.md §3.3).
///
/// Pinned schedule: the first fit happens at day `t0 = train_min_days - 1`
/// on the expanding window `r[0..=t0]`; refits at `t0 + refit_days * k` on
/// `r[0..=t]`.  The first fit starts from the pinned quantile
/// initialization; every later refit warm-starts from the previous fitted
/// (label-sorted) parameters.
///
/// The gate at day t (for `t >= t0`) is the *filtered* probability
/// `p_calm(t) = P(s_t = k* | r_0..r_t)` of the low-vol state under the most
/// recent model, where `k* = argmin_k variances[k, 0]` (ties -> lower
/// label).  A full scaled forward pass runs at each refit day; single
/// [`forward_step`] updates are used in between (identical math, no
/// lookahead).  Before the first fit the gate is 1.
///
/// `mode = "prob"` returns `gate[t] = p_calm(t)`; `mode = "binary"` returns
/// `1.0` if `p_calm(t) >= threshold` else `0.0`.
///
/// Returns the gate values in `[0, 1]` and the fitted models in refit order.
///
/// # Errors
/// Series shorter than `train_min_days`, `train_min_days <= n_states`,
/// `refit_days < 1`, unknown `mode`, or non-finite input.
#[allow(clippy::too_many_arguments)]
pub fn regime_gate(
    index_returns: &[f64],
    n_states: usize,
    train_min_days: usize,
    refit_days: usize,
    mode: &str,
    threshold: f64,
    tol: f64,
    max_iter: usize,
    var_floor: f64,
) -> Result<(Vec<f64>, Vec<GaussianHmm>)> {
    if index_returns.iter().any(|v| !v.is_finite()) {
        return Err(RegimeError::InvalidInput("index_returns contain NaN or inf".into()));
    }
    if mode != "prob" && mode != "binary" {
        return Err(RegimeError::InvalidInput(format!(
            "gate mode must be 'prob' or 'binary', got {mode:?}"
        )));
    }
    let t_total = index_returns.len();
    if train_min_days <= n_states || refit_days < 1 {
        return Err(RegimeError::InvalidInput(
            "train_min_days must exceed n_states and refit_days must be >= 1".into(),
        ));
    }
    if t_total < train_min_days {
        return Err(RegimeError::InvalidInput(format!(
            "series shorter than train_min_days: T={t_total} < {train_min_days}"
        )));
    }

    let full = Matrix::column(index_returns);
    let t0 = train_min_days - 1;
    let mut gate = vec![1.0; t_total];
    let mut models: Vec<GaussianHmm> = Vec::new();
    let mut alpha: Vec<f64> = Vec::new();
    let mut calm_state = 0usize;
    for t in t0..t_total {
        if (t - t0) % refit_days == 0 {
            let mut model = GaussianHmm::new(n_states, tol, max_iter, var_floor)?;
            let warm = models.last().and_then(|m| m.params()).cloned();
            let window = full.head(t + 1);
            model.fit(&window, warm.as_ref())?;
            let params = model.params().expect("fit just succeeded");
            calm_state = argmin_variance(params);
            let filtered = model.filtered_probabilities(&window)?;
            alpha = filtered.row(t).to_vec();
            models.push(model);
        } else {
            let params = models.last().and_then(|m| m.params()).expect("model fitted");
            alpha = forward_step(params, &alpha, &[index_returns[t]]);
        }
        let p_calm = alpha[calm_state];
        gate[t] = if mode == "prob" {
            p_calm
        } else if p_calm >= threshold {
            1.0
        } else {
            0.0
        };
    }
    Ok((gate, models))
}

/// Index of the lowest-variance state on dimension 0 (ties -> lower label).
fn argmin_variance(params: &crate::hmm::HmmParams) -> usize {
    let k = params.startprob.len();
    let mut best = 0;
    for i in 1..k {
        if params.variances.get(i, 0) < params.variances.get(best, 0) {
            best = i;
        }
    }
    best
}

/// Scale each day's positions by that day's gate value.
///
/// `P_filtered[t] = gate[t] * P[t]`; both are day-t decisions, so the
/// filtered position still only trades against the day t+1 return.
///
/// # Errors
/// Length mismatch or non-finite gate values.
pub fn apply_gate(positions: &Matrix, gate: &[f64]) -> Result<Matrix> {
    validate_returns(positions, "positions")?;
    if gate.len() != positions.rows() {
        return Err(RegimeError::InvalidInput(format!(
            "gate length {} does not match positions rows {}",
            gate.len(),
            positions.rows()
        )));
    }
    if gate.iter().any(|v| !v.is_finite()) {
        return Err(RegimeError::InvalidInput("gate contains NaN or inf".into()));
    }
    let mut out = positions.clone();
    for t in 0..out.rows() {
        for a in 0..out.cols() {
            out.set(t, a, positions.get(t, a) * gate[t]);
        }
    }
    Ok(out)
}
