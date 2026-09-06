//! Backtest engine with pinned no-lookahead accounting and metrics.
//!
//! # Accounting (pinned, shared by every strategy — API_SPEC.md §2)
//!
//! Positions `P[t]` are decided at the close of day t.  The trade from
//! `P[t-1]` to `P[t]` executes at that close, so:
//!
//! ```text
//! gross[t]    = sum_a P[t-1, a] * r[t, a]          (gross[0] = 0)
//! turnover[t] = sum_a |P[t, a] - P[t-1, a]|        (P[-1] = 0)
//! net[t]      = gross[t] - (cost_bps / 10^4) * turnover[t]
//! equity[t]   = prod_{u<=t} (1 + net[u])
//! ```
//!
//! The day-t P&L is earned by yesterday's position — the signal never sees
//! the return it trades against — while the cost of moving *into* `P[t]` is
//! charged on day t itself.  `cost_bps = 0` gives `net == gross` exactly.

use crate::error::{RegimeError, Result};
use crate::matrix::Matrix;

/// Trading days per year (pinned).
pub const TRADING_DAYS: f64 = 252.0;

/// Calendar date `YYYY-MM-DD` (only year/month matter for the metrics).
/// Field order gives the derived `Ord` chronological meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Date {
    /// Four-digit year.
    pub year: i32,
    /// Month 1..=12.
    pub month: u32,
    /// Day of month 1..=31.
    pub day: u32,
}

impl Date {
    /// Parse a `YYYY-MM-DD` string (exactly ten characters, month 1..=12,
    /// day 1..=31; calendar validity beyond that is not checked).
    ///
    /// # Errors
    /// `RegimeError::Data` on any malformed input.
    pub fn parse(s: &str) -> Result<Date> {
        let bad = || RegimeError::Data(format!("bad date {s:?} (want YYYY-MM-DD)"));
        let b = s.as_bytes();
        if b.len() != 10 || b[4] != b'-' || b[7] != b'-' {
            return Err(bad());
        }
        if [0, 1, 2, 3, 5, 6, 8, 9].iter().any(|&i| !b[i].is_ascii_digit()) {
            return Err(bad());
        }
        let year = s[0..4].parse::<i32>().map_err(|_| bad())?;
        let month = s[5..7].parse::<u32>().map_err(|_| bad())?;
        let day = s[8..10].parse::<u32>().map_err(|_| bad())?;
        Date::new(year, month, day)
    }

    /// Construct a date, rejecting month outside 1..=12 or day outside 1..=31.
    ///
    /// # Errors
    /// `RegimeError::Data` on an out-of-range month or day.
    pub fn new(year: i32, month: u32, day: u32) -> Result<Date> {
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return Err(RegimeError::Data(format!("bad date {year:04}-{month:02}-{day:02}")));
        }
        Ok(Date { year, month, day })
    }

    fn is_valid(&self) -> bool {
        (1..=12).contains(&self.month) && (1..=31).contains(&self.day)
    }
}

/// Pinned date checks (API_SPEC.md §2.1): length `T`, well-formed, strictly
/// increasing.
fn validate_dates(dates: &[Date], t_len: usize) -> Result<()> {
    if dates.len() != t_len {
        return Err(RegimeError::InvalidInput(format!(
            "dates length {} does not match number of days {t_len}",
            dates.len()
        )));
    }
    for (t, d) in dates.iter().enumerate() {
        if !d.is_valid() {
            return Err(RegimeError::InvalidInput(format!("dates could not be parsed: {d} at t={t}")));
        }
        if t > 0 && dates[t - 1] >= *d {
            return Err(RegimeError::InvalidInput(format!(
                "dates must be strictly increasing (at t={t})"
            )));
        }
    }
    Ok(())
}

/// Pinned net-return checks: non-empty, finite, and `> -1` (a net return of
/// -100 % or worse means the book is wiped out; API_SPEC.md §2).
fn validate_net(net: &[f64]) -> Result<()> {
    if net.is_empty() {
        return Err(RegimeError::InvalidInput("net return series is empty".into()));
    }
    if net.iter().any(|v| !v.is_finite()) {
        return Err(RegimeError::InvalidInput("net returns contain NaN or inf".into()));
    }
    if let Some(t) = net.iter().position(|&v| v <= -1.0) {
        return Err(RegimeError::InvalidInput(format!(
            "equity wiped out on day t={t}: net return {} <= -100%",
            net[t]
        )));
    }
    Ok(())
}

impl std::fmt::Display for Date {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

/// Pinned summary metrics of a daily net-return series (§2.1).
#[derive(Debug, Clone, PartialEq)]
pub struct Metrics {
    /// `252 * mean(net)`.
    pub ann_return: f64,
    /// `sqrt(252) * std(net, ddof=0)`.
    pub ann_vol: f64,
    /// `ann_return / ann_vol` (0 if `ann_vol == 0`).
    pub sharpe: f64,
    /// `min_t (equity[t] / cummax(equity)[t] - 1)`, non-positive.
    pub max_dd: f64,
    /// `ann_return / |max_dd|` (0 if `max_dd == 0`).
    pub calmar: f64,
    /// Fraction of days with `net > 0`.
    pub hit_rate: f64,
    /// Worst calendar-month compounded return (21-day blocks without dates).
    pub worst_month: f64,
}

/// Daily series and summary metrics of one backtest run.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Pre-cost daily portfolio returns, length `T`.
    pub gross_returns: Vec<f64>,
    /// Post-cost daily returns, length `T`.
    pub net_returns: Vec<f64>,
    /// Daily one-sided turnover `sum |dP|`, length `T`.
    pub turnover: Vec<f64>,
    /// Compounded equity curve starting near 1, length `T`.
    pub equity: Vec<f64>,
    /// Pinned summary metrics.
    pub metrics: Metrics,
}

/// Per-state performance row of the crisis table (§2.2).
#[derive(Debug, Clone, PartialEq)]
pub struct StateStats {
    /// Days attributed to the state.
    pub n_days: usize,
    /// `252 * mean(net in state)`.
    pub ann_return: f64,
    /// `sqrt(252) * std(net in state, ddof=0)`.
    pub ann_vol: f64,
    /// `ann_return / ann_vol` (0 if `ann_vol == 0`).
    pub sharpe: f64,
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn std_ddof0(xs: &[f64]) -> f64 {
    let mu = mean(xs);
    (xs.iter().map(|v| (v - mu) * (v - mu)).sum::<f64>() / xs.len() as f64).sqrt()
}

/// Maximum drawdown of an equity curve, as a non-positive fraction.
///
/// # Errors
/// Empty, non-finite or non-positive curve (a drawdown from or to a
/// non-positive equity is undefined; `run_backtest` rejects wipe-outs first).
pub fn max_drawdown(equity: &[f64]) -> Result<f64> {
    if equity.is_empty() {
        return Err(RegimeError::InvalidInput("equity curve is empty".into()));
    }
    if equity.iter().any(|&e| !e.is_finite() || e <= 0.0) {
        return Err(RegimeError::InvalidInput(
            "equity curve must be finite and strictly positive".into(),
        ));
    }
    let mut peak = f64::NEG_INFINITY;
    let mut mdd = f64::INFINITY;
    for &e in equity {
        peak = peak.max(e);
        mdd = mdd.min(e / peak - 1.0);
    }
    Ok(mdd)
}

/// Pinned summary metrics from a daily net-return series.
///
/// With `dates` (strictly increasing), `worst_month` compounds within
/// calendar months; without, it uses consecutive 21-day blocks (any
/// trailing partial block beyond `T/21` full blocks is dropped — pinned,
/// API_SPEC.md §2.1).
///
/// # Errors
/// Empty or non-finite series, a net return `<= -1` (wipe-out), or dates
/// that are the wrong length, malformed or not strictly increasing.
pub fn compute_metrics(net: &[f64], dates: Option<&[Date]>) -> Result<Metrics> {
    validate_net(net)?;
    if let Some(d) = dates {
        validate_dates(d, net.len())?;
    }
    let ann_ret = TRADING_DAYS * mean(net);
    let ann_vol = TRADING_DAYS.sqrt() * std_ddof0(net);
    let sharpe = if ann_vol > 0.0 { ann_ret / ann_vol } else { 0.0 };
    let mut equity = Vec::with_capacity(net.len());
    let mut acc = 1.0;
    for &v in net {
        acc *= 1.0 + v;
        equity.push(acc);
    }
    let mdd = max_drawdown(&equity)?;
    let calmar = if mdd < 0.0 { ann_ret / mdd.abs() } else { 0.0 };
    let hit = net.iter().filter(|&&v| v > 0.0).count() as f64 / net.len() as f64;

    let worst_month = match dates {
        Some(d) => {
            let mut worst = f64::INFINITY;
            let mut i = 0;
            while i < net.len() {
                let key = (d[i].year, d[i].month);
                let mut growth = 1.0;
                while i < net.len() && (d[i].year, d[i].month) == key {
                    growth *= 1.0 + net[i];
                    i += 1;
                }
                worst = worst.min(growth - 1.0);
            }
            worst
        }
        None => {
            let nblk = (net.len() / 21).max(1);
            let mut worst = f64::INFINITY;
            for b in 0..nblk {
                let lo = b * 21;
                let hi = ((b + 1) * 21).min(net.len());
                let growth: f64 = net[lo..hi].iter().map(|v| 1.0 + v).product();
                worst = worst.min(growth - 1.0);
            }
            worst
        }
    };
    Ok(Metrics { ann_return: ann_ret, ann_vol, sharpe, max_dd: mdd, calmar, hit_rate: hit, worst_month })
}

/// Run the pinned accounting on a position/return pair.
///
/// `positions[t][a]` is decided at the close of day t; `returns` must have
/// the same shape.  `cost_bps` is a one-way transaction cost in basis points
/// of turnover; `dates` (optional, length `T`) enables calendar-month
/// metrics.
///
/// # Errors
/// Shape mismatch, empty input (no days or no assets), non-finite values, a
/// return `<= -1`, negative `cost_bps`, invalid `dates`, or a wipe-out
/// (`net[t] <= -1` for some day; the message names `t`) — API_SPEC.md §2.
pub fn run_backtest(
    positions: &Matrix,
    returns: &Matrix,
    cost_bps: f64,
    dates: Option<&[Date]>,
) -> Result<BacktestResult> {
    if positions.rows() != returns.rows() || positions.cols() != returns.cols() {
        return Err(RegimeError::InvalidInput(format!(
            "positions {}x{} and returns {}x{} must have equal shape",
            positions.rows(),
            positions.cols(),
            returns.rows(),
            returns.cols()
        )));
    }
    if positions.rows() == 0 {
        return Err(RegimeError::InvalidInput("empty backtest: no days".into()));
    }
    if positions.cols() == 0 {
        return Err(RegimeError::InvalidInput("empty backtest: no assets".into()));
    }
    if !positions.all_finite() || !returns.all_finite() {
        return Err(RegimeError::InvalidInput("positions/returns contain NaN or inf".into()));
    }
    for t in 0..returns.rows() {
        for a in 0..returns.cols() {
            if returns.get(t, a) <= -1.0 {
                return Err(RegimeError::InvalidInput(format!(
                    "returns contain a return <= -100% at t={t}, asset={a} ({})",
                    returns.get(t, a)
                )));
            }
        }
    }
    if !cost_bps.is_finite() || cost_bps < 0.0 {
        return Err(RegimeError::InvalidInput(format!("cost_bps must be >= 0, got {cost_bps}")));
    }
    if let Some(d) = dates {
        validate_dates(d, positions.rows())?;
    }

    let (t_len, n_assets) = (positions.rows(), positions.cols());
    let mut gross = vec![0.0; t_len];
    for t in 1..t_len {
        let mut s = 0.0;
        for a in 0..n_assets {
            s += positions.get(t - 1, a) * returns.get(t, a);
        }
        gross[t] = s;
    }
    let mut turnover = vec![0.0; t_len];
    for t in 0..t_len {
        let mut s = 0.0;
        for a in 0..n_assets {
            let prev = if t == 0 { 0.0 } else { positions.get(t - 1, a) };
            s += (positions.get(t, a) - prev).abs();
        }
        turnover[t] = s;
    }
    let cost = cost_bps / 1e4;
    let net: Vec<f64> = gross.iter().zip(&turnover).map(|(g, to)| g - cost * to).collect();
    validate_net(&net)?; // wipe-out guard: net[t] <= -1 is an error naming t
    let mut equity = Vec::with_capacity(t_len);
    let mut acc = 1.0;
    for &v in &net {
        acc *= 1.0 + v;
        equity.push(acc);
    }
    let metrics = compute_metrics(&net, dates)?;
    Ok(BacktestResult { gross_returns: gross, net_returns: net, turnover, equity, metrics })
}

/// Annualized performance of a return series conditioned on a state path.
///
/// Used for the crisis-period table: the day-t net return is attributed to
/// the day-t decoded state.  States with no days get all-zero stats.
///
/// # Errors
/// Length mismatch between `net` and `states`, non-finite `net`,
/// `n_states < 1`, or a label outside `[0, n_states)` (pinned: labels are
/// never silently dropped).
pub fn state_conditional_returns(
    net: &[f64],
    states: &[usize],
    n_states: usize,
) -> Result<Vec<StateStats>> {
    if net.len() != states.len() {
        return Err(RegimeError::InvalidInput("net returns and states must have equal length".into()));
    }
    if net.iter().any(|v| !v.is_finite()) {
        return Err(RegimeError::InvalidInput("net returns contain NaN or inf".into()));
    }
    if n_states < 1 {
        return Err(RegimeError::InvalidInput(format!(
            "n_states must be an integer >= 1, got {n_states}"
        )));
    }
    if states.iter().any(|&s| s >= n_states) {
        return Err(RegimeError::InvalidInput(format!(
            "state labels must be integers in [0, {n_states})"
        )));
    }
    let mut out = Vec::with_capacity(n_states);
    for k in 0..n_states {
        let sel: Vec<f64> = net
            .iter()
            .zip(states)
            .filter(|(_, &s)| s == k)
            .map(|(&v, _)| v)
            .collect();
        if sel.is_empty() {
            out.push(StateStats { n_days: 0, ann_return: 0.0, ann_vol: 0.0, sharpe: 0.0 });
            continue;
        }
        let mu = TRADING_DAYS * mean(&sel);
        let sd = TRADING_DAYS.sqrt() * std_ddof0(&sel);
        out.push(StateStats {
            n_days: sel.len(),
            ann_return: mu,
            ann_vol: sd,
            sharpe: if sd > 0.0 { mu / sd } else { 0.0 },
        });
    }
    Ok(out)
}
