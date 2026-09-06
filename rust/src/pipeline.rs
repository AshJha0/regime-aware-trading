//! End-to-end pipeline on the bundled dataset.
//!
//! Loads the CSVs + config, fits full-sample HMMs (K=2, K=3) on the market
//! index, builds the walk-forward regime gate, runs momentum / carry /
//! combined backtests filtered and unfiltered, and computes the
//! crisis-state breakdown.  The demo and the golden tests both call
//! [`run_full_pipeline`] so they can never drift apart.
//!
//! Label conventions (pinned): HMM states are sorted by mean ascending, so
//! on this data state 0 = crisis (lowest mean, highest vol).  The bundled
//! true states use the generator's ordering (0 = calm-bull, 1 = choppy,
//! 2 = crisis); the pinned mapping goes through the state volatilities
//! (calm -> argmin variance, crisis -> argmax variance, choppy -> the rest).

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::backtest::{
    compute_metrics, run_backtest, state_conditional_returns, BacktestResult, Date, Metrics,
    StateStats,
};
use crate::error::{RegimeError, Result};
use crate::hmm::{GaussianHmm, HmmFitResult};
use crate::matrix::Matrix;
use crate::strategies::{
    apply_gate, carry_positions, carry_total_returns, momentum_positions, regime_gate,
};

/// Pinned strategy/backtest configuration (`data/config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// HMM state count for the gate and crisis attribution.
    pub n_states: usize,
    /// EM convergence tolerance.
    pub em_tol: f64,
    /// EM iteration cap.
    pub em_max_iter: usize,
    /// Variance floor.
    pub var_floor: f64,
    /// Momentum trailing window (days).
    pub momentum_lookback: usize,
    /// Annualized per-asset vol target.
    pub vol_target: f64,
    /// EWMA decay for the vol estimate.
    pub ewma_lambda: f64,
    /// Per-asset leverage cap.
    pub leverage_cap: f64,
    /// Carry long book size.
    pub carry_top_n: usize,
    /// Carry short book size.
    pub carry_bottom_n: usize,
    /// Carry rebalance cadence (days).
    pub rebalance_days: usize,
    /// Observations required before the first gate fit.
    pub train_min_days: usize,
    /// Days between walk-forward refits.
    pub refit_days: usize,
    /// Gate mode: `"prob"` or `"binary"`.
    pub gate_mode: String,
    /// Threshold for the binary gate.
    pub gate_threshold: f64,
    /// One-way transaction cost in basis points of turnover.
    pub cost_bps: f64,
    /// Trading days per year.
    pub trading_days: usize,
}

/// Bundled market data.
#[derive(Debug, Clone)]
pub struct Dataset {
    /// Trading dates, length `T`.
    pub dates: Vec<Date>,
    /// Market index daily returns, length `T`.
    pub index_returns: Vec<f64>,
    /// Trend-asset daily returns, shape `(T, 6)`.
    pub trend_returns: Matrix,
    /// Currency spot returns, shape `(T, 6)`.
    pub fx_spot_returns: Matrix,
    /// Annualized rate differentials, shape `(T, 6)`.
    pub fx_rate_diffs: Matrix,
    /// Generator regime path (teaching only), length `T`.
    pub true_states: Vec<usize>,
    /// Pinned configuration.
    pub config: Config,
}

/// One side (unfiltered or filtered) of the combined 50/50 portfolio.
#[derive(Debug, Clone)]
pub struct CombinedLeg {
    /// Daily 50/50 net returns.
    pub net_returns: Vec<f64>,
    /// Pinned metrics of the combined stream.
    pub metrics: Metrics,
}

/// Crisis-state attribution tables (per strategy, per decoded state).
#[derive(Debug, Clone)]
pub struct CrisisTables {
    /// Momentum, unfiltered.
    pub momentum_unfiltered: Vec<StateStats>,
    /// Momentum, regime-filtered.
    pub momentum_filtered: Vec<StateStats>,
    /// Carry, unfiltered.
    pub carry_unfiltered: Vec<StateStats>,
    /// Carry, regime-filtered.
    pub carry_filtered: Vec<StateStats>,
    /// Combined 50/50, unfiltered.
    pub combined_unfiltered: Vec<StateStats>,
    /// Combined 50/50, regime-filtered.
    pub combined_filtered: Vec<StateStats>,
}

/// Everything computed by [`run_full_pipeline`].
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// The loaded dataset.
    pub dataset: Dataset,
    /// Effective configuration (after any overrides).
    pub config: Config,
    /// Full-sample K=2 model on the index.
    pub hmm2: GaussianHmm,
    /// K=2 fit summary.
    pub fit2: HmmFitResult,
    /// Full-sample K=3 model on the index.
    pub hmm3: GaussianHmm,
    /// K=3 fit summary.
    pub fit3: HmmFitResult,
    /// Full-sample K=3 Viterbi path.
    pub viterbi3: Vec<usize>,
    /// Full-sample K=3 smoothed probabilities, shape `(T, 3)`.
    pub smoothed3: Matrix,
    /// Stationary distribution of the K=3 transition matrix.
    pub stationary3: Vec<f64>,
    /// Fraction of days where Viterbi matches the mapped true state.
    pub viterbi_accuracy: f64,
    /// Walk-forward gate values, length `T`.
    pub gate: Vec<f64>,
    /// Number of walk-forward refits.
    pub n_gate_models: usize,
    /// Momentum backtest, unfiltered.
    pub momentum_unfiltered: BacktestResult,
    /// Momentum backtest, regime-filtered.
    pub momentum_filtered: BacktestResult,
    /// Carry backtest, unfiltered.
    pub carry_unfiltered: BacktestResult,
    /// Carry backtest, regime-filtered.
    pub carry_filtered: BacktestResult,
    /// Combined portfolio, unfiltered.
    pub combined_unfiltered: CombinedLeg,
    /// Combined portfolio, regime-filtered.
    pub combined_filtered: CombinedLeg,
    /// Crisis-state attribution tables.
    pub crisis: CrisisTables,
}

fn read_file(path: &Path) -> Result<String> {
    fs::read_to_string(path)
        .map_err(|e| RegimeError::Data(format!("cannot read {}: {e}", path.display())))
}

/// Parse a CSV with a leading `date` column; returns `(dates, columns)` where
/// `columns` holds the named float columns in file order.
fn parse_csv(path: &Path) -> Result<(Vec<Date>, Vec<String>, Matrix)> {
    let text = read_file(path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| RegimeError::Data(format!("{}: empty file", path.display())))?;
    let cols: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
    if cols.first().map(String::as_str) != Some("date") {
        return Err(RegimeError::Data(format!("{}: first column must be 'date'", path.display())));
    }
    let names: Vec<String> = cols[1..].to_vec();
    let mut dates = Vec::new();
    let mut data = Vec::new();
    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != cols.len() {
            return Err(RegimeError::Data(format!("{}: ragged row", path.display())));
        }
        dates.push(Date::parse(fields[0])?);
        for f in &fields[1..] {
            let v: f64 = f
                .trim()
                .parse()
                .map_err(|_| RegimeError::Data(format!("{}: bad float {f:?}", path.display())))?;
            data.push(v);
        }
    }
    let rows = dates.len();
    let matrix = Matrix::from_vec(rows, names.len(), data)?;
    Ok((dates, names, matrix))
}

/// Load the bundled CSVs and config from `data_dir`.
///
/// # Errors
/// Missing files, parse failures, or frames that disagree on length.
pub fn load_dataset<P: AsRef<Path>>(data_dir: P) -> Result<Dataset> {
    let d = data_dir.as_ref();
    for name in ["market_index.csv", "trend_assets.csv", "fx_carry.csv", "true_states.csv", "config.json"]
    {
        if !d.join(name).exists() {
            return Err(RegimeError::Data(format!("missing bundled data file: {}", d.join(name).display())));
        }
    }
    let (dates, _, idx) = parse_csv(&d.join("market_index.csv"))?;
    let (trend_dates, trend_names, trend) = parse_csv(&d.join("trend_assets.csv"))?;
    let (fx_dates, fx_names, fx) = parse_csv(&d.join("fx_carry.csv"))?;
    let (ts_dates, _, ts) = parse_csv(&d.join("true_states.csv"))?;
    let config: Config = serde_json::from_str(&read_file(&d.join("config.json"))?)
        .map_err(|e| RegimeError::Data(format!("config.json: {e}")))?;

    let t_len = dates.len();
    if trend_dates.len() != t_len || fx_dates.len() != t_len || ts_dates.len() != t_len {
        return Err(RegimeError::Data("bundled CSVs have inconsistent lengths".into()));
    }

    let index_returns = idx.col(0);
    let trend_cols: Vec<usize> = trend_names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.starts_with("asset_"))
        .map(|(i, _)| i)
        .collect();
    let spot_cols: Vec<usize> = fx_names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.starts_with("spot_ret_"))
        .map(|(i, _)| i)
        .collect();
    let diff_cols: Vec<usize> = fx_names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.starts_with("rate_diff_"))
        .map(|(i, _)| i)
        .collect();

    let select = |m: &Matrix, cols: &[usize]| -> Matrix {
        let mut out = Matrix::zeros(m.rows(), cols.len());
        for r in 0..m.rows() {
            for (j, &c) in cols.iter().enumerate() {
                out.set(r, j, m.get(r, c));
            }
        }
        out
    };

    if t_len == 0 {
        return Err(RegimeError::Data("bundled CSVs are empty".into()));
    }
    let mut true_states: Vec<usize> = Vec::with_capacity(t_len);
    for v in ts.col(0) {
        if v.fract() != 0.0 || !(0.0..=2.0).contains(&v) {
            return Err(RegimeError::Data(
                "true_states.csv: state labels must be integers in [0, 2]".into(),
            ));
        }
        true_states.push(v as usize);
    }
    Ok(Dataset {
        dates,
        index_returns,
        trend_returns: select(&trend, &trend_cols),
        fx_spot_returns: select(&fx, &spot_cols),
        fx_rate_diffs: select(&fx, &diff_cols),
        true_states,
        config,
    })
}

/// Run the entire study on the bundled data.
///
/// `refit_days` / `train_min_days` optionally override the pinned config
/// (tests may shorten the walk-forward loop; golden values always use the
/// bundled config).
///
/// # Errors
/// Data loading or any downstream validation failure.
pub fn run_full_pipeline<P: AsRef<Path>>(
    data_dir: P,
    refit_days: Option<usize>,
    train_min_days: Option<usize>,
) -> Result<PipelineOutput> {
    let dataset = load_dataset(data_dir)?;
    let mut cfg = dataset.config.clone();
    if let Some(v) = refit_days {
        cfg.refit_days = v;
    }
    if let Some(v) = train_min_days {
        cfg.train_min_days = v;
    }
    let k = cfg.n_states;
    let r_idx = Matrix::column(&dataset.index_returns);

    // Full-sample HMM fits (model selection + crisis attribution).
    let mut hmm2 = GaussianHmm::new(2, cfg.em_tol, cfg.em_max_iter, cfg.var_floor)?;
    let fit2 = hmm2.fit(&r_idx, None)?;
    let mut hmm3 = GaussianHmm::new(3, cfg.em_tol, cfg.em_max_iter, cfg.var_floor)?;
    let fit3 = hmm3.fit(&r_idx, None)?;
    let viterbi3 = hmm3.viterbi(&r_idx)?;
    let smoothed3 = hmm3.smoothed_probabilities(&r_idx)?;
    let stationary3 = hmm3.stationary_distribution()?;

    // Teaching comparison via the pinned volatility mapping of the K=3 fit
    // (independent of the gate's config n_states).
    let k3 = hmm3.n_states;
    let p3 = hmm3.params().expect("fitted");
    let variances: Vec<f64> = (0..k3).map(|i| p3.variances.get(i, 0)).collect();
    let calm_label = argmin(&variances);
    let crisis_label = argmax(&variances);
    let choppy_label = 3 - calm_label - crisis_label;
    let mapping = [calm_label, choppy_label, crisis_label];
    let hits = dataset
        .true_states
        .iter()
        .zip(&viterbi3)
        .filter(|(&ts, &vs)| mapping[ts] == vs)
        .count();
    let viterbi_accuracy = hits as f64 / dataset.true_states.len() as f64;

    // Walk-forward regime gate on the index (no lookahead).
    let (gate, gate_models) = regime_gate(
        &dataset.index_returns,
        k,
        cfg.train_min_days,
        cfg.refit_days,
        &cfg.gate_mode,
        cfg.gate_threshold,
        cfg.em_tol,
        cfg.em_max_iter,
        cfg.var_floor,
    )?;

    let cost = cfg.cost_bps;
    let dates = Some(dataset.dates.as_slice());

    let mom_pos = momentum_positions(
        &dataset.trend_returns,
        cfg.momentum_lookback,
        cfg.vol_target,
        cfg.ewma_lambda,
        cfg.leverage_cap,
    )?;
    let carry_pos = carry_positions(
        &dataset.fx_rate_diffs,
        cfg.carry_top_n,
        cfg.carry_bottom_n,
        cfg.rebalance_days,
    )?;
    let carry_rets = carry_total_returns(&dataset.fx_spot_returns, &dataset.fx_rate_diffs)?;

    let momentum_unfiltered = run_backtest(&mom_pos, &dataset.trend_returns, cost, dates)?;
    let momentum_filtered =
        run_backtest(&apply_gate(&mom_pos, &gate)?, &dataset.trend_returns, cost, dates)?;
    let carry_unfiltered = run_backtest(&carry_pos, &carry_rets, cost, dates)?;
    let carry_filtered = run_backtest(&apply_gate(&carry_pos, &gate)?, &carry_rets, cost, dates)?;

    // Combined portfolio: 50/50 in each strategy's net return stream.
    let combine = |a: &BacktestResult, b: &BacktestResult| -> Result<CombinedLeg> {
        let net: Vec<f64> = a
            .net_returns
            .iter()
            .zip(&b.net_returns)
            .map(|(x, y)| 0.5 * x + 0.5 * y)
            .collect();
        let metrics = compute_metrics(&net, dates)?;
        Ok(CombinedLeg { net_returns: net, metrics })
    };
    let combined_unfiltered = combine(&momentum_unfiltered, &carry_unfiltered)?;
    let combined_filtered = combine(&momentum_filtered, &carry_filtered)?;

    // Crisis table: attribute daily strategy returns to the full-sample
    // K=3 Viterbi state (state 0 = crisis on this data); the table has k3
    // rows regardless of the gate's n_states.
    let crisis = CrisisTables {
        momentum_unfiltered: state_conditional_returns(&momentum_unfiltered.net_returns, &viterbi3, k3)?,
        momentum_filtered: state_conditional_returns(&momentum_filtered.net_returns, &viterbi3, k3)?,
        carry_unfiltered: state_conditional_returns(&carry_unfiltered.net_returns, &viterbi3, k3)?,
        carry_filtered: state_conditional_returns(&carry_filtered.net_returns, &viterbi3, k3)?,
        combined_unfiltered: state_conditional_returns(&combined_unfiltered.net_returns, &viterbi3, k3)?,
        combined_filtered: state_conditional_returns(&combined_filtered.net_returns, &viterbi3, k3)?,
    };

    Ok(PipelineOutput {
        dataset,
        config: cfg,
        hmm2,
        fit2,
        hmm3,
        fit3,
        viterbi3,
        smoothed3,
        stationary3,
        viterbi_accuracy,
        gate,
        n_gate_models: gate_models.len(),
        momentum_unfiltered,
        momentum_filtered,
        carry_unfiltered,
        carry_filtered,
        combined_unfiltered,
        combined_filtered,
        crisis,
    })
}

fn argmin(xs: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..xs.len() {
        if xs[i] < xs[best] {
            best = i;
        }
    }
    best
}

fn argmax(xs: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..xs.len() {
        if xs[i] > xs[best] {
            best = i;
        }
    }
    best
}

/// Build the golden-value map `case name -> {key -> value}` from a full
/// pipeline run.  Case names and keys are pinned in API_SPEC.md §5; the
/// golden test compares this map against `data/golden/golden.json`.
pub fn golden_values(pipe: &PipelineOutput) -> BTreeMap<String, BTreeMap<String, f64>> {
    let mut out: BTreeMap<String, BTreeMap<String, f64>> = BTreeMap::new();
    let mut case = |name: &str, entries: Vec<(String, f64)>| {
        out.insert(name.to_string(), entries.into_iter().collect());
    };
    let p3 = pipe.hmm3.params().expect("fitted");

    case("hmm_k2_loglik", vec![("loglik".into(), pipe.fit2.log_likelihood)]);
    case("hmm_k3_loglik", vec![("loglik".into(), pipe.fit3.log_likelihood)]);
    case(
        "hmm_k3_means",
        (0..3).map(|k| (format!("mean{k}"), p3.means.get(k, 0))).collect(),
    );
    case(
        "hmm_k3_stds",
        (0..3).map(|k| (format!("std{k}"), p3.variances.get(k, 0).sqrt())).collect(),
    );
    case(
        "hmm_k3_transmat",
        (0..3)
            .flat_map(|i| (0..3).map(move |j| (format!("a{i}{j}"), (i, j))))
            .map(|(name, (i, j))| (name, p3.transmat.get(i, j)))
            .collect(),
    );
    let mut counts = [0usize; 3];
    for &s in &pipe.viterbi3 {
        counts[s] += 1;
    }
    case(
        "hmm_k3_viterbi_counts",
        (0..3).map(|k| (format!("count{k}"), counts[k] as f64)).collect(),
    );
    case(
        "hmm_k3_stationary",
        (0..3).map(|k| (format!("pi{k}"), pipe.stationary3[k])).collect(),
    );
    for t in [500usize, 1500] {
        case(
            &format!("hmm_k3_smoothed_t{t}"),
            (0..3).map(|k| (format!("p{k}"), pipe.smoothed3.get(t, k))).collect(),
        );
    }
    case("viterbi_accuracy_vs_true", vec![("accuracy".into(), pipe.viterbi_accuracy)]);

    case("momentum_unfiltered", metric_entries(&pipe.momentum_unfiltered.metrics));
    case("momentum_filtered", metric_entries(&pipe.momentum_filtered.metrics));
    case("carry_unfiltered", metric_entries(&pipe.carry_unfiltered.metrics));
    case("carry_filtered", metric_entries(&pipe.carry_filtered.metrics));
    case(
        "combined_sharpe",
        vec![
            ("sharpe_unfiltered".into(), pipe.combined_unfiltered.metrics.sharpe),
            ("sharpe_filtered".into(), pipe.combined_filtered.metrics.sharpe),
            ("max_dd_unfiltered".into(), pipe.combined_unfiltered.metrics.max_dd),
            ("max_dd_filtered".into(), pipe.combined_filtered.metrics.max_dd),
        ],
    );
    case(
        "crisis_momentum_return",
        vec![("ann_return".into(), pipe.crisis.momentum_unfiltered[0].ann_return)],
    );
    let mean_of = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;
    case(
        "turnover_momentum",
        vec![("mean_turnover".into(), mean_of(&pipe.momentum_unfiltered.turnover))],
    );
    case(
        "turnover_carry",
        vec![("mean_turnover".into(), mean_of(&pipe.carry_unfiltered.turnover))],
    );
    out
}

/// All seven pinned metrics of one strategy case, keyed as in golden.json.
fn metric_entries(m: &Metrics) -> Vec<(String, f64)> {
    vec![
        ("ann_return".to_string(), m.ann_return),
        ("ann_vol".to_string(), m.ann_vol),
        ("sharpe".to_string(), m.sharpe),
        ("max_dd".to_string(), m.max_dd),
        ("calmar".to_string(), m.calmar),
        ("hit_rate".to_string(), m.hit_rate),
        ("worst_month".to_string(), m.worst_month),
    ]
}

/// Inputs of the `carry_rerank_*` golden cases (API_SPEC.md §5): a
/// piecewise-constant differential panel whose ranks cross.
#[derive(Debug, Clone, Deserialize)]
pub struct CarryRerankInputs {
    /// Number of days `T`.
    pub n_days: usize,
    /// Number of currencies `A`.
    pub n_assets: usize,
    /// Ascending day indices (starting at 0) at which `diffs` rows take effect.
    pub switch_days: Vec<usize>,
    /// One differential row per switch day.
    pub diffs: Vec<Vec<f64>>,
    /// Carry long book size.
    pub top_n: usize,
    /// Carry short book size.
    pub bottom_n: usize,
    /// Rebalance cadence (days).
    pub rebalance_days: usize,
    /// One-way cost in bps of turnover.
    pub cost_bps: f64,
}

/// Build the pinned re-ranking panel: `diff[t] = diffs[k]` with `k` the
/// last index whose `switch_days[k] <= t`.
///
/// # Errors
/// Inconsistent inputs (ragged rows, unsorted switch days).
pub fn carry_rerank_panel(inputs: &CarryRerankInputs) -> Result<Matrix> {
    let (t_len, n_assets) = (inputs.n_days, inputs.n_assets);
    let sw = &inputs.switch_days;
    if sw.len() != inputs.diffs.len()
        || sw.first() != Some(&0)
        || sw.windows(2).any(|w| w[1] <= w[0])
    {
        return Err(RegimeError::InvalidInput(
            "carry_rerank inputs: switch_days must start at 0 and be strictly increasing".into(),
        ));
    }
    let mut panel = Matrix::zeros(t_len, n_assets);
    let mut k = 0;
    for t in 0..t_len {
        while k + 1 < sw.len() && sw[k + 1] <= t {
            k += 1;
        }
        let row = &inputs.diffs[k];
        if row.len() != n_assets {
            return Err(RegimeError::InvalidInput("carry_rerank inputs: ragged diffs".into()));
        }
        for a in 0..n_assets {
            panel.set(t, a, row[a]);
        }
    }
    Ok(panel)
}

/// Every `carry_rerank_*` golden scalar computed from the pinned inputs.
///
/// # Errors
/// Any downstream validation failure.
pub fn carry_rerank_values(inputs: &CarryRerankInputs) -> Result<BTreeMap<String, f64>> {
    let diffs = carry_rerank_panel(inputs)?;
    let pos = carry_positions(&diffs, inputs.top_n, inputs.bottom_n, inputs.rebalance_days)?;
    let rets = carry_total_returns(&Matrix::zeros(diffs.rows(), diffs.cols()), &diffs)?;
    let res = run_backtest(&pos, &rets, inputs.cost_bps, None)?;
    let mean_to = res.turnover.iter().sum::<f64>() / res.turnover.len() as f64;
    let mut out = BTreeMap::new();
    out.insert("mean_turnover".to_string(), mean_to);
    out.insert("turnover_day21".to_string(), res.turnover[21]);
    out.insert("turnover_day42".to_string(), res.turnover[42]);
    for (t, a) in [(15, 0), (15, 3), (21, 0), (21, 3), (42, 0), (42, 3)] {
        out.insert(format!("pos_day{t}_asset{a}"), pos.try_get(t, a)?);
    }
    out.insert("ann_return".to_string(), res.metrics.ann_return);
    out.insert("sharpe".to_string(), res.metrics.sharpe);
    out.insert("max_dd".to_string(), res.metrics.max_dd);
    Ok(out)
}
