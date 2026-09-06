//! Backtest engine tests: pinned accounting, no-lookahead, metrics.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use regime::{
    compute_metrics, max_drawdown, run_backtest, state_conditional_returns, Date, Matrix,
};

#[test]
fn accounting_identity() {
    // net = gross - cost*turnover and equity = cumprod(1+net), exactly.
    let p = Matrix::from_rows(&[vec![1.0], vec![0.5], vec![-0.5], vec![-0.5]]).unwrap();
    let r = Matrix::from_rows(&[vec![0.01], vec![0.02], vec![-0.01], vec![0.03]]).unwrap();
    let res = run_backtest(&p, &r, 10.0, None).unwrap();
    let cost = 10.0 / 1e4;
    let expected_gross = [0.0, 1.0 * 0.02, 0.5 * -0.01, -0.5 * 0.03];
    let expected_to = [1.0, 0.5, 1.0, 0.0];
    let mut equity = 1.0;
    for t in 0..4 {
        assert!((res.gross_returns[t] - expected_gross[t]).abs() < 1e-15);
        assert!((res.turnover[t] - expected_to[t]).abs() < 1e-15);
        let net = expected_gross[t] - cost * expected_to[t];
        assert!((res.net_returns[t] - net).abs() < 1e-15);
        equity *= 1.0 + res.net_returns[t];
        assert!((res.equity[t] - equity).abs() < 1e-15);
    }
}

#[test]
fn no_lookahead_shift() {
    // Day-t P&L must come from the day t-1 position; using the same-day
    // position (lookahead) gives a measurably different result.
    let mut rng = StdRng::seed_from_u64(3); // test data only
    let normal = Normal::new(0.0, 0.01).unwrap();
    let r_vals: Vec<f64> = (0..300).map(|_| normal.sample(&mut rng)).collect();
    let p_vals: Vec<f64> = r_vals.iter().map(|v| v.signum()).collect();
    let p = Matrix::column(&p_vals);
    let r = Matrix::column(&r_vals);
    let res = run_backtest(&p, &r, 0.0, None).unwrap();
    let lookahead_pnl: f64 = p_vals.iter().zip(&r_vals).map(|(a, b)| a * b).sum();
    let honest_pnl: f64 = res.gross_returns.iter().sum();
    assert!((lookahead_pnl - honest_pnl).abs() > 0.1); // foresight is very profitable
    let shifted: f64 = (1..300).map(|t| p_vals[t - 1] * r_vals[t]).sum();
    assert!((honest_pnl - shifted).abs() < 1e-12);
}

#[test]
fn zero_cost_path() {
    let p = Matrix::from_rows(&[vec![1.0], vec![1.0], vec![-1.0]]).unwrap();
    let r = Matrix::from_rows(&[vec![0.01], vec![0.02], vec![0.03]]).unwrap();
    let res = run_backtest(&p, &r, 0.0, None).unwrap();
    assert_eq!(res.net_returns, res.gross_returns);
}

#[test]
fn turnover_first_day_is_position_build() {
    let p = Matrix::from_rows(&[vec![0.5, -0.5], vec![0.5, -0.5]]).unwrap();
    let r = Matrix::zeros(2, 2);
    let res = run_backtest(&p, &r, 0.0, None).unwrap();
    assert!((res.turnover[0] - 1.0).abs() < 1e-15);
    assert!(res.turnover[1].abs() < 1e-15);
}

#[test]
fn max_drawdown_known_case() {
    let equity = [1.0, 1.2, 0.6, 0.9, 1.3];
    assert!((max_drawdown(&equity).unwrap() - (0.6 / 1.2 - 1.0)).abs() < 1e-15); // -50%
    assert_eq!(max_drawdown(&[1.0, 1.1, 1.2]).unwrap(), 0.0);
    assert!(max_drawdown(&[]).is_err());
}

#[test]
fn metrics_formulas() {
    let net = [0.01, -0.005, 0.02, 0.0, -0.01];
    let m = compute_metrics(&net, None).unwrap();
    let mean: f64 = net.iter().sum::<f64>() / 5.0;
    let var: f64 = net.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 5.0;
    assert!((m.ann_return - 252.0 * mean).abs() < 1e-15);
    assert!((m.ann_vol - (252.0f64).sqrt() * var.sqrt()).abs() < 1e-15);
    assert!((m.sharpe - m.ann_return / m.ann_vol).abs() < 1e-12);
    assert!((m.hit_rate - 2.0 / 5.0).abs() < 1e-15);
    assert!(m.max_dd <= 0.0);
    // zero-vol series: sharpe defined as 0, no division error
    let z = compute_metrics(&[0.0; 10], None).unwrap();
    assert_eq!(z.sharpe, 0.0);
    assert_eq!(z.calmar, 0.0);
}

#[test]
fn worst_month_with_dates() {
    // 63 business-day-like series spanning three months; the middle stretch
    // is one bad calendar month.
    let mut dates = Vec::new();
    let mut net = Vec::new();
    for i in 0..63 {
        let month = 1 + (i / 21) as u32;
        dates.push(Date { year: 2020, month, day: 1 + (i % 21) as u32 });
        net.push(if (21..42).contains(&i) { -0.01 } else { 0.0 });
    }
    let m = compute_metrics(&net, Some(&dates)).unwrap();
    let expect = (1.0 - 0.01f64).powi(21) - 1.0;
    assert!((m.worst_month - expect).abs() < 1e-12);
    assert!(m.worst_month < -0.05);
}

#[test]
fn worst_month_block_fallback() {
    // Without dates: consecutive 21-day blocks, trailing partial ignored.
    let mut net = vec![0.0; 45];
    for v in net.iter_mut().take(42).skip(21) {
        *v = -0.01;
    }
    net[43] = -0.5; // in the trailing partial block; must not count
    let m = compute_metrics(&net, None).unwrap();
    let expect = (1.0 - 0.01f64).powi(21) - 1.0;
    assert!((m.worst_month - expect).abs() < 1e-12);
}

#[test]
fn state_conditional_returns_partition() {
    let net = [0.01, -0.02, 0.03, 0.0, -0.01, 0.02];
    let states = [0usize, 0, 1, 1, 2, 2];
    let out = state_conditional_returns(&net, &states, 3).unwrap();
    let total: usize = out.iter().map(|s| s.n_days).sum();
    assert_eq!(total, 6);
    let mu0 = 252.0 * (0.01 - 0.02) / 2.0;
    assert!((out[0].ann_return - mu0).abs() < 1e-12);
    let empty = state_conditional_returns(&net, &[0; 6], 2).unwrap();
    assert_eq!(empty[1].n_days, 0);
    assert_eq!(empty[1].sharpe, 0.0);
}

#[test]
fn backtest_validation() {
    assert!(run_backtest(&Matrix::filled(3, 1, 1.0), &Matrix::filled(4, 1, 1.0), 0.0, None).is_err()); // shape
    assert!(run_backtest(&Matrix::filled(3, 1, 1.0), &Matrix::filled(3, 1, 1.0), -1.0, None).is_err()); // cost
    assert!(run_backtest(&Matrix::filled(1, 1, f64::NAN), &Matrix::filled(1, 1, 0.0), 0.0, None).is_err());
    assert!(run_backtest(&Matrix::zeros(0, 1), &Matrix::zeros(0, 1), 0.0, None).is_err()); // empty
    assert!(compute_metrics(&[], None).is_err());
    let dates = [Date { year: 2020, month: 1, day: 1 }];
    assert!(run_backtest(&Matrix::filled(3, 1, 1.0), &Matrix::filled(3, 1, 0.0), 0.0, Some(&dates)).is_err());
}

#[test]
fn property_costs_never_help() {
    // Grid property: for any cost level, higher costs never raise the
    // total net return (turnover is non-negative).
    let mut rng = StdRng::seed_from_u64(5); // test data only
    let n_pos = Normal::new(0.0, 1.0).unwrap();
    let n_ret = Normal::new(0.0, 0.01).unwrap();
    let p_rows: Vec<Vec<f64>> = (0..100).map(|_| (0..3).map(|_| n_pos.sample(&mut rng)).collect()).collect();
    let r_rows: Vec<Vec<f64>> = (0..100).map(|_| (0..3).map(|_| n_ret.sample(&mut rng)).collect()).collect();
    let p = Matrix::from_rows(&p_rows).unwrap();
    let r = Matrix::from_rows(&r_rows).unwrap();
    let mut prev = f64::INFINITY;
    for bps in [0.0, 1.0, 5.0, 10.0, 25.0, 50.0] {
        let res = run_backtest(&p, &r, bps, None).unwrap();
        let total: f64 = res.net_returns.iter().sum();
        assert!(total <= prev + 1e-12);
        prev = total;
    }
}

// ------------------------------------------------------------------------- //
// Robustness: wipe-outs, dates, degenerate shapes
// ------------------------------------------------------------------------- //

#[test]
fn backtest_wipeout_is_error() {
    // PT-5: 4x leverage into a -30% day is a -120% net day -> error naming t.
    let p = Matrix::column(&[4.0, 4.0, 4.0]);
    let r = Matrix::column(&[0.0, -0.3, 0.1]);
    let err = run_backtest(&p, &r, 0.0, None).unwrap_err();
    assert!(err.to_string().contains("wiped out on day t=1"), "{err}");
    assert!(compute_metrics(&[0.01, -1.0, 0.02], None).is_err());
    // exactly -100% net on day 0 via costs alone is a wipe-out too
    let err = run_backtest(&Matrix::filled(1, 1, 1.0), &Matrix::zeros(1, 1), 1e4, None).unwrap_err();
    assert!(err.to_string().contains("t=0"));
    // invariant on accepted input: max_dd >= -1 and equity > 0
    let res = run_backtest(&p, &Matrix::column(&[0.0, -0.2, 0.1]), 0.0, None).unwrap();
    assert!(res.metrics.max_dd >= -1.0);
    assert!(res.equity.iter().all(|&e| e > 0.0));
    assert!((res.equity[1] - 0.2).abs() < 1e-15);
}

#[test]
fn backtest_rejects_returns_below_minus_one() {
    let p = Matrix::filled(3, 1, 1.0);
    for bad in [-1.0, -1.5] {
        let r = Matrix::column(&[0.0, bad, 0.0]);
        let err = run_backtest(&p, &r, 0.0, None).unwrap_err();
        assert!(err.to_string().contains("<= -100%"), "{err}");
    }
}

#[test]
fn compute_metrics_dates_length_mismatch() {
    // PT-7: the public metrics function validates dates itself.
    let three = [
        Date::new(2020, 1, 1).unwrap(),
        Date::new(2020, 1, 2).unwrap(),
        Date::new(2020, 1, 3).unwrap(),
    ];
    assert!(compute_metrics(&[0.0; 5], Some(&three)).is_err());
    assert!(run_backtest(&Matrix::filled(5, 1, 1.0), &Matrix::zeros(5, 1), 0.0, Some(&three)).is_err());
}

#[test]
fn dates_must_be_strictly_increasing() {
    // PT-8 / MAJ-6: duplicated, out-of-order or malformed dates are rejected.
    let p = Matrix::filled(3, 1, 1.0);
    let r = Matrix::zeros(3, 1);
    let dup = [
        Date::new(2020, 1, 2).unwrap(),
        Date::new(2020, 1, 2).unwrap(),
        Date::new(2020, 1, 3).unwrap(),
    ];
    let unsorted = [
        Date::new(2020, 2, 3).unwrap(),
        Date::new(2020, 1, 2).unwrap(),
        Date::new(2020, 1, 3).unwrap(),
    ];
    let malformed = [
        Date { year: 2020, month: 13, day: 1 },
        Date { year: 2020, month: 13, day: 2 },
        Date { year: 2020, month: 13, day: 3 },
    ];
    for dates in [&dup, &unsorted, &malformed] {
        let err = run_backtest(&p, &r, 0.0, Some(dates)).unwrap_err();
        assert!(matches!(err, regime::RegimeError::InvalidInput(_)), "{err}");
        assert!(compute_metrics(&[0.0; 3], Some(dates)).is_err());
    }
    assert!(Date::parse("2020-1-5").is_err());
    assert!(Date::parse("2020-13-01").is_err());
    assert!(Date::parse("2020-01-32").is_err());
    assert_eq!(Date::parse("2020-01-05").unwrap(), Date::new(2020, 1, 5).unwrap());
    // sorted but non-contiguous months: pinned "contiguous runs" == calendar months
    let dates = [
        Date::new(2020, 1, 2).unwrap(),
        Date::new(2020, 1, 31).unwrap(),
        Date::new(2020, 3, 2).unwrap(),
    ];
    let m = compute_metrics(&[-0.01, -0.02, 0.05], Some(&dates)).unwrap();
    assert!((m.worst_month - (0.99 * 0.98 - 1.0)).abs() < 1e-15);
}

#[test]
fn backtest_rejects_zero_assets() {
    // MAJ-7: A = 0 columns is an error in every language.
    let err = run_backtest(&Matrix::zeros(3, 0), &Matrix::zeros(3, 0), 0.0, None).unwrap_err();
    assert!(err.to_string().contains("no assets"), "{err}");
}

#[test]
fn max_drawdown_requires_positive_equity() {
    assert!(max_drawdown(&[1.0, 0.0, 0.5]).is_err());
    assert!(max_drawdown(&[1.0, -0.2]).is_err());
    assert!(max_drawdown(&[1.0, f64::INFINITY]).is_err());
}

#[test]
fn state_conditional_returns_validation() {
    // MIN-13: labels outside [0, K) and K < 1 are errors, never dropped.
    let net = [0.0; 4];
    assert!(state_conditional_returns(&net, &[0, 1, 2, 3], 3).is_err());
    assert!(state_conditional_returns(&net, &[0, 0, 0, 0], 0).is_err());
    assert!(state_conditional_returns(&[0.0, f64::NAN, 0.0, 0.0], &[0, 0, 0, 0], 1).is_err());
}

#[test]
fn worst_month_block_fallback_short_series() {
    // PT-11 complement: fewer than 21 days -> one block covering everything.
    let m = compute_metrics(&[-0.01; 5], None).unwrap();
    assert!((m.worst_month - (0.99f64.powi(5) - 1.0)).abs() < 1e-15);
}
