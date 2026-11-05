//! Strategy tests: momentum sizing, carry schedule/tie-break, regime gate.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use regime::{
    apply_gate, carry_positions, carry_total_returns, ewma_variance, momentum_positions,
    regime_gate, Matrix, portfolio_scale, PORTFOLIO_VOL_INIT_DAYS,
};

fn constant_matrix(t: usize, values: &[f64]) -> Matrix {
    let rows: Vec<Vec<f64>> = (0..t).map(|_| values.to_vec()).collect();
    Matrix::from_rows(&rows).unwrap()
}

fn index_returns() -> Vec<f64> {
    regime::load_dataset("../data").expect("bundled data").index_returns
}

#[test]
fn momentum_rejects_short_series() {
    let r = constant_matrix(252, &[0.001, 0.001]); // exactly lookback long -> too short
    let err = momentum_positions(&r, 252, 0.10, 0.94, 4.0).unwrap_err();
    assert!(err.to_string().contains("shorter than momentum lookback"));
}

#[test]
fn momentum_sign_correctness() {
    // Steady up-trend asset goes long, down-trend short.
    let r = constant_matrix(300, &[0.001, -0.001]);
    let pos = momentum_positions(&r, 252, 0.10, 0.94, 4.0).unwrap();
    for t in 251..300 {
        assert!(pos.get(t, 0) > 0.0);
        assert!(pos.get(t, 1) < 0.0);
    }
    for t in 0..251 {
        assert_eq!(pos.get(t, 0), 0.0); // undefined before first full window
        assert_eq!(pos.get(t, 1), 0.0);
    }
}

#[test]
fn momentum_vol_targeting_caps_leverage() {
    // A near-zero-vol series would want huge leverage; the cap binds.
    let r = constant_matrix(300, &[1e-6]);
    let pos = momentum_positions(&r, 252, 0.10, 0.94, 4.0).unwrap();
    for t in 251..300 {
        assert!((pos.get(t, 0) - 4.0).abs() < 1e-9); // cap / 1 asset
    }
    // With meaningful vol the target is hit instead of the cap.
    let wave: Vec<f64> = (0..300).map(|t| 0.02 * (t as f64).sin()).collect();
    let pos2 = momentum_positions(&Matrix::column(&wave), 252, 0.10, 0.94, 4.0).unwrap();
    for t in 251..300 {
        assert!(pos2.get(t, 0).abs() < 4.0);
    }
}

#[test]
fn momentum_scaling_matches_formula() {
    // Position = sign * min(target/ann_vol, cap) / n_assets exactly.
    let r = constant_matrix(260, &[0.002, 0.003]);
    let pos = momentum_positions(&r, 252, 0.10, 0.94, 4.0).unwrap();
    let sig2 = ewma_variance(&r, 0.94, 252).unwrap();
    let t = 255;
    for a in 0..2 {
        let ann_vol = (252.0 * sig2.get(t, a)).sqrt();
        let expect = (0.10 / ann_vol).min(4.0) / 2.0;
        assert!((pos.get(t, a) - expect).abs() < 1e-12 * expect.abs().max(1.0));
    }
}

#[test]
fn ewma_variance_recursion() {
    let r = Matrix::column(&[0.01, -0.02, 0.005, 0.03]);
    let sig2 = ewma_variance(&r, 0.9, 2).unwrap();
    let seed = (0.01f64 * 0.01 + 0.02 * 0.02) / 2.0;
    assert!(sig2.get(0, 0).is_nan());
    assert!((sig2.get(1, 0) - seed).abs() < 1e-15);
    assert!((sig2.get(2, 0) - (0.9 * seed + 0.1 * 0.005 * 0.005)).abs() < 1e-15);
    assert!((sig2.get(3, 0) - (0.9 * sig2.get(2, 0) + 0.1 * 0.03 * 0.03)).abs() < 1e-15);
}

#[test]
fn carry_tie_break_by_asset_index() {
    // All-equal differentials: longs are assets 0..2, shorts 3..5 (pinned).
    let d = constant_matrix(10, &[0.02; 6]);
    let pos = carry_positions(&d, 3, 3, 21).unwrap();
    for a in 0..3 {
        assert!((pos.get(0, a) - 1.0 / 3.0).abs() < 1e-15);
    }
    for a in 3..6 {
        assert!((pos.get(0, a) + 1.0 / 3.0).abs() < 1e-15);
    }
}

#[test]
fn carry_ranking_and_weights() {
    let d = constant_matrix(5, &[0.01, 0.05, -0.02, 0.03, 0.00, 0.02]);
    let pos = carry_positions(&d, 3, 3, 21).unwrap();
    // descending diff order: 1, 3, 5, 0, 4, 2
    let expect = [-1.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0];
    let mut net = 0.0;
    let mut grossl = 0.0;
    for a in 0..6 {
        assert!((pos.get(0, a) - expect[a]).abs() < 1e-15);
        net += pos.get(0, a);
        grossl += pos.get(0, a).abs();
    }
    assert!(net.abs() < 1e-15);
    assert!((grossl - 2.0).abs() < 1e-15);
}

#[test]
fn carry_rebalance_schedule() {
    // Positions change only at t = 0, 21, 42, ... (pinned schedule).
    let mut rng = StdRng::seed_from_u64(7); // test data only; strategy code is RNG-free
    let normal = Normal::new(0.0, 0.02).unwrap();
    let rows: Vec<Vec<f64>> = (0..70).map(|_| (0..6).map(|_| normal.sample(&mut rng)).collect()).collect();
    let d = Matrix::from_rows(&rows).unwrap();
    let pos = carry_positions(&d, 3, 3, 21).unwrap();
    for t in 1..70 {
        let changed = (0..6).any(|a| pos.get(t, a) != pos.get(t - 1, a));
        if changed {
            assert_eq!(t % 21, 0, "position changed off-schedule at t={t}");
        }
    }
    for a in 0..6 {
        assert_eq!(pos.get(1, a), pos.get(0, a));
        assert_eq!(pos.get(41, a), pos.get(21, a));
    }
}

#[test]
fn carry_single_rebalance_when_window_exceeds_series() {
    // rebalance_days > T: only the day-0 rebalance happens, held throughout.
    let d = constant_matrix(10, &[0.03, 0.02, 0.01, 0.0, -0.01, -0.02]);
    let pos = carry_positions(&d, 3, 3, 500).unwrap();
    for t in 1..10 {
        for a in 0..6 {
            assert_eq!(pos.get(t, a), pos.get(0, a));
        }
    }
}

#[test]
fn carry_validation() {
    let d = constant_matrix(10, &[1.0; 4]);
    assert!(carry_positions(&d, 3, 3, 21).is_err()); // 6 > 4 assets
    assert!(carry_positions(&d, 1, 1, 0).is_err()); // rebalance_days = 0
    assert!(carry_positions(&Matrix::zeros(0, 4), 1, 1, 21).is_err()); // empty
}

#[test]
fn carry_total_returns_accrual() {
    let spot = Matrix::from_rows(&[vec![0.01, 0.0], vec![0.005, -0.01]]).unwrap();
    let diff = Matrix::from_rows(&[vec![0.0252, 0.0504], vec![0.0252, 0.0504]]).unwrap();
    let tot = carry_total_returns(&spot, &diff).unwrap();
    assert_eq!(tot.get(0, 0), 0.01); // day 0: no accrual
    assert_eq!(tot.get(0, 1), 0.0);
    assert!((tot.get(1, 0) - (0.005 + 0.0252 / 252.0)).abs() < 1e-15);
    assert!((tot.get(1, 1) - (-0.01 + 0.0504 / 252.0)).abs() < 1e-15);
    // shape mismatch
    assert!(carry_total_returns(&spot, &Matrix::zeros(3, 2)).is_err());
}

#[test]
fn regime_gate_basics() {
    // Gate is 1 before the first fit, within [0,1] after, deterministic.
    let r = index_returns();
    let r = &r[..700];
    let (gate, models) = regime_gate(r, 3, 400, 200, "prob", 0.5, 1e-8, 500, 1e-8).unwrap();
    for &g in &gate[..399] {
        assert_eq!(g, 1.0);
    }
    for &g in &gate {
        assert!((0.0..=1.0).contains(&g));
    }
    assert_eq!(models.len(), 2); // fits at t=399 and t=599
    let (gate2, _) = regime_gate(r, 3, 400, 200, "prob", 0.5, 1e-8, 500, 1e-8).unwrap();
    assert_eq!(gate, gate2);
}

#[test]
fn regime_gate_binary_mode() {
    let r = index_returns();
    let (gate, _) = regime_gate(&r[..600], 3, 400, 300, "binary", 0.5, 1e-8, 500, 1e-8).unwrap();
    for &g in &gate {
        assert!(g == 0.0 || g == 1.0);
    }
}

#[test]
fn regime_gate_validation() {
    let r = index_returns();
    let err = regime_gate(&r[..100], 3, 400, 63, "prob", 0.5, 1e-8, 500, 1e-8).unwrap_err();
    assert!(err.to_string().contains("shorter than train_min_days"));
    let err = regime_gate(&r[..600], 3, 400, 63, "magic", 0.5, 1e-8, 500, 1e-8).unwrap_err();
    assert!(err.to_string().contains("mode"));
    assert!(regime_gate(&r[..600], 3, 3, 63, "prob", 0.5, 1e-8, 500, 1e-8).is_err());
    assert!(regime_gate(&r[..600], 3, 400, 0, "prob", 0.5, 1e-8, 500, 1e-8).is_err());
}

#[test]
fn apply_gate_scales_positions() {
    let pos = Matrix::filled(5, 2, 1.0);
    let gate = [1.0, 0.5, 0.0, 0.25, 1.0];
    let out = apply_gate(&pos, &gate).unwrap();
    for t in 0..5 {
        assert_eq!(out.get(t, 0), gate[t]);
        assert_eq!(out.get(t, 1), gate[t]);
    }
    assert!(apply_gate(&pos, &gate[..3]).is_err());
    assert!(apply_gate(&pos, &[1.0, f64::NAN, 0.0, 0.0, 0.0]).is_err());
}

// ------------------------------------------------------------------------- //
// Robustness: corrupt returns, re-ranking, gate edges
// ------------------------------------------------------------------------- //

#[test]
fn momentum_rejects_returns_below_minus_one() {
    // PT-4: a -100% (or worse) print is corrupt data, not a signal.
    for bad in [-1.0, -1.5] {
        let mut r = constant_matrix(300, &[0.001]);
        r.set(10, 0, bad);
        let err = momentum_positions(&r, 252, 0.10, 0.94, 4.0).unwrap_err();
        assert!(err.to_string().contains("<= -100%"), "{err}");
        assert!(ewma_variance(&r, 0.94, 252).is_err());
        assert!(carry_total_returns(&r, &Matrix::zeros(300, 1)).is_err());
    }
    // -99.9% is legal (extreme, but a valid simple return).
    let mut r = constant_matrix(300, &[0.001]);
    r.set(10, 0, -0.999);
    assert!(momentum_positions(&r, 252, 0.10, 0.94, 4.0).unwrap().all_finite());
    // Rate differentials are not simple returns: -1.5 is accepted there.
    let d = Matrix::filled(10, 4, -1.5);
    assert_eq!(carry_positions(&d, 1, 1, 21).unwrap().rows(), 10);
}

#[test]
fn carry_rerank_after_crossing() {
    // PT-6: the book flips exactly at the first rebalance after the ranks
    // cross, with turnover 4 (two full round-trips) on that day only.
    let rows: Vec<Vec<f64>> = (0..45)
        .map(|t| if t < 21 { vec![0.04, 0.03, 0.02, 0.01] } else { vec![0.01, 0.02, 0.03, 0.04] })
        .collect();
    let d = Matrix::from_rows(&rows).unwrap();
    let pos = carry_positions(&d, 1, 1, 21).unwrap();
    assert_eq!(pos.row(20), &[1.0, 0.0, 0.0, -1.0]);
    for t in 21..42 {
        assert_eq!(pos.row(t), &[-1.0, 0.0, 0.0, 1.0]);
    }
    for t in 1..45 {
        let to: f64 = (0..4).map(|a| (pos.get(t, a) - pos.get(t - 1, a)).abs()).sum();
        assert_eq!(to, if t == 21 { 4.0 } else { 0.0 }, "turnover at t={t}");
    }
    // Tie at the crossing day -> ascending asset index decides.
    let tie_rows: Vec<Vec<f64>> = (0..45)
        .map(|t| if t < 21 { vec![0.04, 0.03, 0.02, 0.01] } else { vec![0.02; 4] })
        .collect();
    let pos_tie = carry_positions(&Matrix::from_rows(&tie_rows).unwrap(), 1, 1, 21).unwrap();
    assert_eq!(pos_tie.row(21), &[1.0, 0.0, 0.0, -1.0]);
}

#[test]
fn binary_gate_threshold_edges() {
    // PT-12: the '>=' convention at both ends of [0, 1]; outside is an error.
    let r = index_returns();
    let r = &r[..600];
    let (prob, _) = regime_gate(r, 3, 400, 300, "prob", 0.5, 1e-8, 500, 1e-8).unwrap();
    let (lo, _) = regime_gate(r, 3, 400, 300, "binary", 0.0, 1e-8, 500, 1e-8).unwrap();
    assert!(lo[399..].iter().all(|&g| g == 1.0));
    let (hi, _) = regime_gate(r, 3, 400, 300, "binary", 1.0, 1e-8, 500, 1e-8).unwrap();
    for t in 399..600 {
        assert_eq!(hi[t], if prob[t] == 1.0 { 1.0 } else { 0.0 });
    }
    for bad in [1.5, -0.1, f64::NAN] {
        let err = regime_gate(r, 3, 400, 300, "binary", bad, 1e-8, 500, 1e-8).unwrap_err();
        assert!(err.to_string().contains("threshold"), "{err}");
    }
}

#[test]
fn gate_edge_windows() {
    // PT-13: degenerate but legal walk-forward geometries.
    let r = index_returns();
    let (gate, models) = regime_gate(&r[..400], 3, 400, 63, "prob", 0.5, 1e-8, 500, 1e-8).unwrap();
    assert_eq!(gate.len(), 400);
    assert_eq!(models.len(), 1);
    assert!(gate[..399].iter().all(|&g| g == 1.0));
    assert!((0.0..=1.0).contains(&gate[399]));
    let (gate2, models2) = regime_gate(&r[..500], 3, 400, 1000, "prob", 0.5, 1e-8, 500, 1e-8).unwrap();
    assert_eq!(models2.len(), 1);
    assert!(gate2.iter().all(|g| (0.0..=1.0).contains(g)));
    let (gate3, models3) = regime_gate(&r[..4], 3, 4, 1, "prob", 0.5, 1e-8, 500, 1e-8).unwrap();
    assert_eq!(models3.len(), 1);
    assert!(gate3.iter().all(|g| g.is_finite()));
}

#[test]
fn regime_gate_parameter_validation() {
    // MIN-11: HMM settings are validated before any fit.
    let r = index_returns();
    let r = &r[..600];
    let err = regime_gate(r, 1, 400, 63, "prob", 0.5, 1e-8, 500, 1e-8).unwrap_err();
    assert!(err.to_string().contains("n_states"), "{err}");
    assert!(regime_gate(r, 3, 400, 63, "prob", 0.5, 0.0, 500, 1e-8).is_err());
    let mut nan = r.to_vec();
    nan.push(f64::NAN);
    assert!(regime_gate(&nan, 3, 400, 63, "prob", 0.5, 1e-8, 500, 1e-8).is_err());
}

// No-trade band and the regime_gate return domain.

#[test]
fn gate_band_recursion_by_hand() {
    let g = [1.0, 0.8, 0.6, 0.65, 0.2, 0.45, 0.51];
    assert_eq!(regime::apply_gate_band(&g, 0.3).unwrap(), vec![1.0, 1.0, 0.6, 0.6, 0.2, 0.2, 0.51]);
}

#[test]
fn gate_band_is_strict_at_the_edge() {
    assert_eq!(regime::apply_gate_band(&[0.5], 0.5).unwrap(), vec![1.0]);
    assert_eq!(regime::apply_gate_band(&[0.25], 0.5).unwrap(), vec![0.25]);
}

#[test]
fn gate_band_zero_is_identity_and_one_holds_the_start() {
    let g = [0.1, 0.9, 0.9, 0.0, 1.0];
    assert_eq!(regime::apply_gate_band(&g, 0.0).unwrap(), g.to_vec());
    assert_eq!(regime::apply_gate_band(&g, 1.0).unwrap(), vec![1.0; 5]);
}

fn gate_with(r: &[f64], min_days: usize, refit: usize, o: regime::GateOptions) -> regime::Result<Vec<f64>> {
    regime::regime_gate_with(r, 3, min_days, refit, "prob", 0.5, 1e-8, 500, 1e-8, o).map(|x| x.0)
}

#[test]
fn gate_band_range_is_checked() {
    for band in [-0.1, 1.1, f64::NAN] {
        assert!(regime::apply_gate_band(&[1.0], band).is_err());
        let o = regime::GateOptions { gate_band: band, transition_floor: 0.0, calm_guard: 0.0 };
        assert!(gate_with(&[0.0; 10], 5, 63, o).is_err());
    }
}

fn head_700() -> Vec<f64> {
    regime::load_dataset("../data").unwrap().index_returns[..700].to_vec()
}

#[test]
fn regime_gate_band_equals_post_processing() {
    let r = head_700();
    let raw = gate_with(&r, 504, 100, regime::GateOptions::default()).unwrap();
    let o = regime::GateOptions { gate_band: 0.3, transition_floor: 0.0, calm_guard: 0.0 };
    assert_eq!(gate_with(&r, 504, 100, o).unwrap(), regime::apply_gate_band(&raw, 0.3).unwrap());
}

#[test]
fn regime_gate_rejects_a_total_loss_return() {
    let mut r = vec![0.001; 20];
    r[7] = -1.0;
    let err = gate_with(&r, 10, 63, regime::GateOptions::default()).unwrap_err();
    assert!(err.to_string().contains("<= -100% at t=7"), "{err}");
}

#[test]
fn regime_gate_transition_floor_zero_is_bitwise_default() {
    let r = head_700();
    let plain = regime_gate(&r, 3, 504, 100, "prob", 0.5, 1e-8, 500, 1e-8).unwrap().0;
    assert_eq!(gate_with(&r, 504, 100, regime::GateOptions::default()).unwrap(), plain);
    let o = regime::GateOptions { gate_band: 0.0, transition_floor: 1e-6, calm_guard: 0.0 };
    assert!(gate_with(&r, 504, 100, o).unwrap().iter().all(|g| (0.0..=1.0).contains(g)));
}

// --- calm-state guard (API_SPEC 3.3) ---------------------------------------

#[test]
fn winsorize_by_hand() {
    let got = regime::winsorize_mad(&[0.0, 1.0, 2.0, 3.0, 100.0], 2.0);
    assert_eq!(got, vec![0.0, 1.0, 2.0, 3.0, 2.0 + 2.0 * 1.4826]);
}

#[test]
fn winsorize_zero_scale_is_unchanged() {
    let x = [1.0, 1.0, 1.0, 9.0];
    assert_eq!(regime::winsorize_mad(&x, regime::CALM_CLIP_MADS), x.to_vec());
}

#[test]
fn select_skips_thin_states_and_falls_back() {
    let v = [1.0, 2.0, 3.0];
    assert_eq!(regime::select_calm_state(&v, &[0.001, 0.7, 0.299], 0.01), 1);
    assert_eq!(regime::select_calm_state(&v, &[0.3, 0.3, 0.4], 0.5), 2);
}

#[test]
fn calm_guard_range_is_checked() {
    for bad in [-0.1, 1.0] {
        let options = regime::GateOptions { calm_guard: bad, ..Default::default() };
        let err = regime::regime_gate_report(&[0.0; 10], 3, 504, 63, "prob", 0.5, 1e-8, 500, 1e-8, options);
        assert!(err.unwrap_err().to_string().contains("calm_guard"));
    }
}

#[test]
fn portfolio_scale_formula_warm_up_and_cap() {
    let net: Vec<f64> = (0..60).map(|t| 0.002 + 0.0005 * (t % 3) as f64).collect();
    let scale = portfolio_scale(&net, 0.10, 0.94, 4.0, 5).unwrap();
    let seed_day = 5 + PORTFOLIO_VOL_INIT_DAYS - 1;
    assert!(scale[..seed_day].iter().all(|&s| s == 1.0));
    let mut sig2 = net[5..=seed_day].iter().map(|x| x * x).sum::<f64>() / 21.0;
    assert!((scale[seed_day] - (0.10 / (252.0 * sig2).sqrt()).min(4.0)).abs() < 1e-12);
    sig2 = 0.94 * sig2 + 0.06 * net[seed_day + 1] * net[seed_day + 1];
    assert!((scale[seed_day + 1] - (0.10 / (252.0 * sig2).sqrt()).min(4.0)).abs() < 1e-12);
    assert!(portfolio_scale(&vec![0.0; 40], 0.10, 0.94, 4.0, 0).unwrap()[39] == 4.0);
    assert!(portfolio_scale(&net[..10], 0.10, 0.94, 4.0, 5).unwrap().iter().all(|&s| s == 1.0));
    assert!(portfolio_scale(&net, 0.0, 0.94, 4.0, 5).is_err());
    assert!(portfolio_scale(&net, 0.10, 1.0, 4.0, 5).is_err());
    assert!(portfolio_scale(&net, 0.10, 0.94, 0.0, 5).is_err());
}

#[test]
fn portfolio_scale_ignores_the_future() {
    let net: Vec<f64> = (0..200).map(|t| 0.003 * ((t * 7 % 11) as f64 - 5.0) / 5.0).collect();
    let base = portfolio_scale(&net, 0.10, 0.94, 4.0, 20).unwrap();
    let mut changed = net.clone();
    for v in &mut changed[101..] {
        *v = -3.0 * *v + 0.01;
    }
    let other = portfolio_scale(&changed, 0.10, 0.94, 4.0, 20).unwrap();
    assert_eq!(base[..=100], other[..=100]);
    assert_ne!(base[101], other[101]);
}
