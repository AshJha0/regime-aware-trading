//! Strategy tests: momentum sizing, carry schedule/tie-break, regime gate.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

use regime::{
    apply_gate, carry_positions, carry_total_returns, ewma_variance, momentum_positions,
    regime_gate, Matrix,
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
