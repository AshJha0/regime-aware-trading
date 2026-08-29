//! HMM tests: enumeration ground truth, EM properties, edge cases.

use regime::{forward_step, GaussianHmm, HmmParams, Matrix, RegimeError};

/// Fixed tiny model for the enumerated K=2, T=3 case.
fn tiny_params() -> HmmParams {
    HmmParams {
        startprob: vec![0.6, 0.4],
        transmat: Matrix::from_vec(2, 2, vec![0.7, 0.3, 0.2, 0.8]).unwrap(),
        means: Matrix::from_vec(2, 1, vec![-1.0, 1.0]).unwrap(),
        variances: Matrix::from_vec(2, 1, vec![0.5, 2.0]).unwrap(),
    }
}

const TINY_X: [f64; 3] = [-0.5, 0.3, 1.2];

fn norm_pdf(x: f64, mu: f64, var: f64) -> f64 {
    (-0.5 * (x - mu) * (x - mu) / var).exp() / (2.0 * std::f64::consts::PI * var).sqrt()
}

fn path_prob(path: &[usize], x: &[f64], p: &HmmParams) -> f64 {
    let mut prob = p.startprob[path[0]] * norm_pdf(x[0], p.means.get(path[0], 0), p.variances.get(path[0], 0));
    for t in 1..x.len() {
        prob *= p.transmat.get(path[t - 1], path[t])
            * norm_pdf(x[t], p.means.get(path[t], 0), p.variances.get(path[t], 0));
    }
    prob
}

fn all_paths() -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                out.push(vec![a, b, c]);
            }
        }
    }
    out
}

fn tiny_model() -> GaussianHmm {
    let mut hmm = GaussianHmm::with_defaults(2).unwrap();
    hmm.set_params(tiny_params());
    hmm
}

fn index_returns() -> Vec<f64> {
    let ds = regime::load_dataset("../data").expect("bundled data");
    ds.index_returns
}

#[test]
fn forward_likelihood_matches_enumeration() {
    // Scaled forward log-likelihood == brute-force sum over all 8 paths.
    let hmm = tiny_model();
    let p = tiny_params();
    let brute: f64 = all_paths().iter().map(|path| path_prob(path, &TINY_X, &p)).sum();
    let ll = hmm.score(&Matrix::column(&TINY_X)).unwrap();
    assert!((ll.exp() - brute).abs() < 1e-12);
}

#[test]
fn smoothed_probs_match_enumeration() {
    let hmm = tiny_model();
    let p = tiny_params();
    let gamma = hmm.smoothed_probabilities(&Matrix::column(&TINY_X)).unwrap();
    let total: f64 = all_paths().iter().map(|path| path_prob(path, &TINY_X, &p)).sum();
    for t in 0..3 {
        for i in 0..2 {
            let marg: f64 = all_paths()
                .iter()
                .filter(|path| path[t] == i)
                .map(|path| path_prob(path, &TINY_X, &p))
                .sum();
            assert!((gamma.get(t, i) - marg / total).abs() < 1e-12);
        }
    }
}

#[test]
fn viterbi_is_argmax_path_on_enumerated_case() {
    let hmm = tiny_model();
    let p = tiny_params();
    let best = all_paths()
        .into_iter()
        .max_by(|a, b| {
            path_prob(a, &TINY_X, &p)
                .partial_cmp(&path_prob(b, &TINY_X, &p))
                .unwrap()
        })
        .unwrap();
    assert_eq!(hmm.viterbi(&Matrix::column(&TINY_X)).unwrap(), best);
}

#[test]
fn em_monotonic_on_bundled_data() {
    // Every Baum-Welch iteration is non-decreasing in log-likelihood.
    let r = Matrix::column(&index_returns());
    let mut hmm = GaussianHmm::with_defaults(3).unwrap();
    let res = hmm.fit(&r, None).unwrap();
    assert!(res.loglik_history.len() >= 3);
    for w in res.loglik_history.windows(2) {
        assert!(w[1] - w[0] >= -1e-9, "EM decreased: {} -> {}", w[0], w[1]);
    }
    assert!(res.converged);
}

#[test]
fn transitions_row_stochastic_and_labels_sorted() {
    let r = Matrix::column(&index_returns());
    for k in [2usize, 3] {
        let mut hmm = GaussianHmm::with_defaults(k).unwrap();
        hmm.fit(&r, None).unwrap();
        let p = hmm.params().unwrap();
        for i in 0..k {
            let row_sum: f64 = (0..k).map(|j| p.transmat.get(i, j)).sum();
            assert!((row_sum - 1.0).abs() < 1e-12);
            for j in 0..k {
                assert!(p.transmat.get(i, j) >= 0.0);
            }
        }
        // Pinned label convention: means ascending on dimension 0.
        for i in 1..k {
            assert!(p.means.get(i, 0) >= p.means.get(i - 1, 0));
        }
    }
}

#[test]
fn stationary_is_left_eigenvector() {
    let r = Matrix::column(&index_returns());
    let mut hmm = GaussianHmm::with_defaults(3).unwrap();
    hmm.fit(&r, None).unwrap();
    let pi = hmm.stationary_distribution().unwrap();
    let p = hmm.params().unwrap();
    let total: f64 = pi.iter().sum();
    assert!((total - 1.0).abs() < 1e-12);
    for j in 0..3 {
        let piaj: f64 = (0..3).map(|i| pi[i] * p.transmat.get(i, j)).sum();
        assert!((piaj - pi[j]).abs() < 1e-10, "pi P != pi at column {j}");
    }
}

#[test]
fn fit_is_deterministic() {
    // Two fits on identical data produce bitwise-identical parameters.
    let r = index_returns();
    let x = Matrix::column(&r[..600]);
    let mut a = GaussianHmm::with_defaults(3).unwrap();
    let ra = a.fit(&x, None).unwrap();
    let mut b = GaussianHmm::with_defaults(3).unwrap();
    let rb = b.fit(&x, None).unwrap();
    assert_eq!(ra.log_likelihood, rb.log_likelihood);
    assert_eq!(a.params().unwrap().means, b.params().unwrap().means);
    assert_eq!(a.params().unwrap().transmat, b.params().unwrap().transmat);
    assert_eq!(a.params().unwrap().variances, b.params().unwrap().variances);
}

#[test]
fn k1_is_degenerate_error() {
    assert!(matches!(GaussianHmm::with_defaults(1), Err(RegimeError::InvalidInput(_))));
}

#[test]
fn variance_floor_engages() {
    // K exceeding the data's distinct support hits the variance floor.
    let mut data = vec![0.0; 30];
    data.extend(vec![1.0; 30]);
    let x = Matrix::column(&data);
    let mut hmm = GaussianHmm::new(3, 1e-8, 200, 1e-8).unwrap();
    let res = hmm.fit(&x, None).unwrap(); // must not crash or produce zero variance
    let p = hmm.params().unwrap();
    let mut floor_hit = false;
    for i in 0..3 {
        assert!(p.variances.get(i, 0) >= 1e-8);
        if (p.variances.get(i, 0) - 1e-8).abs() < 1e-12 {
            floor_hit = true;
        }
    }
    assert!(floor_hit, "variance floor never engaged");
    assert!(res.log_likelihood.is_finite());
}

#[test]
fn nonconvergence_is_flag_not_error() {
    let r = index_returns();
    let x = Matrix::column(&r[..400]);
    let mut hmm = GaussianHmm::new(3, 1e-8, 3, 1e-8).unwrap();
    let res = hmm.fit(&x, None).unwrap();
    assert!(!res.converged);
    assert_eq!(res.n_iter, 3);
    assert!(res.log_likelihood.is_finite());
}

#[test]
fn probability_outputs_are_valid() {
    let r = index_returns();
    let x = Matrix::column(&r[..600]);
    let mut hmm = GaussianHmm::with_defaults(3).unwrap();
    hmm.fit(&x, None).unwrap();
    let gamma = hmm.smoothed_probabilities(&x).unwrap();
    for t in 0..gamma.rows() {
        let s: f64 = gamma.row(t).iter().sum();
        assert!((s - 1.0).abs() < 1e-10);
        for &v in gamma.row(t) {
            assert!((-1e-12..=1.0 + 1e-12).contains(&v));
        }
    }
    let alpha = hmm.filtered_probabilities(&Matrix::column(&r[..200])).unwrap();
    for t in 0..alpha.rows() {
        let s: f64 = alpha.row(t).iter().sum();
        assert!((s - 1.0).abs() < 1e-10);
    }
}

#[test]
fn forward_step_matches_full_filter() {
    // Incremental filtering (used by the gate) equals the batch forward.
    let r = index_returns();
    let mut hmm = GaussianHmm::with_defaults(3).unwrap();
    hmm.fit(&Matrix::column(&r), None).unwrap();
    let x = &r[..150];
    let full = hmm.filtered_probabilities(&Matrix::column(x)).unwrap();
    let head = hmm.filtered_probabilities(&Matrix::column(&x[..100])).unwrap();
    let mut alpha = head.row(99).to_vec();
    for &xt in x.iter().take(150).skip(100) {
        alpha = forward_step(hmm.params().unwrap(), &alpha, &[xt]);
    }
    for i in 0..3 {
        assert!((alpha[i] - full.get(149, i)).abs() < 1e-12);
    }
}

#[test]
fn two_dimensional_observations() {
    // 2-D diagonal-covariance fit runs; labels sort on dimension 0.
    let r = index_returns();
    let rows: Vec<Vec<f64>> = r[..500].iter().map(|&v| vec![v, v.abs()]).collect();
    let x = Matrix::from_rows(&rows).unwrap();
    let mut hmm = GaussianHmm::with_defaults(2).unwrap();
    let res = hmm.fit(&x, None).unwrap();
    let p = hmm.params().unwrap();
    assert_eq!((p.means.rows(), p.means.cols()), (2, 2));
    assert!(p.means.get(0, 0) <= p.means.get(1, 0));
    assert!(res.log_likelihood.is_finite());
    let gamma = hmm.smoothed_probabilities(&x).unwrap();
    assert_eq!((gamma.rows(), gamma.cols()), (500, 2));
}

#[test]
fn model_selection_prefers_three_states() {
    // Data come from a true 3-state DGP: K=3 wins on AIC and BIC.
    let r = Matrix::column(&index_returns());
    let mut hmm2 = GaussianHmm::with_defaults(2).unwrap();
    let fit2 = hmm2.fit(&r, None).unwrap();
    let mut hmm3 = GaussianHmm::with_defaults(3).unwrap();
    let fit3 = hmm3.fit(&r, None).unwrap();
    assert!(fit3.log_likelihood > fit2.log_likelihood);
    assert!(hmm3.aic(&r).unwrap() < hmm2.aic(&r).unwrap());
    assert!(hmm3.bic(&r).unwrap() < hmm2.bic(&r).unwrap());
}

#[test]
fn input_validation_errors() {
    let mut hmm = GaussianHmm::with_defaults(2).unwrap();
    assert!(hmm.fit(&Matrix::column(&[]), None).is_err()); // empty
    assert!(hmm.fit(&Matrix::column(&[1.0, f64::NAN, 2.0]), None).is_err()); // NaN
    assert!(hmm.fit(&Matrix::filled(10, 3, 1.0), None).is_err()); // D=3 unsupported
    assert!(hmm.fit(&Matrix::column(&[1.0, 2.0]), None).is_err()); // T <= K
    assert!(hmm.score(&Matrix::column(&[1.0])).is_err()); // not fitted
    assert!(GaussianHmm::new(2, 0.0, 500, 1e-8).is_err()); // tol <= 0
    assert!(GaussianHmm::new(2, 1e-8, 0, 1e-8).is_err()); // max_iter < 1
    assert!(GaussianHmm::new(2, 1e-8, 500, 0.0).is_err()); // var_floor <= 0
}

#[test]
fn grid_property_fitted_params_beat_perturbations() {
    // Property-style grid check: the fitted parameters score at least as
    // well as every mean-shifted perturbation of themselves.
    let r = index_returns();
    let x = Matrix::column(&r[..400]);
    let mut hmm = GaussianHmm::with_defaults(2).unwrap();
    let res = hmm.fit(&x, None).unwrap();
    let fitted = hmm.params().unwrap().clone();
    for step in 0..7 {
        let shift = -0.01 + 0.02 * step as f64 / 6.0;
        let mut p = fitted.clone();
        for i in 0..2 {
            p.means.set(i, 0, p.means.get(i, 0) + shift);
        }
        let mut probe = GaussianHmm::with_defaults(2).unwrap();
        probe.set_params(p);
        assert!(probe.score(&x).unwrap() <= res.log_likelihood + 1e-9);
    }
}

#[test]
fn pinned_init_matches_spec() {
    // Quantile means, pooled floored variance, 0.9 self-transitions, uniform pi.
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let hmm = GaussianHmm::with_defaults(2).unwrap();
    let p = hmm.pinned_init(&Matrix::column(&data)).unwrap();
    // q = 0.25 and 0.75 on [1..5]: h = 1.0 -> 2.0 and h = 3.0 -> 4.0.
    assert!((p.means.get(0, 0) - 2.0).abs() < 1e-15);
    assert!((p.means.get(1, 0) - 4.0).abs() < 1e-15);
    let var = 2.0; // population variance of 1..5
    assert!((p.variances.get(0, 0) - var).abs() < 1e-15);
    assert!((p.variances.get(1, 0) - var).abs() < 1e-15);
    assert!((p.transmat.get(0, 0) - 0.9).abs() < 1e-15);
    assert!((p.transmat.get(0, 1) - 0.1).abs() < 1e-15);
    assert!((p.startprob[0] - 0.5).abs() < 1e-15);
}
