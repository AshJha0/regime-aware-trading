"""HMM tests: enumeration ground truth, EM properties, edge cases."""

import itertools

import numpy as np
import pytest

from regime import GaussianHMM, HMMParams, forward_step

# Fixed tiny model for the enumerated K=2, T=3 case.
TINY = HMMParams(
    startprob=np.array([0.6, 0.4]),
    transmat=np.array([[0.7, 0.3], [0.2, 0.8]]),
    means=np.array([[-1.0], [1.0]]),
    variances=np.array([[0.5], [2.0]]),
)
TINY_X = np.array([-0.5, 0.3, 1.2])


def _norm_pdf(x: float, mu: float, var: float) -> float:
    return float(np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2.0 * np.pi * var))


def _path_prob(path, x, p: HMMParams) -> float:
    prob = p.startprob[path[0]] * _norm_pdf(x[0], p.means[path[0], 0], p.variances[path[0], 0])
    for t in range(1, len(x)):
        prob *= p.transmat[path[t - 1], path[t]] * _norm_pdf(
            x[t], p.means[path[t], 0], p.variances[path[t], 0]
        )
    return prob


def _tiny_model() -> GaussianHMM:
    hmm = GaussianHMM(2)
    hmm.params = TINY.copy()
    return hmm


def test_forward_likelihood_matches_enumeration():
    """Scaled forward log-likelihood == brute-force sum over all 8 paths."""
    hmm = _tiny_model()
    brute = sum(_path_prob(p, TINY_X, TINY) for p in itertools.product([0, 1], repeat=3))
    assert abs(np.exp(hmm.score(TINY_X)) - brute) < 1e-12


def test_smoothed_probs_match_enumeration():
    """Smoothed posteriors equal enumerated path-probability marginals."""
    hmm = _tiny_model()
    gamma = hmm.smoothed_probabilities(TINY_X)
    total = sum(_path_prob(p, TINY_X, TINY) for p in itertools.product([0, 1], repeat=3))
    for t in range(3):
        for i in range(2):
            marg = sum(
                _path_prob(p, TINY_X, TINY)
                for p in itertools.product([0, 1], repeat=3)
                if p[t] == i
            )
            assert abs(gamma[t, i] - marg / total) < 1e-12


def test_viterbi_is_argmax_path_on_enumerated_case():
    """Viterbi output equals the exhaustive-argmax state path."""
    hmm = _tiny_model()
    best = max(itertools.product([0, 1], repeat=3), key=lambda p: _path_prob(p, TINY_X, TINY))
    assert tuple(hmm.viterbi(TINY_X)) == best


def test_em_monotonic_on_bundled_data(pipeline_full):
    """Every Baum-Welch iteration is non-decreasing in log-likelihood."""
    hist = np.array(pipeline_full["fit3"].loglik_history)
    assert len(hist) >= 3
    assert np.all(np.diff(hist) >= -1e-9)
    assert pipeline_full["fit3"].converged


def test_transitions_row_stochastic(pipeline_full):
    for key in ("hmm2", "hmm3"):
        A = pipeline_full[key].params.transmat
        assert np.all(A >= 0.0)
        assert np.allclose(A.sum(axis=1), 1.0, atol=1e-12)


def test_state_label_ordering(pipeline_full):
    """Pinned convention: means ascending, state 0 = lowest mean."""
    for key in ("hmm2", "hmm3"):
        means = pipeline_full[key].params.means[:, 0]
        assert np.all(np.diff(means) >= 0.0)


def test_stationary_is_left_eigenvector(pipeline_full):
    hmm = pipeline_full["hmm3"]
    pi = hmm.stationary_distribution()
    assert abs(pi.sum() - 1.0) < 1e-12
    assert np.max(np.abs(pi @ hmm.params.transmat - pi)) < 1e-10
    # cross-check against the exact eigenvector.
    w, v = np.linalg.eig(hmm.params.transmat.T)
    ev = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    ev = ev / ev.sum()
    assert np.max(np.abs(pi - ev)) < 1e-8


def test_fit_is_deterministic(index_returns):
    """Two fits on identical data produce bitwise-identical parameters."""
    x = index_returns[:600]
    a = GaussianHMM(3)
    ra = a.fit(x)
    b = GaussianHMM(3)
    rb = b.fit(x)
    assert ra.log_likelihood == rb.log_likelihood
    assert np.array_equal(a.params.means, b.params.means)
    assert np.array_equal(a.params.transmat, b.params.transmat)
    assert np.array_equal(a.params.variances, b.params.variances)


def test_k1_is_degenerate_error():
    with pytest.raises(ValueError):
        GaussianHMM(1)


def test_variance_floor_engages():
    """K exceeding the data's distinct support hits the variance floor."""
    x = np.array([0.0] * 30 + [1.0] * 30)  # only two distinct values
    hmm = GaussianHMM(3, var_floor=1e-8, max_iter=200)
    res = hmm.fit(x)  # must not crash or produce zero variance
    assert np.all(hmm.params.variances >= 1e-8)
    assert np.any(np.isclose(hmm.params.variances, 1e-8))
    assert np.isfinite(res.log_likelihood)


def test_nonconvergence_is_flag_not_exception(index_returns):
    hmm = GaussianHMM(3, max_iter=3)
    res = hmm.fit(index_returns[:400])
    assert res.converged is False
    assert res.n_iter == 3
    assert np.isfinite(res.log_likelihood)


def test_probability_outputs_are_valid(pipeline_full, index_returns):
    hmm = pipeline_full["hmm3"]
    gamma = pipeline_full["smoothed3"]
    assert np.allclose(gamma.sum(axis=1), 1.0, atol=1e-10)
    assert np.all((gamma >= 0.0) & (gamma <= 1.0 + 1e-12))
    alpha = hmm.filtered_probabilities(index_returns[:200])
    assert np.allclose(alpha.sum(axis=1), 1.0, atol=1e-10)


def test_forward_step_matches_full_filter(pipeline_full, index_returns):
    """Incremental filtering (used by the gate) equals the batch forward."""
    hmm = pipeline_full["hmm3"]
    x = index_returns[:150]
    full = hmm.filtered_probabilities(x)
    alpha = hmm.filtered_probabilities(x[:100])[-1]
    for t in range(100, 150):
        alpha = forward_step(hmm.params, alpha, x[t])
    assert np.max(np.abs(alpha - full[-1])) < 1e-12


def test_two_dimensional_observations(index_returns):
    """2-D diagonal-covariance fit runs; labels sort on dimension 0."""
    x2 = np.column_stack([index_returns[:500], np.abs(index_returns[:500])])
    hmm = GaussianHMM(2)
    res = hmm.fit(x2)
    assert hmm.params.means.shape == (2, 2)
    assert hmm.params.means[0, 0] <= hmm.params.means[1, 0]
    assert np.isfinite(res.log_likelihood)
    assert hmm.smoothed_probabilities(x2).shape == (500, 2)


def test_model_selection_prefers_three_states(pipeline_full, index_returns):
    """Data come from a true 3-state DGP: K=3 wins on AIC and BIC."""
    hmm2, hmm3 = pipeline_full["hmm2"], pipeline_full["hmm3"]
    assert pipeline_full["fit3"].log_likelihood > pipeline_full["fit2"].log_likelihood
    assert hmm3.aic(index_returns) < hmm2.aic(index_returns)
    assert hmm3.bic(index_returns) < hmm2.bic(index_returns)


def test_input_validation_errors():
    hmm = GaussianHMM(2)
    with pytest.raises(ValueError):
        hmm.fit(np.array([]))  # empty
    with pytest.raises(ValueError):
        hmm.fit(np.array([1.0, np.nan, 2.0]))  # NaN
    with pytest.raises(ValueError):
        hmm.fit(np.ones((10, 3)))  # D=3 unsupported
    with pytest.raises(ValueError):
        hmm.fit(np.array([1.0, 2.0]))  # T <= K
    with pytest.raises(ValueError):
        hmm.score(np.array([1.0]))  # not fitted
    with pytest.raises(ValueError):
        GaussianHMM(2, tol=0.0)


def test_grid_property_loglik_increases_with_better_params(index_returns):
    """Property-style grid check: the fitted parameters score at least as
    well as every pinned-init perturbation on a mean grid."""
    x = index_returns[:400]
    hmm = GaussianHMM(2)
    res = hmm.fit(x)
    fitted = hmm.params
    for shift in np.linspace(-0.01, 0.01, 7):
        probe = GaussianHMM(2)
        p = fitted.copy()
        p.means = p.means + shift
        probe.params = p
        assert probe.score(x) <= res.log_likelihood + 1e-9


def test_hmmlearn_cross_check(index_returns):
    """Score the same parameters with hmmlearn; log-likelihoods must agree."""
    hmmlearn = pytest.importorskip("hmmlearn.hmm")
    x = index_returns[:800]
    ours = GaussianHMM(3)
    ours.fit(x)
    ref = hmmlearn.GaussianHMM(n_components=3, covariance_type="diag", init_params="")
    ref.startprob_ = ours.params.startprob.copy()
    # hmmlearn requires strictly positive start probabilities; EM can drive
    # some to ~0, so blend a hair of uniform mass into BOTH models' copies
    # and compare scores of the identical blended parameter set.
    eps = 1e-12
    sp = (ours.params.startprob + eps) / (1.0 + 3 * eps)
    ref.startprob_ = sp
    ref.transmat_ = ours.params.transmat.copy()
    ref.means_ = ours.params.means.copy()
    ref.covars_ = ours.params.variances.copy()
    blended = ours.params.copy()
    blended.startprob = sp
    mine = GaussianHMM(3)
    mine.params = blended
    assert abs(ref.score(x.reshape(-1, 1)) - mine.score(x)) < 1e-6
