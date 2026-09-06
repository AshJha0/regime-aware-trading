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


# --------------------------------------------------------------------- #
# Robustness: dead states, shape validation, monotonicity, zero transitions
# --------------------------------------------------------------------- #

def _dead_state_setup():
    """300 low-vol points and a warm start whose last state sits 5.0 away
    with variance 1e-8: that state has zero posterior support everywhere."""
    x = 0.01 * np.sin(np.arange(300))
    hmm = GaussianHMM(3)
    init = hmm.pinned_init(x)
    init.means[2, 0] = 5.0
    init.variances[2, 0] = 1e-8
    return x, hmm, init


def test_dead_state_is_flagged_not_nan():
    """PT-1: a state with no posterior support stops EM with a clear flag;
    every returned number is finite and the log-likelihood matches the
    returned parameter set exactly."""
    x, hmm, init = _dead_state_setup()
    res = hmm.fit(x, init=init)
    assert res.dead_state == 2
    assert res.converged is False
    assert res.monotone is True
    assert res.n_iter == 1  # stopped after the first E-step (init scored)
    assert np.isfinite(res.log_likelihood)
    assert all(np.isfinite(v) for v in res.loglik_history)
    p = hmm.params
    assert np.all(np.isfinite(p.means)) and np.all(np.isfinite(p.variances))
    assert np.all(np.isfinite(p.transmat)) and np.all(np.isfinite(p.startprob))
    # The returned parameters are the just-scored ones: re-scoring reproduces ll.
    assert hmm.score(x) == pytest.approx(res.log_likelihood, abs=1e-9)
    # A healthy fit on the same data reports no dead state.
    ok = GaussianHMM(3)
    assert ok.fit(x).dead_state is None


def test_forward_step_zero_likelihood_is_error():
    """PT-1b: all mass on a state whose row of A only reaches states under
    which the observation underflows -> error, never NaN."""
    p = HMMParams(
        startprob=np.array([0.5, 0.5]),
        transmat=np.array([[0.0, 1.0], [0.0, 1.0]]),
        means=np.array([[0.0], [100.0]]),
        variances=np.array([[1e-8], [1e-8]]),
    )
    with pytest.raises(ValueError, match="zero likelihood"):
        forward_step(p, np.array([1.0, 0.0]), np.array([0.0]))
    hmm = GaussianHMM(2)
    hmm.params = p
    with pytest.raises(ValueError, match="zero"):
        hmm.score(np.array([0.0, 0.0]))


def test_score_rejects_dimension_mismatch(index_returns):
    """PT-2: a D=1 model must refuse (T, 2) data in every inference call."""
    hmm = GaussianHMM(2)
    x = index_returns[:300]
    hmm.fit(x)
    x2 = np.column_stack([x, x])
    for fn in (hmm.score, hmm.filtered_probabilities, hmm.smoothed_probabilities, hmm.viterbi, hmm.aic, hmm.bic):
        with pytest.raises(ValueError, match="does not match"):
            fn(x2)
    # sanity: the D=1 call still works and equals the fit score
    assert hmm.score(x) == pytest.approx(hmm.fit_result.log_likelihood)


def test_warm_start_shape_validation(index_returns):
    """PT-3: every inconsistent warm start is a ValueError, never a numpy error."""
    x = index_returns[:300]
    good = GaussianHMM(2).pinned_init(x)
    bad_k = GaussianHMM(3)
    with pytest.raises(ValueError, match="n_states"):
        bad_k.fit(x, init=good)  # K=2 params on a K=3 model
    hmm = GaussianHMM(2)
    init_d2 = GaussianHMM(2).pinned_init(np.column_stack([x, x]))
    with pytest.raises(ValueError, match="does not match"):
        hmm.fit(x, init=init_d2)  # D=2 params on D=1 data
    neg = good.copy()
    neg.variances[0, 0] = -1.0
    with pytest.raises(ValueError, match="variances"):
        hmm.fit(x, init=neg)
    nonstoch = good.copy()
    nonstoch.transmat[0] = [0.5, 0.6]
    with pytest.raises(ValueError, match="transmat"):
        hmm.fit(x, init=nonstoch)
    badpi = good.copy()
    badpi.startprob = np.array([0.7, 0.7])
    with pytest.raises(ValueError, match="startprob"):
        hmm.fit(x, init=badpi)
    nan = good.copy()
    nan.means[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        hmm.fit(x, init=nan)
    # forward_step length checks
    with pytest.raises(ValueError, match="alpha_prev"):
        forward_step(good, np.array([1.0, 0.0, 0.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="observation length"):
        forward_step(good, np.array([0.5, 0.5]), np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="alpha_prev"):
        forward_step(good, np.array([0.7, 0.7]), np.array([0.0]))


def test_pinned_init_matches_spec():
    """PT-10: quantile means, pooled floored variance, 0.9 self-transitions."""
    p = GaussianHMM(2).pinned_init(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    # q = 0.25, 0.75 on [1..5]: h = 1.0 -> 2.0, h = 3.0 -> 4.0
    assert p.means[:, 0] == pytest.approx([2.0, 4.0], abs=1e-15)
    assert p.variances[:, 0] == pytest.approx([2.0, 2.0], abs=1e-15)  # population var
    assert p.transmat == pytest.approx(np.array([[0.9, 0.1], [0.1, 0.9]]), abs=1e-15)
    assert p.startprob == pytest.approx([0.5, 0.5], abs=1e-15)
    # K=3 on 30 zeros + 30 ones: quantiles 1/6, 1/2, 5/6 -> 0, 0.5, 1 (h = 29.5 interpolates)
    x = np.array([0.0] * 30 + [1.0] * 30)
    p3 = GaussianHMM(3).pinned_init(x)
    assert p3.means[:, 0] == pytest.approx([0.0, 0.5, 1.0], abs=1e-15)
    assert p3.variances[:, 0] == pytest.approx([0.25] * 3, abs=1e-15)
    assert p3.transmat[0] == pytest.approx([0.9, 0.05, 0.05], abs=1e-15)


def test_viterbi_respects_zero_transitions():
    """PT-14: A[0,1] = 0 means state 0 can never leave; the path must start
    in state 1 despite the first observation favouring state 0."""
    p = HMMParams(
        startprob=np.array([0.5, 0.5]),
        transmat=np.array([[1.0, 0.0], [0.5, 0.5]]),
        means=np.array([[-1.0], [1.0]]),
        variances=np.array([[0.1], [0.1]]),
    )
    x = np.array([-1.0, 1.0, 1.0])
    hmm = GaussianHMM(2)
    hmm.params = p
    path = hmm.viterbi(x)
    assert path.tolist() == [1, 1, 1]
    # cross-check against enumeration in log-space (log 0 = -inf never wins)
    with np.errstate(divide="ignore"):
        best = max(
            itertools.product([0, 1], repeat=3),
            key=lambda s: np.log(p.startprob[s[0]])
            + sum(np.log(p.transmat[s[t - 1], s[t]]) for t in range(1, 3))
            + sum(np.log(_norm_pdf(x[t], p.means[s[t], 0], p.variances[s[t], 0])) for t in range(3)),
        )
    assert tuple(path) == best


def test_em_step_status_classification():
    """PT-15: the pinned (ll, ll_prev) -> {continue, converged, non-monotone} rule."""
    from regime import EMStep, MONOTONE_REL_TOL, em_step_status

    assert em_step_status(10.0, 9.0, 1e-8) is EMStep.CONTINUE
    assert em_step_status(10.0, 10.0 - 5e-9, 1e-8) is EMStep.CONVERGED
    assert em_step_status(10.0, 10.0, 1e-8) is EMStep.CONVERGED  # zero improvement
    # a decrease smaller than the relative tolerance still counts as converged...
    assert em_step_status(1000.0 - 1e-5, 1000.0, 1e-8) is EMStep.CONVERGED
    # ...a decrease beyond it is a monotonicity violation
    assert em_step_status(1000.0 - 2.0 * MONOTONE_REL_TOL * 1000.0, 1000.0, 1e-8) is EMStep.NON_MONOTONE
    assert em_step_status(-2.0, -1.0, 1e-8) is EMStep.NON_MONOTONE  # |ll_prev| < 1 uses max(1, .)
    assert em_step_status(float("nan"), 1.0, 1e-8) is EMStep.NON_MONOTONE
    assert em_step_status(1.0, -np.inf, 1e-8) is EMStep.NON_MONOTONE


def test_em_convergence_flags_decrease(index_returns, monkeypatch):
    """PT-15b: a scorer that decreases must be reported as converged=False,
    monotone=False — never as convergence."""
    x = index_returns[:300]
    hmm = GaussianHMM(2, max_iter=50)
    real_forward = GaussianHMM._forward
    calls = {"n": 0}

    def decreasing(self, logb, params):
        alpha, d, m = real_forward(self, logb, params)
        calls["n"] += 1
        # second E-step: scale the normalizers so the score drops by 1000 nats
        # (far more than any genuine EM improvement on 300 points)
        if calls["n"] == 2:
            d = d * np.exp(-1000.0 / len(d))
        return alpha, d, m

    monkeypatch.setattr(GaussianHMM, "_forward", decreasing)
    res = hmm.fit(x)
    assert res.monotone is False
    assert res.converged is False
    assert res.dead_state is None
    assert res.n_iter == 2
    assert res.loglik_history[1] < res.loglik_history[0]


def test_n_iter_counts_history_entries(index_returns):
    """MIN-7: n_iter == len(history); the post-exhaustion re-score is not counted."""
    hmm = GaussianHMM(3, max_iter=3)
    res = hmm.fit(index_returns[:400])
    assert res.n_iter == 3 == len(res.loglik_history)
    assert res.monotone is True and res.dead_state is None
    assert res.log_likelihood >= res.loglik_history[-1] - 1e-9


def test_stationary_distribution_reports_nonconvergence():
    """MIN-10: a chain whose power-method iterate cycles never converges -> error."""
    hmm = GaussianHMM(3)
    # Cyclic permutation with one split row: from the uniform start the
    # iterate oscillates with period 3 and never meets the 1e-13 tolerance.
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.0]])
    hmm.params = HMMParams(np.array([1.0, 0.0, 0.0]), A, np.zeros((3, 1)), np.ones((3, 1)))
    with pytest.raises(ValueError, match="did not converge"):
        hmm.stationary_distribution(max_iter=50)
    with pytest.raises(ValueError):
        hmm.stationary_distribution(tol=0.0)
    with pytest.raises(ValueError):
        hmm.stationary_distribution(max_iter=0)
    # an aperiodic chain converges and satisfies pi A = pi
    hmm.params.transmat = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    pi = hmm.stationary_distribution()
    assert pi == pytest.approx([1 / 3] * 3, abs=1e-10)
