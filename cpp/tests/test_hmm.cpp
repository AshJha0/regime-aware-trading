/// \file test_hmm.cpp
/// \brief HMM tests: enumeration ground truth, EM properties, edge cases.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "regime/hmm.hpp"
#include "test_common.hpp"

using regime::GaussianHMM;
using regime::HMMParams;
using regime::Matrix;
using regime_tests::full_pipeline;
using regime_tests::index_head;

namespace {

/// Fixed tiny model for the enumerated K=2, T=3 case.
HMMParams tiny_params() {
    HMMParams p;
    p.startprob = {0.6, 0.4};
    p.transmat = Matrix(2, 2);
    p.transmat(0, 0) = 0.7;
    p.transmat(0, 1) = 0.3;
    p.transmat(1, 0) = 0.2;
    p.transmat(1, 1) = 0.8;
    p.means = Matrix(2, 1);
    p.means(0, 0) = -1.0;
    p.means(1, 0) = 1.0;
    p.variances = Matrix(2, 1);
    p.variances(0, 0) = 0.5;
    p.variances(1, 0) = 2.0;
    return p;
}

const std::vector<double> kTinyX = {-0.5, 0.3, 1.2};

constexpr double kPi = 3.14159265358979323846;  // M_PI is not standard C++

double norm_pdf(double x, double mu, double var) {
    return std::exp(-0.5 * (x - mu) * (x - mu) / var) / std::sqrt(2.0 * kPi * var);
}

/// Probability of one complete state path under the tiny model.
double path_prob(const std::vector<int>& path, const std::vector<double>& x,
                 const HMMParams& p) {
    const std::size_t s0 = static_cast<std::size_t>(path[0]);
    double prob = p.startprob[s0] * norm_pdf(x[0], p.means(s0, 0), p.variances(s0, 0));
    for (std::size_t t = 1; t < x.size(); ++t) {
        const std::size_t a = static_cast<std::size_t>(path[t - 1]);
        const std::size_t b = static_cast<std::size_t>(path[t]);
        prob *= p.transmat(a, b) * norm_pdf(x[t], p.means(b, 0), p.variances(b, 0));
    }
    return prob;
}

/// All 8 binary state paths of length 3.
std::vector<std::vector<int>> all_paths() {
    std::vector<std::vector<int>> out;
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
            for (int c = 0; c < 2; ++c) out.push_back({a, b, c});
    return out;
}

GaussianHMM tiny_model() {
    GaussianHMM hmm(2);
    hmm.set_params(tiny_params());
    return hmm;
}

}  // namespace

TEST(HmmTiny, ForwardLikelihoodMatchesEnumeration) {
    GaussianHMM hmm = tiny_model();
    double brute = 0.0;
    for (const auto& p : all_paths()) brute += path_prob(p, kTinyX, tiny_params());
    EXPECT_NEAR(std::exp(hmm.score(regime::to_matrix(kTinyX))), brute, 1e-12);
}

TEST(HmmTiny, SmoothedProbsMatchEnumeration) {
    GaussianHMM hmm = tiny_model();
    Matrix gamma = hmm.smoothed_probabilities(regime::to_matrix(kTinyX));
    double total = 0.0;
    for (const auto& p : all_paths()) total += path_prob(p, kTinyX, tiny_params());
    for (std::size_t t = 0; t < 3; ++t)
        for (int i = 0; i < 2; ++i) {
            double marg = 0.0;
            for (const auto& p : all_paths())
                if (p[t] == i) marg += path_prob(p, kTinyX, tiny_params());
            EXPECT_NEAR(gamma(t, static_cast<std::size_t>(i)), marg / total, 1e-12);
        }
}

TEST(HmmTiny, ViterbiIsArgmaxPathOnEnumeratedCase) {
    GaussianHMM hmm = tiny_model();
    std::vector<int> best;
    double best_prob = -1.0;
    for (const auto& p : all_paths()) {
        const double prob = path_prob(p, kTinyX, tiny_params());
        if (prob > best_prob) {
            best_prob = prob;
            best = p;
        }
    }
    EXPECT_EQ(hmm.viterbi(regime::to_matrix(kTinyX)), best);
}

TEST(HmmEm, MonotonicOnBundledData) {
    const auto& hist = full_pipeline().fit3.loglik_history;
    ASSERT_GE(hist.size(), 3u);
    for (std::size_t i = 1; i < hist.size(); ++i) EXPECT_GE(hist[i] - hist[i - 1], -1e-9);
    EXPECT_TRUE(full_pipeline().fit3.converged);
}

TEST(HmmEm, TransitionsRowStochastic) {
    for (const GaussianHMM* hmm : {&full_pipeline().hmm2, &full_pipeline().hmm3}) {
        const Matrix& A = hmm->params().transmat;
        for (std::size_t i = 0; i < A.rows; ++i) {
            double row = 0.0;
            for (std::size_t j = 0; j < A.cols; ++j) {
                EXPECT_GE(A(i, j), 0.0);
                row += A(i, j);
            }
            EXPECT_NEAR(row, 1.0, 1e-12);
        }
    }
}

TEST(HmmEm, StateLabelOrdering) {
    // Pinned convention: means ascending, state 0 = lowest mean.
    for (const GaussianHMM* hmm : {&full_pipeline().hmm2, &full_pipeline().hmm3}) {
        const Matrix& means = hmm->params().means;
        for (std::size_t k = 1; k < means.rows; ++k)
            EXPECT_GE(means(k, 0), means(k - 1, 0));
    }
}

TEST(HmmEm, StationaryIsLeftEigenvector) {
    const GaussianHMM& hmm = full_pipeline().hmm3;
    const std::vector<double> pi = hmm.stationary_distribution();
    const Matrix& A = hmm.params().transmat;
    double sum = 0.0;
    for (double x : pi) sum += x;
    EXPECT_NEAR(sum, 1.0, 1e-12);
    for (std::size_t j = 0; j < 3; ++j) {
        double piA = 0.0;
        for (std::size_t i = 0; i < 3; ++i) piA += pi[i] * A(i, j);
        EXPECT_NEAR(piA, pi[j], 1e-10);  // pi P = pi
    }
}

TEST(HmmEm, FitIsDeterministic) {
    // Two fits on identical data produce bitwise-identical parameters.
    const Matrix x = index_head(600);
    GaussianHMM a(3), b(3);
    const auto ra = a.fit(x);
    const auto rb = b.fit(x);
    EXPECT_EQ(ra.log_likelihood, rb.log_likelihood);
    EXPECT_EQ(a.params().means.data, b.params().means.data);
    EXPECT_EQ(a.params().transmat.data, b.params().transmat.data);
    EXPECT_EQ(a.params().variances.data, b.params().variances.data);
}

TEST(HmmEdge, K1IsDegenerateError) {
    EXPECT_THROW(GaussianHMM(1), std::invalid_argument);
}

TEST(HmmEdge, VarianceFloorEngages) {
    // K exceeding the data's distinct support hits the variance floor.
    std::vector<double> x;
    for (int i = 0; i < 30; ++i) x.push_back(0.0);
    for (int i = 0; i < 30; ++i) x.push_back(1.0);
    GaussianHMM hmm(3, 1e-8, 200, 1e-8);
    const auto res = hmm.fit(regime::to_matrix(x));  // must not crash
    bool any_at_floor = false;
    for (double v : hmm.params().variances.data) {
        EXPECT_GE(v, 1e-8);
        if (std::abs(v - 1e-8) < 1e-12) any_at_floor = true;
    }
    EXPECT_TRUE(any_at_floor);
    EXPECT_TRUE(std::isfinite(res.log_likelihood));
}

TEST(HmmEdge, NonConvergenceIsFlagNotException) {
    GaussianHMM hmm(3, 1e-8, 3, 1e-8);
    const auto res = hmm.fit(index_head(400));
    EXPECT_FALSE(res.converged);
    EXPECT_EQ(res.n_iter, 3);
    EXPECT_TRUE(std::isfinite(res.log_likelihood));
}

TEST(HmmInference, ProbabilityOutputsAreValid) {
    const Matrix& gamma = full_pipeline().smoothed3;
    for (std::size_t t = 0; t < gamma.rows; ++t) {
        double row = 0.0;
        for (std::size_t i = 0; i < gamma.cols; ++i) {
            EXPECT_GE(gamma(t, i), 0.0);
            EXPECT_LE(gamma(t, i), 1.0 + 1e-12);
            row += gamma(t, i);
        }
        ASSERT_NEAR(row, 1.0, 1e-10);
    }
    Matrix alpha = full_pipeline().hmm3.filtered_probabilities(index_head(200));
    for (std::size_t t = 0; t < alpha.rows; ++t) {
        double row = 0.0;
        for (std::size_t i = 0; i < alpha.cols; ++i) row += alpha(t, i);
        ASSERT_NEAR(row, 1.0, 1e-10);
    }
}

TEST(HmmInference, ForwardStepMatchesFullFilter) {
    // Incremental filtering (used by the gate) equals the batch forward.
    const GaussianHMM& hmm = full_pipeline().hmm3;
    const Matrix x = index_head(150);
    const Matrix full = hmm.filtered_probabilities(x);
    Matrix head = index_head(100);
    Matrix alpha_head = hmm.filtered_probabilities(head);
    std::vector<double> alpha(3);
    for (std::size_t k = 0; k < 3; ++k) alpha[k] = alpha_head(99, k);
    for (std::size_t t = 100; t < 150; ++t)
        alpha = regime::forward_step(hmm.params(), alpha, {x(t, 0)});
    for (std::size_t k = 0; k < 3; ++k) EXPECT_NEAR(alpha[k], full(149, k), 1e-12);
}

TEST(HmmInference, TwoDimensionalObservations) {
    // 2-D diagonal-covariance fit runs; labels sort on dimension 0.
    const auto& r = regime_tests::index_returns();
    Matrix x2(500, 2);
    for (std::size_t t = 0; t < 500; ++t) {
        x2(t, 0) = r[t];
        x2(t, 1) = std::abs(r[t]);
    }
    GaussianHMM hmm(2);
    const auto res = hmm.fit(x2);
    EXPECT_EQ(hmm.params().means.rows, 2u);
    EXPECT_EQ(hmm.params().means.cols, 2u);
    EXPECT_LE(hmm.params().means(0, 0), hmm.params().means(1, 0));
    EXPECT_TRUE(std::isfinite(res.log_likelihood));
    const Matrix gamma = hmm.smoothed_probabilities(x2);
    EXPECT_EQ(gamma.rows, 500u);
    EXPECT_EQ(gamma.cols, 2u);
}

TEST(HmmSelection, PrefersThreeStates) {
    // Data come from a true 3-state DGP: K=3 wins on AIC and BIC.
    const auto& pipe = full_pipeline();
    const Matrix r = regime::to_matrix(pipe.dataset.index_returns);
    EXPECT_GT(pipe.fit3.log_likelihood, pipe.fit2.log_likelihood);
    EXPECT_LT(pipe.hmm3.aic(r), pipe.hmm2.aic(r));
    EXPECT_LT(pipe.hmm3.bic(r), pipe.hmm2.bic(r));
}

TEST(HmmEdge, InputValidationErrors) {
    GaussianHMM hmm(2);
    EXPECT_THROW(hmm.fit(Matrix(0, 1)), std::invalid_argument);  // empty
    Matrix bad(3, 1);
    bad(0, 0) = 1.0;
    bad(1, 0) = std::nan("");
    bad(2, 0) = 2.0;
    EXPECT_THROW(hmm.fit(bad), std::invalid_argument);           // NaN
    EXPECT_THROW(hmm.fit(Matrix(10, 3, 1.0)), std::invalid_argument);  // D=3
    EXPECT_THROW(hmm.fit(Matrix(2, 1, 1.0)), std::invalid_argument);   // T <= K
    EXPECT_THROW(hmm.score(Matrix(1, 1, 1.0)), std::invalid_argument); // not fitted
    EXPECT_THROW(GaussianHMM(2, 0.0), std::invalid_argument);          // tol <= 0
    EXPECT_THROW(GaussianHMM(2, 1e-8, 0), std::invalid_argument);      // max_iter < 1
    EXPECT_THROW(GaussianHMM(2, 1e-8, 500, 0.0), std::invalid_argument);  // var_floor <= 0
}

TEST(HmmProperty, FittedParamsScoreAtLeastAsWellAsPerturbations) {
    // Property-style grid check: the fitted parameters score at least as
    // well as every mean-shifted perturbation.
    const Matrix x = index_head(400);
    GaussianHMM hmm(2);
    const auto res = hmm.fit(x);
    for (int i = 0; i < 7; ++i) {
        const double shift = -0.01 + i * (0.02 / 6.0);
        HMMParams p = hmm.params();
        for (double& mval : p.means.data) mval += shift;
        GaussianHMM probe(2);
        probe.set_params(p);
        EXPECT_LE(probe.score(x), res.log_likelihood + 1e-9);
    }
}

// ------------------------------------------------------------------------- //
// Robustness: dead states, shape validation, monotonicity, zero transitions
// ------------------------------------------------------------------------- //

namespace {

/// 300 low-vol points and a warm start whose last state sits 5.0 away with
/// variance 1e-8: that state has zero posterior support everywhere.
std::pair<Matrix, HMMParams> dead_state_setup(GaussianHMM& hmm) {
    Matrix x(300, 1);
    for (std::size_t t = 0; t < 300; ++t) x(t, 0) = 0.01 * std::sin(static_cast<double>(t));
    HMMParams init = hmm.pinned_init(x);
    init.means(2, 0) = 5.0;
    init.variances(2, 0) = 1e-8;
    return {x, init};
}

bool all_finite(const std::vector<double>& v) {
    for (double x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

}  // namespace

TEST(HmmRobust, DeadStateIsFlaggedNotNan) {
    // PT-1: a state with no posterior support stops EM with a clear flag.
    GaussianHMM hmm(3);
    auto [x, init] = dead_state_setup(hmm);
    const auto res = hmm.fit(x, init);
    EXPECT_EQ(res.dead_state, 2);
    EXPECT_FALSE(res.converged);
    EXPECT_TRUE(res.monotone);
    EXPECT_EQ(res.n_iter, 1);  // stopped after the first E-step (init scored)
    EXPECT_TRUE(std::isfinite(res.log_likelihood));
    EXPECT_TRUE(all_finite(res.loglik_history));
    const HMMParams& p = hmm.params();
    EXPECT_TRUE(all_finite(p.means.data));
    EXPECT_TRUE(all_finite(p.variances.data));
    EXPECT_TRUE(all_finite(p.transmat.data));
    EXPECT_TRUE(all_finite(p.startprob));
    // The returned parameters are the just-scored ones.
    EXPECT_NEAR(hmm.score(x), res.log_likelihood, 1e-9);
    GaussianHMM ok(3);
    EXPECT_EQ(ok.fit(x).dead_state, -1);
}

TEST(HmmRobust, ForwardStepZeroLikelihoodIsError) {
    // PT-1b: all mass on a state whose row of A only reaches states under
    // which the observation underflows -> error, never NaN.
    HMMParams p;
    p.startprob = {0.5, 0.5};
    p.transmat = Matrix(2, 2);
    p.transmat(0, 1) = 1.0;
    p.transmat(1, 1) = 1.0;
    p.means = Matrix(2, 1);
    p.means(1, 0) = 100.0;
    p.variances = Matrix(2, 1, 1e-8);
    EXPECT_THROW(regime::forward_step(p, {1.0, 0.0}, {0.0}), std::invalid_argument);
    GaussianHMM hmm(2);
    hmm.set_params(p);
    EXPECT_THROW(hmm.score(Matrix(2, 1, 0.0)), std::invalid_argument);
}

TEST(HmmRobust, ScoreRejectsDimensionMismatch) {
    // PT-2: a D=1 model must refuse (T, 2) data in every inference call.
    GaussianHMM hmm(2);
    const Matrix x = index_head(300);
    const auto fit = hmm.fit(x);
    Matrix x2(300, 2);
    for (std::size_t t = 0; t < 300; ++t) x2(t, 0) = x2(t, 1) = x(t, 0);
    EXPECT_THROW(hmm.score(x2), std::invalid_argument);
    EXPECT_THROW(hmm.filtered_probabilities(x2), std::invalid_argument);
    EXPECT_THROW(hmm.smoothed_probabilities(x2), std::invalid_argument);
    EXPECT_THROW(hmm.viterbi(x2), std::invalid_argument);
    EXPECT_THROW(hmm.aic(x2), std::invalid_argument);
    EXPECT_THROW(hmm.bic(x2), std::invalid_argument);
    EXPECT_NEAR(hmm.score(x), fit.log_likelihood, 1e-9);
}

TEST(HmmRobust, WarmStartShapeValidation) {
    // PT-3: every inconsistent warm start is std::invalid_argument, never UB.
    const Matrix x = index_head(300);
    const HMMParams good = GaussianHMM(2).pinned_init(x);
    GaussianHMM k3(3);
    EXPECT_THROW(k3.fit(x, good), std::invalid_argument);  // K=2 params on K=3 model
    GaussianHMM hmm(2);
    Matrix x2(300, 2);
    for (std::size_t t = 0; t < 300; ++t) x2(t, 0) = x2(t, 1) = x(t, 0);
    const HMMParams init_d2 = GaussianHMM(2).pinned_init(x2);
    EXPECT_THROW(hmm.fit(x, init_d2), std::invalid_argument);  // D=2 params on D=1 data
    HMMParams neg = good;
    neg.variances(0, 0) = -1.0;
    EXPECT_THROW(hmm.fit(x, neg), std::invalid_argument);
    HMMParams nonstoch = good;
    nonstoch.transmat(0, 0) = 0.5;
    nonstoch.transmat(0, 1) = 0.6;
    EXPECT_THROW(hmm.fit(x, nonstoch), std::invalid_argument);
    HMMParams badpi = good;
    badpi.startprob = {0.7, 0.7};
    EXPECT_THROW(hmm.fit(x, badpi), std::invalid_argument);
    HMMParams nan = good;
    nan.means(0, 0) = std::nan("");
    EXPECT_THROW(hmm.fit(x, nan), std::invalid_argument);
    HMMParams ragged = good;
    ragged.means.data.pop_back();  // storage no longer matches rows*cols
    EXPECT_THROW(hmm.fit(x, ragged), std::invalid_argument);
    EXPECT_THROW(hmm.set_params(badpi), std::invalid_argument);
    EXPECT_THROW(k3.set_params(good), std::invalid_argument);
    // forward_step length checks
    EXPECT_THROW(regime::forward_step(good, {1.0, 0.0, 0.0}, {0.0}), std::invalid_argument);
    EXPECT_THROW(regime::forward_step(good, {0.5, 0.5}, {0.0, 0.0}), std::invalid_argument);
    EXPECT_THROW(regime::forward_step(good, {0.7, 0.7}, {0.0}), std::invalid_argument);
    EXPECT_THROW(regime::forward_step(good, {0.5, 0.5}, {std::nan("")}), std::invalid_argument);
}

TEST(HmmRobust, PinnedInitMatchesSpec) {
    // PT-10: quantile means, pooled floored variance, 0.9 self-transitions.
    const HMMParams p = GaussianHMM(2).pinned_init(regime::to_matrix({1.0, 2.0, 3.0, 4.0, 5.0}));
    EXPECT_NEAR(p.means(0, 0), 2.0, 1e-15);  // q = 0.25 -> h = 1.0
    EXPECT_NEAR(p.means(1, 0), 4.0, 1e-15);  // q = 0.75 -> h = 3.0
    EXPECT_NEAR(p.variances(0, 0), 2.0, 1e-15);  // population variance of 1..5
    EXPECT_NEAR(p.variances(1, 0), 2.0, 1e-15);
    EXPECT_NEAR(p.transmat(0, 0), 0.9, 1e-15);
    EXPECT_NEAR(p.transmat(0, 1), 0.1, 1e-15);
    EXPECT_NEAR(p.startprob[0], 0.5, 1e-15);
    std::vector<double> x(60, 0.0);
    for (std::size_t i = 30; i < 60; ++i) x[i] = 1.0;
    const HMMParams p3 = GaussianHMM(3).pinned_init(regime::to_matrix(x));
    EXPECT_NEAR(p3.means(0, 0), 0.0, 1e-15);  // q = 1/6 -> h = 9.83
    EXPECT_NEAR(p3.means(1, 0), 0.5, 1e-15);  // q = 1/2 -> h = 29.5 interpolates
    EXPECT_NEAR(p3.means(2, 0), 1.0, 1e-15);  // q = 5/6 -> h = 49.17
    for (std::size_t k = 0; k < 3; ++k) EXPECT_NEAR(p3.variances(k, 0), 0.25, 1e-15);
    EXPECT_NEAR(p3.transmat(0, 1), 0.05, 1e-15);
}

TEST(HmmRobust, ViterbiRespectsZeroTransitions) {
    // PT-14: A[0][1] = 0 means state 0 can never leave; the path must start
    // in state 1 despite the first observation favouring state 0.
    HMMParams p;
    p.startprob = {0.5, 0.5};
    p.transmat = Matrix(2, 2);
    p.transmat(0, 0) = 1.0;
    p.transmat(1, 0) = 0.5;
    p.transmat(1, 1) = 0.5;
    p.means = Matrix(2, 1);
    p.means(0, 0) = -1.0;
    p.means(1, 0) = 1.0;
    p.variances = Matrix(2, 1, 0.1);
    const std::vector<double> x = {-1.0, 1.0, 1.0};
    GaussianHMM hmm(2);
    hmm.set_params(p);
    const std::vector<int> path = hmm.viterbi(regime::to_matrix(x));
    EXPECT_EQ(path, (std::vector<int>{1, 1, 1}));
    // Cross-check against enumeration (a zero transition kills the path).
    std::vector<int> best;
    double best_prob = -1.0;
    for (const auto& s : all_paths()) {
        const double prob = path_prob(s, x, p);
        if (prob > best_prob) {
            best_prob = prob;
            best = s;
        }
    }
    EXPECT_EQ(path, best);
}

TEST(HmmRobust, EmStepStatusClassification) {
    // PT-15: the pinned (ll, ll_prev) -> {continue, converged, non-monotone} rule.
    using regime::EmStep;
    using regime::em_step_status;
    EXPECT_EQ(em_step_status(10.0, 9.0, 1e-8), EmStep::Continue);
    EXPECT_EQ(em_step_status(10.0, 10.0 - 5e-9, 1e-8), EmStep::Converged);
    EXPECT_EQ(em_step_status(10.0, 10.0, 1e-8), EmStep::Converged);
    EXPECT_EQ(em_step_status(1000.0 - 1e-5, 1000.0, 1e-8), EmStep::Converged);  // tiny decrease
    EXPECT_EQ(em_step_status(1000.0 - 2.0 * regime::kMonotoneRelTol * 1000.0, 1000.0, 1e-8),
              EmStep::NonMonotone);
    EXPECT_EQ(em_step_status(-2.0, -1.0, 1e-8), EmStep::NonMonotone);  // max(1, |ll_prev|)
    EXPECT_EQ(em_step_status(std::nan(""), 1.0, 1e-8), EmStep::NonMonotone);
    EXPECT_EQ(em_step_status(1.0, -std::numeric_limits<double>::infinity(), 1e-8),
              EmStep::NonMonotone);
}

TEST(HmmRobust, NIterCountsHistoryEntries) {
    // MIN-7: n_iter == history size; the post-exhaustion re-score is not counted.
    GaussianHMM hmm(3, 1e-8, 3, 1e-8);
    const auto res = hmm.fit(index_head(400));
    EXPECT_EQ(res.n_iter, 3);
    EXPECT_EQ(res.loglik_history.size(), 3u);
    EXPECT_TRUE(res.monotone);
    EXPECT_EQ(res.dead_state, -1);
    EXPECT_GE(res.log_likelihood, res.loglik_history.back() - 1e-9);
}

TEST(HmmRobust, StationaryDistributionReportsNonConvergence) {
    // MIN-10: a chain whose power-method iterate cycles -> error, never a
    // silently unconverged vector.
    GaussianHMM hmm(3);
    HMMParams p;
    p.startprob = {1.0, 0.0, 0.0};
    p.transmat = Matrix(3, 3);
    p.transmat(0, 1) = 1.0;
    p.transmat(1, 2) = 1.0;
    p.transmat(2, 0) = 0.5;
    p.transmat(2, 1) = 0.5;
    p.means = Matrix(3, 1, 0.0);
    p.variances = Matrix(3, 1, 1.0);
    hmm.set_params(p);
    EXPECT_THROW(hmm.stationary_distribution(1e-13, 50), std::invalid_argument);
    EXPECT_THROW(hmm.stationary_distribution(0.0), std::invalid_argument);
    EXPECT_THROW(hmm.stationary_distribution(1e-13, 0), std::invalid_argument);
    // an aperiodic chain converges to the uniform vector
    p.transmat = Matrix(3, 3);
    p.transmat(0, 0) = p.transmat(0, 1) = 0.5;
    p.transmat(1, 1) = p.transmat(1, 2) = 0.5;
    p.transmat(2, 0) = p.transmat(2, 2) = 0.5;
    hmm.set_params(p);
    const auto pi = hmm.stationary_distribution();
    for (double v : pi) EXPECT_NEAR(v, 1.0 / 3.0, 1e-10);
}
