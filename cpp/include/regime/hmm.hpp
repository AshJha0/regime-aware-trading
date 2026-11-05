#ifndef REGIME_HMM_HPP
#define REGIME_HMM_HPP

/// \file hmm.hpp
/// \brief Diagonal-covariance Gaussian HMM with scaled forward-backward and
///        deterministic (RNG-free) Baum-Welch EM.
///
/// Scaled forward-backward (pinned in API_SPEC.md section 1.1): each step is
/// normalized by d_t after factoring out the per-step maximum log-emission
/// m_t, so the log-likelihood is the sum of the log normalizers
/// sum_t (log d_t + m_t).  The backward pass shares the same d_t, which makes
/// the posteriors gamma_t(i) = alpha^_t(i) * beta^_t(i) already normalized.
///
/// Everything is deterministic: initialization uses pinned quantile-based
/// means (section 1.3) and no random number generator appears anywhere.
///
/// Dead-state guard (pinned, API_SPEC 1.4): after each E-step the
/// transition-row support sum_{t<T} gamma_t(i) of every state is checked
/// against kDeadStateSupport; a state below it has no posterior mass
/// (Rabiner 1989 section V.B, Bilmes 1998 section 4) and the M-step would
/// produce 0/0.  EM then stops with the just-scored parameters, converged =
/// false and dead_state = i.  No NaN ever leaves fit().  A zero forward
/// normalizer (data impossible under every reachable state) throws.

#include <cstddef>
#include <vector>

#include "regime/matrix.hpp"

namespace regime {

/// Pinned dead-state threshold on sum_{t<T} gamma_t(i) (API_SPEC 1.4).
constexpr double kDeadStateSupport = 1e-12;
/// Pinned relative tolerance for a log-likelihood decrease to count as a
/// monotonicity violation (API_SPEC 1.4).
constexpr double kMonotoneRelTol = 1e-6;
/// Pinned tolerance on sum(startprob) and on every transition row sum.
constexpr double kStochasticTol = 1e-9;

/// Classification of one E-step score against the previous one (pinned).
enum class EmStep { Continue, Converged, NonMonotone };

/// Pinned rule (API_SPEC 1.4): NonMonotone if
/// ll - ll_prev < -kMonotoneRelTol * max(1, |ll_prev|) or either score is
/// non-finite; else Converged if ll - ll_prev < tol; else Continue.
EmStep em_step_status(double ll, double ll_prev, double tol);

/// Parameter set of a diagonal-covariance Gaussian HMM.
struct HMMParams {
    std::vector<double> startprob;  ///< Initial state distribution, length K.
    Matrix transmat;                ///< Row-stochastic transition matrix, K x K.
    Matrix means;                   ///< State means, K x D.
    Matrix variances;               ///< Per-dimension state variances, K x D.
};

/// Validate a parameter set against a model size (pinned, API_SPEC 1.9):
/// startprob length K, transmat K x K, means/variances K x D, all finite,
/// variances > 0, startprob and every transition row non-negative and
/// summing to 1 within kStochasticTol.
/// \throws std::invalid_argument on any violation.
void validate_params(const HMMParams& p, std::size_t n_states, std::size_t n_dims);

/// Add \p floor to every transition entry and renormalise the rows
/// (API_SPEC 1.3): EM multiplies the old A_ij into the new one, so an entry
/// at or near zero can never recover.  floor == 0 returns an exact copy.
/// \throws std::invalid_argument if floor is negative or non-finite.
Matrix apply_transition_floor(const Matrix& transmat, double floor);

/// Outcome of a Baum-Welch fit.
struct HMMFitResult {
    double log_likelihood{0.0};          ///< Log-likelihood of the returned parameters
                                         ///< (always finite; the score of exactly
                                         ///< the returned parameter set).
    int n_iter{0};                       ///< Number of E-steps that appended to
                                         ///< loglik_history (the consistency re-score
                                         ///< after max_iter exhaustion is not counted).
    bool converged{false};               ///< True iff the improvement fell below tol
                                         ///< with no monotonicity violation and no
                                         ///< dead state (a flag, never an exception).
    std::vector<double> loglik_history;  ///< Log-likelihood at each counted E-step.
    bool monotone{true};                 ///< False iff an E-step decreased the
                                         ///< log-likelihood beyond kMonotoneRelTol
                                         ///< (EM stops there; converged is false).
    int dead_state{-1};                  ///< Lowest state whose support fell below
                                         ///< kDeadStateSupport (EM stops there;
                                         ///< converged is false), else -1.
};

/// Diagonal-covariance Gaussian HMM with deterministic Baum-Welch EM.
///
/// Observations are (T x D) matrices with D in {1, 2}; a 1-D series is a
/// (T x 1) matrix (see regime::to_matrix).  States are relabeled after EM so
/// means on dimension 0 are ascending (state 0 = lowest mean) — the pinned
/// cross-language label convention.
class GaussianHMM {
public:
    /// \param n_states Number of hidden states K (>= 2; K = 1 is degenerate).
    /// \param tol      Convergence tolerance on the log-likelihood improvement.
    /// \param max_iter Maximum number of EM iterations (E-steps).
    /// \param var_floor Lower bound applied to every variance after each
    ///                  update (guards zero-variance collapse).
    /// \param transition_floor Added to each transition entry of a warm-start
    ///                  init before EM, rows renormalised (API_SPEC 1.3);
    ///                  ignored on a cold start.  Default 0 (off).
    /// \throws std::invalid_argument on invalid hyper-parameters.
    explicit GaussianHMM(int n_states, double tol = 1e-8, int max_iter = 500,
                         double var_floor = 1e-8, double transition_floor = 0.0);

    /// Deterministic EM starting point (pinned rule, API_SPEC 1.3):
    /// per-dimension means at quantiles q_k = (2k+1)/(2K) (linear
    /// interpolation), full-sample variance (ddof = 0, floored) for every
    /// state, 0.9 self-transitions, uniform initial distribution.
    HMMParams pinned_init(const Matrix& X) const;

    /// Fit by Baum-Welch EM from the pinned initialization.
    /// \throws std::invalid_argument on invalid observations or T <= K.
    HMMFitResult fit(const Matrix& X);

    /// Fit with a warm-start parameter set (used by walk-forward refits).
    /// \throws std::invalid_argument if \p init disagrees with (K, D) or is
    ///         not a valid stochastic parameter set (validate_params).
    HMMFitResult fit(const Matrix& X, const HMMParams& init);

    /// Log-likelihood of \p X under the fitted parameters.
    /// \throws std::invalid_argument if unfitted, X is invalid, the fitted
    ///         parameters disagree with X.cols / n_states, or an observation
    ///         has zero likelihood under every reachable state.
    double score(const Matrix& X) const;

    /// Filtered posteriors P(s_t = i | x_1..t), shape (T x K).
    Matrix filtered_probabilities(const Matrix& X) const;

    /// Smoothed posteriors P(s_t = i | x_1..T), shape (T x K).
    Matrix smoothed_probabilities(const Matrix& X) const;

    /// Most likely state path (log-space Viterbi); ties break toward the
    /// lower state index.
    std::vector<int> viterbi(const Matrix& X) const;

    /// Stationary distribution of the fitted transition matrix by the pinned
    /// power method: pi <- pi A with L1 renormalization until
    /// max|pi_new - pi| < tol.
    /// \throws std::invalid_argument if unfitted, tol <= 0, max_iter < 1, or
    ///         the iteration does not converge within max_iter (periodic or
    ///         reducible chain) — an unconverged vector is never returned.
    std::vector<double> stationary_distribution(double tol = 1e-13,
                                                int max_iter = 10000) const;

    /// Free parameter count: 2*K*D + K*(K-1) + (K-1).
    int n_parameters(int n_dims) const;

    /// Akaike information criterion -2 LL + 2p (lower is better).
    double aic(const Matrix& X) const;

    /// Bayesian information criterion -2 LL + p ln T (lower is better).
    double bic(const Matrix& X) const;

    /// Fitted parameters. \throws std::invalid_argument if not fitted.
    const HMMParams& params() const;

    /// Install parameters directly (testing / cross-checks).
    /// \throws std::invalid_argument unless p is a valid parameter set with
    ///         n_states rows and D in {1, 2} (validate_params).
    void set_params(const HMMParams& p);

    /// True once parameters are available (fit or set_params).
    bool is_fitted() const { return fitted_; }

    /// Number of hidden states K.
    int n_states() const { return n_states_; }

    /// Result of the last fit() call.
    const HMMFitResult& fit_result() const { return fit_result_; }

private:
    HMMFitResult fit_impl(const Matrix& X, const HMMParams* init);
    /// Validate X and the fitted parameters' agreement with (n_states, X.cols).
    const HMMParams& prepare(const Matrix& X) const;

    int n_states_;
    double tol_;
    int max_iter_;
    double var_floor_;
    double transition_floor_;
    bool fitted_{false};
    HMMParams params_;
    HMMFitResult fit_result_;
};

/// One incremental scaled-forward update for online filtering: given
/// alpha_prev = P(s_{t-1} | x_1..t-1) and a new observation x_t (length D),
/// return P(s_t | x_1..t).  Identical math to one forward step; used by the
/// regime gate between refits so the filter never re-reads the past.
/// \throws std::invalid_argument if params is invalid, alpha_prev is not a
///         length-K probability vector, x_t is not length D / finite, or the
///         observation has zero likelihood under every reachable state.
std::vector<double> forward_step(const HMMParams& params,
                                 const std::vector<double>& alpha_prev,
                                 const std::vector<double>& x_t);

/// Streaming filter: forward_step with the set-up paid once (API_SPEC 1.10).
///
/// The constructor validates and copies the parameters, precomputes the
/// log-variance terms and sizes every buffer; update() then checks only the
/// observation and allocates nothing (except when it throws).  Without an
/// alpha_prev the first update applies startprob with no transition, so a
/// run over a series reproduces GaussianHMM::filtered_probabilities.
class RegimeFilter {
public:
    /// \throws std::invalid_argument if params is invalid (validate_params).
    explicit RegimeFilter(const HMMParams& params);

    /// Start from a filtered posterior at t-1: the first update then equals
    /// forward_step(params, alpha_prev, x).
    /// \throws std::invalid_argument on invalid params or alpha_prev.
    RegimeFilter(const HMMParams& params, const std::vector<double>& alpha_prev);

    /// Restart from startprob, with no transition on the first update.
    void reset();

    /// Restart from a filtered posterior at t-1.
    /// \throws std::invalid_argument if alpha_prev is not a length-K
    ///         probability vector; the filter is then unchanged.
    void reset(const std::vector<double>& alpha_prev);

    /// Advance one observation of length n and return P(s_t | x_1..t).  The
    /// reference stays valid for the life of the filter; its contents change
    /// on the next update.
    /// \throws std::invalid_argument if n != D, x is not finite, or x has
    ///         zero likelihood under every reachable state; the filter is
    ///         then unchanged.
    const std::vector<double>& update(const double* x, std::size_t n);

    /// Univariate overload (requires D == 1).
    const std::vector<double>& update(double x) { return update(&x, 1); }

    /// Vector overload.
    const std::vector<double>& update(const std::vector<double>& x) {
        return update(x.data(), x.size());
    }

    /// Current filtered posterior (startprob before any update).
    const std::vector<double>& alpha() const { return alpha_; }

    /// True once there is a filtered posterior to transition from.
    bool started() const { return started_; }

    std::size_t n_states() const { return K_; }
    std::size_t n_dims() const { return D_; }

private:
    std::size_t K_;
    std::size_t D_;
    std::vector<double> startprob_;
    std::vector<double> transmat_;   ///< Row-major K x K.
    std::vector<double> means_;      ///< Row-major K x D.
    std::vector<double> variances_;  ///< Row-major K x D.
    std::vector<double> log_const_;  ///< sum_d (log 2 pi + log var_id), length K.
    std::vector<double> logb_;       ///< Scratch, length K.
    std::vector<double> next_;       ///< Scratch, length K; swapped with alpha_.
    std::vector<double> alpha_;
    bool started_{false};
};

}  // namespace regime

#endif  // REGIME_HMM_HPP
