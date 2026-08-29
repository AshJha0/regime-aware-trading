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

#include <vector>

#include "regime/matrix.hpp"

namespace regime {

/// Parameter set of a diagonal-covariance Gaussian HMM.
struct HMMParams {
    std::vector<double> startprob;  ///< Initial state distribution, length K.
    Matrix transmat;                ///< Row-stochastic transition matrix, K x K.
    Matrix means;                   ///< State means, K x D.
    Matrix variances;               ///< Per-dimension state variances, K x D.
};

/// Outcome of a Baum-Welch fit.
struct HMMFitResult {
    double log_likelihood{0.0};          ///< Log-likelihood of the returned parameters.
    int n_iter{0};                       ///< Number of E-steps performed.
    bool converged{false};               ///< False iff max_iter E-steps ran without
                                         ///< the tolerance being met (a flag, never
                                         ///< an exception).
    std::vector<double> loglik_history;  ///< Log-likelihood at each E-step
                                         ///< (non-decreasing by EM monotonicity).
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
    /// \throws std::invalid_argument on invalid hyper-parameters.
    explicit GaussianHMM(int n_states, double tol = 1e-8, int max_iter = 500,
                         double var_floor = 1e-8);

    /// Deterministic EM starting point (pinned rule, API_SPEC 1.3):
    /// per-dimension means at quantiles q_k = (2k+1)/(2K) (linear
    /// interpolation), full-sample variance (ddof = 0, floored) for every
    /// state, 0.9 self-transitions, uniform initial distribution.
    HMMParams pinned_init(const Matrix& X) const;

    /// Fit by Baum-Welch EM from the pinned initialization.
    /// \throws std::invalid_argument on invalid observations or T <= K.
    HMMFitResult fit(const Matrix& X);

    /// Fit with a warm-start parameter set (used by walk-forward refits).
    HMMFitResult fit(const Matrix& X, const HMMParams& init);

    /// Log-likelihood of \p X under the fitted parameters.
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
    void set_params(const HMMParams& p);

    /// True once parameters are available (fit or set_params).
    bool is_fitted() const { return fitted_; }

    /// Number of hidden states K.
    int n_states() const { return n_states_; }

    /// Result of the last fit() call.
    const HMMFitResult& fit_result() const { return fit_result_; }

private:
    HMMFitResult fit_impl(const Matrix& X, const HMMParams* init);

    int n_states_;
    double tol_;
    int max_iter_;
    double var_floor_;
    bool fitted_{false};
    HMMParams params_;
    HMMFitResult fit_result_;
};

/// One incremental scaled-forward update for online filtering: given
/// alpha_prev = P(s_{t-1} | x_1..t-1) and a new observation x_t (length D),
/// return P(s_t | x_1..t).  Identical math to one forward step; used by the
/// regime gate between refits so the filter never re-reads the past.
std::vector<double> forward_step(const HMMParams& params,
                                 const std::vector<double>& alpha_prev,
                                 const std::vector<double>& x_t);

}  // namespace regime

#endif  // REGIME_HMM_HPP
