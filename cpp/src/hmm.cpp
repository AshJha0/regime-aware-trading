/// \file hmm.cpp
/// \brief Gaussian HMM: scaled forward-backward, Baum-Welch EM, Viterbi.
///
/// All summations run left-to-right over time to keep results deterministic;
/// the golden tolerances absorb the rounding differences vs the vectorized
/// Python reference.

#include "regime/hmm.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace regime {

namespace {

constexpr double kLog2Pi = 1.8378770664093454;  // log(2*pi)
constexpr double kNegInf = -std::numeric_limits<double>::infinity();

/// Validate an observation matrix: non-empty, D in {1, 2}, all finite.
void validate_observations(const Matrix& X) {
    if (X.rows == 0) throw std::invalid_argument("observation sequence is empty");
    if (X.cols != 1 && X.cols != 2)
        throw std::invalid_argument("only univariate or 2-D observations supported, got D=" +
                                    std::to_string(X.cols));
    for (double v : X.data)
        if (!std::isfinite(v)) throw std::invalid_argument("observations contain NaN or inf");
}

/// Linear-interpolation quantile (numpy default) of a sorted sample.
double quantile_sorted(const std::vector<double>& sorted, double q) {
    const std::size_t n = sorted.size();
    if (n == 1) return sorted[0];
    const double h = q * static_cast<double>(n - 1);
    const auto lo = static_cast<std::size_t>(std::floor(h));
    if (lo + 1 >= n) return sorted[n - 1];
    return sorted[lo] + (h - static_cast<double>(lo)) * (sorted[lo + 1] - sorted[lo]);
}

/// Log emission densities, shape (T x K):
/// log b_t(i) = -0.5 * sum_d [log(2 pi) + log var_id + (x_td - mu_id)^2 / var_id].
Matrix log_emissions(const Matrix& X, const HMMParams& p) {
    const std::size_t T = X.rows, D = X.cols, K = p.means.rows;
    Matrix logb(T, K);
    for (std::size_t t = 0; t < T; ++t) {
        for (std::size_t i = 0; i < K; ++i) {
            double s = 0.0;
            for (std::size_t d = 0; d < D; ++d) {
                const double v = p.variances(i, d);
                const double diff = X(t, d) - p.means(i, d);
                s += kLog2Pi + std::log(v) + diff * diff / v;
            }
            logb(t, i) = -0.5 * s;
        }
    }
    return logb;
}

/// Scaled forward pass.  Fills alpha (T x K), the normalizers d (length T)
/// and the per-step emission shifts m; log-likelihood = sum(log d + m).
void forward_pass(const Matrix& logb, const HMMParams& p, Matrix& alpha,
                  std::vector<double>& d, std::vector<double>& m) {
    const std::size_t T = logb.rows, K = logb.cols;
    alpha = Matrix(T, K);
    d.assign(T, 0.0);
    m.assign(T, 0.0);
    std::vector<double> bt(K);
    for (std::size_t t = 0; t < T; ++t) {
        double mx = logb(t, 0);
        for (std::size_t i = 1; i < K; ++i) mx = std::max(mx, logb(t, i));
        m[t] = mx;
        for (std::size_t i = 0; i < K; ++i) bt[i] = std::exp(logb(t, i) - mx);
        double dt = 0.0;
        if (t == 0) {
            for (std::size_t i = 0; i < K; ++i) {
                const double a = p.startprob[i] * bt[i];
                alpha(0, i) = a;
                dt += a;
            }
        } else {
            for (std::size_t j = 0; j < K; ++j) {
                double s = 0.0;
                for (std::size_t i = 0; i < K; ++i) s += alpha(t - 1, i) * p.transmat(i, j);
                const double a = s * bt[j];
                alpha(t, j) = a;
                dt += a;
            }
        }
        d[t] = dt;
        for (std::size_t i = 0; i < K; ++i) alpha(t, i) /= dt;
    }
}

/// Scaled backward pass sharing the forward normalizers d.
Matrix backward_pass(const Matrix& logb, const HMMParams& p,
                     const std::vector<double>& d, const std::vector<double>& m) {
    const std::size_t T = logb.rows, K = logb.cols;
    Matrix beta(T, K);
    for (std::size_t i = 0; i < K; ++i) beta(T - 1, i) = 1.0;
    std::vector<double> bt(K);
    for (std::size_t t = T - 1; t-- > 0;) {
        for (std::size_t j = 0; j < K; ++j) bt[j] = std::exp(logb(t + 1, j) - m[t + 1]);
        for (std::size_t i = 0; i < K; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < K; ++j) s += p.transmat(i, j) * bt[j] * beta(t + 1, j);
            beta(t, i) = s / d[t + 1];
        }
    }
    return beta;
}

double loglik_from(const std::vector<double>& d, const std::vector<double>& m) {
    double ll = 0.0;
    for (std::size_t t = 0; t < d.size(); ++t) ll += std::log(d[t]) + m[t];
    return ll;
}

}  // namespace

GaussianHMM::GaussianHMM(int n_states, double tol, int max_iter, double var_floor)
    : n_states_(n_states), tol_(tol), max_iter_(max_iter), var_floor_(var_floor) {
    if (n_states < 2)
        throw std::invalid_argument("n_states must be >= 2 (K=1 is degenerate), got " +
                                    std::to_string(n_states));
    if (tol <= 0.0 || max_iter < 1 || var_floor <= 0.0)
        throw std::invalid_argument("tol and var_floor must be > 0 and max_iter >= 1");
}

HMMParams GaussianHMM::pinned_init(const Matrix& X) const {
    validate_observations(X);
    const std::size_t K = static_cast<std::size_t>(n_states_), D = X.cols, T = X.rows;
    HMMParams p;
    p.means = Matrix(K, D);
    p.variances = Matrix(K, D);
    for (std::size_t d = 0; d < D; ++d) {
        std::vector<double> col(T);
        for (std::size_t t = 0; t < T; ++t) col[t] = X(t, d);
        std::sort(col.begin(), col.end());
        for (std::size_t k = 0; k < K; ++k) {
            const double q = (2.0 * static_cast<double>(k) + 1.0) / (2.0 * static_cast<double>(K));
            p.means(k, d) = quantile_sorted(col, q);
        }
        double mean = 0.0;
        for (std::size_t t = 0; t < T; ++t) mean += col[t];
        mean /= static_cast<double>(T);
        double var = 0.0;
        for (std::size_t t = 0; t < T; ++t) var += (col[t] - mean) * (col[t] - mean);
        var /= static_cast<double>(T);  // ddof = 0
        var = std::max(var, var_floor_);
        for (std::size_t k = 0; k < K; ++k) p.variances(k, d) = var;
    }
    p.transmat = Matrix(K, K, 0.1 / static_cast<double>(K - 1));
    for (std::size_t k = 0; k < K; ++k) p.transmat(k, k) = 0.9;
    p.startprob.assign(K, 1.0 / static_cast<double>(K));
    return p;
}

HMMFitResult GaussianHMM::fit(const Matrix& X) { return fit_impl(X, nullptr); }

HMMFitResult GaussianHMM::fit(const Matrix& X, const HMMParams& init) {
    return fit_impl(X, &init);
}

HMMFitResult GaussianHMM::fit_impl(const Matrix& X, const HMMParams* init) {
    validate_observations(X);
    const std::size_t T = X.rows, D = X.cols;
    const std::size_t K = static_cast<std::size_t>(n_states_);
    if (T <= K)
        throw std::invalid_argument("need more observations than states: T=" +
                                    std::to_string(T) + ", K=" + std::to_string(n_states_));
    HMMParams params = init ? *init : pinned_init(X);

    std::vector<double> history;
    double ll_prev = kNegInf;
    double ll = kNegInf;
    bool converged = false;
    int n_iter = 0;
    Matrix alpha;
    std::vector<double> d, m;

    for (int it = 0; it < max_iter_; ++it) {
        // E-step: scores the CURRENT parameters, so on convergence the
        // recorded log-likelihood matches the returned parameter set.
        Matrix logb = log_emissions(X, params);
        forward_pass(logb, params, alpha, d, m);
        ll = loglik_from(d, m);
        history.push_back(ll);
        ++n_iter;
        if (n_iter > 1 && ll - ll_prev < tol_) {
            converged = true;
            break;
        }
        ll_prev = ll;
        Matrix beta = backward_pass(logb, params, d, m);

        // gamma_t(i) = alpha^_t(i) * beta^_t(i); rows already sum to 1.
        Matrix gamma(T, K);
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t i = 0; i < K; ++i) gamma(t, i) = alpha(t, i) * beta(t, i);

        // xi summed over t: xi_sum(i,j) = A_ij * sum_t alpha_t(i) * w_t(j),
        // with w_t(j) = bt_{t+1}(j) * beta_{t+1}(j) / d_{t+1}.
        Matrix xi_sum(K, K);
        std::vector<double> w(K);
        for (std::size_t t = 0; t + 1 < T; ++t) {
            for (std::size_t j = 0; j < K; ++j)
                w[j] = std::exp(logb(t + 1, j) - m[t + 1]) * beta(t + 1, j) / d[t + 1];
            for (std::size_t i = 0; i < K; ++i)
                for (std::size_t j = 0; j < K; ++j) xi_sum(i, j) += alpha(t, i) * w[j];
        }
        for (std::size_t i = 0; i < K; ++i)
            for (std::size_t j = 0; j < K; ++j) xi_sum(i, j) *= params.transmat(i, j);

        // M-step.
        std::vector<double> gsum(K, 0.0), gsum_trans(K, 0.0);
        for (std::size_t t = 0; t < T; ++t)
            for (std::size_t i = 0; i < K; ++i) {
                gsum[i] += gamma(t, i);
                if (t + 1 < T) gsum_trans[i] += gamma(t, i);
            }
        for (std::size_t i = 0; i < K; ++i) params.startprob[i] = gamma(0, i);
        for (std::size_t i = 0; i < K; ++i)
            for (std::size_t j = 0; j < K; ++j) params.transmat(i, j) = xi_sum(i, j) / gsum_trans[i];
        for (std::size_t i = 0; i < K; ++i)
            for (std::size_t dd = 0; dd < D; ++dd) {
                double s = 0.0;
                for (std::size_t t = 0; t < T; ++t) s += gamma(t, i) * X(t, dd);
                params.means(i, dd) = s / gsum[i];
            }
        // Variance update uses the NEW means; floored every iteration.
        for (std::size_t i = 0; i < K; ++i)
            for (std::size_t dd = 0; dd < D; ++dd) {
                double s = 0.0;
                for (std::size_t t = 0; t < T; ++t) {
                    const double diff = X(t, dd) - params.means(i, dd);
                    s += gamma(t, i) * diff * diff;
                }
                params.variances(i, dd) = std::max(s / gsum[i], var_floor_);
            }
    }

    if (!converged) {
        // max_iter exhausted: params had one more M-step than the last
        // recorded E-step, so score them once for a consistent report.
        Matrix logb = log_emissions(X, params);
        forward_pass(logb, params, alpha, d, m);
        ll = loglik_from(d, m);
    }

    // Pinned label convention: sort states by means[:, 0] ascending
    // (stable; state 0 = lowest mean).  Applied once, after EM finishes.
    std::vector<std::size_t> order(K);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&params](std::size_t a, std::size_t b) {
        return params.means(a, 0) < params.means(b, 0);
    });
    HMMParams sorted;
    sorted.startprob.resize(K);
    sorted.transmat = Matrix(K, K);
    sorted.means = Matrix(K, D);
    sorted.variances = Matrix(K, D);
    for (std::size_t i = 0; i < K; ++i) {
        sorted.startprob[i] = params.startprob[order[i]];
        for (std::size_t j = 0; j < K; ++j) sorted.transmat(i, j) = params.transmat(order[i], order[j]);
        for (std::size_t dd = 0; dd < D; ++dd) {
            sorted.means(i, dd) = params.means(order[i], dd);
            sorted.variances(i, dd) = params.variances(order[i], dd);
        }
    }
    params_ = std::move(sorted);
    fitted_ = true;
    fit_result_ = HMMFitResult{ll, n_iter, converged, std::move(history)};
    return fit_result_;
}

const HMMParams& GaussianHMM::params() const {
    if (!fitted_) throw std::invalid_argument("model is not fitted; call fit() first");
    return params_;
}

void GaussianHMM::set_params(const HMMParams& p) {
    params_ = p;
    fitted_ = true;
}

double GaussianHMM::score(const Matrix& X) const {
    const HMMParams& p = params();
    validate_observations(X);
    Matrix logb = log_emissions(X, p);
    Matrix alpha;
    std::vector<double> d, m;
    forward_pass(logb, p, alpha, d, m);
    return loglik_from(d, m);
}

Matrix GaussianHMM::filtered_probabilities(const Matrix& X) const {
    const HMMParams& p = params();
    validate_observations(X);
    Matrix logb = log_emissions(X, p);
    Matrix alpha;
    std::vector<double> d, m;
    forward_pass(logb, p, alpha, d, m);
    return alpha;
}

Matrix GaussianHMM::smoothed_probabilities(const Matrix& X) const {
    const HMMParams& p = params();
    validate_observations(X);
    Matrix logb = log_emissions(X, p);
    Matrix alpha;
    std::vector<double> d, m;
    forward_pass(logb, p, alpha, d, m);
    Matrix beta = backward_pass(logb, p, d, m);
    Matrix gamma(X.rows, logb.cols);
    for (std::size_t t = 0; t < X.rows; ++t)
        for (std::size_t i = 0; i < logb.cols; ++i) gamma(t, i) = alpha(t, i) * beta(t, i);
    return gamma;
}

std::vector<int> GaussianHMM::viterbi(const Matrix& X) const {
    const HMMParams& p = params();
    validate_observations(X);
    Matrix logb = log_emissions(X, p);
    const std::size_t T = X.rows, K = logb.cols;
    Matrix logA(K, K);
    for (std::size_t i = 0; i < K; ++i)
        for (std::size_t j = 0; j < K; ++j)
            logA(i, j) = p.transmat(i, j) > 0.0 ? std::log(p.transmat(i, j)) : kNegInf;
    std::vector<double> delta(K), next(K);
    for (std::size_t i = 0; i < K; ++i)
        delta[i] = (p.startprob[i] > 0.0 ? std::log(p.startprob[i]) : kNegInf) + logb(0, i);
    std::vector<std::vector<int>> psi(T, std::vector<int>(K, 0));
    for (std::size_t t = 1; t < T; ++t) {
        for (std::size_t j = 0; j < K; ++j) {
            int best_i = 0;
            double best = delta[0] + logA(0, j);
            for (std::size_t i = 1; i < K; ++i) {
                const double cand = delta[i] + logA(i, j);
                if (cand > best) {  // strict '>' keeps the lower index on ties
                    best = cand;
                    best_i = static_cast<int>(i);
                }
            }
            psi[t][j] = best_i;
            next[j] = best + logb(t, j);
        }
        delta = next;
    }
    std::vector<int> states(T);
    int last = 0;
    for (std::size_t i = 1; i < K; ++i)
        if (delta[i] > delta[static_cast<std::size_t>(last)]) last = static_cast<int>(i);
    states[T - 1] = last;
    for (std::size_t t = T - 1; t-- > 0;)
        states[t] = psi[t + 1][static_cast<std::size_t>(states[t + 1])];
    return states;
}

std::vector<double> GaussianHMM::stationary_distribution(double tol, int max_iter) const {
    const HMMParams& p = params();
    const std::size_t K = static_cast<std::size_t>(n_states_);
    std::vector<double> pi(K, 1.0 / static_cast<double>(K)), nxt(K);
    for (int it = 0; it < max_iter; ++it) {
        double total = 0.0;
        for (std::size_t j = 0; j < K; ++j) {
            double s = 0.0;
            for (std::size_t i = 0; i < K; ++i) s += pi[i] * p.transmat(i, j);
            nxt[j] = s;
            total += s;
        }
        double diff = 0.0;
        for (std::size_t j = 0; j < K; ++j) {
            nxt[j] /= total;
            diff = std::max(diff, std::abs(nxt[j] - pi[j]));
        }
        if (diff < tol) return nxt;
        pi = nxt;
    }
    return pi;
}

int GaussianHMM::n_parameters(int n_dims) const {
    const int K = n_states_;
    return 2 * K * n_dims + K * (K - 1) + (K - 1);
}

double GaussianHMM::aic(const Matrix& X) const {
    return -2.0 * score(X) + 2.0 * n_parameters(static_cast<int>(X.cols));
}

double GaussianHMM::bic(const Matrix& X) const {
    return -2.0 * score(X) +
           n_parameters(static_cast<int>(X.cols)) * std::log(static_cast<double>(X.rows));
}

std::vector<double> forward_step(const HMMParams& params,
                                 const std::vector<double>& alpha_prev,
                                 const std::vector<double>& x_t) {
    const std::size_t K = params.means.rows, D = params.means.cols;
    std::vector<double> logb(K);
    double mx = kNegInf;
    for (std::size_t i = 0; i < K; ++i) {
        double s = 0.0;
        for (std::size_t d = 0; d < D; ++d) {
            const double v = params.variances(i, d);
            const double diff = x_t[d] - params.means(i, d);
            s += kLog2Pi + std::log(v) + diff * diff / v;
        }
        logb[i] = -0.5 * s;
        mx = std::max(mx, logb[i]);
    }
    std::vector<double> a(K);
    double total = 0.0;
    for (std::size_t j = 0; j < K; ++j) {
        double s = 0.0;
        for (std::size_t i = 0; i < K; ++i) s += alpha_prev[i] * params.transmat(i, j);
        a[j] = s * std::exp(logb[j] - mx);
        total += a[j];
    }
    for (std::size_t j = 0; j < K; ++j) a[j] /= total;
    return a;
}

}  // namespace regime
