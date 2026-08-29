/// \file strategies.cpp
/// \brief Momentum, FX carry, and the walk-forward HMM regime gate.

#include "regime/strategies.hpp"

#include "regime/backtest.hpp"  // kTradingDays

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace regime {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// Validate a (T x A) return-like matrix: non-empty and all finite.
void validate_matrix(const Matrix& r, const char* name) {
    if (r.rows == 0 || r.cols == 0)
        throw std::invalid_argument(std::string(name) + " must be a non-empty 2-D array");
    for (double v : r.data)
        if (!std::isfinite(v)) throw std::invalid_argument(std::string(name) + " contain NaN or inf");
}

/// sign(x) in {-1, 0, +1}.
double sign(double x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }

}  // namespace

Matrix ewma_variance(const Matrix& returns, double lam, int init_window) {
    validate_matrix(returns, "returns");
    const std::size_t T = returns.rows, A = returns.cols;
    if (!(lam > 0.0 && lam < 1.0))
        throw std::invalid_argument("ewma lambda must be in (0, 1), got " + std::to_string(lam));
    if (init_window < 1 || static_cast<std::size_t>(init_window) > T)
        throw std::invalid_argument("init_window must be in [1, T], got " +
                                    std::to_string(init_window));
    const std::size_t w = static_cast<std::size_t>(init_window);
    Matrix sig2(T, A, kNaN);
    for (std::size_t a = 0; a < A; ++a) {
        double seed = 0.0;
        for (std::size_t t = 0; t < w; ++t) seed += returns(t, a) * returns(t, a);
        seed /= static_cast<double>(w);  // zero-mean convention
        sig2(w - 1, a) = seed;
        for (std::size_t t = w; t < T; ++t)
            sig2(t, a) = lam * sig2(t - 1, a) + (1.0 - lam) * returns(t, a) * returns(t, a);
    }
    return sig2;
}

Matrix momentum_positions(const Matrix& returns, int lookback, double vol_target,
                          double ewma_lambda, double leverage_cap) {
    validate_matrix(returns, "returns");
    const std::size_t T = returns.rows, A = returns.cols;
    if (lookback < 1)
        throw std::invalid_argument("lookback must be >= 1, got " + std::to_string(lookback));
    if (T <= static_cast<std::size_t>(lookback))
        throw std::invalid_argument("series shorter than momentum lookback: T=" +
                                    std::to_string(T) + " <= lookback=" + std::to_string(lookback));
    if (vol_target <= 0.0 || leverage_cap <= 0.0)
        throw std::invalid_argument("vol_target and leverage_cap must be > 0");
    const std::size_t lb = static_cast<std::size_t>(lookback);

    // Trailing cumulative return via cumulative growth ratios:
    // cumret[t] = growth[t] / growth[t - lookback] - 1 (growth[t] = prod(1+r)).
    Matrix growth(T, A);
    for (std::size_t a = 0; a < A; ++a) {
        double g = 1.0;
        for (std::size_t t = 0; t < T; ++t) {
            g *= 1.0 + returns(t, a);
            growth(t, a) = g;
        }
    }
    Matrix sig2 = ewma_variance(returns, ewma_lambda, lookback);

    Matrix pos(T, A, 0.0);
    for (std::size_t t = lb - 1; t < T; ++t) {
        for (std::size_t a = 0; a < A; ++a) {
            const double cumret = t == lb - 1 ? growth(t, a) - 1.0
                                              : growth(t, a) / growth(t - lb, a) - 1.0;
            const double ann_vol = std::sqrt(kTradingDays * sig2(t, a));
            const double lev = ann_vol > 0.0 ? std::min(vol_target / ann_vol, leverage_cap)
                                             : leverage_cap;
            pos(t, a) = sign(cumret) * lev / static_cast<double>(A);
        }
    }
    return pos;
}

Matrix carry_positions(const Matrix& rate_diffs, int top_n, int bottom_n,
                       int rebalance_days) {
    validate_matrix(rate_diffs, "rate differentials");
    const std::size_t T = rate_diffs.rows, A = rate_diffs.cols;
    if (top_n < 1 || bottom_n < 1 ||
        static_cast<std::size_t>(top_n) + static_cast<std::size_t>(bottom_n) > A)
        throw std::invalid_argument("need top_n>=1, bottom_n>=1, top_n+bottom_n<=A=" +
                                    std::to_string(A) + "; got " + std::to_string(top_n) + "," +
                                    std::to_string(bottom_n));
    if (rebalance_days < 1)
        throw std::invalid_argument("rebalance_days must be >= 1, got " +
                                    std::to_string(rebalance_days));

    Matrix pos(T, A, 0.0);
    std::vector<double> current(A, 0.0);
    std::vector<std::size_t> order(A);
    for (std::size_t t = 0; t < T; ++t) {
        if (t % static_cast<std::size_t>(rebalance_days) == 0) {
            // Stable sort on descending differential; asset index breaks ties.
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(),
                             [&rate_diffs, t](std::size_t a, std::size_t b) {
                                 return rate_diffs(t, a) > rate_diffs(t, b);
                             });
            std::fill(current.begin(), current.end(), 0.0);
            for (int i = 0; i < top_n; ++i)
                current[order[static_cast<std::size_t>(i)]] = 1.0 / top_n;
            for (int i = 0; i < bottom_n; ++i)
                current[order[A - 1 - static_cast<std::size_t>(i)]] = -1.0 / bottom_n;
        }
        for (std::size_t a = 0; a < A; ++a) pos(t, a) = current[a];
    }
    return pos;
}

Matrix carry_total_returns(const Matrix& spot_returns, const Matrix& rate_diffs) {
    validate_matrix(spot_returns, "spot returns");
    validate_matrix(rate_diffs, "rate differentials");
    if (spot_returns.rows != rate_diffs.rows || spot_returns.cols != rate_diffs.cols)
        throw std::invalid_argument("spot returns and differentials must have equal shape");
    Matrix total = spot_returns;
    for (std::size_t t = 1; t < total.rows; ++t)
        for (std::size_t a = 0; a < total.cols; ++a)
            total(t, a) += rate_diffs(t - 1, a) / kTradingDays;
    return total;
}

RegimeGateResult regime_gate(const std::vector<double>& index_returns, int n_states,
                             int train_min_days, int refit_days, const std::string& mode,
                             double threshold, double tol, int max_iter, double var_floor) {
    for (double v : index_returns)
        if (!std::isfinite(v)) throw std::invalid_argument("index_returns contain NaN or inf");
    if (mode != "prob" && mode != "binary")
        throw std::invalid_argument("gate mode must be 'prob' or 'binary', got '" + mode + "'");
    const std::size_t T = index_returns.size();
    if (train_min_days <= n_states || refit_days < 1)
        throw std::invalid_argument(
            "train_min_days must exceed n_states and refit_days must be >= 1");
    if (T < static_cast<std::size_t>(train_min_days))
        throw std::invalid_argument("series shorter than train_min_days: T=" + std::to_string(T) +
                                    " < " + std::to_string(train_min_days));

    const std::size_t t0 = static_cast<std::size_t>(train_min_days) - 1;
    RegimeGateResult out;
    out.gate.assign(T, 1.0);
    std::vector<double> alpha;
    std::size_t calm_state = 0;
    for (std::size_t t = t0; t < T; ++t) {
        if ((t - t0) % static_cast<std::size_t>(refit_days) == 0) {
            GaussianHMM model(n_states, tol, max_iter, var_floor);
            Matrix window(t + 1, 1);
            for (std::size_t u = 0; u <= t; ++u) window(u, 0) = index_returns[u];
            if (out.models.empty()) {
                model.fit(window);
            } else {
                model.fit(window, out.models.back().params());  // warm start
            }
            // Pinned gate state: lowest variance on dimension 0 (ties -> lower label).
            const Matrix& vars = model.params().variances;
            calm_state = 0;
            for (std::size_t k = 1; k < vars.rows; ++k)
                if (vars(k, 0) < vars(calm_state, 0)) calm_state = k;
            Matrix filt = model.filtered_probabilities(window);
            alpha.assign(filt.cols, 0.0);
            for (std::size_t k = 0; k < filt.cols; ++k) alpha[k] = filt(t, k);
            out.models.push_back(std::move(model));
        } else {
            alpha = forward_step(out.models.back().params(), alpha, {index_returns[t]});
        }
        const double p_calm = alpha[calm_state];
        out.gate[t] = mode == "prob" ? p_calm : (p_calm >= threshold ? 1.0 : 0.0);
    }
    return out;
}

Matrix apply_gate(const Matrix& positions, const std::vector<double>& gate) {
    validate_matrix(positions, "positions");
    if (gate.size() != positions.rows)
        throw std::invalid_argument("gate length does not match positions");
    for (double g : gate)
        if (!std::isfinite(g)) throw std::invalid_argument("gate contains NaN or inf");
    Matrix out = positions;
    for (std::size_t t = 0; t < out.rows; ++t)
        for (std::size_t a = 0; a < out.cols; ++a) out(t, a) *= gate[t];
    return out;
}

}  // namespace regime
