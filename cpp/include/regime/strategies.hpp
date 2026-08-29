#ifndef REGIME_STRATEGIES_HPP
#define REGIME_STRATEGIES_HPP

/// \file strategies.hpp
/// \brief Time-series momentum, FX carry, and the walk-forward HMM regime gate.
///
/// No-lookahead convention (shared with the backtest engine): every position
/// matrix P produced here has P[t] = the position decided at the close of day
/// t using information up to and including day t.  The backtest applies
/// P[t-1] to the day-t return, so a signal can never see the return it is
/// traded against.  The regime gate at day t likewise uses only observations
/// x_0..x_t (filtered, not smoothed, probabilities) and models fitted on data
/// through day t.

#include <string>
#include <vector>

#include "regime/hmm.hpp"
#include "regime/matrix.hpp"

namespace regime {

/// EWMA variance per asset with the pinned initialization: at
/// t = init_window - 1 the variance is the mean of squared returns over the
/// first init_window days (zero-mean convention); afterwards
/// sig2[t] = lam * sig2[t-1] + (1 - lam) * r[t]^2.  Entries before
/// init_window - 1 are NaN (undefined).
/// \throws std::invalid_argument on invalid lambda/window or non-finite input.
Matrix ewma_variance(const Matrix& returns, double lam, int init_window);

/// Time-series momentum positions with vol-targeted sizing (API_SPEC 3.1).
/// Signal at day t (defined for t >= lookback - 1) is the sign of the
/// trailing cumulative simple return; sizing is
/// min(vol_target / ann_vol, leverage_cap) per asset (cap when ann_vol == 0),
/// divided by the asset count.  Positions are zero before the first full
/// lookback window.
/// \throws std::invalid_argument if T <= lookback or parameters are invalid.
Matrix momentum_positions(const Matrix& returns, int lookback = 252,
                          double vol_target = 0.10, double ewma_lambda = 0.94,
                          double leverage_cap = 4.0);

/// FX carry positions (API_SPEC 3.2).  Rebalances at day indices
/// 0, rebalance_days, 2*rebalance_days, ...; ranks currencies by that day's
/// annualized rate differential descending with ties broken by ascending
/// asset index; top_n get +1/top_n each, bottom_n get -1/bottom_n each.
/// \throws std::invalid_argument if top_n + bottom_n exceeds the asset count
///         or the schedule/params are invalid.
Matrix carry_positions(const Matrix& rate_diffs, int top_n = 3, int bottom_n = 3,
                       int rebalance_days = 21);

/// Total currency returns: spot move plus accrued carry (pinned accrual):
/// day-t total return is spot[t] + diff[t-1] / 252 (day 0 accrues nothing).
/// \throws std::invalid_argument on shape mismatch or non-finite input.
Matrix carry_total_returns(const Matrix& spot_returns, const Matrix& rate_diffs);

/// Result of the walk-forward regime gate.
struct RegimeGateResult {
    std::vector<double> gate;        ///< Gate values in [0, 1], length T.
    std::vector<GaussianHMM> models; ///< Fitted models in refit order.
};

/// Walk-forward HMM regime gate on a market proxy (API_SPEC 3.3).
///
/// First fit at t0 = train_min_days - 1 on the expanding window r[0..t0]
/// using the pinned initialization; refits at t0 + refit_days*k warm-start
/// from the previous fitted (label-sorted) parameters.  The gate at day t is
/// based on the filtered probability of the low-vol state
/// k* = argmin_k variances[k, 0] under the model most recently fitted at or
/// before t; a full scaled forward pass runs at each refit day and single
/// forward_step updates in between.  Before the first fit the gate is 1.
///
/// \param mode "prob" (gate = p_calm) or "binary" (1 if p_calm >= threshold).
/// \throws std::invalid_argument if the series is shorter than
///         train_min_days, the mode is unknown, or inputs are invalid.
RegimeGateResult regime_gate(const std::vector<double>& index_returns,
                             int n_states = 3, int train_min_days = 504,
                             int refit_days = 63, const std::string& mode = "prob",
                             double threshold = 0.5, double tol = 1e-8,
                             int max_iter = 500, double var_floor = 1e-8);

/// Scale each day's positions by that day's gate value:
/// P_filtered[t] = gate[t] * P[t] (both day-t decisions, so the filtered
/// position still only trades against the day t+1 return).
/// \throws std::invalid_argument on length mismatch or non-finite gate.
Matrix apply_gate(const Matrix& positions, const std::vector<double>& gate);

}  // namespace regime

#endif  // REGIME_STRATEGIES_HPP
