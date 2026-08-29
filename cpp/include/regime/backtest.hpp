#ifndef REGIME_BACKTEST_HPP
#define REGIME_BACKTEST_HPP

/// \file backtest.hpp
/// \brief Backtest engine with pinned no-lookahead accounting and metrics.
///
/// Accounting (API_SPEC section 2): positions P[t] are decided at the close
/// of day t, so day-t P&L is earned by yesterday's position while the cost of
/// trading into P[t] is charged on day t itself:
///
///     gross[t]    = sum_a P[t-1,a] * r[t,a]     (gross[0] = 0)
///     turnover[t] = sum_a |P[t,a] - P[t-1,a]|   (P[-1] = 0)
///     net[t]      = gross[t] - (cost_bps/1e4) * turnover[t]
///     equity[t]   = prod_{u<=t} (1 + net[u])

#include <map>
#include <string>
#include <vector>

#include "regime/matrix.hpp"

namespace regime {

/// Days per year used by every annualization in the project.
constexpr int kTradingDays = 252;

/// Pinned summary metrics of a daily net-return series (API_SPEC 2.1).
struct Metrics {
    double ann_return{0.0};   ///< 252 * mean(net).
    double ann_vol{0.0};      ///< sqrt(252) * std(net, ddof=0).
    double sharpe{0.0};       ///< ann_return / ann_vol (0 if ann_vol == 0).
    double max_dd{0.0};       ///< min_t (equity/cummax(equity) - 1), <= 0.
    double calmar{0.0};       ///< ann_return / |max_dd| (0 if max_dd == 0).
    double hit_rate{0.0};     ///< Fraction of days with net > 0.
    double worst_month{0.0};  ///< Worst compounded calendar month (or 21-day block).
};

/// Daily series and summary metrics of one backtest run.
struct BacktestResult {
    std::vector<double> gross_returns;  ///< Pre-cost daily portfolio returns.
    std::vector<double> net_returns;    ///< Post-cost daily returns.
    std::vector<double> turnover;       ///< Daily one-sided turnover sum|dP|.
    std::vector<double> equity;         ///< Compounded equity curve.
    Metrics metrics;                    ///< Pinned summary metrics.
};

/// Annualized per-state performance for the crisis table (API_SPEC 2.2).
struct StateStats {
    int n_days{0};
    double ann_return{0.0};
    double ann_vol{0.0};
    double sharpe{0.0};
};

/// Maximum drawdown of an equity curve, as a non-positive fraction.
/// \throws std::invalid_argument on an empty curve.
double max_drawdown(const std::vector<double>& equity);

/// Pinned summary metrics from a daily net-return series.
/// \param net   Daily net returns.
/// \param dates Optional ISO dates (YYYY-MM-DD, length T) for the
///              worst-calendar-month metric; when null, worst_month uses
///              consecutive 21-day blocks instead.
/// \throws std::invalid_argument on empty or non-finite input.
Metrics compute_metrics(const std::vector<double>& net,
                        const std::vector<std::string>* dates = nullptr);

/// Run the pinned accounting on a position/return pair.
/// \param positions P[t,a] decided at close of day t, shape (T x A).
/// \param returns   Asset simple returns r[t,a], same shape.
/// \param cost_bps  One-way transaction cost in basis points of turnover.
/// \param dates     Optional dates (length T) for calendar-month metrics.
/// \throws std::invalid_argument on shape mismatch, non-finite input,
///         negative cost, or dates length mismatch.
BacktestResult run_backtest(const Matrix& positions, const Matrix& returns,
                            double cost_bps = 0.0,
                            const std::vector<std::string>* dates = nullptr);

/// Annualized performance of a return series conditioned on a state path:
/// day-t net return is attributed to the day-t decoded state.
/// \throws std::invalid_argument on length mismatch.
std::vector<StateStats> state_conditional_returns(const std::vector<double>& net,
                                                  const std::vector<int>& states,
                                                  int n_states);

}  // namespace regime

#endif  // REGIME_BACKTEST_HPP
