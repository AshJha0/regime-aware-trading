/// \file test_backtest.cpp
/// \brief Backtest engine tests: pinned accounting, no-lookahead, metrics.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "regime/backtest.hpp"

using regime::Matrix;

namespace {

Matrix column(std::vector<double> v) {
    Matrix m(v.size(), 1);
    for (std::size_t t = 0; t < v.size(); ++t) m(t, 0) = v[t];
    return m;
}

}  // namespace

TEST(Backtest, AccountingIdentity) {
    // net = gross - cost*turnover and equity = cumprod(1+net), exactly.
    Matrix P = column({1.0, 0.5, -0.5, -0.5});
    Matrix r = column({0.01, 0.02, -0.01, 0.03});
    auto res = regime::run_backtest(P, r, 10.0);
    const double cost = 10.0 / 1e4;
    const double eg[4] = {0.0, 1.0 * 0.02, 0.5 * -0.01, -0.5 * 0.03};
    const double eto[4] = {1.0, 0.5, 1.0, 0.0};
    double eq = 1.0;
    for (std::size_t t = 0; t < 4; ++t) {
        EXPECT_NEAR(res.gross_returns[t], eg[t], 1e-15);
        EXPECT_NEAR(res.turnover[t], eto[t], 1e-15);
        EXPECT_NEAR(res.net_returns[t], eg[t] - cost * eto[t], 1e-15);
        eq *= 1.0 + res.net_returns[t];
        EXPECT_NEAR(res.equity[t], eq, 1e-15);
    }
}

TEST(Backtest, NoLookaheadShift) {
    // Day-t P&L must come from the day t-1 position; using the same-day
    // position (lookahead) gives a measurably different result.
    std::mt19937 gen(3);  // test data only
    std::normal_distribution<double> nd(0.0, 0.01);
    const std::size_t T = 300;
    Matrix r(T, 1), P(T, 1);
    for (std::size_t t = 0; t < T; ++t) {
        r(t, 0) = nd(gen);
        P(t, 0) = r(t, 0) > 0.0 ? 1.0 : (r(t, 0) < 0.0 ? -1.0 : 0.0);  // foresight
    }
    auto res = regime::run_backtest(P, r, 0.0);
    double lookahead_pnl = 0.0, honest_pnl = 0.0, shifted = 0.0;
    for (std::size_t t = 0; t < T; ++t) {
        lookahead_pnl += P(t, 0) * r(t, 0);
        honest_pnl += res.gross_returns[t];
        if (t + 1 < T) shifted += P(t, 0) * r(t + 1, 0);
    }
    EXPECT_GT(std::abs(lookahead_pnl - honest_pnl), 0.1);  // foresight is very profitable
    EXPECT_NEAR(honest_pnl, shifted, 1e-15);  // honest accounting == explicit shift
}

TEST(Backtest, ZeroCostPath) {
    Matrix P = column({1.0, 1.0, -1.0});
    Matrix r = column({0.01, 0.02, 0.03});
    auto res = regime::run_backtest(P, r, 0.0);
    EXPECT_EQ(res.net_returns, res.gross_returns);
}

TEST(Backtest, TurnoverFirstDayIsPositionBuild) {
    Matrix P(2, 2);
    P(0, 0) = 0.5;
    P(0, 1) = -0.5;
    P(1, 0) = 0.5;
    P(1, 1) = -0.5;
    Matrix r(2, 2, 0.0);
    auto res = regime::run_backtest(P, r);
    EXPECT_NEAR(res.turnover[0], 1.0, 1e-15);
    EXPECT_NEAR(res.turnover[1], 0.0, 1e-15);
}

TEST(Backtest, MaxDrawdownKnownCase) {
    EXPECT_NEAR(regime::max_drawdown({1.0, 1.2, 0.6, 0.9, 1.3}), 0.6 / 1.2 - 1.0, 1e-15);
    EXPECT_NEAR(regime::max_drawdown({1.0, 1.1, 1.2}), 0.0, 1e-15);
}

TEST(Backtest, MetricsFormulas) {
    const std::vector<double> net = {0.01, -0.005, 0.02, 0.0, -0.01};
    auto m = regime::compute_metrics(net);
    double mean = 0.0;
    for (double x : net) mean += x;
    mean /= 5.0;
    double var = 0.0;
    for (double x : net) var += (x - mean) * (x - mean);
    var /= 5.0;
    EXPECT_NEAR(m.ann_return, 252.0 * mean, 1e-14);
    EXPECT_NEAR(m.ann_vol, std::sqrt(252.0 * var), 1e-14);
    EXPECT_NEAR(m.sharpe, m.ann_return / m.ann_vol, 1e-14);
    EXPECT_NEAR(m.hit_rate, 2.0 / 5.0, 1e-15);
    EXPECT_LE(m.max_dd, 0.0);
    // Zero-vol series: sharpe defined as 0, no division error.
    auto z = regime::compute_metrics(std::vector<double>(10, 0.0));
    EXPECT_EQ(z.sharpe, 0.0);
    EXPECT_EQ(z.calmar, 0.0);
}

TEST(Backtest, WorstMonthWithDates) {
    // Three synthetic months of 21 business days each; the middle month is bad.
    std::vector<std::string> dates;
    std::vector<double> net(63, 0.0);
    const char* months[3] = {"2020-01", "2020-02", "2020-03"};
    for (int m = 0; m < 3; ++m)
        for (int d = 1; d <= 21; ++d) {
            char buf[16];
            std::snprintf(buf, sizeof(buf), "%s-%02d", months[m], d);
            dates.push_back(buf);
        }
    for (int t = 21; t < 42; ++t) net[static_cast<std::size_t>(t)] = -0.01;
    auto m = regime::compute_metrics(net, &dates);
    const double feb = std::pow(0.99, 21) - 1.0;
    EXPECT_NEAR(m.worst_month, feb, 1e-12);
    EXPECT_LT(m.worst_month, -0.05);
}

TEST(Backtest, StateConditionalReturnsPartition) {
    const std::vector<double> net = {0.01, -0.02, 0.03, 0.0, -0.01, 0.02};
    const std::vector<int> states = {0, 0, 1, 1, 2, 2};
    auto out = regime::state_conditional_returns(net, states, 3);
    int total = 0;
    for (const auto& s : out) total += s.n_days;
    EXPECT_EQ(total, 6);
    EXPECT_NEAR(out[0].ann_return, 252.0 * (0.01 - 0.02) / 2.0, 1e-14);
    auto empty = regime::state_conditional_returns(net, std::vector<int>(6, 0), 2);
    EXPECT_EQ(empty[1].n_days, 0);
    EXPECT_EQ(empty[1].sharpe, 0.0);
}

TEST(Backtest, Validation) {
    EXPECT_THROW(regime::run_backtest(Matrix(3, 1, 1.0), Matrix(4, 1, 1.0)),
                 std::invalid_argument);  // shape mismatch
    EXPECT_THROW(regime::run_backtest(Matrix(3, 1, 1.0), Matrix(3, 1, 1.0), -1.0),
                 std::invalid_argument);  // negative cost
    Matrix nanP(1, 1);
    nanP(0, 0) = std::nan("");
    EXPECT_THROW(regime::run_backtest(nanP, Matrix(1, 1, 0.0)), std::invalid_argument);
    EXPECT_THROW(regime::run_backtest(Matrix(0, 1), Matrix(0, 1)), std::invalid_argument);
    EXPECT_THROW(regime::compute_metrics({}), std::invalid_argument);
    std::vector<std::string> dates = {"2020-01-01"};
    EXPECT_THROW(regime::run_backtest(Matrix(3, 1, 1.0), Matrix(3, 1, 1.0), 0.0, &dates),
                 std::invalid_argument);  // dates length mismatch
}

TEST(Backtest, PropertyCostsNeverHelp) {
    // Grid property: for any cost level, higher costs never raise the
    // total net return (turnover is non-negative).
    std::mt19937 gen(5);  // test data only
    std::normal_distribution<double> nd(0.0, 1.0);
    Matrix P(100, 3), r(100, 3);
    for (double& v : P.data) v = nd(gen);
    for (double& v : r.data) v = 0.01 * nd(gen);
    double prev_total = std::numeric_limits<double>::infinity();
    for (double bps : {0.0, 1.0, 5.0, 10.0, 25.0, 50.0}) {
        auto res = regime::run_backtest(P, r, bps);
        double total = 0.0;
        for (double x : res.net_returns) total += x;
        EXPECT_LE(total, prev_total + 1e-12);
        prev_total = total;
    }
}

// ------------------------------------------------------------------------- //
// Robustness: wipe-outs, dates, degenerate shapes
// ------------------------------------------------------------------------- //

TEST(Backtest, WipeoutIsError) {
    // PT-5: 4x leverage into a -30% day is a -120% net day -> error naming t.
    const Matrix P = column({4.0, 4.0, 4.0});
    const Matrix r = column({0.0, -0.3, 0.1});
    try {
        regime::run_backtest(P, r, 0.0);
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find("wiped out on day t=1"), std::string::npos);
    }
    EXPECT_THROW(regime::compute_metrics({0.01, -1.0, 0.02}), std::invalid_argument);
    // exactly -100% net on day 0 via costs alone is a wipe-out too
    EXPECT_THROW(regime::run_backtest(Matrix(1, 1, 1.0), Matrix(1, 1, 0.0), 1e4),
                 std::invalid_argument);
    // invariant on accepted input: max_dd >= -1 and equity > 0
    const auto res = regime::run_backtest(P, column({0.0, -0.2, 0.1}));
    EXPECT_GE(res.metrics.max_dd, -1.0);
    for (double e : res.equity) EXPECT_GT(e, 0.0);
    EXPECT_NEAR(res.equity[1], 0.2, 1e-15);
}

TEST(Backtest, RejectsReturnsBelowMinusOne) {
    const Matrix P(3, 1, 1.0);
    Matrix r = column({0.0, -1.0, 0.0});
    EXPECT_THROW(regime::run_backtest(P, r), std::invalid_argument);
    r(1, 0) = -1.5;
    EXPECT_THROW(regime::run_backtest(P, r), std::invalid_argument);
}

TEST(Backtest, ComputeMetricsDatesLengthMismatch) {
    // PT-7 / MAJ-5: the public metrics function validates dates itself.
    const std::vector<std::string> three = {"2020-01-01", "2020-01-02", "2020-01-03"};
    EXPECT_THROW(regime::compute_metrics(std::vector<double>(5, 0.0), &three), std::invalid_argument);
    EXPECT_THROW(regime::run_backtest(Matrix(5, 1, 1.0), Matrix(5, 1, 0.0), 0.0, &three),
                 std::invalid_argument);
}

TEST(Backtest, DatesMustBeStrictlyIncreasing) {
    // PT-8 / MAJ-6: duplicated, out-of-order or malformed dates are rejected.
    const Matrix P(3, 1, 1.0), r(3, 1, 0.0);
    const std::vector<std::vector<std::string>> bad = {
        {"2020-01-02", "2020-01-02", "2020-01-03"},
        {"2020-02-03", "2020-01-02", "2020-01-03"},
        {"2020-1-5", "2020-01-06", "2020-01-07"},   // not YYYY-MM-DD
        {"2020-13-01", "2020-13-02", "2020-13-03"}, // month 13
    };
    for (const auto& dates : bad) {
        EXPECT_THROW(regime::run_backtest(P, r, 0.0, &dates), std::invalid_argument);
        EXPECT_THROW(regime::compute_metrics(std::vector<double>(3, 0.0), &dates),
                     std::invalid_argument);
    }
    // sorted but non-contiguous months: pinned "contiguous runs" == calendar months
    const std::vector<std::string> dates = {"2020-01-02", "2020-01-31", "2020-03-02"};
    const std::vector<double> net = {-0.01, -0.02, 0.05};
    const auto m = regime::compute_metrics(net, &dates);
    EXPECT_NEAR(m.worst_month, 0.99 * 0.98 - 1.0, 1e-15);
}

TEST(Backtest, WorstMonthBlockFallback) {
    // PT-11: without dates, consecutive 21-day blocks; trailing partial dropped.
    std::vector<double> net(45, 0.0);
    for (std::size_t t = 21; t < 42; ++t) net[t] = -0.01;
    net[43] = -0.5;  // in the dropped tail; must not count
    const auto m = regime::compute_metrics(net);
    EXPECT_NEAR(m.worst_month, std::pow(0.99, 21) - 1.0, 1e-12);
    EXPECT_NEAR(regime::compute_metrics(std::vector<double>(5, -0.01)).worst_month,
                std::pow(0.99, 5) - 1.0, 1e-15);
}

TEST(Backtest, RejectsZeroAssets) {
    // MAJ-7: A = 0 columns is an error in every language.
    EXPECT_THROW(regime::run_backtest(Matrix(3, 0), Matrix(3, 0)), std::invalid_argument);
}

TEST(Backtest, MaxDrawdownRequiresPositiveEquity) {
    EXPECT_THROW(regime::max_drawdown({1.0, 0.0, 0.5}), std::invalid_argument);
    EXPECT_THROW(regime::max_drawdown({1.0, -0.2}), std::invalid_argument);
    EXPECT_THROW(regime::max_drawdown({1.0, std::numeric_limits<double>::infinity()}),
                 std::invalid_argument);
}

TEST(Backtest, StateConditionalReturnsValidation) {
    // MIN-13: labels outside [0, K) and K < 1 are errors, never dropped.
    const std::vector<double> net(4, 0.0);
    EXPECT_THROW(regime::state_conditional_returns(net, {0, 1, 2, 3}, 3), std::invalid_argument);
    EXPECT_THROW(regime::state_conditional_returns(net, {0, -1, 0, 0}, 3), std::invalid_argument);
    EXPECT_THROW(regime::state_conditional_returns(net, {0, 0, 0, 0}, 0), std::invalid_argument);
    EXPECT_THROW(regime::state_conditional_returns({0.0, std::nan(""), 0.0, 0.0}, {0, 0, 0, 0}, 1),
                 std::invalid_argument);
}
