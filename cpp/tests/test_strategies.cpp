/// \file test_strategies.cpp
/// \brief Strategy tests: momentum sizing, carry schedule/tie-break, regime gate.

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <set>
#include <vector>

#include "regime/strategies.hpp"
#include "test_common.hpp"

using regime::Matrix;
using regime_tests::index_returns;

namespace {

Matrix constant_matrix(std::size_t T, std::vector<double> per_asset) {
    Matrix m(T, per_asset.size());
    for (std::size_t t = 0; t < T; ++t)
        for (std::size_t a = 0; a < per_asset.size(); ++a) m(t, a) = per_asset[a];
    return m;
}

std::vector<double> index_slice(std::size_t n) {
    const auto& r = index_returns();
    return std::vector<double>(r.begin(), r.begin() + static_cast<long>(n));
}

}  // namespace

TEST(Momentum, RejectsShortSeries) {
    // Exactly lookback long -> too short.
    Matrix r = constant_matrix(252, {0.001, 0.001});
    EXPECT_THROW(regime::momentum_positions(r, 252), std::invalid_argument);
}

TEST(Momentum, SignCorrectness) {
    // Steady up-trend asset goes long, down-trend short.
    Matrix r = constant_matrix(300, {0.001, -0.001});
    Matrix pos = regime::momentum_positions(r, 252);
    for (std::size_t t = 0; t < 251; ++t) {
        EXPECT_EQ(pos(t, 0), 0.0);  // undefined before first full window
        EXPECT_EQ(pos(t, 1), 0.0);
    }
    for (std::size_t t = 251; t < 300; ++t) {
        EXPECT_GT(pos(t, 0), 0.0);
        EXPECT_LT(pos(t, 1), 0.0);
    }
}

TEST(Momentum, VolTargetingCapsLeverage) {
    // A near-zero-vol series would want huge leverage; the cap binds.
    Matrix r = constant_matrix(300, {1e-6});
    Matrix pos = regime::momentum_positions(r, 252, 0.10, 0.94, 4.0);
    for (std::size_t t = 251; t < 300; ++t) EXPECT_NEAR(pos(t, 0), 4.0, 1e-12);
    // And with meaningful vol the target is hit instead of the cap.
    Matrix r2(300, 1);
    for (std::size_t t = 0; t < 300; ++t) r2(t, 0) = 0.02 * std::sin(static_cast<double>(t));
    Matrix pos2 = regime::momentum_positions(r2, 252);
    for (std::size_t t = 251; t < 300; ++t) EXPECT_LT(std::abs(pos2(t, 0)), 4.0);
}

TEST(Momentum, ScalingMatchesFormula) {
    // Position = sign * min(target/ann_vol, cap) / n_assets exactly.
    Matrix r = constant_matrix(260, {0.002, 0.003});
    const int lb = 252;
    Matrix pos = regime::momentum_positions(r, lb);
    Matrix sig2 = regime::ewma_variance(r, 0.94, lb);
    const std::size_t t = 255;
    for (std::size_t a = 0; a < 2; ++a) {
        const double ann_vol = std::sqrt(252.0 * sig2(t, a));
        const double expect = std::min(0.10 / ann_vol, 4.0) / 2.0;
        EXPECT_NEAR(pos(t, a), expect, 1e-15);
    }
}

TEST(Ewma, VarianceRecursion) {
    Matrix r(4, 1);
    r(0, 0) = 0.01;
    r(1, 0) = -0.02;
    r(2, 0) = 0.005;
    r(3, 0) = 0.03;
    Matrix sig2 = regime::ewma_variance(r, 0.9, 2);
    const double seed = (0.01 * 0.01 + 0.02 * 0.02) / 2.0;
    EXPECT_TRUE(std::isnan(sig2(0, 0)));
    EXPECT_NEAR(sig2(1, 0), seed, 1e-16);
    EXPECT_NEAR(sig2(2, 0), 0.9 * seed + 0.1 * 0.005 * 0.005, 1e-16);
    EXPECT_NEAR(sig2(3, 0), 0.9 * sig2(2, 0) + 0.1 * 0.03 * 0.03, 1e-16);
}

TEST(Carry, TieBreakByAssetIndex) {
    // All-equal differentials: longs are assets 0..2, shorts 3..5 (pinned).
    Matrix d(10, 6, 0.02);
    Matrix pos = regime::carry_positions(d, 3, 3, 21);
    for (std::size_t a = 0; a < 3; ++a) EXPECT_NEAR(pos(0, a), 1.0 / 3.0, 1e-15);
    for (std::size_t a = 3; a < 6; ++a) EXPECT_NEAR(pos(0, a), -1.0 / 3.0, 1e-15);
}

TEST(Carry, RankingAndWeights) {
    Matrix d = constant_matrix(5, {0.01, 0.05, -0.02, 0.03, 0.00, 0.02});
    Matrix pos = regime::carry_positions(d);
    // Descending diff order: 1, 3, 5, 0, 4, 2.
    const double e[6] = {-1.0 / 3, 1.0 / 3, -1.0 / 3, 1.0 / 3, -1.0 / 3, 1.0 / 3};
    double net = 0.0, gross = 0.0;
    for (std::size_t a = 0; a < 6; ++a) {
        EXPECT_NEAR(pos(0, a), e[a], 1e-15);
        net += pos(0, a);
        gross += std::abs(pos(0, a));
    }
    EXPECT_NEAR(net, 0.0, 1e-15);
    EXPECT_NEAR(gross, 2.0, 1e-15);
}

TEST(Carry, RebalanceSchedule) {
    // Positions change only at t = 0, 21, 42, ... (pinned schedule).
    const std::size_t T = 70;
    std::mt19937 gen(7);  // test data only; strategy code is RNG-free
    std::normal_distribution<double> nd(0.0, 0.02);
    Matrix d(T, 6);
    for (double& v : d.data) v = nd(gen);
    Matrix pos = regime::carry_positions(d, 3, 3, 21);
    std::set<std::size_t> changes;
    for (std::size_t t = 1; t < T; ++t)
        for (std::size_t a = 0; a < 6; ++a)
            if (pos(t, a) != pos(t - 1, a)) changes.insert(t);
    for (std::size_t c : changes) EXPECT_TRUE(c == 21 || c == 42 || c == 63) << "change at " << c;
    for (std::size_t a = 0; a < 6; ++a) {
        EXPECT_EQ(pos(1, a), pos(0, a));
        EXPECT_EQ(pos(21, a), pos(41, a));
    }
}

TEST(Carry, SingleRebalanceWhenWindowExceedsSeries) {
    // rebalance_days > T: only the day-0 rebalance happens, held throughout.
    Matrix d = constant_matrix(10, {0.03, 0.02, 0.01, 0.0, -0.01, -0.02});
    Matrix pos = regime::carry_positions(d, 3, 3, 500);
    for (std::size_t t = 1; t < 10; ++t)
        for (std::size_t a = 0; a < 6; ++a) EXPECT_EQ(pos(t, a), pos(0, a));
}

TEST(Carry, Validation) {
    Matrix d(10, 4, 1.0);
    EXPECT_THROW(regime::carry_positions(d, 3, 3), std::invalid_argument);  // 6 > 4 assets
    EXPECT_THROW(regime::carry_positions(d, 1, 1, 0), std::invalid_argument);
    EXPECT_THROW(regime::carry_positions(Matrix(0, 4)), std::invalid_argument);  // empty
}

TEST(Carry, TotalReturnsAccrual) {
    Matrix spot(2, 2), diff(2, 2);
    spot(0, 0) = 0.01;
    spot(0, 1) = 0.0;
    spot(1, 0) = 0.005;
    spot(1, 1) = -0.01;
    diff(0, 0) = 0.0252;
    diff(0, 1) = 0.0504;
    diff(1, 0) = 0.0252;
    diff(1, 1) = 0.0504;
    Matrix tot = regime::carry_total_returns(spot, diff);
    EXPECT_EQ(tot(0, 0), spot(0, 0));  // day 0: no accrual
    EXPECT_EQ(tot(0, 1), spot(0, 1));
    EXPECT_NEAR(tot(1, 0), 0.005 + 0.0252 / 252.0, 1e-15);
    EXPECT_NEAR(tot(1, 1), -0.01 + 0.0504 / 252.0, 1e-15);
}

TEST(RegimeGate, Basics) {
    // Gate is 1 before the first fit, within [0,1] after, deterministic.
    const std::vector<double> r = index_slice(700);
    auto res = regime::regime_gate(r, 3, 400, 200);
    for (std::size_t t = 0; t < 399; ++t) EXPECT_EQ(res.gate[t], 1.0);
    for (double g : res.gate) {
        EXPECT_GE(g, 0.0);
        EXPECT_LE(g, 1.0);
    }
    EXPECT_EQ(res.models.size(), 2u);  // fits at t=399 and t=599
    auto res2 = regime::regime_gate(r, 3, 400, 200);
    EXPECT_EQ(res.gate, res2.gate);
}

TEST(RegimeGate, BinaryMode) {
    const std::vector<double> r = index_slice(600);
    auto res = regime::regime_gate(r, 3, 400, 300, "binary", 0.5);
    for (double g : res.gate) EXPECT_TRUE(g == 0.0 || g == 1.0);
}

TEST(RegimeGate, Validation) {
    EXPECT_THROW(regime::regime_gate(index_slice(100), 3, 400), std::invalid_argument);
    EXPECT_THROW(regime::regime_gate(index_slice(600), 3, 400, 63, "magic"),
                 std::invalid_argument);
}

TEST(RegimeGate, ApplyGate) {
    Matrix pos(5, 2, 1.0);
    const std::vector<double> gate = {1.0, 0.5, 0.0, 0.25, 1.0};
    Matrix out = regime::apply_gate(pos, gate);
    for (std::size_t t = 0; t < 5; ++t) EXPECT_NEAR(out(t, 0), gate[t], 1e-15);
    EXPECT_THROW(regime::apply_gate(pos, {1.0, 0.5, 0.0}), std::invalid_argument);
}
