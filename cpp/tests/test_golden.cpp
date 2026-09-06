/// \file test_golden.cpp
/// \brief Cross-language golden suite: every case in data/golden/golden.json
///        must match the pinned pipeline within its tolerance.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "mini_json.hpp"
#include "regime/backtest.hpp"
#include "regime/strategies.hpp"
#include "test_common.hpp"

using regime_tests::full_pipeline;

namespace {

const mini_json::Value& golden() {
    static const mini_json::Value doc = [] {
        const std::string path = regime_tests::data_dir() + "/golden/golden.json";
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open " + path);
        std::stringstream buf;
        buf << in.rdbuf();
        return mini_json::parse(buf.str());
    }();
    return doc;
}

double metric_by_key(const regime::Metrics& m, const std::string& key) {
    if (key == "ann_return") return m.ann_return;
    if (key == "ann_vol") return m.ann_vol;
    if (key == "sharpe") return m.sharpe;
    if (key == "max_dd") return m.max_dd;
    if (key == "calmar") return m.calmar;
    if (key == "hit_rate") return m.hit_rate;
    if (key == "worst_month") return m.worst_month;
    throw std::runtime_error("unknown metric key: " + key);
}

/// Build the pinned re-ranking differential panel from a case's `inputs`
/// (API_SPEC 5): diff[t] = diffs[k] with k the last index whose
/// switch_days[k] <= t.
regime::Matrix carry_rerank_panel(const mini_json::Value& inputs) {
    const auto T = static_cast<std::size_t>(inputs.at("n_days").num());
    const auto A = static_cast<std::size_t>(inputs.at("n_assets").num());
    const auto& switch_days = inputs.at("switch_days").arr;
    const auto& diffs = inputs.at("diffs").arr;
    if (switch_days.size() != diffs.size() || switch_days.empty() || switch_days[0].num() != 0.0)
        throw std::runtime_error("carry_rerank inputs: bad switch_days");
    regime::Matrix panel(T, A);
    std::size_t k = 0;
    for (std::size_t t = 0; t < T; ++t) {
        while (k + 1 < switch_days.size() &&
               static_cast<std::size_t>(switch_days[k + 1].num()) <= t)
            ++k;
        const auto& row = diffs[k].arr;
        if (row.size() != A) throw std::runtime_error("carry_rerank inputs: ragged diffs");
        for (std::size_t a = 0; a < A; ++a) panel(t, a) = row[a].num();
    }
    return panel;
}

/// Every `carry_rerank_*` scalar computed by the C++ port from `inputs`.
double carry_rerank_value(const mini_json::Value& inputs, const std::string& key) {
    const regime::Matrix diffs = carry_rerank_panel(inputs);
    const regime::Matrix pos = regime::carry_positions(
        diffs, static_cast<int>(inputs.at("top_n").num()),
        static_cast<int>(inputs.at("bottom_n").num()),
        static_cast<int>(inputs.at("rebalance_days").num()));
    const regime::Matrix rets =
        regime::carry_total_returns(regime::Matrix(diffs.rows, diffs.cols, 0.0), diffs);
    const regime::BacktestResult res =
        regime::run_backtest(pos, rets, inputs.at("cost_bps").num());
    if (key == "mean_turnover") {
        double mean = 0.0;
        for (double x : res.turnover) mean += x;
        return mean / static_cast<double>(res.turnover.size());
    }
    if (key == "turnover_day21") return res.turnover[21];
    if (key == "turnover_day42") return res.turnover[42];
    if (key.rfind("pos_day", 0) == 0) {  // pos_day<T>_asset<A>
        const std::size_t us = key.find("_asset");
        const std::size_t t = std::stoul(key.substr(7, us - 7));
        const std::size_t a = std::stoul(key.substr(us + 6));
        return pos(t, a);
    }
    return metric_by_key(res.metrics, key);
}

/// Value computed by the C++ pipeline for one (case, key) pair.
double computed_value(const std::string& case_name, const std::string& key,
                      const mini_json::Value& inputs) {
    const regime::PipelineResult& pipe = full_pipeline();
    const regime::HMMParams& p3 = pipe.hmm3.params();

    if (case_name == "hmm_k2_loglik") return pipe.fit2.log_likelihood;
    if (case_name == "hmm_k3_loglik") return pipe.fit3.log_likelihood;
    if (case_name == "hmm_k3_means") {
        const std::size_t k = static_cast<std::size_t>(key.back() - '0');
        return p3.means(k, 0);
    }
    if (case_name == "hmm_k3_stds") {
        const std::size_t k = static_cast<std::size_t>(key.back() - '0');
        return std::sqrt(p3.variances(k, 0));
    }
    if (case_name == "hmm_k3_transmat") {
        // key = "aIJ", row-major.
        const std::size_t i = static_cast<std::size_t>(key[1] - '0');
        const std::size_t j = static_cast<std::size_t>(key[2] - '0');
        return p3.transmat(i, j);
    }
    if (case_name == "hmm_k3_viterbi_counts") {
        const int k = key.back() - '0';
        int count = 0;
        for (int s : pipe.viterbi3)
            if (s == k) ++count;
        return static_cast<double>(count);
    }
    if (case_name == "hmm_k3_stationary") {
        const std::size_t k = static_cast<std::size_t>(key.back() - '0');
        return pipe.stationary3[k];
    }
    if (case_name == "hmm_k3_smoothed_t500" || case_name == "hmm_k3_smoothed_t1500") {
        const std::size_t t = case_name == "hmm_k3_smoothed_t500" ? 500 : 1500;
        const std::size_t k = static_cast<std::size_t>(key.back() - '0');
        return pipe.smoothed3(t, k);
    }
    if (case_name == "viterbi_accuracy_vs_true") return pipe.viterbi_accuracy;
    if (case_name == "momentum_unfiltered" || case_name == "momentum_filtered" ||
        case_name == "carry_unfiltered" || case_name == "carry_filtered")
        return metric_by_key(pipe.results.at(case_name).metrics, key);
    if (case_name == "combined_sharpe") {
        if (key == "sharpe_unfiltered") return pipe.combined.at("unfiltered").metrics.sharpe;
        if (key == "sharpe_filtered") return pipe.combined.at("filtered").metrics.sharpe;
        if (key == "max_dd_unfiltered") return pipe.combined.at("unfiltered").metrics.max_dd;
        if (key == "max_dd_filtered") return pipe.combined.at("filtered").metrics.max_dd;
    }
    if (case_name == "carry_rerank_positions" || case_name == "carry_rerank_backtest")
        return carry_rerank_value(inputs, key);
    if (case_name == "crisis_momentum_return")
        return pipe.crisis.at("momentum_unfiltered")[0].ann_return;
    if (case_name == "turnover_momentum" || case_name == "turnover_carry") {
        const std::string res_name =
            case_name == "turnover_momentum" ? "momentum_unfiltered" : "carry_unfiltered";
        const auto& to = pipe.results.at(res_name).turnover;
        double mean = 0.0;
        for (double x : to) mean += x;
        return mean / static_cast<double>(to.size());
    }
    throw std::runtime_error("unhandled golden case/key: " + case_name + "/" + key);
}

}  // namespace

TEST(Golden, AllCasesMatchWithinTolerance) {
    const auto& cases = golden().at("cases").arr;
    ASSERT_EQ(cases.size(), 20u);
    int checked = 0;
    for (const auto& c : cases) {
        const std::string name = c.at("name").string();
        const double tol = c.at("tol").num();
        for (const auto& [key, expect] : c.at("expect").obj) {
            const double got = computed_value(name, key, c.at("inputs"));
            if (tol == 0.0) {
                EXPECT_EQ(got, expect.num()) << name << "." << key;
            } else {
                EXPECT_NEAR(got, expect.num(), tol) << name << "." << key;
            }
            ++checked;
        }
    }
    EXPECT_EQ(checked, 77);  // every scalar across the 20 cases
}

TEST(Golden, CarryRerankPanelReallyReranks) {
    // The bundled differentials never cross; the embedded panel must.
    const auto& cases = golden().at("cases").arr;
    const mini_json::Value* rr = nullptr;
    for (const auto& c : cases)
        if (c.at("name").string() == "carry_rerank_positions") rr = &c;
    ASSERT_NE(rr, nullptr);
    EXPECT_EQ(rr->at("tol").num(), 0.0);
    const regime::Matrix panel = carry_rerank_panel(rr->at("inputs"));
    EXPECT_EQ(panel.rows, 63u);
    EXPECT_EQ(panel(9, 0), 0.04);
    EXPECT_EQ(panel(10, 0), 0.01);
    EXPECT_EQ(rr->at("expect").at("pos_day15_asset0").num(), 1.0);   // not yet rebalanced
    EXPECT_EQ(rr->at("expect").at("pos_day21_asset0").num(), -1.0);  // flipped at day 21
    EXPECT_EQ(rr->at("expect").at("turnover_day21").num(), 4.0);
}

TEST(Golden, ShorterRefitWindowAlsoRuns) {
    // Tests are allowed a faster walk-forward loop: overriding the refit
    // cadence must work and still produce a valid gate (golden values always
    // use the bundled pinned config, checked above).
    const regime::PipelineResult pipe =
        regime::run_full_pipeline(regime_tests::data_dir(), 750, 1000);
    for (std::size_t t = 0; t < 999; ++t) EXPECT_EQ(pipe.gate[t], 1.0);
    for (double g : pipe.gate) {
        EXPECT_GE(g, 0.0);
        EXPECT_LE(g, 1.0);
    }
    EXPECT_EQ(pipe.config.refit_days, 750);
    EXPECT_EQ(pipe.config.train_min_days, 1000);
}

TEST(Golden, SemanticInvariants) {
    const regime::PipelineResult& pipe = full_pipeline();
    // Crisis-state momentum return is negative (the sign case).
    EXPECT_LT(pipe.crisis.at("momentum_unfiltered")[0].ann_return, 0.0);
    // The filter improves the combined Sharpe...
    EXPECT_GT(pipe.combined.at("filtered").metrics.sharpe,
              pipe.combined.at("unfiltered").metrics.sharpe);
    // ...and shrinks momentum drawdown at the cost of total return.
    EXPECT_GT(pipe.results.at("momentum_filtered").metrics.max_dd,
              pipe.results.at("momentum_unfiltered").metrics.max_dd);
    EXPECT_LT(pipe.results.at("momentum_filtered").metrics.ann_return,
              pipe.results.at("momentum_unfiltered").metrics.ann_return);
    EXPECT_GT(pipe.combined.at("filtered").metrics.max_dd,
              pipe.combined.at("unfiltered").metrics.max_dd);
    // Stationary distribution sums to 1; Viterbi counts partition the sample.
    double pi_sum = 0.0;
    for (double x : pipe.stationary3) pi_sum += x;
    EXPECT_NEAR(pi_sum, 1.0, 1e-12);
    EXPECT_EQ(pipe.viterbi3.size(), 2000u);
}
