/// \file test_golden.cpp
/// \brief Cross-language golden suite: every case in data/golden/golden.json
///        must match the pinned pipeline within its tolerance.

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#include "mini_json.hpp"
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

/// Value computed by the C++ pipeline for one (case, key) pair.
double computed_value(const std::string& case_name, const std::string& key) {
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
        case_name == "carry_unfiltered" || case_name == "carry_filtered") {
        const regime::Metrics& m = pipe.results.at(case_name).metrics;
        if (key == "ann_return") return m.ann_return;
        if (key == "sharpe") return m.sharpe;
        if (key == "max_dd") return m.max_dd;
    }
    if (case_name == "combined_sharpe") {
        if (key == "sharpe_unfiltered") return pipe.combined.at("unfiltered").metrics.sharpe;
        if (key == "sharpe_filtered") return pipe.combined.at("filtered").metrics.sharpe;
    }
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
    ASSERT_EQ(cases.size(), 18u);
    int checked = 0;
    for (const auto& c : cases) {
        const std::string name = c.at("name").string();
        const double tol = c.at("tol").num();
        for (const auto& [key, expect] : c.at("expect").obj) {
            const double got = computed_value(name, key);
            if (tol == 0.0) {
                EXPECT_EQ(got, expect.num()) << name << "." << key;
            } else {
                EXPECT_NEAR(got, expect.num(), tol) << name << "." << key;
            }
            ++checked;
        }
    }
    EXPECT_GE(checked, 40);  // 18 cases, several keys each
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
    // Stationary distribution sums to 1; Viterbi counts partition the sample.
    double pi_sum = 0.0;
    for (double x : pipe.stationary3) pi_sum += x;
    EXPECT_NEAR(pi_sum, 1.0, 1e-12);
    EXPECT_EQ(pipe.viterbi3.size(), 2000u);
}
