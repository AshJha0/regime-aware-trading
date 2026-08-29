/// \file demo.cpp
/// \brief End-to-end demo: regime detection + regime-aware momentum and carry.
///
/// Prints the fitted regime table, Viterbi-vs-true accuracy, the strategy
/// comparison (filtered vs unfiltered) and the crisis-state breakdown.
/// Mirrors python/demo.py.  Data directory: first CLI argument, defaulting to
/// the bundled ../data (compiled in as REGIME_DATA_DIR).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "regime/pipeline.hpp"

#ifndef REGIME_DATA_DIR
#define REGIME_DATA_DIR "../data"
#endif

namespace {

const char* kLine =
    "------------------------------------------------------------------------------";

void print_metrics_row(const std::string& label, const regime::Metrics& m) {
    std::printf("  %-22s %+7.2f%% %7.2f%% %7.2f %+7.2f%% %7.2f %5.1f%% %+8.2f%%\n",
                label.c_str(), 100.0 * m.ann_return, 100.0 * m.ann_vol, m.sharpe,
                100.0 * m.max_dd, m.calmar, 100.0 * m.hit_rate, 100.0 * m.worst_month);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data_dir = argc > 1 ? argv[1] : REGIME_DATA_DIR;
    try {
        std::printf(
            "Regime-Aware Trading — HMM + momentum + FX carry (bundled synthetic data)\n%s\n",
            kLine);
        regime::PipelineResult pipe = regime::run_full_pipeline(data_dir);
        const auto& ds = pipe.dataset;
        const int K = pipe.config.n_states;
        const regime::Matrix r = regime::to_matrix(ds.index_returns);

        // ---------------- model selection ---------------- //
        std::printf("Data: %zu trading days, %s .. %s\n", ds.index_returns.size(),
                    ds.dates.front().c_str(), ds.dates.back().c_str());
        std::printf("\nModel selection on the market index (diagonal Gaussian HMM):\n");
        std::printf("  %2s %12s %12s %12s %6s %5s\n", "K", "loglik", "AIC", "BIC", "iters", "conv");
        const regime::GaussianHMM* models[2] = {&pipe.hmm2, &pipe.hmm3};
        const regime::HMMFitResult* fits[2] = {&pipe.fit2, &pipe.fit3};
        for (int i = 0; i < 2; ++i)
            std::printf("  %2d %12.4f %12.4f %12.4f %6d %5s\n", i + 2,
                        fits[i]->log_likelihood, models[i]->aic(r), models[i]->bic(r),
                        fits[i]->n_iter, fits[i]->converged ? "True" : "False");

        // ---------------- regime table ---------------- //
        const regime::HMMParams& p = pipe.hmm3.params();
        std::vector<std::string> names(static_cast<std::size_t>(K), "?");
        std::size_t hi = 0, lo = 0;
        for (std::size_t k = 1; k < static_cast<std::size_t>(K); ++k) {
            if (p.variances(k, 0) > p.variances(hi, 0)) hi = k;
            if (p.variances(k, 0) < p.variances(lo, 0)) lo = k;
        }
        names[hi] = "crisis";
        names[lo] = "calm-bull";
        for (auto& n : names)
            if (n == "?") n = "choppy";
        std::printf("\nFitted K=3 regimes (states sorted by mean; 0 = lowest):\n");
        std::printf("  %5s %10s %10s %9s %11s\n", "state", "label", "mean(ann)", "vol(ann)",
                    "stationary");
        for (std::size_t k = 0; k < static_cast<std::size_t>(K); ++k)
            std::printf("  %5zu %10s %+9.2f%% %8.2f%% %11.4f\n", k, names[k].c_str(),
                        100.0 * 252.0 * p.means(k, 0),
                        100.0 * std::sqrt(252.0 * p.variances(k, 0)), pipe.stationary3[k]);
        std::printf("  transition matrix (rows sum to 1):\n");
        for (std::size_t i = 0; i < static_cast<std::size_t>(K); ++i) {
            std::printf("    ");
            for (std::size_t j = 0; j < static_cast<std::size_t>(K); ++j)
                std::printf("%.4f%s", p.transmat(i, j), j + 1 < static_cast<std::size_t>(K) ? "  " : "");
            std::printf("\n");
        }

        std::vector<int> counts(static_cast<std::size_t>(K), 0);
        for (int s : pipe.viterbi3) ++counts[static_cast<std::size_t>(s)];
        std::vector<int> true_counts(static_cast<std::size_t>(K), 0);
        for (int s : ds.true_states) ++true_counts[static_cast<std::size_t>(s)];
        std::printf("\nViterbi decoding vs true (generator) states: accuracy = %.2f%%\n",
                    100.0 * pipe.viterbi_accuracy);
        std::printf("  decoded days per state [%d, %d, %d]  (true calm/choppy/crisis [%d, %d, %d])\n",
                    counts[0], counts[1], counts[2], true_counts[0], true_counts[1],
                    true_counts[2]);

        // ---------------- strategy comparison ---------------- //
        std::printf("%s\n", kLine);
        std::printf("Strategies (cost %.0f bps, gate mode '%s', refit every %dd on expanding window):\n",
                    pipe.config.cost_bps, pipe.config.gate_mode.c_str(), pipe.config.refit_days);
        std::printf("  %-22s %8s %8s %7s %8s %7s %6s %9s\n", "strategy", "ann ret", "ann vol",
                    "Sharpe", "maxDD", "Calmar", "hit", "worst mo");
        for (const char* name : {"momentum unfiltered", "momentum filtered", "carry unfiltered",
                                 "carry filtered"}) {
            std::string key(name);
            std::replace(key.begin(), key.end(), ' ', '_');
            print_metrics_row(name, pipe.results.at(key).metrics);
        }
        for (const char* tag : {"unfiltered", "filtered"})
            print_metrics_row(std::string("combined ") + tag, pipe.combined.at(tag).metrics);

        // ---------------- crisis breakdown ---------------- //
        std::printf("%s\n", kLine);
        std::printf("Annualized net return by decoded regime (state 0 = lowest mean):\n");
        std::printf("  %-22s %9s %9s %10s\n", "strategy", names[0].c_str(), names[1].c_str(),
                    names[2].c_str());
        for (const char* key : {"momentum_unfiltered", "momentum_filtered", "carry_unfiltered",
                                "carry_filtered", "combined_unfiltered", "combined_filtered"}) {
            const auto& c = pipe.crisis.at(key);
            std::string label(key);
            std::replace(label.begin(), label.end(), '_', ' ');
            std::printf("  %-22s %+8.2f%% %+8.2f%% %+9.2f%%\n", label.c_str(),
                        100.0 * c[0].ann_return, 100.0 * c[1].ann_return, 100.0 * c[2].ann_return);
        }

        const regime::Metrics& mu = pipe.combined.at("unfiltered").metrics;
        const regime::Metrics& mf = pipe.combined.at("filtered").metrics;
        std::printf("%s\n", kLine);
        std::printf("Takeaway: momentum and carry both lose money in the high-vol crisis\n");
        std::printf("regime; scaling by the filtered P(calm) improves the combined Sharpe\n");
        std::printf("(%.2f -> %.2f) and maxDD (%+.2f%% -> %+.2f%%)\n", mu.sharpe, mf.sharpe,
                    100.0 * mu.max_dd, 100.0 * mf.max_dd);
        std::printf("at the cost of total return (%+.2f%% -> %+.2f%%) — the filter\n",
                    100.0 * mu.ann_return, 100.0 * mf.ann_return);
        std::printf("buys risk reduction, not free performance.\n");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "demo failed: %s\n", e.what());
        return 1;
    }
}
