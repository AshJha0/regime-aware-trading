#ifndef REGIME_PIPELINE_HPP
#define REGIME_PIPELINE_HPP

/// \file pipeline.hpp
/// \brief End-to-end pipeline on the bundled dataset.
///
/// Loads the CSVs + config, fits full-sample HMMs (K=2, K=3) on the market
/// index, builds the walk-forward regime gate, runs momentum / carry /
/// combined backtests filtered and unfiltered, and computes the crisis-state
/// breakdown.  The demo and the golden tests both call run_full_pipeline so
/// they can never drift apart.
///
/// Label conventions (pinned): HMM states are sorted by mean ascending, so on
/// this data state 0 = crisis (lowest mean, highest vol).  The bundled true
/// states use the generator's ordering (0 = calm-bull, 1 = choppy,
/// 2 = crisis); the pinned volatility mapping of the full-sample K=3 fit
/// (calm -> argmin variance, crisis -> argmax variance) aligns them for the
/// Viterbi-accuracy teaching number.  The crisis table always uses that K=3
/// fit, whatever config n_states the walk-forward gate uses.

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "regime/backtest.hpp"
#include "regime/hmm.hpp"
#include "regime/matrix.hpp"

namespace regime {

/// Pinned strategy/backtest configuration (mirrors data/config.json).
struct PipelineConfig {
    int n_states{3};
    double em_tol{1e-8};
    int em_max_iter{500};
    double var_floor{1e-8};
    int momentum_lookback{252};
    double vol_target{0.10};
    double ewma_lambda{0.94};
    double leverage_cap{4.0};
    int carry_top_n{3};
    int carry_bottom_n{3};
    int rebalance_days{21};
    int train_min_days{504};
    int refit_days{63};
    std::string gate_mode{"prob"};
    double gate_threshold{0.5};
    double cost_bps{5.0};
    int trading_days{252};
};

/// Bundled market data.
struct Dataset {
    std::vector<std::string> dates;      ///< Trading dates (YYYY-MM-DD), length T.
    std::vector<double> index_returns;   ///< Market index daily returns, length T.
    Matrix trend_returns;                ///< Trend-asset daily returns, T x 6.
    Matrix fx_spot_returns;              ///< Currency spot returns, T x 6.
    Matrix fx_rate_diffs;                ///< Annualized rate differentials, T x 6.
    std::vector<int> true_states;        ///< Generator regime path (teaching only).
    PipelineConfig config;               ///< Pinned configuration.
};

/// Combined-portfolio leg: 50/50 in each strategy's net return stream.
struct CombinedResult {
    std::vector<double> net_returns;
    Metrics metrics;
};

/// Everything the demo and the golden tests need from one full run.
struct PipelineResult {
    Dataset dataset;
    PipelineConfig config;
    GaussianHMM hmm2;                    ///< Full-sample K=2 fit on the index.
    GaussianHMM hmm3;                    ///< Full-sample K=3 fit on the index.
    HMMFitResult fit2;
    HMMFitResult fit3;
    std::vector<int> viterbi3;           ///< Full-sample K=3 Viterbi path.
    Matrix smoothed3;                    ///< Full-sample K=3 smoothed posteriors.
    std::vector<double> stationary3;     ///< Stationary distribution of the K=3 fit.
    double viterbi_accuracy{0.0};        ///< Viterbi vs mapped true states.
    std::vector<double> gate;            ///< Walk-forward regime gate values.
    std::map<std::string, BacktestResult> results;      ///< momentum/carry x un/filtered.
    std::map<std::string, CombinedResult> combined;     ///< "unfiltered", "filtered".
    std::map<std::string, std::vector<StateStats>> crisis;  ///< Per-strategy state table.
};

/// Load the bundled CSVs and config from \p data_dir.
/// \throws std::invalid_argument if files are missing or lengths disagree.
Dataset load_dataset(const std::string& data_dir);

/// Run the entire study on the bundled data with the pinned config.
PipelineResult run_full_pipeline(const std::string& data_dir);

/// Run the study with optional overrides of the walk-forward cadence
/// (tests may shorten the loop; golden values always use the bundled
/// config).  std::nullopt keeps the config value.
PipelineResult run_full_pipeline(const std::string& data_dir, std::optional<int> refit_days,
                                 std::optional<int> train_min_days);

}  // namespace regime

#endif  // REGIME_PIPELINE_HPP
