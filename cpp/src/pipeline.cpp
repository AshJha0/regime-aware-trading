/// \file pipeline.cpp
/// \brief Bundled-data loading and the end-to-end study.

#include "regime/pipeline.hpp"

#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "regime/strategies.hpp"

namespace regime {

namespace {

/// Split one CSV line on commas (the bundled files have no quoting).
std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string field;
    std::stringstream ss(line);
    while (std::getline(ss, field, ',')) out.push_back(field);
    return out;
}

/// Read a CSV file into header + rows of strings.
void read_csv(const std::string& path, std::vector<std::string>& header,
              std::vector<std::vector<std::string>>& rows) {
    std::ifstream in(path);
    if (!in) throw std::invalid_argument("missing bundled data file: " + path);
    std::string line;
    bool first = true;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (first) {
            header = split_csv(line);
            first = false;
        } else {
            rows.push_back(split_csv(line));
        }
    }
    if (first) throw std::invalid_argument("empty bundled data file: " + path);
}

/// Extract the raw token following "key": in a flat JSON object.
std::string json_raw(const std::string& text, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    std::size_t pos = text.find(quoted);
    if (pos == std::string::npos)
        throw std::invalid_argument("config.json is missing key '" + key + "'");
    pos = text.find(':', pos + quoted.size());
    if (pos == std::string::npos)
        throw std::invalid_argument("config.json is malformed at key '" + key + "'");
    ++pos;
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
    std::size_t end = pos;
    if (pos < text.size() && text[pos] == '"') {
        end = text.find('"', pos + 1);
        if (end == std::string::npos)
            throw std::invalid_argument("config.json has an unterminated string for '" + key + "'");
        return text.substr(pos + 1, end - pos - 1);
    }
    while (end < text.size() && text[end] != ',' && text[end] != '}' &&
           !std::isspace(static_cast<unsigned char>(text[end])))
        ++end;
    return text.substr(pos, end - pos);
}

double json_number(const std::string& text, const std::string& key) {
    return std::stod(json_raw(text, key));
}

/// Integer config value; a non-integral number (e.g. 3.5) is rejected so the
/// C++/Rust/Java loaders agree on the same file.
int json_int(const std::string& text, const std::string& key) {
    const double v = json_number(text, key);
    if (!std::isfinite(v) || v != std::floor(v) || std::abs(v) > 1e9)
        throw std::invalid_argument("config.json key '" + key + "' must be an integer, got " +
                                    std::to_string(v));
    return static_cast<int>(v);
}

PipelineConfig load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::invalid_argument("missing bundled data file: " + path);
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string text = buf.str();
    PipelineConfig c;
    c.n_states = json_int(text, "n_states");
    c.em_tol = json_number(text, "em_tol");
    c.em_max_iter = json_int(text, "em_max_iter");
    c.var_floor = json_number(text, "var_floor");
    c.momentum_lookback = json_int(text, "momentum_lookback");
    c.vol_target = json_number(text, "vol_target");
    c.ewma_lambda = json_number(text, "ewma_lambda");
    c.leverage_cap = json_number(text, "leverage_cap");
    c.carry_top_n = json_int(text, "carry_top_n");
    c.carry_bottom_n = json_int(text, "carry_bottom_n");
    c.rebalance_days = json_int(text, "rebalance_days");
    c.train_min_days = json_int(text, "train_min_days");
    c.refit_days = json_int(text, "refit_days");
    c.gate_mode = json_raw(text, "gate_mode");
    c.gate_threshold = json_number(text, "gate_threshold");
    c.cost_bps = json_number(text, "cost_bps");
    c.trading_days = json_int(text, "trading_days");
    return c;
}

/// Column indices of every header starting with \p prefix, in header order.
std::vector<std::size_t> columns_with_prefix(const std::vector<std::string>& header,
                                             const std::string& prefix) {
    std::vector<std::size_t> idx;
    for (std::size_t c = 0; c < header.size(); ++c)
        if (header[c].rfind(prefix, 0) == 0) idx.push_back(c);
    return idx;
}

Matrix extract(const std::vector<std::vector<std::string>>& rows,
               const std::vector<std::size_t>& cols) {
    Matrix m(rows.size(), cols.size());
    for (std::size_t t = 0; t < rows.size(); ++t)
        for (std::size_t a = 0; a < cols.size(); ++a) m(t, a) = std::stod(rows[t][cols[a]]);
    return m;
}

}  // namespace

Dataset load_dataset(const std::string& data_dir) {
    Dataset ds;
    std::vector<std::string> header;
    std::vector<std::vector<std::string>> rows;

    read_csv(data_dir + "/market_index.csv", header, rows);
    if (rows.empty()) throw std::invalid_argument("bundled CSVs are empty");
    for (const auto& row : rows) {
        if (row.size() < 2) throw std::invalid_argument("market_index.csv: ragged row");
        ds.dates.push_back(row[0]);
        ds.index_returns.push_back(std::stod(row[1]));
    }
    const std::size_t T = rows.size();

    header.clear();
    rows.clear();
    read_csv(data_dir + "/trend_assets.csv", header, rows);
    if (rows.size() != T) throw std::invalid_argument("bundled CSVs have inconsistent lengths");
    ds.trend_returns = extract(rows, columns_with_prefix(header, "asset_"));

    header.clear();
    rows.clear();
    read_csv(data_dir + "/fx_carry.csv", header, rows);
    if (rows.size() != T) throw std::invalid_argument("bundled CSVs have inconsistent lengths");
    ds.fx_spot_returns = extract(rows, columns_with_prefix(header, "spot_ret_"));
    ds.fx_rate_diffs = extract(rows, columns_with_prefix(header, "rate_diff_"));

    header.clear();
    rows.clear();
    read_csv(data_dir + "/true_states.csv", header, rows);
    if (rows.size() != T) throw std::invalid_argument("bundled CSVs have inconsistent lengths");
    for (const auto& row : rows) {
        if (row.size() < 2) throw std::invalid_argument("true_states.csv: ragged row");
        const int state = std::stoi(row[1]);
        if (state < 0 || state > 2)
            throw std::invalid_argument("true_states.csv: state labels must be integers in [0, 2]");
        ds.true_states.push_back(state);
    }

    ds.config = load_config(data_dir + "/config.json");
    return ds;
}

PipelineResult run_full_pipeline(const std::string& data_dir) {
    return run_full_pipeline(data_dir, std::nullopt, std::nullopt);
}

PipelineResult run_full_pipeline(const std::string& data_dir, std::optional<int> refit_days,
                                 std::optional<int> train_min_days) {
    Dataset ds = load_dataset(data_dir);
    PipelineConfig cfg = ds.config;  // copy: ds is moved into the result below
    if (refit_days) cfg.refit_days = *refit_days;
    if (train_min_days) cfg.train_min_days = *train_min_days;
    const int K = cfg.n_states;
    const Matrix r_idx = to_matrix(ds.index_returns);

    // Full-sample HMM fits (model selection + crisis attribution).
    GaussianHMM hmm2(2, cfg.em_tol, cfg.em_max_iter, cfg.var_floor);
    HMMFitResult fit2 = hmm2.fit(r_idx);
    GaussianHMM hmm3(3, cfg.em_tol, cfg.em_max_iter, cfg.var_floor);
    HMMFitResult fit3 = hmm3.fit(r_idx);
    std::vector<int> viterbi3 = hmm3.viterbi(r_idx);
    Matrix smoothed3 = hmm3.smoothed_probabilities(r_idx);
    std::vector<double> stationary3 = hmm3.stationary_distribution();

    // Teaching comparison via the pinned volatility mapping of the K=3 fit:
    // generator label calm-bull -> argmin variance, crisis -> argmax
    // variance, choppy -> rest.  K3 is hmm3's state count (3), independent
    // of the gate's config n_states.
    const int K3 = hmm3.n_states();
    const Matrix& vars3 = hmm3.params().variances;
    int calm_label = 0, crisis_label = 0;
    for (int k = 1; k < K3; ++k) {
        if (vars3(static_cast<std::size_t>(k), 0) < vars3(static_cast<std::size_t>(calm_label), 0))
            calm_label = k;
        if (vars3(static_cast<std::size_t>(k), 0) > vars3(static_cast<std::size_t>(crisis_label), 0))
            crisis_label = k;
    }
    const int choppy_label = 3 - calm_label - crisis_label;
    const int label_map[3] = {calm_label, choppy_label, crisis_label};
    std::size_t agree = 0;
    for (std::size_t t = 0; t < ds.true_states.size(); ++t)
        if (viterbi3[t] == label_map[ds.true_states[t]]) ++agree;
    const double viterbi_accuracy =
        static_cast<double>(agree) / static_cast<double>(ds.true_states.size());

    // Walk-forward regime gate on the index (no lookahead).
    RegimeGateResult gate_res =
        regime_gate(ds.index_returns, K, cfg.train_min_days, cfg.refit_days, cfg.gate_mode,
                    cfg.gate_threshold, cfg.em_tol, cfg.em_max_iter, cfg.var_floor);

    const double cost = cfg.cost_bps;

    Matrix mom_pos = momentum_positions(ds.trend_returns, cfg.momentum_lookback, cfg.vol_target,
                                        cfg.ewma_lambda, cfg.leverage_cap);
    Matrix carry_pos = carry_positions(ds.fx_rate_diffs, cfg.carry_top_n, cfg.carry_bottom_n,
                                       cfg.rebalance_days);
    Matrix carry_rets = carry_total_returns(ds.fx_spot_returns, ds.fx_rate_diffs);

    PipelineResult pipe{
        std::move(ds), cfg, std::move(hmm2), std::move(hmm3), fit2, fit3,
        std::move(viterbi3), std::move(smoothed3), std::move(stationary3),
        viterbi_accuracy, std::move(gate_res.gate), {}, {}, {}};

    const std::vector<std::string>* dates = &pipe.dataset.dates;
    pipe.results["momentum_unfiltered"] = run_backtest(mom_pos, pipe.dataset.trend_returns, cost, dates);
    pipe.results["momentum_filtered"] =
        run_backtest(apply_gate(mom_pos, pipe.gate), pipe.dataset.trend_returns, cost, dates);
    pipe.results["carry_unfiltered"] = run_backtest(carry_pos, carry_rets, cost, dates);
    pipe.results["carry_filtered"] =
        run_backtest(apply_gate(carry_pos, pipe.gate), carry_rets, cost, dates);

    // Combined portfolio: 50/50 in each strategy's net return stream.
    for (const std::string tag : {"unfiltered", "filtered"}) {
        const auto& mom = pipe.results.at("momentum_" + tag).net_returns;
        const auto& car = pipe.results.at("carry_" + tag).net_returns;
        CombinedResult comb;
        comb.net_returns.resize(mom.size());
        for (std::size_t t = 0; t < mom.size(); ++t)
            comb.net_returns[t] = 0.5 * mom[t] + 0.5 * car[t];
        comb.metrics = compute_metrics(comb.net_returns, dates);
        pipe.combined[tag] = std::move(comb);
    }

    // Crisis table: attribute daily strategy returns to the full-sample
    // K=3 Viterbi state (state 0 = crisis on this data); the table has K3
    // rows regardless of the gate's n_states.
    for (const auto& [name, res] : pipe.results)
        pipe.crisis[name] = state_conditional_returns(res.net_returns, pipe.viterbi3, K3);
    for (const std::string tag : {"unfiltered", "filtered"})
        pipe.crisis["combined_" + tag] =
            state_conditional_returns(pipe.combined.at(tag).net_returns, pipe.viterbi3, K3);

    return pipe;
}

}  // namespace regime
