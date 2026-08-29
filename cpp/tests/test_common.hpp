#ifndef REGIME_TESTS_TEST_COMMON_HPP
#define REGIME_TESTS_TEST_COMMON_HPP

/// \file test_common.hpp
/// \brief Shared fixtures: the bundled dataset and one full-pipeline run,
///        computed once per test binary (the pinned walk-forward loop is the
///        expensive part and every golden case reads from the same run).

#include <string>

#include "regime/pipeline.hpp"

#ifndef REGIME_DATA_DIR
#define REGIME_DATA_DIR "../data"
#endif

namespace regime_tests {

inline const std::string& data_dir() {
    static const std::string dir = REGIME_DATA_DIR;
    return dir;
}

/// Full pipeline on the bundled data with the pinned config (run once).
inline const regime::PipelineResult& full_pipeline() {
    static const regime::PipelineResult pipe = regime::run_full_pipeline(data_dir());
    return pipe;
}

/// Bundled market-index returns (loaded once, without the pipeline).
inline const std::vector<double>& index_returns() {
    static const std::vector<double> r = regime::load_dataset(data_dir()).index_returns;
    return r;
}

/// First n index returns as a (n x 1) observation matrix.
inline regime::Matrix index_head(std::size_t n) {
    const auto& r = index_returns();
    regime::Matrix m(n, 1);
    for (std::size_t t = 0; t < n; ++t) m(t, 0) = r[t];
    return m;
}

}  // namespace regime_tests

#endif  // REGIME_TESTS_TEST_COMMON_HPP
