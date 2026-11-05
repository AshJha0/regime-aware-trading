/// \file bench_filter.cpp
/// \brief Per-call latency of forward_step and RegimeFilter::update.
///
/// Fits K = 3 on the bundled index, then times each call on the bundled
/// returns, cycled.  This measures one CPU routine on warm caches.  The model
/// is daily-frequency: there is no market-data, order or network path here,
/// so these numbers say nothing about a trading system's latency.
///
///   cmake --build build --target bench_filter && build/bench_filter [data_dir] [samples]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "regime/hmm.hpp"
#include "regime/pipeline.hpp"

#if defined(__x86_64__)
#include <x86intrin.h>
#endif

#ifndef REGIME_DATA_DIR
#define REGIME_DATA_DIR "../data"
#endif

namespace {

/// Tick source: fenced TSC on x86-64, steady_clock elsewhere.
inline std::uint64_t ticks() {
#if defined(__x86_64__)
    _mm_lfence();
    const std::uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
#else
    return static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
#endif
}

double ns_per_tick() {
    using clock = std::chrono::steady_clock;
    const auto w0 = clock::now();
    const std::uint64_t t0 = ticks();
    while (clock::now() - w0 < std::chrono::milliseconds(200)) {
    }
    const std::uint64_t t1 = ticks();
    const double ns = std::chrono::duration<double, std::nano>(clock::now() - w0).count();
    return ns / static_cast<double>(t1 - t0);
}

void report(const char* name, std::vector<std::uint64_t>& s, double scale) {
    double sum = 0.0;
    for (std::uint64_t v : s) sum += static_cast<double>(v);
    std::sort(s.begin(), s.end());
    const auto q = [&](double p) {
        return static_cast<double>(s[static_cast<std::size_t>(p * (s.size() - 1))]) * scale;
    };
    std::printf("%-22s mean %8.1f  p50 %8.1f  p90 %8.1f  p99 %8.1f  p99.9 %9.1f  max %11.1f ns\n",
                name, sum / static_cast<double>(s.size()) * scale, q(0.5), q(0.9), q(0.99),
                q(0.999), static_cast<double>(s.back()) * scale);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : REGIME_DATA_DIR;
    const std::size_t n = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 2000000;
    const std::size_t warmup = 100000;
    const std::vector<double> r = regime::load_dataset(dir).index_returns;
    regime::GaussianHMM model(3);
    model.fit(regime::to_matrix(r));
    const regime::HMMParams& p = model.params();
    const double scale = ns_per_tick();

    std::vector<std::uint64_t> samples(n);
    volatile double sink = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t a = ticks();
        const std::uint64_t b = ticks();
        samples[i] = b - a;
    }
    std::printf("samples per routine: %zu, ns per tick: %.4f, K=3 D=1\n", n, scale);
    report("timer floor", samples, scale);

    std::vector<double> alpha = p.startprob;
    std::vector<double> x(1);
    for (std::size_t i = 0; i < warmup + n; ++i) {
        x[0] = r[i % r.size()];
        const std::uint64_t a = ticks();
        alpha = regime::forward_step(p, alpha, x);
        const std::uint64_t b = ticks();
        if (i >= warmup) samples[i - warmup] = b - a;
    }
    sink = sink + alpha[0];
    report("forward_step", samples, scale);

    regime::RegimeFilter filt(p);
    for (std::size_t i = 0; i < warmup + n; ++i) {
        const double v = r[i % r.size()];
        const std::uint64_t a = ticks();
        const std::vector<double>& out = filt.update(v);
        const std::uint64_t b = ticks();
        sink = sink + out[0];
        if (i >= warmup) samples[i - warmup] = b - a;
    }
    report("RegimeFilter::update", samples, scale);
    return sink > -1.0 ? 0 : 1;
}
