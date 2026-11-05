/// \file test_filter_alloc.cpp
/// \brief Counts global operator new calls to pin that RegimeFilter::update
///        allocates nothing (API_SPEC 1.10).  Replacing operator new affects
///        the whole test binary, which only adds a counter.

#include <gtest/gtest.h>

#include <atomic>
#include <cstdlib>
#include <new>
#include <vector>

#include "regime/hmm.hpp"
#include "test_common.hpp"

namespace {
std::atomic<long> g_news{0};
}

void* operator new(std::size_t n) {
    ++g_news;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

TEST(RegimeFilter, UpdateDoesNotAllocate) {
    regime::GaussianHMM m(3);
    m.fit(regime_tests::index_head(600));
    regime::RegimeFilter filt(m.params());
    const std::vector<double>& r = regime_tests::index_returns();
    const std::vector<double>& alpha = filt.alpha();
    double sink = 0.0;
    const long before = g_news.load();
    for (double x : r) sink += filt.update(x)[0];
    const long after = g_news.load();
    EXPECT_EQ(after - before, 0);
    EXPECT_GT(sink, 0.0);
    EXPECT_EQ(alpha.size(), 3u);

    // The counter itself works: forward_step allocates.
    const long fs_before = g_news.load();
    regime::forward_step(m.params(), filt.alpha(), {0.001});
    EXPECT_GT(g_news.load(), fs_before);
}
