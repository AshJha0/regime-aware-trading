/// \file backtest.cpp
/// \brief Pinned no-lookahead accounting, summary metrics, crisis attribution.

#include "regime/backtest.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace regime {

namespace {

void require_finite(const std::vector<double>& v, const char* what) {
    for (double x : v)
        if (!std::isfinite(x)) throw std::invalid_argument(std::string(what) + " contain NaN or inf");
}

/// Pinned net-return checks: non-empty, finite, and > -1 (a net return of
/// -100% or worse means the book is wiped out; API_SPEC 2).
void validate_net(const std::vector<double>& net) {
    if (net.empty()) throw std::invalid_argument("net return series is empty");
    require_finite(net, "net returns");
    for (std::size_t t = 0; t < net.size(); ++t)
        if (net[t] <= -1.0)
            throw std::invalid_argument("equity wiped out on day t=" + std::to_string(t) +
                                        ": net return " + std::to_string(net[t]) + " <= -100%");
}

/// True iff s is a well-formed ISO date "YYYY-MM-DD" with month 1..12 and
/// day 1..31 (calendar validity beyond that is not checked).
bool iso_date_ok(const std::string& s) {
    if (s.size() != 10 || s[4] != '-' || s[7] != '-') return false;
    for (std::size_t i : {0u, 1u, 2u, 3u, 5u, 6u, 8u, 9u})
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    const int month = (s[5] - '0') * 10 + (s[6] - '0');
    const int day = (s[8] - '0') * 10 + (s[9] - '0');
    return month >= 1 && month <= 12 && day >= 1 && day <= 31;
}

/// Pinned date checks (API_SPEC 2.1): length T, ISO format, strictly
/// increasing (ISO strings order lexicographically).
void validate_dates(const std::vector<std::string>& dates, std::size_t T) {
    if (dates.size() != T)
        throw std::invalid_argument("dates length " + std::to_string(dates.size()) +
                                    " does not match number of days " + std::to_string(T));
    for (std::size_t t = 0; t < T; ++t) {
        if (!iso_date_ok(dates[t]))
            throw std::invalid_argument("dates could not be parsed: '" + dates[t] +
                                        "' is not YYYY-MM-DD");
        if (t > 0 && !(dates[t - 1] < dates[t]))
            throw std::invalid_argument("dates must be strictly increasing (at t=" +
                                        std::to_string(t) + ")");
    }
}

}  // namespace

double max_drawdown(const std::vector<double>& equity) {
    if (equity.empty()) throw std::invalid_argument("equity curve is empty");
    for (double e : equity)
        if (!std::isfinite(e) || e <= 0.0)
            throw std::invalid_argument("equity curve must be finite and strictly positive");
    double peak = equity[0];
    double mdd = 0.0;
    for (double e : equity) {
        peak = std::max(peak, e);
        mdd = std::min(mdd, e / peak - 1.0);
    }
    return mdd;
}

Metrics compute_metrics(const std::vector<double>& net,
                        const std::vector<std::string>* dates) {
    validate_net(net);
    const std::size_t T = net.size();
    if (dates != nullptr) validate_dates(*dates, T);

    double mean = 0.0;
    for (double x : net) mean += x;
    mean /= static_cast<double>(T);
    double var = 0.0;
    for (double x : net) var += (x - mean) * (x - mean);
    var /= static_cast<double>(T);  // ddof = 0

    Metrics m;
    m.ann_return = kTradingDays * mean;
    m.ann_vol = std::sqrt(static_cast<double>(kTradingDays)) * std::sqrt(var);
    m.sharpe = m.ann_vol > 0.0 ? m.ann_return / m.ann_vol : 0.0;

    std::vector<double> equity(T);
    double eq = 1.0;
    for (std::size_t t = 0; t < T; ++t) {
        eq *= 1.0 + net[t];
        equity[t] = eq;
    }
    m.max_dd = max_drawdown(equity);
    m.calmar = m.max_dd < 0.0 ? m.ann_return / std::abs(m.max_dd) : 0.0;

    std::size_t hits = 0;
    for (double x : net)
        if (x > 0.0) ++hits;
    m.hit_rate = static_cast<double>(hits) / static_cast<double>(T);

    // Worst month: calendar months when dates are given (group key = the
    // YYYY-MM prefix; dates are sorted, so groups are contiguous), otherwise
    // consecutive 21-day blocks (trailing partial block dropped, matching
    // the Python fallback).
    double worst = std::numeric_limits<double>::infinity();
    if (dates != nullptr) {
        std::string cur_month;
        double comp = 1.0;
        bool open = false;
        for (std::size_t t = 0; t < T; ++t) {
            const std::string month = (*dates)[t].substr(0, 7);
            if (!open || month != cur_month) {
                if (open) worst = std::min(worst, comp - 1.0);
                cur_month = month;
                comp = 1.0;
                open = true;
            }
            comp *= 1.0 + net[t];
        }
        if (open) worst = std::min(worst, comp - 1.0);
    } else {
        std::size_t nblk = T / 21;
        if (nblk == 0) nblk = 1;
        for (std::size_t b = 0; b < nblk; ++b) {
            double comp = 1.0;
            const std::size_t end = std::min((b + 1) * 21, T);
            for (std::size_t t = b * 21; t < end; ++t) comp *= 1.0 + net[t];
            worst = std::min(worst, comp - 1.0);
        }
    }
    m.worst_month = worst;
    return m;
}

BacktestResult run_backtest(const Matrix& positions, const Matrix& returns,
                            double cost_bps, const std::vector<std::string>* dates) {
    if (positions.rows != returns.rows || positions.cols != returns.cols)
        throw std::invalid_argument("positions and returns must have equal shape");
    if (positions.rows == 0) throw std::invalid_argument("empty backtest: no days");
    if (positions.cols == 0) throw std::invalid_argument("empty backtest: no assets");
    if (positions.data.size() != positions.rows * positions.cols ||
        returns.data.size() != returns.rows * returns.cols)
        throw std::invalid_argument("positions/returns storage does not match their shape");
    require_finite(positions.data, "positions");
    require_finite(returns.data, "returns");
    for (std::size_t t = 0; t < returns.rows; ++t)
        for (std::size_t a = 0; a < returns.cols; ++a)
            if (returns(t, a) <= -1.0)
                throw std::invalid_argument("returns contain a return <= -100% at t=" +
                                            std::to_string(t) + ", asset=" + std::to_string(a) +
                                            " (" + std::to_string(returns(t, a)) + ")");
    if (!std::isfinite(cost_bps) || cost_bps < 0.0)
        throw std::invalid_argument("cost_bps must be >= 0, got " + std::to_string(cost_bps));
    if (dates != nullptr) validate_dates(*dates, positions.rows);

    const std::size_t T = positions.rows, A = positions.cols;
    BacktestResult res;
    res.gross_returns.assign(T, 0.0);
    res.turnover.assign(T, 0.0);
    res.net_returns.assign(T, 0.0);
    res.equity.assign(T, 0.0);
    const double cost = cost_bps / 1e4;
    double eq = 1.0;
    for (std::size_t t = 0; t < T; ++t) {
        double gross = 0.0, to = 0.0;
        for (std::size_t a = 0; a < A; ++a) {
            const double prev = t > 0 ? positions(t - 1, a) : 0.0;
            if (t > 0) gross += prev * returns(t, a);
            to += std::abs(positions(t, a) - prev);
        }
        res.gross_returns[t] = gross;
        res.turnover[t] = to;
        res.net_returns[t] = gross - cost * to;
    }
    validate_net(res.net_returns);  // wipe-out guard: net[t] <= -1 is an error naming t
    for (std::size_t t = 0; t < T; ++t) {
        eq *= 1.0 + res.net_returns[t];
        res.equity[t] = eq;
    }
    res.metrics = compute_metrics(res.net_returns, dates);
    return res;
}

std::vector<StateStats> state_conditional_returns(const std::vector<double>& net,
                                                  const std::vector<int>& states,
                                                  int n_states) {
    if (net.size() != states.size())
        throw std::invalid_argument("net returns and states must have equal length");
    require_finite(net, "net returns");
    if (n_states < 1)
        throw std::invalid_argument("n_states must be an integer >= 1, got " +
                                    std::to_string(n_states));
    for (int s : states)
        if (s < 0 || s >= n_states)
            throw std::invalid_argument("state labels must be integers in [0, " +
                                        std::to_string(n_states) + ")");
    std::vector<StateStats> out(static_cast<std::size_t>(n_states));
    for (int k = 0; k < n_states; ++k) {
        double sum = 0.0;
        int n = 0;
        for (std::size_t t = 0; t < net.size(); ++t)
            if (states[t] == k) {
                sum += net[t];
                ++n;
            }
        StateStats& s = out[static_cast<std::size_t>(k)];
        s.n_days = n;
        if (n == 0) continue;
        const double mean = sum / n;
        double var = 0.0;
        for (std::size_t t = 0; t < net.size(); ++t)
            if (states[t] == k) var += (net[t] - mean) * (net[t] - mean);
        var /= n;  // ddof = 0
        s.ann_return = kTradingDays * mean;
        s.ann_vol = std::sqrt(static_cast<double>(kTradingDays)) * std::sqrt(var);
        s.sharpe = s.ann_vol > 0.0 ? s.ann_return / s.ann_vol : 0.0;
    }
    return out;
}

}  // namespace regime
