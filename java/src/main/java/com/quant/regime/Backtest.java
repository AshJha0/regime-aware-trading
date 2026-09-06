package com.quant.regime;

import java.time.LocalDate;

/**
 * Backtest engine with pinned no-lookahead accounting and metrics.
 *
 * <p><b>Accounting</b> (pinned, shared by every strategy). Positions
 * {@code P[t]} are decided at the close of day t; the trade from
 * {@code P[t-1]} to {@code P[t]} executes at that close, so
 *
 * <pre>
 *   gross[t]    = sum_a P[t-1, a] * r[t, a]          (gross[0] = 0)
 *   turnover[t] = sum_a |P[t, a] - P[t-1, a]|        (P[-1] = 0)
 *   net[t]      = gross[t] - (cost_bps / 10^4) * turnover[t]
 *   equity[t]   = prod_{u&lt;=t} (1 + net[u])
 * </pre>
 *
 * The day-t P&amp;L is earned by yesterday's position — the signal never sees
 * the return it trades against — while the cost of moving into {@code P[t]}
 * is charged on day t itself. {@code cost_bps = 0} gives net == gross
 * exactly.
 */
public final class Backtest {

    /** Trading days per year (pinned annualization constant). */
    public static final int TRADING_DAYS = 252;

    private Backtest() {
    }

    /**
     * Pinned summary metrics of a daily net-return series.
     *
     * @param annReturn {@code 252 * mean(net)}
     * @param annVol {@code sqrt(252) * std(net, ddof=0)}
     * @param sharpe {@code annReturn / annVol} (0 if annVol == 0)
     * @param maxDd minimum of {@code equity/cummax(equity) - 1} (non-positive)
     * @param calmar {@code annReturn / |maxDd|} (0 if maxDd == 0)
     * @param hitRate fraction of days with net &gt; 0
     * @param worstMonth worst calendar-month compounded net return
     *     (21-day blocks when no dates are given)
     */
    public record Metrics(double annReturn, double annVol, double sharpe, double maxDd,
            double calmar, double hitRate, double worstMonth) {
    }

    /**
     * Daily series and summary metrics of one backtest run.
     *
     * @param grossReturns pre-cost daily portfolio returns, length T
     * @param netReturns post-cost daily returns, length T
     * @param turnover daily one-sided turnover {@code sum|dP|}, length T
     * @param equity compounded equity curve, length T
     * @param metrics pinned summary metrics
     */
    public record Result(double[] grossReturns, double[] netReturns, double[] turnover,
            double[] equity, Metrics metrics) {
    }

    /**
     * Per-state performance for the crisis-attribution table.
     *
     * @param nDays days attributed to the state
     * @param annReturn annualized mean net return in the state
     * @param annVol annualized volatility (ddof = 0) in the state
     * @param sharpe annualized Sharpe (0 when annVol == 0)
     */
    public record StateStats(int nDays, double annReturn, double annVol, double sharpe) {
    }

    /** Pinned net-return checks: non-empty, finite, and &gt; -1 (wipe-out; API_SPEC 2). */
    private static void validateNet(double[] net) {
        if (net == null || net.length == 0) {
            throw new IllegalArgumentException("net return series is empty");
        }
        for (int t = 0; t < net.length; t++) {
            if (!Double.isFinite(net[t])) {
                throw new IllegalArgumentException("net returns contain NaN or inf");
            }
            if (net[t] <= -1.0) {
                throw new IllegalArgumentException(
                        "equity wiped out on day t=" + t + ": net return " + net[t] + " <= -100%");
            }
        }
    }

    /** Pinned date checks (API_SPEC 2.1): length T, non-null, strictly increasing. */
    private static void validateDates(LocalDate[] dates, int t2) {
        if (dates.length != t2) {
            throw new IllegalArgumentException(
                    "dates length " + dates.length + " does not match number of days " + t2);
        }
        for (int t = 0; t < t2; t++) {
            if (dates[t] == null) {
                throw new IllegalArgumentException("dates could not be parsed: null at t=" + t);
            }
            if (t > 0 && !dates[t - 1].isBefore(dates[t])) {
                throw new IllegalArgumentException("dates must be strictly increasing (at t=" + t + ")");
            }
        }
    }

    /**
     * Maximum drawdown of an equity curve, as a non-positive fraction.
     *
     * @param equity equity curve, length &gt;= 1, finite and strictly positive
     * @return {@code min_t (equity[t] / cummax(equity)[t] - 1)}
     * @throws IllegalArgumentException on an empty, non-finite or non-positive curve
     */
    public static double maxDrawdown(double[] equity) {
        if (equity == null || equity.length == 0) {
            throw new IllegalArgumentException("equity curve is empty");
        }
        for (double e : equity) {
            if (!Double.isFinite(e) || e <= 0.0) {
                throw new IllegalArgumentException("equity curve must be finite and strictly positive");
            }
        }
        double peak = Double.NEGATIVE_INFINITY;
        double mdd = Double.POSITIVE_INFINITY;
        for (double e : equity) {
            peak = Math.max(peak, e);
            mdd = Math.min(mdd, e / peak - 1.0);
        }
        return mdd;
    }

    /**
     * Pinned summary metrics from a daily net-return series.
     *
     * @param net daily net returns, length T (finite, each &gt; -1)
     * @param dates optional dates (length T, strictly increasing) for the
     *     worst-calendar-month metric; null falls back to consecutive 21-day
     *     blocks (trailing partial block dropped — pinned, API_SPEC 2.1)
     * @return the metrics
     * @throws IllegalArgumentException on empty/non-finite input, a net return
     *     &lt;= -1 (wipe-out), or dates that are the wrong length, null, or
     *     not strictly increasing
     */
    public static Metrics computeMetrics(double[] net, LocalDate[] dates) {
        validateNet(net);
        if (dates != null) {
            validateDates(dates, net.length);
        }
        int t2 = net.length;
        double mean = 0.0;
        for (double v : net) {
            mean += v;
        }
        mean /= t2;
        double var = 0.0;
        int hits = 0;
        for (double v : net) {
            double d = v - mean;
            var += d * d;
            if (v > 0.0) {
                hits++;
            }
        }
        var /= t2;
        double annRet = TRADING_DAYS * mean;
        double annVol = Math.sqrt(TRADING_DAYS) * Math.sqrt(var);
        double sharpe = annVol > 0.0 ? annRet / annVol : 0.0;
        double[] equity = new double[t2];
        double eq = 1.0;
        for (int t = 0; t < t2; t++) {
            eq *= 1.0 + net[t];
            equity[t] = eq;
        }
        double mdd = maxDrawdown(equity);
        double calmar = mdd < 0.0 ? annRet / Math.abs(mdd) : 0.0;
        double hitRate = (double) hits / t2;

        double worstMonth;
        if (dates != null) {
            worstMonth = Double.POSITIVE_INFINITY;
            double block = 1.0;
            for (int t = 0; t < t2; t++) {
                block *= 1.0 + net[t];
                boolean monthEnds = t == t2 - 1
                        || dates[t + 1].getYear() != dates[t].getYear()
                        || dates[t + 1].getMonthValue() != dates[t].getMonthValue();
                if (monthEnds) {
                    worstMonth = Math.min(worstMonth, block - 1.0);
                    block = 1.0;
                }
            }
        } else {
            int nblk = Math.max(t2 / 21, 1);
            worstMonth = Double.POSITIVE_INFINITY;
            for (int b = 0; b < nblk; b++) {
                double block = 1.0;
                for (int t = b * 21; t < Math.min((b + 1) * 21, t2); t++) {
                    block *= 1.0 + net[t];
                }
                worstMonth = Math.min(worstMonth, block - 1.0);
            }
        }
        return new Metrics(annRet, annVol, sharpe, mdd, calmar, hitRate, worstMonth);
    }

    /**
     * Runs the pinned accounting on a position/return pair.
     *
     * @param positions {@code P[t, a]} decided at close of day t, T x A
     * @param returns asset simple returns {@code r[t, a]}, T x A
     * @param costBps one-way transaction cost in basis points of turnover
     * @param dates optional dates (length T) for calendar-month metrics
     * @return the backtest result
     * @throws IllegalArgumentException on shape mismatch, empty input (no days
     *     or no assets), non-finite input, a return &lt;= -1, negative cost,
     *     invalid dates, or a wipe-out ({@code net[t] <= -1}; the message
     *     names t) — API_SPEC 2
     */
    public static Result run(double[][] positions, double[][] returns, double costBps, LocalDate[] dates) {
        if (positions == null || returns == null || positions.length == 0) {
            throw new IllegalArgumentException("empty backtest: no days");
        }
        if (positions.length != returns.length) {
            throw new IllegalArgumentException("positions and returns must have equal shape");
        }
        int t2 = positions.length;
        if (positions[0] == null || positions[0].length == 0) {
            throw new IllegalArgumentException("empty backtest: no assets");
        }
        int a2 = positions[0].length;
        for (int t = 0; t < t2; t++) {
            if (positions[t] == null || returns[t] == null
                    || positions[t].length != a2 || returns[t].length != a2) {
                throw new IllegalArgumentException("positions and returns must have equal shape");
            }
            for (int a = 0; a < a2; a++) {
                if (!Double.isFinite(positions[t][a]) || !Double.isFinite(returns[t][a])) {
                    throw new IllegalArgumentException("positions/returns contain NaN or inf");
                }
                if (returns[t][a] <= -1.0) {
                    throw new IllegalArgumentException("returns contain a return <= -100% at t=" + t
                            + ", asset=" + a + " (" + returns[t][a] + ")");
                }
            }
        }
        if (!Double.isFinite(costBps) || costBps < 0.0) {
            throw new IllegalArgumentException("cost_bps must be >= 0, got " + costBps);
        }
        if (dates != null) {
            validateDates(dates, t2);
        }
        double[] gross = new double[t2];
        double[] turnover = new double[t2];
        double[] net = new double[t2];
        double[] equity = new double[t2];
        double cost = costBps / 1e4;
        double eq = 1.0;
        for (int t = 0; t < t2; t++) {
            double g = 0.0;
            double to = 0.0;
            for (int a = 0; a < a2; a++) {
                double prev = t > 0 ? positions[t - 1][a] : 0.0;
                if (t > 0) {
                    g += prev * returns[t][a];
                }
                to += Math.abs(positions[t][a] - prev);
            }
            gross[t] = g;
            turnover[t] = to;
            net[t] = g - cost * to;
        }
        validateNet(net); // wipe-out guard: net[t] <= -1 is an error naming t
        for (int t = 0; t < t2; t++) {
            eq *= 1.0 + net[t];
            equity[t] = eq;
        }
        return new Result(gross, net, turnover, equity, computeMetrics(net, dates));
    }

    /**
     * Annualized performance of a return series conditioned on a state path.
     *
     * <p>Used for the crisis-period table: the day-t net return is attributed
     * to the day-t decoded state.
     *
     * @param net daily net returns, length T
     * @param states integer state labels per day, length T
     * @param nStates number of states K
     * @return per-state statistics, index = state label
     * @throws IllegalArgumentException when lengths disagree, net is
     *     non-finite, {@code nStates < 1}, or a label is outside
     *     {@code [0, nStates)} (pinned: labels are never silently dropped)
     */
    public static StateStats[] stateConditionalReturns(double[] net, int[] states, int nStates) {
        if (net == null || states == null || net.length != states.length) {
            throw new IllegalArgumentException("net returns and states must have equal length");
        }
        for (double v : net) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("net returns contain NaN or inf");
            }
        }
        if (nStates < 1) {
            throw new IllegalArgumentException("n_states must be an integer >= 1, got " + nStates);
        }
        for (int s : states) {
            if (s < 0 || s >= nStates) {
                throw new IllegalArgumentException("state labels must be integers in [0, " + nStates + ")");
            }
        }
        StateStats[] out = new StateStats[nStates];
        for (int k = 0; k < nStates; k++) {
            int n = 0;
            double mean = 0.0;
            for (int t = 0; t < net.length; t++) {
                if (states[t] == k) {
                    n++;
                    mean += net[t];
                }
            }
            if (n == 0) {
                out[k] = new StateStats(0, 0.0, 0.0, 0.0);
                continue;
            }
            mean /= n;
            double var = 0.0;
            for (int t = 0; t < net.length; t++) {
                if (states[t] == k) {
                    double d = net[t] - mean;
                    var += d * d;
                }
            }
            var /= n;
            double mu = TRADING_DAYS * mean;
            double sd = Math.sqrt(TRADING_DAYS) * Math.sqrt(var);
            out[k] = new StateStats(n, mu, sd, sd > 0.0 ? mu / sd : 0.0);
        }
        return out;
    }
}
