package com.quant.regime;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Trading strategies: time-series momentum, FX carry, and the HMM regime gate.
 *
 * <p><b>No-lookahead convention</b> (shared with the backtest engine): every
 * position matrix produced here has {@code P[t]} = the position decided at
 * the close of day t using information up to and including day t. The
 * backtest applies {@code P[t-1]} to the day-t return, so a signal can never
 * see the return it is traded against. The regime gate at day t likewise
 * uses only observations {@code x_0..x_t} (filtered, not smoothed,
 * probabilities) and models fitted on data through day t.
 */
public final class Strategies {

    /** Trading days per year (pinned annualization constant). */
    public static final int TRADING_DAYS = 252;

    private Strategies() {
    }

    private static double[][] validate(double[][] r, String name) {
        return validate(r, name, false);
    }

    /**
     * Non-empty, rectangular, all finite. With {@code simpleReturns} (pinned,
     * API_SPEC 3) every entry must also be &gt; -1: a -100 % (or worse) day
     * is corrupt data and would make every trailing growth ratio 0/0.
     */
    private static double[][] validate(double[][] r, String name, boolean simpleReturns) {
        if (r == null || r.length == 0 || r[0] == null || r[0].length == 0) {
            throw new IllegalArgumentException(name + " must be a non-empty 2-D array");
        }
        int a = r[0].length;
        for (int t = 0; t < r.length; t++) {
            double[] row = r[t];
            if (row == null || row.length != a) {
                throw new IllegalArgumentException(name + " is ragged");
            }
            for (int j = 0; j < a; j++) {
                if (!Double.isFinite(row[j])) {
                    throw new IllegalArgumentException(name + " contain NaN or inf");
                }
                if (simpleReturns && row[j] <= -1.0) {
                    throw new IllegalArgumentException(name + " contain a return <= -100% at t=" + t
                            + ", asset=" + j + " (" + row[j] + ")");
                }
            }
        }
        return r;
    }

    /**
     * EWMA variance per asset with a pinned initialization.
     *
     * <p>Pinned rule: at {@code t = initWindow - 1} the variance is the mean
     * of squared returns over the first {@code initWindow} days (zero-mean
     * convention), and for {@code t >= initWindow}
     * {@code sigma2[t] = lam * sigma2[t-1] + (1 - lam) * r[t]^2}. Entries
     * before {@code initWindow - 1} are NaN (undefined).
     *
     * @param returns daily returns, T x A
     * @param lam EWMA decay in (0, 1), e.g. 0.94
     * @param initWindow seed window length (the momentum lookback)
     * @return T x A daily-frequency variances
     * @throws IllegalArgumentException on invalid inputs
     */
    public static double[][] ewmaVariance(double[][] returns, double lam, int initWindow) {
        double[][] r = validate(returns, "returns", true);
        int t2 = r.length;
        int a2 = r[0].length;
        if (!(lam > 0.0 && lam < 1.0)) {
            throw new IllegalArgumentException("ewma lambda must be in (0, 1), got " + lam);
        }
        if (initWindow < 1 || initWindow > t2) {
            throw new IllegalArgumentException("init_window must be in [1, T], got " + initWindow);
        }
        double[][] sig2 = new double[t2][a2];
        for (double[] row : sig2) {
            Arrays.fill(row, Double.NaN);
        }
        for (int a = 0; a < a2; a++) {
            double seed = 0.0;
            for (int t = 0; t < initWindow; t++) {
                seed += r[t][a] * r[t][a];
            }
            sig2[initWindow - 1][a] = seed / initWindow;
            for (int t = initWindow; t < t2; t++) {
                sig2[t][a] = lam * sig2[t - 1][a] + (1.0 - lam) * r[t][a] * r[t][a];
            }
        }
        return sig2;
    }

    /**
     * Time-series momentum positions with vol-targeted sizing.
     *
     * <p>Signal at day t (defined for {@code t >= lookback - 1}): the sign of
     * the cumulative simple return {@code prod(1 + r) - 1} over the trailing
     * {@code lookback} days ending at t (sign in {-1, 0, +1}). Sizing:
     * per-asset leverage {@code min(volTarget / annVol[t], leverageCap)}
     * where {@code annVol = sqrt(252 * sigma2_EWMA)}; a zero EWMA vol gets
     * the cap. Positions are divided by the number of assets (equal risk
     * budget) and are zero before the first full lookback window.
     *
     * @param returns daily simple returns, T x A
     * @param lookback trailing window length in days
     * @param volTarget annualized per-asset volatility target
     * @param ewmaLambda EWMA decay for the vol estimate
     * @param leverageCap maximum per-asset leverage
     * @return positions T x A; {@code P[t]} is decided at close of day t
     * @throws IllegalArgumentException if the series is not longer than the
     *     lookback, or any parameter is invalid
     */
    public static double[][] momentumPositions(double[][] returns, int lookback, double volTarget,
            double ewmaLambda, double leverageCap) {
        double[][] r = validate(returns, "returns", true);
        int t2 = r.length;
        int a2 = r[0].length;
        if (lookback < 1) {
            throw new IllegalArgumentException("lookback must be >= 1, got " + lookback);
        }
        if (t2 <= lookback) {
            throw new IllegalArgumentException(
                    "series shorter than momentum lookback: T=" + t2 + " <= lookback=" + lookback);
        }
        if (volTarget <= 0.0 || leverageCap <= 0.0) {
            throw new IllegalArgumentException("vol_target and leverage_cap must be > 0");
        }

        // Trailing cumulative return via cumulative growth ratios.
        double[][] growth = new double[t2][a2];
        for (int a = 0; a < a2; a++) {
            double g = 1.0;
            for (int t = 0; t < t2; t++) {
                g *= 1.0 + r[t][a];
                growth[t][a] = g;
            }
        }
        double[][] sig2 = ewmaVariance(r, ewmaLambda, lookback);
        double[][] pos = new double[t2][a2];
        for (int t = lookback - 1; t < t2; t++) {
            for (int a = 0; a < a2; a++) {
                double cumret = t == lookback - 1
                        ? growth[t][a] - 1.0
                        : growth[t][a] / growth[t - lookback][a] - 1.0;
                double signal = Math.signum(cumret);
                double annVol = Math.sqrt(TRADING_DAYS * sig2[t][a]);
                double lev = annVol > 0.0 ? Math.min(volTarget / annVol, leverageCap) : leverageCap;
                pos[t][a] = signal * lev / a2;
            }
        }
        return pos;
    }

    /**
     * FX carry positions: long high-differential, short low-differential.
     *
     * <p>Pinned schedule: rebalance at day indices 0, 21, 42, ... (every
     * {@code rebalanceDays} trading days); positions are held unchanged
     * between rebalances. At each rebalance the currencies are ranked by that
     * day's differential, descending; ties are broken by ascending asset
     * index. The top {@code topN} get {@code +1/topN} each and the bottom
     * {@code bottomN} get {@code -1/bottomN} each (equal notional).
     *
     * @param rateDiffs annualized interest differentials, T x A
     * @param topN number of longs
     * @param bottomN number of shorts
     * @param rebalanceDays trading days between rebalances
     * @return positions T x A; {@code P[t]} decided at close of day t
     * @throws IllegalArgumentException if {@code topN + bottomN} exceeds the
     *     asset count or the schedule/params are invalid
     */
    public static double[][] carryPositions(double[][] rateDiffs, int topN, int bottomN, int rebalanceDays) {
        double[][] d = validate(rateDiffs, "rate differentials");
        int t2 = d.length;
        int a2 = d[0].length;
        if (topN < 1 || bottomN < 1 || topN + bottomN > a2) {
            throw new IllegalArgumentException(
                    "need top_n>=1, bottom_n>=1, top_n+bottom_n<=A=" + a2 + "; got " + topN + "," + bottomN);
        }
        if (rebalanceDays < 1) {
            throw new IllegalArgumentException("rebalance_days must be >= 1, got " + rebalanceDays);
        }
        double[][] pos = new double[t2][a2];
        double[] current = new double[a2];
        Integer[] order = new Integer[a2];
        for (int t = 0; t < t2; t++) {
            if (t % rebalanceDays == 0) {
                final double[] diffs = d[t];
                for (int a = 0; a < a2; a++) {
                    order[a] = a;
                }
                // Descending differential; ascending asset index breaks ties.
                Arrays.sort(order, (x, y) -> {
                    if (diffs[x] > diffs[y]) {
                        return -1;
                    }
                    if (diffs[x] < diffs[y]) {
                        return 1;
                    }
                    return Integer.compare(x, y);
                });
                Arrays.fill(current, 0.0);
                for (int i = 0; i < topN; i++) {
                    current[order[i]] = 1.0 / topN;
                }
                for (int i = a2 - bottomN; i < a2; i++) {
                    current[order[i]] = -1.0 / bottomN;
                }
            }
            System.arraycopy(current, 0, pos[t], 0, a2);
        }
        return pos;
    }

    /**
     * Total currency returns: spot move plus accrued carry.
     *
     * <p>Pinned accrual: the day-t total return is
     * {@code spot[t] + diff[t-1] / 252} — the rate differential set at the
     * previous close (known in advance, no lookahead). Day 0 accrues nothing.
     *
     * @param spotReturns daily spot returns, T x A
     * @param rateDiffs annualized differentials, T x A
     * @return total returns, T x A
     * @throws IllegalArgumentException on shape mismatch or invalid input
     */
    public static double[][] carryTotalReturns(double[][] spotReturns, double[][] rateDiffs) {
        double[][] s = validate(spotReturns, "spot returns", true);
        double[][] d = validate(rateDiffs, "rate differentials");
        if (s.length != d.length || s[0].length != d[0].length) {
            throw new IllegalArgumentException("spot returns and differentials must have equal shape");
        }
        int t2 = s.length;
        int a2 = s[0].length;
        double[][] total = new double[t2][a2];
        for (int t = 0; t < t2; t++) {
            for (int a = 0; a < a2; a++) {
                total[t][a] = s[t][a] + (t > 0 ? d[t - 1][a] / TRADING_DAYS : 0.0);
            }
        }
        return total;
    }

    /**
     * Result of the walk-forward regime gate.
     *
     * @param gate gate values in [0, 1], length T
     * @param models fitted models in refit order
     */
    public record GateResult(double[] gate, List<GaussianHmm> models) {
    }

    /**
     * Walk-forward HMM regime gate on a market proxy.
     *
     * <p>Pinned schedule: the first fit happens at day
     * {@code t0 = trainMinDays - 1} on the expanding window {@code r[0..t0]};
     * refits at {@code t0 + refitDays * k} on {@code r[0..t]}. The first fit
     * starts from the pinned quantile initialization; every subsequent refit
     * warm-starts from the previous fitted (label-sorted) parameters.
     *
     * <p>The gate at day t (for {@code t >= t0}) is the <em>filtered</em>
     * probability {@code p_calm(t) = P(s_t = k* | r_0..r_t)} of the
     * low-vol/bull state under the model most recently fitted at or before t,
     * with the pinned identification {@code k* = argmin_k variances[k][0]}
     * (ties toward the lower label). A full scaled forward pass runs at each
     * refit day; single {@link GaussianHmm#forwardStep} updates are used in
     * between (identical math, no recomputation, no lookahead). Before the
     * first fit the gate is 1.
     *
     * <p>Mode "prob" returns {@code gate[t] = p_calm(t)}; mode "binary"
     * returns 1 if {@code p_calm(t) >= threshold} else 0.
     *
     * @param indexReturns market proxy daily returns, length T
     * @param nStates HMM state count K
     * @param trainMinDays observations required before the first fit
     * @param refitDays trading days between refits
     * @param mode "prob" (scale by probability) or "binary" (hard gate)
     * @param threshold probability threshold for the binary gate
     * @param tol EM tolerance
     * @param maxIter EM iteration cap
     * @param varFloor EM variance floor
     * @return gate values and the fitted models
     * @throws IllegalArgumentException if the series is shorter than
     *     {@code trainMinDays}, the mode is unknown, {@code threshold} is
     *     outside [0, 1], the HMM settings are invalid ({@code nStates < 2}
     *     etc.), or a refit fails (the message then names the refit day t)
     */
    public static GateResult regimeGate(double[] indexReturns, int nStates, int trainMinDays,
            int refitDays, String mode, double threshold, double tol, int maxIter, double varFloor) {
        if (indexReturns == null) {
            throw new IllegalArgumentException("index_returns must be 1-D");
        }
        for (double v : indexReturns) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("index_returns contain NaN or inf");
            }
        }
        if (!"prob".equals(mode) && !"binary".equals(mode)) {
            throw new IllegalArgumentException("gate mode must be 'prob' or 'binary', got '" + mode + "'");
        }
        if (!(threshold >= 0.0 && threshold <= 1.0)) {
            throw new IllegalArgumentException("gate threshold must be in [0, 1], got " + threshold);
        }
        // Validates nStates >= 2, tol/varFloor > 0, maxIter >= 1 up front.
        new GaussianHmm(nStates, tol, maxIter, varFloor);
        int t2 = indexReturns.length;
        if (trainMinDays <= nStates || refitDays < 1) {
            throw new IllegalArgumentException(
                    "train_min_days must exceed n_states and refit_days must be >= 1");
        }
        if (t2 < trainMinDays) {
            throw new IllegalArgumentException(
                    "series shorter than train_min_days: T=" + t2 + " < " + trainMinDays);
        }
        int t0 = trainMinDays - 1;
        double[] gate = new double[t2];
        Arrays.fill(gate, 1.0);
        List<GaussianHmm> models = new ArrayList<>();
        GaussianHmm model = null;
        double[] alpha = null;
        int calmState = 0;
        for (int t = t0; t < t2; t++) {
            if ((t - t0) % refitDays == 0) {
                model = new GaussianHmm(nStates, tol, maxIter, varFloor);
                HmmParams warm = models.isEmpty() ? null : models.get(models.size() - 1).params().copy();
                double[] window = Arrays.copyOfRange(indexReturns, 0, t + 1);
                try {
                    model.fit(window, warm);
                    double[][] filt = model.filteredProbabilities(window);
                    alpha = filt[filt.length - 1];
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException("regime_gate refit at t=" + t + ": " + e.getMessage(), e);
                }
                models.add(model);
                calmState = argminVariance(model.params());
            } else {
                try {
                    alpha = GaussianHmm.forwardStep(model.params(), alpha, new double[] {indexReturns[t]});
                } catch (IllegalArgumentException e) {
                    throw new IllegalArgumentException(
                            "regime_gate forward step at t=" + t + ": " + e.getMessage(), e);
                }
            }
            double pCalm = alpha[calmState];
            gate[t] = "prob".equals(mode) ? pCalm : (pCalm >= threshold ? 1.0 : 0.0);
        }
        return new GateResult(gate, models);
    }

    /** Pinned gate-state rule: lowest variance on dimension 0, ties to the lower label. */
    private static int argminVariance(HmmParams p) {
        int arg = 0;
        for (int k = 1; k < p.variances.length; k++) {
            if (p.variances[k][0] < p.variances[arg][0]) {
                arg = k;
            }
        }
        return arg;
    }

    /**
     * Scales each day's positions by that day's gate value.
     *
     * <p>{@code P_filtered[t] = gate[t] * P[t]}; both are day-t decisions, so
     * the filtered position still only trades against the day t+1 return.
     *
     * @param positions positions T x A
     * @param gate gate values, length T
     * @return filtered positions T x A
     * @throws IllegalArgumentException on length mismatch or non-finite gate
     */
    public static double[][] applyGate(double[][] positions, double[] gate) {
        double[][] p = validate(positions, "positions");
        if (gate == null || gate.length != p.length) {
            throw new IllegalArgumentException("gate length does not match positions");
        }
        for (double g : gate) {
            if (!Double.isFinite(g)) {
                throw new IllegalArgumentException("gate contains NaN or inf");
            }
        }
        double[][] out = new double[p.length][p[0].length];
        for (int t = 0; t < p.length; t++) {
            for (int a = 0; a < p[0].length; a++) {
                out[t][a] = p[t][a] * gate[t];
            }
        }
        return out;
    }
}
