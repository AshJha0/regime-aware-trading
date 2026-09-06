package com.quant.regime;

import java.util.Arrays;

/**
 * Diagonal-covariance Gaussian HMM with deterministic Baum-Welch EM.
 *
 * <p>Everything here is deterministic: initialization is pinned
 * (quantile-based means, 0.9 self-transition prior, sample variance) and no
 * random number generator is used anywhere.
 *
 * <p><b>Scaled forward-backward</b> (pinned: scaled, not log-space). To guard
 * the emission densities against under/overflow the per-step maximum
 * log-emission {@code m_t = max_i log b_t(i)} is factored out, giving
 * {@code bt_t(i) = exp(log b_t(i) - m_t) <= 1}. The forward recursion
 * normalizes each step by {@code d_t = sum_j a~_t(j)}; the true per-step
 * normalizer is {@code c_t = d_t * exp(m_t)} and the log-likelihood is the
 * sum of the log normalizers {@code sum_t (log d_t + m_t)}. The backward
 * pass shares the same {@code d_t}, so the posteriors
 * {@code gamma_t(i) = a^_t(i) * b^_t(i)} need no further normalization.
 *
 * <p><b>Dead-state guard</b> (pinned, API_SPEC 1.4). After each E-step the
 * transition-row support {@code sum_{t<T} gamma_t(i)} of every state is
 * checked against {@link #DEAD_STATE_SUPPORT}; a state below it has no
 * posterior mass (Rabiner 1989 section V.B, Bilmes 1998 section 4) and the
 * M-step would produce 0/0. EM then stops with the just-scored parameters,
 * {@code converged = false} and {@code deadState = i}. No NaN ever leaves
 * {@link #fit}. A zero forward normalizer (data impossible under every
 * reachable state) throws.
 */
public final class GaussianHmm {

    private static final double LOG_2PI = Math.log(2.0 * Math.PI);

    /** Pinned dead-state threshold on {@code sum_{t<T} gamma_t(i)} (API_SPEC 1.4). */
    public static final double DEAD_STATE_SUPPORT = 1e-12;
    /** Pinned relative tolerance for a log-likelihood decrease to count as non-monotone. */
    public static final double MONOTONE_REL_TOL = 1e-6;
    /** Pinned tolerance on {@code sum(startprob)} and on every transition row sum. */
    public static final double STOCHASTIC_TOL = 1e-9;

    /** Classification of one E-step score against the previous one (pinned). */
    public enum EmStep {
        /** Improvement at or above the tolerance: keep iterating. */
        CONTINUE,
        /** Improvement below the tolerance (and no violation): converged. */
        CONVERGED,
        /** A decrease beyond the relative tolerance or a non-finite score. */
        NON_MONOTONE
    }

    /**
     * Pinned rule (API_SPEC 1.4): {@code NON_MONOTONE} if
     * {@code ll - llPrev < -MONOTONE_REL_TOL * max(1, |llPrev|)} or either score
     * is non-finite; else {@code CONVERGED} if {@code ll - llPrev < tol}; else
     * {@code CONTINUE}.
     *
     * @param ll current E-step score
     * @param llPrev previous E-step score
     * @param tol convergence tolerance
     * @return the classification
     */
    public static EmStep emStepStatus(double ll, double llPrev, double tol) {
        if (!Double.isFinite(ll) || !Double.isFinite(llPrev)) {
            return EmStep.NON_MONOTONE;
        }
        double delta = ll - llPrev;
        if (delta < -MONOTONE_REL_TOL * Math.max(1.0, Math.abs(llPrev))) {
            return EmStep.NON_MONOTONE;
        }
        if (delta < tol) {
            return EmStep.CONVERGED;
        }
        return EmStep.CONTINUE;
    }

    /**
     * Validates a parameter set against a model size (pinned, API_SPEC 1.9):
     * {@code startprob} length K, {@code transmat} K x K, {@code means} and
     * {@code variances} K x D, every entry finite, variances &gt; 0,
     * {@code startprob} and every transition row non-negative and summing to
     * 1 within {@link #STOCHASTIC_TOL}.
     *
     * @param p parameter set
     * @param nStates expected K
     * @param nDims expected D
     * @throws IllegalArgumentException on any violation (never an
     *     {@code ArrayIndexOutOfBoundsException} or {@code NullPointerException})
     */
    public static void validateParams(HmmParams p, int nStates, int nDims) {
        if (p == null || p.startprob == null || p.transmat == null || p.means == null
                || p.variances == null) {
            throw new IllegalArgumentException("parameter set is null");
        }
        int k = nStates;
        int d = nDims;
        if (p.startprob.length != k) {
            throw new IllegalArgumentException(
                    "startprob length " + p.startprob.length + " does not match n_states=" + k);
        }
        checkShape(p.transmat, k, k, "transmat", "n_states=" + k);
        checkShape(p.means, k, d, "means", "(n_states, D)=(" + k + ", " + d + ")");
        checkShape(p.variances, k, d, "variances", "(n_states, D)=(" + k + ", " + d + ")");
        for (double v : p.startprob) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("startprob contain NaN or inf");
            }
        }
        checkFinite(p.transmat, "transmat");
        checkFinite(p.means, "means");
        checkFinite(p.variances, "variances");
        for (double[] row : p.variances) {
            for (double v : row) {
                if (!(v > 0.0)) {
                    throw new IllegalArgumentException("variances must be > 0");
                }
            }
        }
        double sp = 0.0;
        for (double v : p.startprob) {
            if (v < 0.0) {
                throw new IllegalArgumentException("startprob must be non-negative and sum to 1");
            }
            sp += v;
        }
        if (Math.abs(sp - 1.0) > STOCHASTIC_TOL) {
            throw new IllegalArgumentException("startprob must be non-negative and sum to 1");
        }
        for (double[] row : p.transmat) {
            double sum = 0.0;
            for (double v : row) {
                if (v < 0.0) {
                    throw new IllegalArgumentException("transmat rows must be non-negative and sum to 1");
                }
                sum += v;
            }
            if (Math.abs(sum - 1.0) > STOCHASTIC_TOL) {
                throw new IllegalArgumentException("transmat rows must be non-negative and sum to 1");
            }
        }
    }

    private static void checkShape(double[][] m, int rows, int cols, String name, String want) {
        if (m.length != rows) {
            throw new IllegalArgumentException(name + " shape " + m.length + "x? does not match " + want);
        }
        for (double[] row : m) {
            if (row == null || row.length != cols) {
                throw new IllegalArgumentException(name + " shape " + rows + "x"
                        + (row == null ? "null" : String.valueOf(row.length)) + " does not match " + want);
            }
        }
    }

    private static void checkFinite(double[][] m, String name) {
        for (double[] row : m) {
            for (double v : row) {
                if (!Double.isFinite(v)) {
                    throw new IllegalArgumentException(name + " contain NaN or inf");
                }
            }
        }
    }

    private final int nStates;
    private final double tol;
    private final int maxIter;
    private final double varFloor;

    private HmmParams params;
    private HmmFitResult fitResult;

    /**
     * Creates an unfitted model with the pinned default EM settings
     * (tol 1e-8, max 500 iterations, variance floor 1e-8).
     *
     * @param nStates number of hidden states K (must be at least 2)
     */
    public GaussianHmm(int nStates) {
        this(nStates, 1e-8, 500, 1e-8);
    }

    /**
     * Creates an unfitted model.
     *
     * @param nStates number of hidden states K; must be at least 2 (K = 1 is
     *     a degenerate model and is rejected)
     * @param tol convergence tolerance on the log-likelihood improvement
     * @param maxIter maximum number of EM iterations
     * @param varFloor lower bound applied to every variance after each update
     *     (guards zero-variance states when K exceeds the data support)
     * @throws IllegalArgumentException on invalid settings
     */
    public GaussianHmm(int nStates, double tol, int maxIter, double varFloor) {
        if (nStates < 2) {
            throw new IllegalArgumentException(
                    "n_states must be an integer >= 2 (K=1 is degenerate), got " + nStates);
        }
        if (tol <= 0.0 || maxIter < 1 || varFloor <= 0.0) {
            throw new IllegalArgumentException("tol and var_floor must be > 0 and max_iter >= 1");
        }
        this.nStates = nStates;
        this.tol = tol;
        this.maxIter = maxIter;
        this.varFloor = varFloor;
    }

    /** Returns the number of hidden states K. */
    public int nStates() {
        return nStates;
    }

    /** Returns the fitted (label-sorted) parameters, or null before fitting. */
    public HmmParams params() {
        return params;
    }

    /**
     * Sets the parameters directly (for scoring hand-built models in tests).
     *
     * @param p parameter set; must be valid for {@code nStates} with D in {1, 2}
     * @throws IllegalArgumentException if invalid ({@link #validateParams})
     */
    public void setParams(HmmParams p) {
        if (p == null || p.means == null || p.means.length == 0 || p.means[0] == null
                || (p.means[0].length != 1 && p.means[0].length != 2)) {
            throw new IllegalArgumentException("only univariate or 2-D parameters supported");
        }
        validateParams(p, nStates, p.means[0].length);
        this.params = p;
    }

    /** Returns the result of the last {@link #fit}, or null before fitting. */
    public HmmFitResult fitResult() {
        return fitResult;
    }

    // ------------------------------------------------------------------ //
    // Observation validation
    // ------------------------------------------------------------------ //

    /**
     * Wraps a univariate series as a T x 1 observation matrix.
     *
     * @param x series of length T
     * @return T x 1 matrix sharing no storage with {@code x}
     */
    public static double[][] column(double[] x) {
        double[][] out = new double[x.length][1];
        for (int t = 0; t < x.length; t++) {
            out[t][0] = x[t];
        }
        return out;
    }

    private static double[][] validate(double[][] x) {
        if (x == null || x.length == 0) {
            throw new IllegalArgumentException("observation sequence is empty");
        }
        int d = x[0].length;
        if (d != 1 && d != 2) {
            throw new IllegalArgumentException(
                    "only univariate or 2-D observations supported, got D=" + d);
        }
        for (double[] row : x) {
            if (row.length != d) {
                throw new IllegalArgumentException("ragged observation matrix");
            }
            for (double v : row) {
                if (!Double.isFinite(v)) {
                    throw new IllegalArgumentException("observations contain NaN or inf");
                }
            }
        }
        return x;
    }

    private HmmParams requireFitted() {
        if (params == null) {
            throw new IllegalArgumentException("model is not fitted; call fit() first");
        }
        return params;
    }

    /** Validates x and the fitted parameters' agreement with (nStates, D). */
    private HmmParams prepare(double[][] x) {
        HmmParams p = requireFitted();
        validate(x);
        validateParams(p, nStates, x[0].length);
        return p;
    }

    // ------------------------------------------------------------------ //
    // Initialization (pinned, RNG-free)
    // ------------------------------------------------------------------ //

    /**
     * Deterministic EM starting point.
     *
     * <p>Pinned rule: state k gets per-dimension means at the empirical
     * quantile {@code q_k = (2k + 1) / (2K)} (linear interpolation), every
     * state gets the full-sample variance (ddof = 0, floored), the transition
     * matrix has 0.9 self-transitions with the remaining mass spread
     * uniformly, and the initial distribution is uniform.
     *
     * @param x observations, T x D
     * @return the pinned initial parameters
     */
    public HmmParams pinnedInit(double[][] x) {
        x = validate(x);
        int k = nStates;
        int d = x[0].length;
        double[][] means = new double[k][d];
        double[][] variances = new double[k][d];
        for (int dim = 0; dim < d; dim++) {
            double[] col = new double[x.length];
            for (int t = 0; t < x.length; t++) {
                col[t] = x[t][dim];
            }
            Arrays.sort(col);
            double mean = 0.0;
            for (int t = 0; t < x.length; t++) {
                mean += x[t][dim];
            }
            mean /= x.length;
            double var = 0.0;
            for (int t = 0; t < x.length; t++) {
                double diff = x[t][dim] - mean;
                var += diff * diff;
            }
            var = Math.max(var / x.length, varFloor);
            for (int s = 0; s < k; s++) {
                double q = (2.0 * s + 1.0) / (2.0 * k);
                means[s][dim] = quantileSorted(col, q);
                variances[s][dim] = var;
            }
        }
        double[][] transmat = new double[k][k];
        for (int i = 0; i < k; i++) {
            Arrays.fill(transmat[i], 0.1 / (k - 1));
            transmat[i][i] = 0.9;
        }
        double[] startprob = new double[k];
        Arrays.fill(startprob, 1.0 / k);
        return new HmmParams(startprob, transmat, means, variances);
    }

    /**
     * Linear-interpolation quantile of a sorted sample (numpy default):
     * index {@code h = q * (n - 1)} on the sorted values,
     * {@code v = x[floor(h)] + (h - floor(h)) * (x[floor(h)+1] - x[floor(h)])}.
     */
    private static double quantileSorted(double[] sorted, double q) {
        int n = sorted.length;
        double h = q * (n - 1);
        int lo = (int) Math.floor(h);
        if (lo >= n - 1) {
            return sorted[n - 1];
        }
        return sorted[lo] + (h - lo) * (sorted[lo + 1] - sorted[lo]);
    }

    // ------------------------------------------------------------------ //
    // Emissions and forward/backward
    // ------------------------------------------------------------------ //

    /** Log emission densities, T x K: {@code log b_t(i) = sum_d log N(x_td; mu_id, var_id)}. */
    private static double[][] logEmissions(double[][] x, HmmParams p) {
        int t2 = x.length;
        int k = p.means.length;
        int d = x[0].length;
        double[][] logb = new double[t2][k];
        for (int t = 0; t < t2; t++) {
            for (int i = 0; i < k; i++) {
                double s = 0.0;
                for (int dim = 0; dim < d; dim++) {
                    double v = p.variances[i][dim];
                    double diff = x[t][dim] - p.means[i][dim];
                    s += LOG_2PI + Math.log(v) + diff * diff / v;
                }
                logb[t][i] = -0.5 * s;
            }
        }
        return logb;
    }

    /** Scaled forward pass: alpha (T x K), normalizers d and shifts m. */
    private record Forward(double[][] alpha, double[] d, double[] m) {
    }

    private static Forward forward(double[][] logb, HmmParams p) {
        int t2 = logb.length;
        int k = logb[0].length;
        double[] m = new double[t2];
        double[][] bt = shiftedEmissions(logb, m);
        double[][] alpha = new double[t2][k];
        double[] d = new double[t2];
        double sum = 0.0;
        for (int i = 0; i < k; i++) {
            alpha[0][i] = p.startprob[i] * bt[0][i];
            sum += alpha[0][i];
        }
        if (!(sum > 0.0)) {
            throw zeroNormalizer(0);
        }
        d[0] = sum;
        for (int i = 0; i < k; i++) {
            alpha[0][i] /= sum;
        }
        for (int t = 1; t < t2; t++) {
            sum = 0.0;
            for (int j = 0; j < k; j++) {
                double a = 0.0;
                for (int i = 0; i < k; i++) {
                    a += alpha[t - 1][i] * p.transmat[i][j];
                }
                a *= bt[t][j];
                alpha[t][j] = a;
                sum += a;
            }
            if (!(sum > 0.0)) {
                throw zeroNormalizer(t);
            }
            d[t] = sum;
            for (int j = 0; j < k; j++) {
                alpha[t][j] /= sum;
            }
        }
        return new Forward(alpha, d, m);
    }

    private static IllegalArgumentException zeroNormalizer(int t) {
        return new IllegalArgumentException("forward normalizer is zero at t=" + t
                + ": the observation has zero likelihood under every reachable state"
                + " (parameters and data are incompatible)");
    }

    /** Fills {@code m} with per-row maxima and returns {@code exp(logb - m)}. */
    private static double[][] shiftedEmissions(double[][] logb, double[] m) {
        int t2 = logb.length;
        int k = logb[0].length;
        double[][] bt = new double[t2][k];
        for (int t = 0; t < t2; t++) {
            double mx = logb[t][0];
            for (int i = 1; i < k; i++) {
                mx = Math.max(mx, logb[t][i]);
            }
            m[t] = mx;
            for (int i = 0; i < k; i++) {
                bt[t][i] = Math.exp(logb[t][i] - mx);
            }
        }
        return bt;
    }

    /** Scaled backward pass sharing the forward normalizers {@code d}. */
    private static double[][] backward(double[][] logb, HmmParams p, double[] d, double[] m) {
        int t2 = logb.length;
        int k = logb[0].length;
        double[][] bt = shiftedEmissions(logb, new double[t2]);
        double[][] beta = new double[t2][k];
        Arrays.fill(beta[t2 - 1], 1.0);
        for (int t = t2 - 2; t >= 0; t--) {
            for (int i = 0; i < k; i++) {
                double s = 0.0;
                for (int j = 0; j < k; j++) {
                    s += p.transmat[i][j] * bt[t + 1][j] * beta[t + 1][j];
                }
                beta[t][i] = s / d[t + 1];
            }
        }
        return beta;
    }

    private static double logLik(double[] d, double[] m) {
        double s = 0.0;
        for (int t = 0; t < d.length; t++) {
            s += Math.log(d[t]) + m[t];
        }
        return s;
    }

    // ------------------------------------------------------------------ //
    // Public inference API
    // ------------------------------------------------------------------ //

    /**
     * Log-likelihood of the observations under the fitted parameters.
     *
     * @param x observations, T x D
     * @return {@code log p(x_1..x_T)}
     */
    public double score(double[][] x) {
        HmmParams p = prepare(x);
        Forward f = forward(logEmissions(x, p), p);
        return logLik(f.d, f.m);
    }

    /** Univariate overload of {@link #score(double[][])}. */
    public double score(double[] x) {
        return score(column(x));
    }

    /**
     * Filtered posteriors {@code P(s_t = i | x_1..t)}, T x K.
     *
     * @param x observations, T x D
     * @return filtered probability rows
     */
    public double[][] filteredProbabilities(double[][] x) {
        HmmParams p = prepare(x);
        return forward(logEmissions(x, p), p).alpha;
    }

    /** Univariate overload of {@link #filteredProbabilities(double[][])}. */
    public double[][] filteredProbabilities(double[] x) {
        return filteredProbabilities(column(x));
    }

    /**
     * Smoothed posteriors {@code P(s_t = i | x_1..T)}, T x K.
     *
     * @param x observations, T x D
     * @return smoothed probability rows (each sums to 1)
     */
    public double[][] smoothedProbabilities(double[][] x) {
        HmmParams p = prepare(x);
        double[][] logb = logEmissions(x, p);
        Forward f = forward(logb, p);
        double[][] beta = backward(logb, p, f.d, f.m);
        int k = p.means.length;
        double[][] gamma = new double[x.length][k];
        for (int t = 0; t < x.length; t++) {
            for (int i = 0; i < k; i++) {
                gamma[t][i] = f.alpha[t][i] * beta[t][i];
            }
        }
        return gamma;
    }

    /** Univariate overload of {@link #smoothedProbabilities(double[][])}. */
    public double[][] smoothedProbabilities(double[] x) {
        return smoothedProbabilities(column(x));
    }

    /**
     * Most likely state path (log-space Viterbi), length T.
     *
     * <p>Ties are broken toward the lower state index (first argmax).
     *
     * @param x observations, T x D
     * @return state indices per day
     */
    public int[] viterbi(double[][] x) {
        HmmParams p = prepare(x);
        double[][] logb = logEmissions(x, p);
        int t2 = logb.length;
        int k = logb[0].length;
        double[][] logA = new double[k][k];
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                logA[i][j] = Math.log(p.transmat[i][j]);
            }
        }
        double[] delta = new double[k];
        for (int i = 0; i < k; i++) {
            delta[i] = Math.log(p.startprob[i]) + logb[0][i];
        }
        int[][] psi = new int[t2][k];
        double[] next = new double[k];
        for (int t = 1; t < t2; t++) {
            for (int j = 0; j < k; j++) {
                int arg = 0;
                double best = delta[0] + logA[0][j];
                for (int i = 1; i < k; i++) {
                    double cand = delta[i] + logA[i][j];
                    if (cand > best) {
                        best = cand;
                        arg = i;
                    }
                }
                psi[t][j] = arg;
                next[j] = best + logb[t][j];
            }
            System.arraycopy(next, 0, delta, 0, k);
        }
        int[] states = new int[t2];
        int arg = 0;
        for (int i = 1; i < k; i++) {
            if (delta[i] > delta[arg]) {
                arg = i;
            }
        }
        states[t2 - 1] = arg;
        for (int t = t2 - 2; t >= 0; t--) {
            states[t] = psi[t + 1][states[t + 1]];
        }
        return states;
    }

    /** Univariate overload of {@link #viterbi(double[][])}. */
    public int[] viterbi(double[] x) {
        return viterbi(column(x));
    }

    /**
     * Stationary distribution of the fitted transition matrix.
     *
     * <p>Pinned power method: start from the uniform vector and iterate
     * {@code pi <- pi A} with L1 renormalization until
     * {@code max|pi_new - pi| < 1e-13} (or 10000 iterations).
     *
     * @return probability vector with {@code pi A = pi}
     */
    public double[] stationaryDistribution() {
        return stationaryDistribution(1e-13, 10000);
    }

    /**
     * Stationary distribution with explicit power-method controls.
     *
     * @param tol convergence tolerance on {@code max|pi_new - pi|}
     * @param maxIterations iteration cap
     * @return probability vector with {@code pi A = pi}
     * @throws IllegalArgumentException if unfitted, {@code tol <= 0},
     *     {@code maxIterations < 1}, or the iteration does not converge within
     *     the cap (periodic or reducible chain) — an unconverged vector is
     *     never returned as stationary
     */
    public double[] stationaryDistribution(double tol, int maxIterations) {
        HmmParams p = requireFitted();
        int k = nStates;
        validateParams(p, k, p.means[0].length);
        if (!(tol > 0.0) || maxIterations < 1) {
            throw new IllegalArgumentException("stationary_distribution: tol must be > 0 and max_iter >= 1");
        }
        double[] pi = new double[k];
        Arrays.fill(pi, 1.0 / k);
        for (int it = 0; it < maxIterations; it++) {
            double[] nxt = new double[k];
            for (int j = 0; j < k; j++) {
                double s = 0.0;
                for (int i = 0; i < k; i++) {
                    s += pi[i] * p.transmat[i][j];
                }
                nxt[j] = s;
            }
            double sum = 0.0;
            for (int j = 0; j < k; j++) {
                sum += nxt[j];
            }
            double maxDiff = 0.0;
            for (int j = 0; j < k; j++) {
                nxt[j] /= sum;
                maxDiff = Math.max(maxDiff, Math.abs(nxt[j] - pi[j]));
            }
            if (maxDiff < tol) {
                return nxt;
            }
            pi = nxt;
        }
        throw new IllegalArgumentException("stationary distribution did not converge in " + maxIterations
                + " power-method iterations (periodic or reducible transition matrix)");
    }

    /**
     * Free parameter count for the diagonal-covariance model:
     * K*D means + K*D variances + K(K-1) transition + (K-1) initial.
     *
     * @param nDims observation dimension D
     * @return parameter count p
     */
    public int nParameters(int nDims) {
        return 2 * nStates * nDims + nStates * (nStates - 1) + (nStates - 1);
    }

    /**
     * Akaike information criterion {@code -2 LL + 2 p} (lower is better).
     *
     * @param x observations, T x D
     * @return AIC of the fitted model on {@code x}
     */
    public double aic(double[][] x) {
        x = validate(x);
        return -2.0 * score(x) + 2.0 * nParameters(x[0].length);
    }

    /** Univariate overload of {@link #aic(double[][])}. */
    public double aic(double[] x) {
        return aic(column(x));
    }

    /**
     * Bayesian information criterion {@code -2 LL + p ln T} (lower is better).
     *
     * @param x observations, T x D
     * @return BIC of the fitted model on {@code x}
     */
    public double bic(double[][] x) {
        x = validate(x);
        return -2.0 * score(x) + nParameters(x[0].length) * Math.log(x.length);
    }

    /** Univariate overload of {@link #bic(double[][])}. */
    public double bic(double[] x) {
        return bic(column(x));
    }

    // ------------------------------------------------------------------ //
    // Baum-Welch EM
    // ------------------------------------------------------------------ //

    /**
     * Fits by Baum-Welch EM from the pinned deterministic starting point.
     *
     * @param x observations, T x D with T &gt; K
     * @return the fit result; the fitted parameters (states relabeled so
     *     means on dimension 0 are ascending) land in {@link #params()}
     */
    public HmmFitResult fit(double[][] x) {
        return fit(x, null);
    }

    /** Univariate overload of {@link #fit(double[][])}. */
    public HmmFitResult fit(double[] x) {
        return fit(column(x), null);
    }

    /** Univariate overload of {@link #fit(double[][], HmmParams)}. */
    public HmmFitResult fit(double[] x, HmmParams init) {
        return fit(column(x), init);
    }

    /**
     * Fits by Baum-Welch EM.
     *
     * <p>Pinned loop semantics: each iteration first scores the current
     * parameters (E-step); if the improvement over the previous score is
     * below {@code tol} the fit is converged and the just-scored parameters
     * are returned (no further M-step). If {@code maxIter} E-steps run
     * without convergence, the converged flag is false and the returned
     * log-likelihood is one extra E-step score of the final parameters so it
     * always matches the returned parameter set.
     *
     * @param x observations, T x D with T &gt; K
     * @param init optional warm-start parameters (used by the walk-forward
     *     refit schedule); null selects {@link #pinnedInit}
     * @return the fit result; a dead state or a monotonicity violation is
     *     reported through {@code deadState} / {@code monotone} with
     *     {@code converged = false}, never as an exception
     * @throws IllegalArgumentException on invalid observations, T &lt;= K, a
     *     warm start whose shapes/stochasticity disagree with (K, D)
     *     (API_SPEC 1.9), or an observation with zero likelihood under every
     *     reachable state
     */
    public HmmFitResult fit(double[][] x, HmmParams init) {
        x = validate(x);
        int t2 = x.length;
        if (t2 <= nStates) {
            throw new IllegalArgumentException(
                    "need more observations than states: T=" + t2 + ", K=" + nStates);
        }
        int k = nStates;
        int d = x[0].length;
        if (init != null) {
            validateParams(init, k, d);
        }
        HmmParams p = init != null ? init.copy() : pinnedInit(x);

        double[] history = new double[maxIter];
        double llPrev = Double.NEGATIVE_INFINITY;
        double ll = Double.NEGATIVE_INFINITY;
        boolean converged = false;
        boolean monotone = true;
        int deadState = -1;
        boolean exhausted = true;
        int nIter = 0;
        for (int it = 0; it < maxIter; it++) {
            // E-step: scores the CURRENT parameters, so on convergence the
            // recorded log-likelihood matches the returned parameter set.
            double[][] logb = logEmissions(x, p);
            Forward f = forward(logb, p);
            ll = logLik(f.d, f.m);
            if (!Double.isFinite(ll)) { // unreachable after the d_t > 0 guard; hard stop
                throw new IllegalArgumentException("EM produced a non-finite log-likelihood");
            }
            history[nIter] = ll;
            nIter++;
            if (nIter > 1) {
                EmStep status = emStepStatus(ll, llPrev, tol);
                if (status == EmStep.NON_MONOTONE) {
                    monotone = false;
                    exhausted = false;
                    break;
                }
                if (status == EmStep.CONVERGED) {
                    converged = true;
                    exhausted = false;
                    break;
                }
            }
            llPrev = ll;
            double[][] beta = backward(logb, p, f.d, f.m);
            double[][] alpha = f.alpha;
            double[][] gamma = new double[t2][k];
            for (int t = 0; t < t2; t++) {
                for (int i = 0; i < k; i++) {
                    gamma[t][i] = alpha[t][i] * beta[t][i];
                }
            }

            // Dead-state guard (pinned): a state without posterior support
            // cannot be re-estimated; stop with the just-scored parameters.
            double[] gsum = new double[k];
            double[] gsumTrans = new double[k];
            for (int t = 0; t < t2; t++) {
                for (int i = 0; i < k; i++) {
                    gsum[i] += gamma[t][i];
                    if (t < t2 - 1) {
                        gsumTrans[i] += gamma[t][i];
                    }
                }
            }
            for (int i = 0; i < k; i++) {
                if (gsumTrans[i] < DEAD_STATE_SUPPORT) {
                    deadState = i;
                    break;
                }
            }
            if (deadState >= 0) {
                exhausted = false;
                break;
            }

            // xi summed over t: xi_sum(i,j) = A_ij * sum_t alpha_t(i) *
            // bt_{t+1}(j) * beta_{t+1}(j) / d_{t+1}.
            double[][] bt = shiftedEmissions(logb, new double[t2]);
            double[][] xiSum = new double[k][k];
            for (int t = 0; t < t2 - 1; t++) {
                for (int i = 0; i < k; i++) {
                    double ai = alpha[t][i];
                    for (int j = 0; j < k; j++) {
                        xiSum[i][j] += ai * bt[t + 1][j] * beta[t + 1][j] / f.d[t + 1];
                    }
                }
            }
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    xiSum[i][j] *= p.transmat[i][j];
                }
            }

            // M-step.
            p.startprob = gamma[0].clone();
            double[][] newA = new double[k][k];
            for (int i = 0; i < k; i++) {
                for (int j = 0; j < k; j++) {
                    newA[i][j] = xiSum[i][j] / gsumTrans[i];
                }
            }
            p.transmat = newA;
            double[][] newMeans = new double[k][d];
            for (int t = 0; t < t2; t++) {
                for (int i = 0; i < k; i++) {
                    for (int dim = 0; dim < d; dim++) {
                        newMeans[i][dim] += gamma[t][i] * x[t][dim];
                    }
                }
            }
            for (int i = 0; i < k; i++) {
                for (int dim = 0; dim < d; dim++) {
                    newMeans[i][dim] /= gsum[i];
                }
            }
            double[][] newVars = new double[k][d];
            for (int t = 0; t < t2; t++) {
                for (int i = 0; i < k; i++) {
                    for (int dim = 0; dim < d; dim++) {
                        double diff = x[t][dim] - newMeans[i][dim];
                        newVars[i][dim] += gamma[t][i] * diff * diff;
                    }
                }
            }
            for (int i = 0; i < k; i++) {
                for (int dim = 0; dim < d; dim++) {
                    newVars[i][dim] = Math.max(newVars[i][dim] / gsum[i], varFloor);
                }
            }
            p.means = newMeans;
            p.variances = newVars;
        }
        if (exhausted) {
            // maxIter exhausted: params had one more M-step than the last
            // recorded E-step, so score them once for a consistent report.
            Forward f = forward(logEmissions(x, p), p);
            ll = logLik(f.d, f.m);
        }

        // Pinned label convention: sort states by means[:, 0] ascending
        // (state 0 = lowest mean).  Applied once, after EM finishes.
        Integer[] order = new Integer[k];
        for (int i = 0; i < k; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(p.means[a][0], p.means[b][0]));
        double[] sp = new double[k];
        double[][] a2 = new double[k][k];
        double[][] mu2 = new double[k][];
        double[][] v2 = new double[k][];
        for (int i = 0; i < k; i++) {
            sp[i] = p.startprob[order[i]];
            mu2[i] = p.means[order[i]].clone();
            v2[i] = p.variances[order[i]].clone();
            for (int j = 0; j < k; j++) {
                a2[i][j] = p.transmat[order[i]][order[j]];
            }
        }
        this.params = new HmmParams(sp, a2, mu2, v2);
        this.fitResult = new HmmFitResult(ll, nIter, converged, Arrays.copyOf(history, nIter),
                monotone, deadState);
        return fitResult;
    }

    // ------------------------------------------------------------------ //
    // Incremental filtering
    // ------------------------------------------------------------------ //

    /**
     * One incremental scaled-forward update for online filtering.
     *
     * <p>Given the filtered posterior {@code P(s_{t-1} | x_1..t-1)} and a new
     * observation, returns {@code P(s_t | x_1..t)}. Identical math to one
     * step of the batch forward pass; used by the regime gate between refits
     * so the filter never re-reads the past.
     *
     * @param p current HMM parameters (validated; K from startprob, D from means)
     * @param alphaPrev filtered probabilities at t-1, length K, non-negative,
     *     summing to 1 within {@link #STOCHASTIC_TOL}
     * @param xt new observation, length D
     * @return filtered probabilities at t, length K
     * @throws IllegalArgumentException on invalid parameters, a length
     *     mismatch, non-finite input, or an observation with zero likelihood
     *     under every reachable state
     */
    public static double[] forwardStep(HmmParams p, double[] alphaPrev, double[] xt) {
        if (p == null || p.startprob == null || p.means == null || p.means.length == 0
                || p.means[0] == null) {
            throw new IllegalArgumentException("parameter set is null");
        }
        int k = p.startprob.length;
        int d = p.means[0].length;
        validateParams(p, k, d);
        if (alphaPrev == null || alphaPrev.length != k) {
            throw new IllegalArgumentException("alpha_prev length "
                    + (alphaPrev == null ? "null" : String.valueOf(alphaPrev.length))
                    + " does not match n_states=" + k);
        }
        double asum = 0.0;
        for (double v : alphaPrev) {
            if (!Double.isFinite(v) || v < 0.0) {
                throw new IllegalArgumentException("alpha_prev must be finite, non-negative and sum to 1");
            }
            asum += v;
        }
        if (Math.abs(asum - 1.0) > STOCHASTIC_TOL) {
            throw new IllegalArgumentException("alpha_prev must be finite, non-negative and sum to 1");
        }
        if (xt == null || xt.length != d) {
            throw new IllegalArgumentException("observation length "
                    + (xt == null ? "null" : String.valueOf(xt.length)) + " does not match D=" + d);
        }
        for (double v : xt) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("observation contains NaN or inf");
            }
        }
        double[] logb = new double[k];
        double mx = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < k; i++) {
            double s = 0.0;
            for (int dim = 0; dim < d; dim++) {
                double v = p.variances[i][dim];
                double diff = xt[dim] - p.means[i][dim];
                s += LOG_2PI + Math.log(v) + diff * diff / v;
            }
            logb[i] = -0.5 * s;
            mx = Math.max(mx, logb[i]);
        }
        double[] a = new double[k];
        double sum = 0.0;
        for (int j = 0; j < k; j++) {
            double dot = 0.0;
            for (int i = 0; i < k; i++) {
                dot += alphaPrev[i] * p.transmat[i][j];
            }
            a[j] = dot * Math.exp(logb[j] - mx);
            sum += a[j];
        }
        if (!(sum > 0.0)) {
            throw new IllegalArgumentException(
                    "forward step: the observation has zero likelihood under every reachable state");
        }
        for (int j = 0; j < k; j++) {
            a[j] /= sum;
        }
        return a;
    }
}
