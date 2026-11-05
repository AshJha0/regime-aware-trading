package com.quant.regime;

/**
 * Streaming filter: {@link GaussianHmm#forwardStep} with the set-up paid once
 * (API_SPEC 1.10).
 *
 * <p>The constructor validates and copies the parameters and precomputes the
 * log-variance terms; {@link #update(double[])} then checks only the
 * observation. Without an {@code alphaPrev} the first update applies
 * {@code startprob} with no transition, so a run over a series reproduces
 * {@link GaussianHmm#filteredProbabilities(double[])}.
 */
public final class RegimeFilter {

    private static final double LOG_2PI = Math.log(2.0 * Math.PI);

    private final int k;
    private final int d;
    private final double[] startprob;
    private final double[][] transmat;
    private final double[][] means;
    private final double[][] variances;
    private final double[] logConst;
    private final double[] logb;
    private final double[] one = new double[1];
    private double[] next;
    private double[] alpha;
    private boolean started;

    /**
     * Creates a filter that starts from {@code startprob}.
     *
     * @param p HMM parameters
     * @throws IllegalArgumentException on invalid parameters
     */
    public RegimeFilter(HmmParams p) {
        this(p, null);
    }

    /**
     * Creates a filter from a filtered posterior at t-1: the first update
     * equals {@code forwardStep(p, alphaPrev, x)}.
     *
     * @param p HMM parameters
     * @param alphaPrev filtered posterior at t-1, or null for {@code startprob}
     * @throws IllegalArgumentException on invalid parameters or alphaPrev
     */
    public RegimeFilter(HmmParams p, double[] alphaPrev) {
        if (p == null || p.startprob == null || p.means == null || p.means.length == 0
                || p.means[0] == null) {
            throw new IllegalArgumentException("parameter set is null");
        }
        k = p.startprob.length;
        d = p.means[0].length;
        GaussianHmm.validateParams(p, k, d);
        if (d < 1) {
            throw new IllegalArgumentException("means must have at least one dimension");
        }
        HmmParams own = p.copy();
        startprob = own.startprob;
        transmat = own.transmat;
        means = own.means;
        variances = own.variances;
        logConst = new double[k];
        for (int i = 0; i < k; i++) {
            for (int dim = 0; dim < d; dim++) {
                logConst[i] += LOG_2PI + Math.log(variances[i][dim]);
            }
        }
        logb = new double[k];
        next = new double[k];
        alpha = new double[k];
        reset(alphaPrev);
    }

    /** Restarts from {@code startprob}, with no transition on the first update. */
    public void reset() {
        reset(null);
    }

    /**
     * Restarts from a filtered posterior at t-1.
     *
     * @param alphaPrev filtered posterior at t-1, or null for {@code startprob}
     * @throws IllegalArgumentException if alphaPrev is not a length-K
     *     probability vector; the filter is then unchanged
     */
    public void reset(double[] alphaPrev) {
        if (alphaPrev == null) {
            System.arraycopy(startprob, 0, alpha, 0, k);
            started = false;
            return;
        }
        if (alphaPrev.length != k) {
            throw new IllegalArgumentException(
                    "alpha_prev length " + alphaPrev.length + " does not match n_states=" + k);
        }
        double asum = 0.0;
        for (double v : alphaPrev) {
            if (!Double.isFinite(v) || v < 0.0) {
                throw new IllegalArgumentException("alpha_prev must be finite, non-negative and sum to 1");
            }
            asum += v;
        }
        if (Math.abs(asum - 1.0) > GaussianHmm.STOCHASTIC_TOL) {
            throw new IllegalArgumentException("alpha_prev must be finite, non-negative and sum to 1");
        }
        System.arraycopy(alphaPrev, 0, alpha, 0, k);
        started = true;
    }

    /**
     * Univariate overload of {@link #update(double[])} (requires D == 1).
     *
     * @param x new observation
     * @return the filtered posterior (see {@link #update(double[])})
     */
    public double[] update(double x) {
        one[0] = x;
        return update(one);
    }

    /**
     * Advances one observation.
     *
     * @param x new observation, length D
     * @return P(s_t | x_1..t) in an internal buffer that the next update
     *     overwrites; use {@link #alpha()} for a copy
     * @throws IllegalArgumentException if x is not length D and finite, or has
     *     zero likelihood under every reachable state; the filter is then
     *     unchanged
     */
    public double[] update(double[] x) {
        if (x == null || x.length != d) {
            throw new IllegalArgumentException("observation length "
                    + (x == null ? "null" : String.valueOf(x.length)) + " does not match D=" + d);
        }
        for (double v : x) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("observation contains NaN or inf");
            }
        }
        double mx = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < k; i++) {
            double s = logConst[i];
            for (int dim = 0; dim < d; dim++) {
                double diff = x[dim] - means[i][dim];
                s += diff * diff / variances[i][dim];
            }
            logb[i] = -0.5 * s;
            mx = Math.max(mx, logb[i]);
        }
        double total = 0.0;
        for (int j = 0; j < k; j++) {
            double prior = alpha[j];
            if (started) {
                prior = 0.0;
                for (int i = 0; i < k; i++) {
                    prior += alpha[i] * transmat[i][j];
                }
            }
            next[j] = prior * Math.exp(logb[j] - mx);
            total += next[j];
        }
        if (!(total > 0.0)) {
            throw new IllegalArgumentException(
                    "forward step: the observation has zero likelihood under every reachable state");
        }
        for (int j = 0; j < k; j++) {
            next[j] /= total;
        }
        double[] swap = alpha;
        alpha = next;
        next = swap;
        started = true;
        return alpha;
    }

    /** Returns a copy of the current filtered posterior ({@code startprob} before any update). */
    public double[] alpha() {
        return alpha.clone();
    }

    /** Returns true once there is a filtered posterior to transition from. */
    public boolean started() {
        return started;
    }

    /** Returns the number of states K. */
    public int nStates() {
        return k;
    }

    /** Returns the observation dimension D. */
    public int nDims() {
        return d;
    }
}
