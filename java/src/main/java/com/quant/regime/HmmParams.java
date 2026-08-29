package com.quant.regime;

/**
 * Parameter set of a diagonal-covariance Gaussian HMM.
 *
 * <p>A mutable data holder (mirrors the Python {@code HMMParams} dataclass):
 * the Baum-Welch M-step rewrites the arrays in place via reassignment.
 */
public final class HmmParams {

    /** Initial state distribution, length K. */
    public double[] startprob;
    /** Row-stochastic transition matrix, K x K. */
    public double[][] transmat;
    /** State means, K x D. */
    public double[][] means;
    /** Per-dimension state variances, K x D. */
    public double[][] variances;

    /**
     * Builds a parameter set (arrays are stored by reference).
     *
     * @param startprob initial distribution, length K
     * @param transmat transition matrix, K x K
     * @param means state means, K x D
     * @param variances state variances, K x D
     */
    public HmmParams(double[] startprob, double[][] transmat, double[][] means, double[][] variances) {
        this.startprob = startprob;
        this.transmat = transmat;
        this.means = means;
        this.variances = variances;
    }

    /**
     * Returns a deep copy of the parameter set.
     *
     * @return an independent copy
     */
    public HmmParams copy() {
        return new HmmParams(
                startprob.clone(),
                copyMatrix(transmat),
                copyMatrix(means),
                copyMatrix(variances));
    }

    private static double[][] copyMatrix(double[][] m) {
        double[][] out = new double[m.length][];
        for (int i = 0; i < m.length; i++) {
            out[i] = m[i].clone();
        }
        return out;
    }
}
