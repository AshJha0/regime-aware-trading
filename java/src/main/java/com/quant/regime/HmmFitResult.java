package com.quant.regime;

/**
 * Outcome of a Baum-Welch fit.
 *
 * @param logLikelihood log-likelihood of the returned parameter set
 * @param nIter number of E-steps performed
 * @param converged true iff the log-likelihood improvement fell below the
 *     tolerance before {@code maxIter} was reached; non-convergence is
 *     reported through this flag, never as an exception
 * @param loglikHistory log-likelihood at each E-step (non-decreasing)
 */
public record HmmFitResult(double logLikelihood, int nIter, boolean converged, double[] loglikHistory) {
}
