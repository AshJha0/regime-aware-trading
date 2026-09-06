package com.quant.regime;

/**
 * Outcome of a Baum-Welch fit.
 *
 * @param logLikelihood log-likelihood of the returned parameter set (always
 *     finite; the score of exactly the returned parameters)
 * @param nIter number of E-steps that appended to {@code loglikHistory} (the
 *     consistency re-score after {@code maxIter} exhaustion is not counted)
 * @param converged true iff the log-likelihood improvement fell below the
 *     tolerance before {@code maxIter} was reached, with no monotonicity
 *     violation and no dead state; non-convergence is reported through this
 *     flag, never as an exception
 * @param loglikHistory log-likelihood at each counted E-step
 * @param monotone false iff some E-step decreased the log-likelihood by more
 *     than {@code MONOTONE_REL_TOL * max(1, |ll_prev|)} (EM stops there and
 *     {@code converged} is false)
 * @param deadState index of the lowest state whose posterior support
 *     {@code sum_{t<T} gamma_t(i)} fell below {@code DEAD_STATE_SUPPORT} (EM
 *     stops there and {@code converged} is false), or -1 if none
 */
public record HmmFitResult(double logLikelihood, int nIter, boolean converged, double[] loglikHistory,
        boolean monotone, int deadState) {
}
