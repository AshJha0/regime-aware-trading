package com.quant.regime;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** HMM tests: enumeration ground truth, EM properties, edge cases. */
public class HmmTest {

    /** Fixed tiny model for the enumerated K=2, T=3 case. */
    private static HmmParams tinyParams() {
        return new HmmParams(
                new double[] {0.6, 0.4},
                new double[][] {{0.7, 0.3}, {0.2, 0.8}},
                new double[][] {{-1.0}, {1.0}},
                new double[][] {{0.5}, {2.0}});
    }

    private static final double[] TINY_X = {-0.5, 0.3, 1.2};

    private static GaussianHmm tinyModel() {
        GaussianHmm hmm = new GaussianHmm(2);
        hmm.setParams(tinyParams());
        return hmm;
    }

    private static double normPdf(double x, double mu, double var) {
        return Math.exp(-0.5 * (x - mu) * (x - mu) / var) / Math.sqrt(2.0 * Math.PI * var);
    }

    private static double pathProb(int[] path, double[] x, HmmParams p) {
        double prob = p.startprob[path[0]] * normPdf(x[0], p.means[path[0]][0], p.variances[path[0]][0]);
        for (int t = 1; t < x.length; t++) {
            prob *= p.transmat[path[t - 1]][path[t]]
                    * normPdf(x[t], p.means[path[t]][0], p.variances[path[t]][0]);
        }
        return prob;
    }

    /** All 8 binary state paths of length 3. */
    private static int[][] allPaths() {
        int[][] paths = new int[8][3];
        for (int i = 0; i < 8; i++) {
            paths[i] = new int[] {(i >> 2) & 1, (i >> 1) & 1, i & 1};
        }
        return paths;
    }

    @Test
    public void forwardLikelihoodMatchesEnumeration() {
        GaussianHmm hmm = tinyModel();
        double brute = 0.0;
        for (int[] path : allPaths()) {
            brute += pathProb(path, TINY_X, tinyParams());
        }
        assertEquals(brute, Math.exp(hmm.score(TINY_X)), 1e-12);
    }

    @Test
    public void smoothedProbsMatchEnumeration() {
        GaussianHmm hmm = tinyModel();
        double[][] gamma = hmm.smoothedProbabilities(TINY_X);
        double total = 0.0;
        for (int[] path : allPaths()) {
            total += pathProb(path, TINY_X, tinyParams());
        }
        for (int t = 0; t < 3; t++) {
            for (int i = 0; i < 2; i++) {
                double marg = 0.0;
                for (int[] path : allPaths()) {
                    if (path[t] == i) {
                        marg += pathProb(path, TINY_X, tinyParams());
                    }
                }
                assertEquals("gamma[" + t + "][" + i + "]", marg / total, gamma[t][i], 1e-12);
            }
        }
    }

    @Test
    public void viterbiIsArgmaxPathOnEnumeratedCase() {
        GaussianHmm hmm = tinyModel();
        int[] best = null;
        double bestProb = -1.0;
        for (int[] path : allPaths()) {
            double prob = pathProb(path, TINY_X, tinyParams());
            if (prob > bestProb) {
                bestProb = prob;
                best = path;
            }
        }
        assertArrayEquals(best, hmm.viterbi(TINY_X));
    }

    @Test
    public void emMonotonicOnBundledData() {
        HmmFitResult fit3 = TestData.pipeline().fit3();
        double[] hist = fit3.loglikHistory();
        assertTrue(hist.length >= 3);
        for (int i = 1; i < hist.length; i++) {
            assertTrue("EM iteration " + i + " decreased the log-likelihood",
                    hist[i] - hist[i - 1] >= -1e-9);
        }
        assertTrue(fit3.converged());
    }

    @Test
    public void transitionsRowStochastic() {
        for (GaussianHmm hmm : new GaussianHmm[] {TestData.pipeline().hmm2(), TestData.pipeline().hmm3()}) {
            double[][] a = hmm.params().transmat;
            for (double[] row : a) {
                double sum = 0.0;
                for (double v : row) {
                    assertTrue(v >= 0.0);
                    sum += v;
                }
                assertEquals(1.0, sum, 1e-12);
            }
        }
    }

    @Test
    public void stateLabelOrderingIsMeanAscending() {
        for (GaussianHmm hmm : new GaussianHmm[] {TestData.pipeline().hmm2(), TestData.pipeline().hmm3()}) {
            double[][] means = hmm.params().means;
            for (int i = 1; i < means.length; i++) {
                assertTrue(means[i][0] >= means[i - 1][0]);
            }
        }
    }

    @Test
    public void stationaryIsLeftEigenvector() {
        GaussianHmm hmm = TestData.pipeline().hmm3();
        double[] pi = hmm.stationaryDistribution();
        double sum = 0.0;
        for (double v : pi) {
            sum += v;
        }
        assertEquals(1.0, sum, 1e-12);
        double[][] a = hmm.params().transmat;
        for (int j = 0; j < pi.length; j++) {
            double piA = 0.0;
            for (int i = 0; i < pi.length; i++) {
                piA += pi[i] * a[i][j];
            }
            assertEquals("pi P != pi at " + j, pi[j], piA, 1e-10);
        }
    }

    @Test
    public void fitIsDeterministic() {
        double[] x = TestData.head(TestData.indexReturns(), 600);
        GaussianHmm a = new GaussianHmm(3);
        HmmFitResult ra = a.fit(x);
        GaussianHmm b = new GaussianHmm(3);
        HmmFitResult rb = b.fit(x);
        assertEquals(ra.logLikelihood(), rb.logLikelihood(), 0.0);
        for (int k = 0; k < 3; k++) {
            assertArrayEquals(a.params().means[k], b.params().means[k], 0.0);
            assertArrayEquals(a.params().transmat[k], b.params().transmat[k], 0.0);
            assertArrayEquals(a.params().variances[k], b.params().variances[k], 0.0);
        }
    }

    @Test
    public void k1IsDegenerateError() {
        assertThrows(IllegalArgumentException.class, () -> new GaussianHmm(1));
    }

    @Test
    public void varianceFloorEngages() {
        // K exceeding the data's distinct support hits the variance floor.
        double[] x = new double[60];
        for (int i = 30; i < 60; i++) {
            x[i] = 1.0;
        }
        GaussianHmm hmm = new GaussianHmm(3, 1e-8, 200, 1e-8);
        HmmFitResult res = hmm.fit(x); // must not crash or produce zero variance
        boolean anyAtFloor = false;
        for (double[] row : hmm.params().variances) {
            for (double v : row) {
                assertTrue(v >= 1e-8);
                anyAtFloor |= Math.abs(v - 1e-8) < 1e-12;
            }
        }
        assertTrue("no variance pinned at the floor", anyAtFloor);
        assertTrue(Double.isFinite(res.logLikelihood()));
    }

    @Test
    public void nonConvergenceIsFlagNotException() {
        GaussianHmm hmm = new GaussianHmm(3, 1e-8, 3, 1e-8);
        HmmFitResult res = hmm.fit(TestData.head(TestData.indexReturns(), 400));
        assertFalse(res.converged());
        assertEquals(3, res.nIter());
        assertTrue(Double.isFinite(res.logLikelihood()));
    }

    @Test
    public void probabilityOutputsAreValid() {
        double[][] gamma = TestData.pipeline().smoothed3();
        for (double[] row : gamma) {
            double sum = 0.0;
            for (double v : row) {
                assertTrue(v >= 0.0 && v <= 1.0 + 1e-12);
                sum += v;
            }
            assertEquals(1.0, sum, 1e-10);
        }
        double[][] alpha = TestData.pipeline().hmm3()
                .filteredProbabilities(TestData.head(TestData.indexReturns(), 200));
        for (double[] row : alpha) {
            double sum = 0.0;
            for (double v : row) {
                sum += v;
            }
            assertEquals(1.0, sum, 1e-10);
        }
    }

    @Test
    public void forwardStepMatchesFullFilter() {
        // Incremental filtering (used by the gate) equals the batch forward.
        GaussianHmm hmm = TestData.pipeline().hmm3();
        double[] x = TestData.head(TestData.indexReturns(), 150);
        double[][] full = hmm.filteredProbabilities(x);
        double[][] first = hmm.filteredProbabilities(TestData.head(x, 100));
        double[] alpha = first[99];
        for (int t = 100; t < 150; t++) {
            alpha = GaussianHmm.forwardStep(hmm.params(), alpha, new double[] {x[t]});
        }
        assertArrayEquals(full[149], alpha, 1e-12);
    }

    @Test
    public void twoDimensionalObservations() {
        // 2-D diagonal-covariance fit runs; labels sort on dimension 0.
        double[] r = TestData.head(TestData.indexReturns(), 500);
        double[][] x2 = new double[500][2];
        for (int t = 0; t < 500; t++) {
            x2[t][0] = r[t];
            x2[t][1] = Math.abs(r[t]);
        }
        GaussianHmm hmm = new GaussianHmm(2);
        HmmFitResult res = hmm.fit(x2);
        assertEquals(2, hmm.params().means.length);
        assertEquals(2, hmm.params().means[0].length);
        assertTrue(hmm.params().means[0][0] <= hmm.params().means[1][0]);
        assertTrue(Double.isFinite(res.logLikelihood()));
        assertEquals(500, hmm.smoothedProbabilities(x2).length);
    }

    @Test
    public void modelSelectionPrefersThreeStates() {
        // Data come from a true 3-state DGP: K=3 wins on AIC and BIC.
        Pipeline.Result pipe = TestData.pipeline();
        double[] r = TestData.indexReturns();
        assertTrue(pipe.fit3().logLikelihood() > pipe.fit2().logLikelihood());
        assertTrue(pipe.hmm3().aic(r) < pipe.hmm2().aic(r));
        assertTrue(pipe.hmm3().bic(r) < pipe.hmm2().bic(r));
    }

    @Test
    public void inputValidationErrors() {
        GaussianHmm hmm = new GaussianHmm(2);
        assertThrows(IllegalArgumentException.class, () -> hmm.fit(new double[0]));
        assertThrows(IllegalArgumentException.class, () -> hmm.fit(new double[] {1.0, Double.NaN, 2.0}));
        assertThrows(IllegalArgumentException.class, () -> hmm.fit(new double[][] {{1.0, 2.0, 3.0},
                {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}})); // D=3 unsupported
        assertThrows(IllegalArgumentException.class, () -> hmm.fit(new double[] {1.0, 2.0})); // T <= K
        assertThrows(IllegalArgumentException.class, () -> hmm.score(new double[] {1.0})); // not fitted
        assertThrows(IllegalArgumentException.class, () -> new GaussianHmm(2, 0.0, 500, 1e-8));
        assertThrows(IllegalArgumentException.class, () -> new GaussianHmm(2, 1e-8, 0, 1e-8));
        assertThrows(IllegalArgumentException.class, () -> new GaussianHmm(2, 1e-8, 500, 0.0));
    }

    @Test
    public void gridPropertyFittedParamsScoreBest() {
        // Property-style grid check: the fitted parameters score at least as
        // well as every mean-shifted perturbation.
        double[] x = TestData.head(TestData.indexReturns(), 400);
        GaussianHmm hmm = new GaussianHmm(2);
        HmmFitResult res = hmm.fit(x);
        for (int s = 0; s < 7; s++) {
            double shift = -0.01 + s * (0.02 / 6.0);
            HmmParams p = hmm.params().copy();
            for (double[] mu : p.means) {
                for (int d = 0; d < mu.length; d++) {
                    mu[d] += shift;
                }
            }
            GaussianHmm probe = new GaussianHmm(2);
            probe.setParams(p);
            assertTrue("shift " + shift + " scored better than the fit",
                    probe.score(x) <= res.logLikelihood() + 1e-9);
        }
    }
}
