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

    // ------------------------------------------------------------------ //
    // Robustness: dead states, shape validation, monotonicity, zero transitions
    // ------------------------------------------------------------------ //

    /**
     * 300 low-vol points and a warm start whose last state sits 5.0 away with
     * variance 1e-8: that state has zero posterior support everywhere.
     */
    private static double[] deadStateData() {
        double[] x = new double[300];
        for (int t = 0; t < 300; t++) {
            x[t] = 0.01 * Math.sin(t);
        }
        return x;
    }

    private static void assertAllFinite(double[][] m) {
        for (double[] row : m) {
            for (double v : row) {
                assertTrue(Double.isFinite(v));
            }
        }
    }

    @Test
    public void deadStateIsFlaggedNotNan() {
        // PT-1: a state with no posterior support stops EM with a clear flag.
        double[] x = deadStateData();
        GaussianHmm hmm = new GaussianHmm(3);
        HmmParams init = hmm.pinnedInit(GaussianHmm.column(x));
        init.means[2][0] = 5.0;
        init.variances[2][0] = 1e-8;
        HmmFitResult res = hmm.fit(x, init);
        assertEquals(2, res.deadState());
        assertFalse(res.converged());
        assertTrue(res.monotone());
        assertEquals(1, res.nIter()); // stopped after the first E-step (init scored)
        assertTrue(Double.isFinite(res.logLikelihood()));
        for (double v : res.loglikHistory()) {
            assertTrue(Double.isFinite(v));
        }
        assertAllFinite(hmm.params().means);
        assertAllFinite(hmm.params().variances);
        assertAllFinite(hmm.params().transmat);
        for (double v : hmm.params().startprob) {
            assertTrue(Double.isFinite(v));
        }
        // The returned parameters are the just-scored ones.
        assertEquals(res.logLikelihood(), hmm.score(x), 1e-9);
        GaussianHmm ok = new GaussianHmm(3);
        assertEquals(-1, ok.fit(x).deadState());
    }

    @Test
    public void forwardStepZeroLikelihoodIsError() {
        // PT-1b: all mass on a state whose row of A only reaches states under
        // which the observation underflows -> error, never NaN.
        HmmParams p = new HmmParams(
                new double[] {0.5, 0.5},
                new double[][] {{0.0, 1.0}, {0.0, 1.0}},
                new double[][] {{0.0}, {100.0}},
                new double[][] {{1e-8}, {1e-8}});
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> GaussianHmm.forwardStep(p, new double[] {1.0, 0.0}, new double[] {0.0}));
        assertTrue(e.getMessage().contains("zero likelihood"));
        GaussianHmm hmm = new GaussianHmm(2);
        hmm.setParams(p);
        assertThrows(IllegalArgumentException.class, () -> hmm.score(new double[] {0.0, 0.0}));
    }

    @Test
    public void scoreRejectsDimensionMismatch() {
        // PT-2: a D=1 model must refuse (T, 2) data in every inference call.
        double[] x = TestData.head(TestData.indexReturns(), 300);
        GaussianHmm hmm = new GaussianHmm(2);
        HmmFitResult fit = hmm.fit(x);
        double[][] x2 = new double[300][2];
        for (int t = 0; t < 300; t++) {
            x2[t][0] = x[t];
            x2[t][1] = x[t];
        }
        assertThrows(IllegalArgumentException.class, () -> hmm.score(x2));
        assertThrows(IllegalArgumentException.class, () -> hmm.filteredProbabilities(x2));
        assertThrows(IllegalArgumentException.class, () -> hmm.smoothedProbabilities(x2));
        assertThrows(IllegalArgumentException.class, () -> hmm.viterbi(x2));
        assertThrows(IllegalArgumentException.class, () -> hmm.aic(x2));
        assertThrows(IllegalArgumentException.class, () -> hmm.bic(x2));
        assertEquals(fit.logLikelihood(), hmm.score(x), 1e-9);
    }

    @Test
    public void warmStartShapeValidation() {
        // PT-3: every inconsistent warm start is IllegalArgumentException,
        // never ArrayIndexOutOfBoundsException.
        double[] x = TestData.head(TestData.indexReturns(), 300);
        HmmParams good = new GaussianHmm(2).pinnedInit(GaussianHmm.column(x));
        GaussianHmm k3 = new GaussianHmm(3);
        assertThrows(IllegalArgumentException.class, () -> k3.fit(x, good)); // K=2 params on K=3
        GaussianHmm hmm = new GaussianHmm(2);
        double[][] x2 = new double[300][2];
        for (int t = 0; t < 300; t++) {
            x2[t][0] = x[t];
            x2[t][1] = x[t];
        }
        HmmParams initD2 = new GaussianHmm(2).pinnedInit(x2);
        assertThrows(IllegalArgumentException.class, () -> hmm.fit(x, initD2)); // D=2 params, D=1 data
        HmmParams neg = good.copy();
        neg.variances[0][0] = -1.0;
        assertThrows(IllegalArgumentException.class, () -> hmm.fit(x, neg));
        HmmParams nonstoch = good.copy();
        nonstoch.transmat[0] = new double[] {0.5, 0.6};
        assertThrows(IllegalArgumentException.class, () -> hmm.fit(x, nonstoch));
        HmmParams badpi = good.copy();
        badpi.startprob = new double[] {0.7, 0.7};
        assertThrows(IllegalArgumentException.class, () -> hmm.fit(x, badpi));
        HmmParams nan = good.copy();
        nan.means[0][0] = Double.NaN;
        assertThrows(IllegalArgumentException.class, () -> hmm.fit(x, nan));
        HmmParams ragged = good.copy();
        ragged.means[1] = new double[] {0.0, 0.0};
        assertThrows(IllegalArgumentException.class, () -> hmm.fit(x, ragged));
        assertThrows(IllegalArgumentException.class, () -> hmm.setParams(badpi));
        assertThrows(IllegalArgumentException.class, () -> k3.setParams(good));
        assertThrows(IllegalArgumentException.class, () -> hmm.setParams(null));
        // forwardStep length checks
        assertThrows(IllegalArgumentException.class,
                () -> GaussianHmm.forwardStep(good, new double[] {1.0, 0.0, 0.0}, new double[] {0.0}));
        assertThrows(IllegalArgumentException.class,
                () -> GaussianHmm.forwardStep(good, new double[] {0.5, 0.5}, new double[] {0.0, 0.0}));
        assertThrows(IllegalArgumentException.class,
                () -> GaussianHmm.forwardStep(good, new double[] {0.7, 0.7}, new double[] {0.0}));
        assertThrows(IllegalArgumentException.class,
                () -> GaussianHmm.forwardStep(good, new double[] {0.5, 0.5}, new double[] {Double.NaN}));
        assertThrows(IllegalArgumentException.class,
                () -> GaussianHmm.forwardStep(good, null, new double[] {0.0}));
    }

    @Test
    public void pinnedInitMatchesSpec() {
        // PT-10: quantile means, pooled floored variance, 0.9 self-transitions.
        HmmParams p = new GaussianHmm(2).pinnedInit(GaussianHmm.column(new double[] {1.0, 2.0, 3.0, 4.0, 5.0}));
        assertEquals(2.0, p.means[0][0], 1e-15); // q = 0.25 -> h = 1.0
        assertEquals(4.0, p.means[1][0], 1e-15); // q = 0.75 -> h = 3.0
        assertEquals(2.0, p.variances[0][0], 1e-15); // population variance of 1..5
        assertEquals(2.0, p.variances[1][0], 1e-15);
        assertEquals(0.9, p.transmat[0][0], 1e-15);
        assertEquals(0.1, p.transmat[0][1], 1e-15);
        assertEquals(0.5, p.startprob[0], 1e-15);
        double[] x = new double[60];
        for (int i = 30; i < 60; i++) {
            x[i] = 1.0;
        }
        HmmParams p3 = new GaussianHmm(3).pinnedInit(GaussianHmm.column(x));
        assertEquals(0.0, p3.means[0][0], 1e-15); // q = 1/6 -> h = 9.83
        assertEquals(0.5, p3.means[1][0], 1e-15); // q = 1/2 -> h = 29.5 interpolates
        assertEquals(1.0, p3.means[2][0], 1e-15); // q = 5/6 -> h = 49.17
        for (int k = 0; k < 3; k++) {
            assertEquals(0.25, p3.variances[k][0], 1e-15);
        }
        assertEquals(0.05, p3.transmat[0][1], 1e-15);
    }

    @Test
    public void viterbiRespectsZeroTransitions() {
        // PT-14: A[0][1] = 0 means state 0 can never leave; the path must start
        // in state 1 despite the first observation favouring state 0.
        HmmParams p = new HmmParams(
                new double[] {0.5, 0.5},
                new double[][] {{1.0, 0.0}, {0.5, 0.5}},
                new double[][] {{-1.0}, {1.0}},
                new double[][] {{0.1}, {0.1}});
        double[] x = {-1.0, 1.0, 1.0};
        GaussianHmm hmm = new GaussianHmm(2);
        hmm.setParams(p);
        int[] path = hmm.viterbi(x);
        assertArrayEquals(new int[] {1, 1, 1}, path);
        int[] best = null;
        double bestProb = -1.0;
        for (int[] s : allPaths()) {
            double prob = pathProb(s, x, p); // a zero transition kills the path
            if (prob > bestProb) {
                bestProb = prob;
                best = s;
            }
        }
        assertArrayEquals(best, path);
    }

    @Test
    public void emStepStatusClassification() {
        // PT-15: the pinned (ll, llPrev) -> {continue, converged, non-monotone} rule.
        assertEquals(GaussianHmm.EmStep.CONTINUE, GaussianHmm.emStepStatus(10.0, 9.0, 1e-8));
        assertEquals(GaussianHmm.EmStep.CONVERGED, GaussianHmm.emStepStatus(10.0, 10.0 - 5e-9, 1e-8));
        assertEquals(GaussianHmm.EmStep.CONVERGED, GaussianHmm.emStepStatus(10.0, 10.0, 1e-8));
        assertEquals(GaussianHmm.EmStep.CONVERGED, GaussianHmm.emStepStatus(1000.0 - 1e-5, 1000.0, 1e-8));
        assertEquals(GaussianHmm.EmStep.NON_MONOTONE, GaussianHmm.emStepStatus(
                1000.0 - 2.0 * GaussianHmm.MONOTONE_REL_TOL * 1000.0, 1000.0, 1e-8));
        assertEquals(GaussianHmm.EmStep.NON_MONOTONE, GaussianHmm.emStepStatus(-2.0, -1.0, 1e-8));
        assertEquals(GaussianHmm.EmStep.NON_MONOTONE, GaussianHmm.emStepStatus(Double.NaN, 1.0, 1e-8));
        assertEquals(GaussianHmm.EmStep.NON_MONOTONE,
                GaussianHmm.emStepStatus(1.0, Double.NEGATIVE_INFINITY, 1e-8));
    }

    @Test
    public void nIterCountsHistoryEntries() {
        // MIN-7: nIter == history length; the post-exhaustion re-score is not counted.
        GaussianHmm hmm = new GaussianHmm(3, 1e-8, 3, 1e-8);
        HmmFitResult res = hmm.fit(TestData.head(TestData.indexReturns(), 400));
        assertEquals(3, res.nIter());
        assertEquals(3, res.loglikHistory().length);
        assertTrue(res.monotone());
        assertEquals(-1, res.deadState());
        assertTrue(res.logLikelihood() >= res.loglikHistory()[2] - 1e-9);
    }

    @Test
    public void stationaryDistributionReportsNonConvergence() {
        // MIN-10: a chain whose power-method iterate cycles -> error, never a
        // silently unconverged vector.
        GaussianHmm hmm = new GaussianHmm(3);
        HmmParams p = new HmmParams(
                new double[] {1.0, 0.0, 0.0},
                new double[][] {{0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}, {0.5, 0.5, 0.0}},
                new double[][] {{0.0}, {0.0}, {0.0}},
                new double[][] {{1.0}, {1.0}, {1.0}});
        hmm.setParams(p);
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> hmm.stationaryDistribution(1e-13, 50));
        assertTrue(e.getMessage().contains("did not converge"));
        assertThrows(IllegalArgumentException.class, () -> hmm.stationaryDistribution(0.0, 100));
        assertThrows(IllegalArgumentException.class, () -> hmm.stationaryDistribution(1e-13, 0));
        // an aperiodic chain converges to the uniform vector
        hmm.setParams(new HmmParams(
                new double[] {1.0, 0.0, 0.0},
                new double[][] {{0.5, 0.5, 0.0}, {0.0, 0.5, 0.5}, {0.5, 0.0, 0.5}},
                new double[][] {{0.0}, {0.0}, {0.0}},
                new double[][] {{1.0}, {1.0}, {1.0}}));
        for (double v : hmm.stationaryDistribution()) {
            assertEquals(1.0 / 3.0, v, 1e-10);
        }
    }
}
