package com.quant.regime;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import org.junit.Test;

/** Strategy tests: momentum sizing, carry schedule/tie-break, regime gate. */
public class StrategiesTest {

    private static double[][] constant(int t, double... perAsset) {
        double[][] out = new double[t][perAsset.length];
        for (double[] row : out) {
            System.arraycopy(perAsset, 0, row, 0, perAsset.length);
        }
        return out;
    }

    @Test
    public void momentumRejectsShortSeries() {
        double[][] r = constant(252, 0.001, 0.001); // exactly lookback long -> too short
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Strategies.momentumPositions(r, 252, 0.10, 0.94, 4.0));
        assertTrue(e.getMessage().contains("shorter than momentum lookback"));
    }

    @Test
    public void momentumSignCorrectness() {
        // Steady up-trend asset goes long, down-trend short.
        double[][] r = constant(300, 0.001, -0.001);
        double[][] pos = Strategies.momentumPositions(r, 252, 0.10, 0.94, 4.0);
        for (int t = 0; t < 300; t++) {
            if (t < 251) {
                assertEquals(0.0, pos[t][0], 0.0); // undefined before first full window
                assertEquals(0.0, pos[t][1], 0.0);
            } else {
                assertTrue(pos[t][0] > 0.0);
                assertTrue(pos[t][1] < 0.0);
            }
        }
    }

    @Test
    public void momentumVolTargetingCapsLeverage() {
        // A near-zero-vol series would want huge leverage; the cap binds.
        double[][] r = constant(300, 1e-6);
        double[][] pos = Strategies.momentumPositions(r, 252, 0.10, 0.94, 4.0);
        for (int t = 251; t < 300; t++) {
            assertEquals(4.0, pos[t][0], 1e-12); // cap / 1 asset
        }
        // and with meaningful vol the target is hit instead of the cap:
        double[][] r2 = new double[300][1];
        for (int t = 0; t < 300; t++) {
            r2[t][0] = 0.02 * Math.sin(t); // deterministic, sd ~1.4%
        }
        double[][] pos2 = Strategies.momentumPositions(r2, 252, 0.10, 0.94, 4.0);
        for (int t = 251; t < 300; t++) {
            assertTrue(Math.abs(pos2[t][0]) < 4.0);
        }
    }

    @Test
    public void momentumScalingMatchesFormula() {
        // Position = sign * min(target/ann_vol, cap) / n_assets exactly.
        double[][] r = constant(260, 0.002, 0.003);
        int lb = 252;
        double[][] pos = Strategies.momentumPositions(r, lb, 0.10, 0.94, 4.0);
        double[][] sig2 = Strategies.ewmaVariance(r, 0.94, lb);
        int t = 255;
        for (int a = 0; a < 2; a++) {
            double annVol = Math.sqrt(252.0 * sig2[t][a]);
            double expect = Math.min(0.10 / annVol, 4.0) / 2.0;
            assertEquals(expect, pos[t][a], 1e-12 * expect);
        }
    }

    @Test
    public void ewmaVarianceRecursion() {
        double[][] r = {{0.01}, {-0.02}, {0.005}, {0.03}};
        double[][] sig2 = Strategies.ewmaVariance(r, 0.9, 2);
        double seed = (0.01 * 0.01 + 0.02 * 0.02) / 2.0;
        assertTrue(Double.isNaN(sig2[0][0]));
        assertEquals(seed, sig2[1][0], 1e-18);
        assertEquals(0.9 * seed + 0.1 * 0.005 * 0.005, sig2[2][0], 1e-18);
        assertEquals(0.9 * sig2[2][0] + 0.1 * 0.03 * 0.03, sig2[3][0], 1e-18);
    }

    @Test
    public void carryTieBreakByAssetIndex() {
        // All-equal differentials: longs are assets 0..2, shorts 3..5 (pinned).
        double[][] d = constant(10, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02);
        double[][] pos = Strategies.carryPositions(d, 3, 3, 21);
        for (int a = 0; a < 3; a++) {
            assertEquals(1.0 / 3.0, pos[0][a], 1e-15);
        }
        for (int a = 3; a < 6; a++) {
            assertEquals(-1.0 / 3.0, pos[0][a], 1e-15);
        }
    }

    @Test
    public void carryRankingAndWeights() {
        double[][] d = constant(5, 0.01, 0.05, -0.02, 0.03, 0.00, 0.02);
        double[][] pos = Strategies.carryPositions(d, 3, 3, 21);
        // descending diff order: 1, 3, 5, 0, 4, 2
        double third = 1.0 / 3.0;
        assertArrayEquals(new double[] {-third, third, -third, third, -third, third}, pos[0], 1e-15);
        double sum = 0.0;
        double absSum = 0.0;
        for (double v : pos[0]) {
            sum += v;
            absSum += Math.abs(v);
        }
        assertEquals(0.0, sum, 1e-15);
        assertEquals(2.0, absSum, 1e-15);
    }

    @Test
    public void carryRebalanceSchedule() {
        // Positions change only at t = 0, 21, 42, ... (pinned schedule).
        int t2 = 70;
        double[][] d = new double[t2][6];
        for (int t = 0; t < t2; t++) {
            for (int a = 0; a < 6; a++) {
                d[t][a] = 0.02 * Math.sin(1.7 * t + 3.1 * a); // deterministic
            }
        }
        double[][] pos = Strategies.carryPositions(d, 3, 3, 21);
        for (int t = 1; t < t2; t++) {
            boolean changed = !Arrays.equals(pos[t], pos[t - 1]);
            if (changed) {
                assertEquals("change off the pinned schedule at t=" + t, 0, t % 21);
            }
        }
        assertArrayEquals(pos[0], pos[1], 0.0);
        assertArrayEquals(pos[21], pos[41], 0.0);
    }

    @Test
    public void carrySingleRebalanceWhenWindowExceedsSeries() {
        // rebalance_days > T: only the day-0 rebalance happens, held throughout.
        double[][] d = constant(10, 0.03, 0.02, 0.01, 0.0, -0.01, -0.02);
        double[][] pos = Strategies.carryPositions(d, 3, 3, 500);
        for (int t = 1; t < 10; t++) {
            assertArrayEquals(pos[0], pos[t], 0.0);
        }
    }

    @Test
    public void carryValidation() {
        double[][] d = constant(10, 1.0, 1.0, 1.0, 1.0);
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.carryPositions(d, 3, 3, 21)); // 6 > 4 assets
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.carryPositions(d, 1, 1, 0));
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.carryPositions(new double[0][4], 1, 1, 21)); // empty
    }

    @Test
    public void carryTotalReturnsAccrual() {
        double[][] spot = {{0.01, 0.0}, {0.005, -0.01}};
        double[][] diff = {{0.0252, 0.0504}, {0.0252, 0.0504}};
        double[][] tot = Strategies.carryTotalReturns(spot, diff);
        assertArrayEquals(spot[0], tot[0], 0.0); // day 0: no accrual
        assertEquals(0.005 + 0.0252 / 252.0, tot[1][0], 1e-15);
        assertEquals(-0.01 + 0.0504 / 252.0, tot[1][1], 1e-15);
    }

    @Test
    public void regimeGateBasics() {
        // Gate is 1 before the first fit, within [0,1] after, deterministic.
        double[] r = TestData.head(TestData.indexReturns(), 700);
        Strategies.GateResult res = Strategies.regimeGate(r, 3, 400, 200, "prob", 0.5, 1e-8, 500, 1e-8);
        for (int t = 0; t < 399; t++) {
            assertEquals(1.0, res.gate()[t], 0.0);
        }
        for (double g : res.gate()) {
            assertTrue(g >= 0.0 && g <= 1.0);
        }
        assertEquals(2, res.models().size()); // fits at t=399 and t=599
        Strategies.GateResult res2 = Strategies.regimeGate(r, 3, 400, 200, "prob", 0.5, 1e-8, 500, 1e-8);
        assertArrayEquals(res.gate(), res2.gate(), 0.0);
    }

    @Test
    public void regimeGateBinaryMode() {
        double[] r = TestData.head(TestData.indexReturns(), 600);
        Strategies.GateResult res = Strategies.regimeGate(r, 3, 400, 300, "binary", 0.5, 1e-8, 500, 1e-8);
        for (double g : res.gate()) {
            assertTrue(g == 0.0 || g == 1.0);
        }
    }

    @Test
    public void regimeGateValidation() {
        double[] r = TestData.indexReturns();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Strategies.regimeGate(TestData.head(r, 100), 3, 400, 63, "prob", 0.5, 1e-8, 500, 1e-8));
        assertTrue(e.getMessage().contains("shorter than train_min_days"));
        e = assertThrows(IllegalArgumentException.class,
                () -> Strategies.regimeGate(TestData.head(r, 600), 3, 400, 63, "magic", 0.5, 1e-8, 500, 1e-8));
        assertTrue(e.getMessage().contains("mode"));
    }

    @Test
    public void applyGateScalesPositions() {
        double[][] pos = constant(5, 1.0, 1.0);
        double[] gate = {1.0, 0.5, 0.0, 0.25, 1.0};
        double[][] out = Strategies.applyGate(pos, gate);
        for (int t = 0; t < 5; t++) {
            assertEquals(gate[t], out[t][0], 1e-15);
        }
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.applyGate(pos, new double[] {1.0, 0.5, 0.0}));
    }

    // ------------------------------------------------------------------ //
    // Robustness: corrupt returns, re-ranking, gate edges
    // ------------------------------------------------------------------ //

    @Test
    public void momentumRejectsReturnsBelowMinusOne() {
        // PT-4: a -100% (or worse) print is corrupt data, not a signal.
        for (double bad : new double[] {-1.0, -1.5}) {
            double[][] r = constant(300, 0.001);
            r[10][0] = bad;
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> Strategies.momentumPositions(r, 252, 0.10, 0.94, 4.0));
            assertTrue(e.getMessage().contains("<= -100%"));
            assertThrows(IllegalArgumentException.class, () -> Strategies.ewmaVariance(r, 0.94, 252));
            assertThrows(IllegalArgumentException.class,
                    () -> Strategies.carryTotalReturns(r, new double[300][1]));
        }
        // -99.9% is legal (extreme, but a valid simple return).
        double[][] r = constant(300, 0.001);
        r[10][0] = -0.999;
        for (double[] row : Strategies.momentumPositions(r, 252, 0.10, 0.94, 4.0)) {
            assertTrue(Double.isFinite(row[0]));
        }
        // Rate differentials are not simple returns: -1.5 is accepted there.
        assertEquals(10, Strategies.carryPositions(constant(10, -1.5, -1.5, -1.5, -1.5), 1, 1, 21).length);
    }

    @Test
    public void carryRerankAfterCrossing() {
        // PT-6: the book flips exactly at the first rebalance after the ranks
        // cross, with turnover 4 (two full round-trips) on that day only.
        double[][] d = new double[45][];
        for (int t = 0; t < 45; t++) {
            d[t] = t < 21 ? new double[] {0.04, 0.03, 0.02, 0.01} : new double[] {0.01, 0.02, 0.03, 0.04};
        }
        double[][] pos = Strategies.carryPositions(d, 1, 1, 21);
        assertArrayEquals(new double[] {1.0, 0.0, 0.0, -1.0}, pos[20], 0.0);
        for (int t = 21; t < 42; t++) {
            assertArrayEquals(new double[] {-1.0, 0.0, 0.0, 1.0}, pos[t], 0.0);
        }
        for (int t = 1; t < 45; t++) {
            double to = 0.0;
            for (int a = 0; a < 4; a++) {
                to += Math.abs(pos[t][a] - pos[t - 1][a]);
            }
            assertEquals("turnover at t=" + t, t == 21 ? 4.0 : 0.0, to, 0.0);
        }
        // Tie at the crossing day -> ascending asset index decides.
        double[][] tie = new double[45][];
        for (int t = 0; t < 45; t++) {
            tie[t] = t < 21 ? new double[] {0.04, 0.03, 0.02, 0.01} : new double[] {0.02, 0.02, 0.02, 0.02};
        }
        assertArrayEquals(new double[] {1.0, 0.0, 0.0, -1.0}, Strategies.carryPositions(tie, 1, 1, 21)[21], 0.0);
    }

    @Test
    public void binaryGateThresholdEdges() {
        // PT-12: the '>=' convention at both ends of [0, 1]; outside is an error.
        double[] r = TestData.head(TestData.indexReturns(), 600);
        double[] prob = Strategies.regimeGate(r, 3, 400, 300, "prob", 0.5, 1e-8, 500, 1e-8).gate();
        double[] lo = Strategies.regimeGate(r, 3, 400, 300, "binary", 0.0, 1e-8, 500, 1e-8).gate();
        double[] hi = Strategies.regimeGate(r, 3, 400, 300, "binary", 1.0, 1e-8, 500, 1e-8).gate();
        for (int t = 399; t < 600; t++) {
            assertEquals(1.0, lo[t], 0.0);
            assertEquals(prob[t] == 1.0 ? 1.0 : 0.0, hi[t], 0.0);
        }
        for (double bad : new double[] {1.5, -0.1, Double.NaN}) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> Strategies.regimeGate(r, 3, 400, 300, "binary", bad, 1e-8, 500, 1e-8));
            assertTrue(e.getMessage().contains("threshold"));
        }
    }

    @Test
    public void gateEdgeWindows() {
        // PT-13: degenerate but legal walk-forward geometries.
        double[] r = TestData.indexReturns();
        Strategies.GateResult one = Strategies.regimeGate(TestData.head(r, 400), 3, 400, 63, "prob", 0.5,
                1e-8, 500, 1e-8);
        assertEquals(400, one.gate().length);
        assertEquals(1, one.models().size());
        for (int t = 0; t < 399; t++) {
            assertEquals(1.0, one.gate()[t], 0.0);
        }
        assertTrue(one.gate()[399] >= 0.0 && one.gate()[399] <= 1.0);
        Strategies.GateResult two = Strategies.regimeGate(TestData.head(r, 500), 3, 400, 1000, "prob", 0.5,
                1e-8, 500, 1e-8);
        assertEquals(1, two.models().size());
        for (double g : two.gate()) {
            assertTrue(g >= 0.0 && g <= 1.0);
        }
        Strategies.GateResult tiny = Strategies.regimeGate(TestData.head(r, 4), 3, 4, 1, "prob", 0.5,
                1e-8, 500, 1e-8);
        assertEquals(1, tiny.models().size());
        for (double g : tiny.gate()) {
            assertTrue(Double.isFinite(g));
        }
    }

    @Test
    public void regimeGateParameterValidation() {
        // MAJ-7/MIN-11: schedule and HMM settings are validated before any fit.
        double[] r = TestData.head(TestData.indexReturns(), 600);
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.regimeGate(r, 3, 3, 63, "prob", 0.5, 1e-8, 500, 1e-8));   // train <= K
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.regimeGate(r, 3, 400, 0, "prob", 0.5, 1e-8, 500, 1e-8));  // refit < 1
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.regimeGate(r, 1, 400, 63, "prob", 0.5, 1e-8, 500, 1e-8)); // K < 2
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.regimeGate(r, 3, 400, 63, "prob", 0.5, 0.0, 500, 1e-8));  // tol <= 0
        double[] nan = Arrays.copyOf(r, 601);
        nan[600] = Double.NaN;
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.regimeGate(nan, 3, 400, 63, "prob", 0.5, 1e-8, 500, 1e-8));
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.regimeGate(null, 3, 400, 63, "prob", 0.5, 1e-8, 500, 1e-8));
    }

    @Test
    public void applyGateRejectsNan() {
        double[][] pos = constant(5, 1.0, 1.0);
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.applyGate(pos, new double[] {1.0, Double.NaN, 0.0, 0.0, 0.0}));
    }

    // No-trade band and the regime_gate return domain.

    private static double[] head700() {
        return Arrays.copyOf(TestData.indexReturns(), 700);
    }

    @Test
    public void gateBandRecursionByHand() {
        assertArrayEquals(new double[] {1.0, 1.0, 0.6, 0.6, 0.2, 0.2, 0.51},
                Strategies.applyGateBand(new double[] {1.0, 0.8, 0.6, 0.65, 0.2, 0.45, 0.51}, 0.3), 0.0);
    }

    @Test
    public void gateBandIsStrictAtTheEdge() {
        assertEquals(1.0, Strategies.applyGateBand(new double[] {0.5}, 0.5)[0], 0.0);
        assertEquals(0.25, Strategies.applyGateBand(new double[] {0.25}, 0.5)[0], 0.0);
    }

    @Test
    public void gateBandZeroIsIdentityAndOneHoldsTheStart() {
        double[] g = {0.1, 0.9, 0.9, 0.0, 1.0};
        assertArrayEquals(g, Strategies.applyGateBand(g, 0.0), 0.0);
        assertArrayEquals(new double[] {1, 1, 1, 1, 1}, Strategies.applyGateBand(g, 1.0), 0.0);
    }

    @Test
    public void gateBandRangeIsChecked() {
        for (double band : new double[] {-0.1, 1.1, Double.NaN}) {
            assertThrows(IllegalArgumentException.class,
                    () -> Strategies.applyGateBand(new double[] {1.0}, band));
            assertThrows(IllegalArgumentException.class, () -> Strategies.regimeGate(new double[10], 3,
                    5, 63, "prob", 0.5, 1e-8, 500, 1e-8, band, 0.0));
        }
    }

    @Test
    public void regimeGateBandEqualsPostProcessing() {
        double[] r = head700();
        double[] raw = Strategies.regimeGate(r, 3, 504, 100, "prob", 0.5, 1e-8, 500, 1e-8).gate();
        double[] banded = Strategies.regimeGate(r, 3, 504, 100, "prob", 0.5, 1e-8, 500, 1e-8, 0.3, 0.0)
                .gate();
        assertArrayEquals(Strategies.applyGateBand(raw, 0.3), banded, 0.0);
    }

    @Test
    public void regimeGateRejectsATotalLossReturn() {
        double[] r = new double[20];
        Arrays.fill(r, 0.001);
        r[7] = -1.0;
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Strategies.regimeGate(r, 3, 10, 63, "prob", 0.5, 1e-8, 500, 1e-8));
        assertTrue(e.getMessage(), e.getMessage().contains("<= -100% at t=7"));
    }

    @Test
    public void regimeGateTransitionFloorZeroIsBitwiseDefault() {
        double[] r = head700();
        double[] plain = Strategies.regimeGate(r, 3, 504, 100, "prob", 0.5, 1e-8, 500, 1e-8).gate();
        double[] zero = Strategies.regimeGate(r, 3, 504, 100, "prob", 0.5, 1e-8, 500, 1e-8, 0.0, 0.0)
                .gate();
        assertArrayEquals(plain, zero, 0.0);
        for (double g : Strategies.regimeGate(r, 3, 504, 100, "prob", 0.5, 1e-8, 500, 1e-8, 0.0, 1e-6)
                .gate()) {
            assertTrue(g >= 0.0 && g <= 1.0);
        }
    }

    // --- calm-state guard (API_SPEC 3.3) ---

    @Test
    public void winsorizeByHand() {
        assertArrayEquals(new double[] {0.0, 1.0, 2.0, 3.0, 2.0 + 2.0 * 1.4826},
                Strategies.winsorizeMad(new double[] {0.0, 1.0, 2.0, 3.0, 100.0}, 2.0), 0.0);
    }

    @Test
    public void winsorizeZeroScaleIsUnchanged() {
        double[] x = {1.0, 1.0, 1.0, 9.0};
        assertArrayEquals(x, Strategies.winsorizeMad(x, Strategies.CALM_CLIP_MADS), 0.0);
    }

    @Test
    public void selectSkipsThinStatesAndFallsBack() {
        double[] v = {1.0, 2.0, 3.0};
        assertEquals(1, Strategies.selectCalmState(v, new double[] {0.001, 0.7, 0.299}, 0.01));
        assertEquals(2, Strategies.selectCalmState(v, new double[] {0.3, 0.3, 0.4}, 0.5));
    }

    @Test
    public void calmGuardRangeIsChecked() {
        for (double bad : new double[] {-0.1, 1.0}) {
            try {
                Strategies.regimeGate(new double[10], 3, 504, 63, "prob", 0.5, 1e-8, 500, 1e-8, 0.0, 0.0, bad);
                org.junit.Assert.fail("expected a range error");
            } catch (IllegalArgumentException e) {
                assertTrue(e.getMessage().contains("calm_guard"));
            }
        }
    }

    @Test
    public void portfolioScaleFormulaWarmUpAndCap() {
        double[] net = new double[60];
        for (int t = 0; t < net.length; t++) {
            net[t] = 0.002 + 0.0005 * (t % 3);
        }
        double[] scale = Strategies.portfolioScale(net, 0.10, 0.94, 4.0, 5);
        int seedDay = 5 + Strategies.PORTFOLIO_VOL_INIT_DAYS - 1;
        for (int t = 0; t < seedDay; t++) {
            assertEquals(1.0, scale[t], 0.0);
        }
        double sig2 = 0.0;
        for (int t = 5; t <= seedDay; t++) {
            sig2 += net[t] * net[t];
        }
        sig2 /= 21.0;
        assertEquals(Math.min(0.10 / Math.sqrt(252.0 * sig2), 4.0), scale[seedDay], 1e-12);
        sig2 = 0.94 * sig2 + 0.06 * net[seedDay + 1] * net[seedDay + 1];
        assertEquals(Math.min(0.10 / Math.sqrt(252.0 * sig2), 4.0), scale[seedDay + 1], 1e-12);
        assertEquals(4.0, Strategies.portfolioScale(new double[40], 0.10, 0.94, 4.0, 0)[39], 0.0);
        for (double s : Strategies.portfolioScale(Arrays.copyOf(net, 10), 0.10, 0.94, 4.0, 5)) {
            assertEquals(1.0, s, 0.0);
        }
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.portfolioScale(net, 0.0, 0.94, 4.0, 5));
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.portfolioScale(net, 0.10, 1.0, 4.0, 5));
        assertThrows(IllegalArgumentException.class,
                () -> Strategies.portfolioScale(net, 0.10, 0.94, 0.0, 5));
    }

    @Test
    public void portfolioScaleIgnoresTheFuture() {
        double[] net = new double[200];
        for (int t = 0; t < net.length; t++) {
            net[t] = 0.003 * ((t * 7 % 11) - 5.0) / 5.0;
        }
        double[] base = Strategies.portfolioScale(net, 0.10, 0.94, 4.0, 20);
        double[] changed = net.clone();
        for (int t = 101; t < changed.length; t++) {
            changed[t] = -3.0 * changed[t] + 0.01;
        }
        double[] other = Strategies.portfolioScale(changed, 0.10, 0.94, 4.0, 20);
        assertArrayEquals(Arrays.copyOf(base, 101), Arrays.copyOf(other, 101), 0.0);
        assertTrue(base[101] != other[101]);
    }
}
