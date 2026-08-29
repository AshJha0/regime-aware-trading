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
}
