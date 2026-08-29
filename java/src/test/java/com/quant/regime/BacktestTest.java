package com.quant.regime;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.time.LocalDate;
import java.util.Random;
import org.junit.Test;

/** Backtest engine tests: pinned accounting, no-lookahead, metrics. */
public class BacktestTest {

    @Test
    public void accountingIdentity() {
        // net = gross - cost*turnover and equity = cumprod(1+net), exactly.
        double[][] p = {{1.0}, {0.5}, {-0.5}, {-0.5}};
        double[][] r = {{0.01}, {0.02}, {-0.01}, {0.03}};
        Backtest.Result res = Backtest.run(p, r, 10.0, null);
        double cost = 10.0 / 1e4;
        double[] expectedGross = {0.0, 1.0 * 0.02, 0.5 * -0.01, -0.5 * 0.03};
        double[] expectedTo = {1.0, 0.5, 1.0, 0.0};
        assertArrayEquals(expectedGross, res.grossReturns(), 1e-15);
        assertArrayEquals(expectedTo, res.turnover(), 1e-15);
        double eq = 1.0;
        for (int t = 0; t < 4; t++) {
            assertEquals(expectedGross[t] - cost * expectedTo[t], res.netReturns()[t], 1e-15);
            eq *= 1.0 + res.netReturns()[t];
            assertEquals(eq, res.equity()[t], 1e-15);
        }
    }

    @Test
    public void noLookaheadShift() {
        // Day-t P&L must come from the day t-1 position; using the same-day
        // position (lookahead) gives a measurably different result.
        Random rng = new Random(3); // test data only; the engine is RNG-free
        double[][] r = new double[300][1];
        double[][] p = new double[300][1];
        for (int t = 0; t < 300; t++) {
            r[t][0] = 0.01 * rng.nextGaussian();
            p[t][0] = Math.signum(r[t][0]); // 'perfect foresight' decided at t
        }
        Backtest.Result res = Backtest.run(p, r, 0.0, null);
        double lookahead = 0.0;
        double honest = 0.0;
        double shifted = 0.0;
        for (int t = 0; t < 300; t++) {
            lookahead += p[t][0] * r[t][0];
            honest += res.grossReturns()[t];
            if (t >= 1) {
                shifted += p[t - 1][0] * r[t][0];
            }
        }
        assertTrue("foresight should be very profitable", Math.abs(lookahead - honest) > 0.1);
        assertEquals(shifted, honest, 1e-15);
    }

    @Test
    public void zeroCostPath() {
        double[][] p = {{1.0}, {1.0}, {-1.0}};
        double[][] r = {{0.01}, {0.02}, {0.03}};
        Backtest.Result res = Backtest.run(p, r, 0.0, null);
        assertArrayEquals(res.grossReturns(), res.netReturns(), 0.0);
    }

    @Test
    public void turnoverFirstDayIsPositionBuild() {
        double[][] p = {{0.5, -0.5}, {0.5, -0.5}};
        double[][] r = new double[2][2];
        Backtest.Result res = Backtest.run(p, r, 0.0, null);
        assertEquals(1.0, res.turnover()[0], 1e-15);
        assertEquals(0.0, res.turnover()[1], 1e-15);
    }

    @Test
    public void maxDrawdownKnownCase() {
        assertEquals(0.6 / 1.2 - 1.0,
                Backtest.maxDrawdown(new double[] {1.0, 1.2, 0.6, 0.9, 1.3}), 1e-15);
        assertEquals(0.0, Backtest.maxDrawdown(new double[] {1.0, 1.1, 1.2}), 1e-15);
    }

    @Test
    public void metricsFormulas() {
        double[] net = {0.01, -0.005, 0.02, 0.0, -0.01};
        Backtest.Metrics m = Backtest.computeMetrics(net, null);
        double mean = 0.0;
        for (double v : net) {
            mean += v;
        }
        mean /= net.length;
        double var = 0.0;
        for (double v : net) {
            var += (v - mean) * (v - mean);
        }
        var /= net.length;
        assertEquals(252.0 * mean, m.annReturn(), 1e-15);
        assertEquals(Math.sqrt(252.0) * Math.sqrt(var), m.annVol(), 1e-15);
        assertEquals(m.annReturn() / m.annVol(), m.sharpe(), 1e-15);
        assertEquals(2.0 / 5.0, m.hitRate(), 1e-15);
        assertTrue(m.maxDd() <= 0.0);
        // zero-vol series: sharpe defined as 0, no division error
        Backtest.Metrics z = Backtest.computeMetrics(new double[10], null);
        assertEquals(0.0, z.sharpe(), 0.0);
        assertEquals(0.0, z.calmar(), 0.0);
    }

    @Test
    public void worstMonthWithDates() {
        // 63 business days from 2020-01-01; one bad calendar stretch.
        LocalDate[] dates = new LocalDate[63];
        LocalDate d = LocalDate.of(2020, 1, 1);
        for (int i = 0; i < 63; i++) {
            while (d.getDayOfWeek().getValue() > 5) {
                d = d.plusDays(1);
            }
            dates[i] = d;
            d = d.plusDays(1);
        }
        double[] net = new double[63];
        for (int t = 21; t < 42; t++) {
            net[t] = -0.01;
        }
        Backtest.Metrics m = Backtest.computeMetrics(net, dates);
        // brute-force calendar-month compounding
        double worst = Double.POSITIVE_INFINITY;
        double block = 1.0;
        for (int t = 0; t < 63; t++) {
            block *= 1.0 + net[t];
            if (t == 62 || dates[t + 1].getMonthValue() != dates[t].getMonthValue()) {
                worst = Math.min(worst, block - 1.0);
                block = 1.0;
            }
        }
        assertEquals(worst, m.worstMonth(), 1e-15);
        assertTrue(m.worstMonth() < -0.05);
    }

    @Test
    public void stateConditionalReturnsPartition() {
        double[] net = {0.01, -0.02, 0.03, 0.0, -0.01, 0.02};
        int[] states = {0, 0, 1, 1, 2, 2};
        Backtest.StateStats[] out = Backtest.stateConditionalReturns(net, states, 3);
        int total = 0;
        for (Backtest.StateStats s : out) {
            total += s.nDays();
        }
        assertEquals(6, total);
        assertEquals(252.0 * (0.01 - 0.02) / 2.0, out[0].annReturn(), 1e-12);
        Backtest.StateStats[] empty = Backtest.stateConditionalReturns(net, new int[6], 2);
        assertEquals(0, empty[1].nDays());
        assertEquals(0.0, empty[1].sharpe(), 0.0);
    }

    @Test
    public void backtestValidation() {
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.run(new double[3][1], new double[4][1], 0.0, null));
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.run(new double[3][1], new double[3][1], -1.0, null));
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.run(new double[][] {{Double.NaN}}, new double[][] {{0.0}}, 0.0, null));
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.run(new double[0][0], new double[0][0], 0.0, null));
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.run(new double[3][1], new double[3][1], 0.0, new LocalDate[2]));
        assertThrows(IllegalArgumentException.class,
                () -> Backtest.computeMetrics(new double[0], null));
    }

    @Test
    public void propertyCostsNeverHelp() {
        // Grid property: for any cost level, higher costs never raise the
        // total net return (turnover is non-negative).
        Random rng = new Random(5); // test data only
        double[][] p = new double[100][3];
        double[][] r = new double[100][3];
        for (int t = 0; t < 100; t++) {
            for (int a = 0; a < 3; a++) {
                p[t][a] = rng.nextGaussian();
                r[t][a] = 0.01 * rng.nextGaussian();
            }
        }
        double prev = Double.POSITIVE_INFINITY;
        for (double bps : new double[] {0.0, 1.0, 5.0, 10.0, 25.0, 50.0}) {
            double total = 0.0;
            for (double v : Backtest.run(p, r, bps, null).netReturns()) {
                total += v;
            }
            assertTrue("cost " + bps + " helped", prev >= total - 1e-12);
            prev = total;
        }
    }
}
