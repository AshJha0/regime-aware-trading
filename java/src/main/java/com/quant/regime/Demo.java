package com.quant.regime;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

/**
 * End-to-end demo: regime detection + regime-aware momentum and carry.
 *
 * <p>Run from {@code java/} after {@code ./build.sh}:
 * {@code ./demo.sh} (optionally passing an alternative data directory).
 * Prints the fitted regime table, Viterbi-vs-true accuracy, the strategy
 * comparison (filtered vs unfiltered) and the crisis-state breakdown.
 */
public final class Demo {

    private static final String LINE = "-".repeat(78);

    private Demo() {
    }

    /**
     * Entry point.
     *
     * @param args optional: args[0] = data directory (default ../data)
     */
    public static void main(String[] args) {
        Path dataDir = Paths.get(args.length > 0 ? args[0] : "../data");
        System.out.println("Regime-Aware Trading — HMM + momentum + FX carry (bundled synthetic data)");
        System.out.println(LINE);
        Pipeline.Result pipe = Pipeline.runFullPipeline(dataDir);
        Pipeline.Dataset ds = pipe.dataset();
        int k = pipe.config().nStates();
        double[] r = ds.indexReturns();

        System.out.printf("Data: %d trading days, %s .. %s%n", r.length,
                ds.dates()[0], ds.dates()[r.length - 1]);
        System.out.println("\nModel selection on the market index (diagonal Gaussian HMM):");
        System.out.printf("  %2s %12s %12s %12s %6s %5s%n", "K", "loglik", "AIC", "BIC", "iters", "conv");
        printModelRow(2, pipe.hmm2(), pipe.fit2(), r);
        printModelRow(3, pipe.hmm3(), pipe.fit3(), r);

        HmmParams p = pipe.hmm3().params();
        double[] pi = pipe.stationary3();
        String[] names = regimeNames(p, k);
        System.out.println("\nFitted K=3 regimes (states sorted by mean; 0 = lowest):");
        System.out.printf("  %5s %10s %10s %9s %11s%n", "state", "label", "mean(ann)", "vol(ann)", "stationary");
        for (int i = 0; i < k; i++) {
            System.out.printf(Locale.ROOT, "  %5d %10s %+9.2f%% %8.2f%% %11.4f%n",
                    i, names[i], 100.0 * 252.0 * p.means[i][0],
                    100.0 * Math.sqrt(252.0 * p.variances[i][0]), pi[i]);
        }
        System.out.println("  transition matrix (rows sum to 1):");
        for (int i = 0; i < k; i++) {
            StringBuilder sb = new StringBuilder("    ");
            for (int j = 0; j < k; j++) {
                sb.append(String.format(Locale.ROOT, "%.4f", p.transmat[i][j]));
                if (j < k - 1) {
                    sb.append("  ");
                }
            }
            System.out.println(sb);
        }

        int[] counts = new int[k];
        int[] trueCounts = new int[k];
        for (int t = 0; t < r.length; t++) {
            counts[pipe.viterbi3()[t]]++;
            trueCounts[ds.trueStates()[t]]++;
        }
        System.out.printf(Locale.ROOT, "%nViterbi decoding vs true (generator) states: accuracy = %.2f%%%n",
                100.0 * pipe.viterbiAccuracy());
        System.out.printf("  decoded days per state [%d, %d, %d]  (true calm/choppy/crisis [%d, %d, %d])%n",
                counts[0], counts[1], counts[2], trueCounts[0], trueCounts[1], trueCounts[2]);

        System.out.println(LINE);
        Pipeline.Config cfg = pipe.config();
        System.out.printf(Locale.ROOT,
                "Strategies (cost %.0f bps, gate mode '%s', refit every %dd on expanding window):%n",
                cfg.costBps(), cfg.gateMode(), cfg.refitDays());
        System.out.printf("  %-22s %8s %8s %7s %8s %7s %6s %9s%n",
                "strategy", "ann ret", "ann vol", "Sharpe", "maxDD", "Calmar", "hit", "worst mo");
        for (String name : new String[] {"momentum_unfiltered", "momentum_filtered",
                "carry_unfiltered", "carry_filtered"}) {
            printMetricsRow(name.replace('_', ' '), pipe.results().get(name).metrics());
        }
        for (String tag : new String[] {"unfiltered", "filtered"}) {
            printMetricsRow("combined " + tag, pipe.combinedMetrics().get(tag));
        }

        System.out.println(LINE);
        System.out.println("Annualized net return by decoded regime (state 0 = lowest mean):");
        System.out.printf("  %-22s %9s %9s %10s%n", "strategy", names[0], names[1], names[2]);
        for (String name : new String[] {"momentum_unfiltered", "momentum_filtered",
                "carry_unfiltered", "carry_filtered", "combined_unfiltered", "combined_filtered"}) {
            Backtest.StateStats[] c = pipe.crisis().get(name);
            System.out.printf(Locale.ROOT, "  %-22s %+8.2f%% %+8.2f%% %+9.2f%%%n",
                    name.replace('_', ' '),
                    100.0 * c[0].annReturn(), 100.0 * c[1].annReturn(), 100.0 * c[2].annReturn());
        }

        Backtest.Metrics mu = pipe.combinedMetrics().get("unfiltered");
        Backtest.Metrics mf = pipe.combinedMetrics().get("filtered");
        System.out.println(LINE);
        System.out.println("Takeaway: momentum and carry both lose money in the high-vol crisis");
        System.out.println("regime; scaling by the filtered P(calm) improves the combined Sharpe");
        System.out.printf(Locale.ROOT, "(%.2f -> %.2f) and maxDD (%+.2f%% -> %+.2f%%)%n",
                mu.sharpe(), mf.sharpe(), 100.0 * mu.maxDd(), 100.0 * mf.maxDd());
        System.out.printf(Locale.ROOT,
                "at the cost of total return (%+.2f%% -> %+.2f%%) — the filter%n",
                100.0 * mu.annReturn(), 100.0 * mf.annReturn());
        System.out.println("buys risk reduction, not free performance.");
    }

    private static void printModelRow(int k, GaussianHmm model, HmmFitResult res, double[] r) {
        System.out.printf(Locale.ROOT, "  %2d %12.4f %12.4f %12.4f %6d %5s%n",
                k, res.logLikelihood(), model.aic(r), model.bic(r), res.nIter(), res.converged());
    }

    /** Interpretation labels via the pinned volatility mapping. */
    private static String[] regimeNames(HmmParams p, int k) {
        int calm = 0;
        int crisis = 0;
        for (int i = 1; i < k; i++) {
            if (p.variances[i][0] < p.variances[calm][0]) {
                calm = i;
            }
            if (p.variances[i][0] > p.variances[crisis][0]) {
                crisis = i;
            }
        }
        String[] names = new String[k];
        for (int i = 0; i < k; i++) {
            names[i] = i == crisis ? "crisis" : (i == calm ? "calm-bull" : "choppy");
        }
        return names;
    }

    private static void printMetricsRow(String label, Backtest.Metrics m) {
        System.out.printf(Locale.ROOT, "  %-22s %+7.2f%% %7.2f%% %7.2f %+7.2f%% %7.2f %5.1f%% %+8.2f%%%n",
                label, 100.0 * m.annReturn(), 100.0 * m.annVol(), m.sharpe(),
                100.0 * m.maxDd(), m.calmar(), 100.0 * m.hitRate(), 100.0 * m.worstMonth());
    }
}
