package com.quant.regime;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * End-to-end pipeline on the bundled dataset.
 *
 * <p>Loads the CSVs + config, fits full-sample HMMs (K=2, K=3) on the market
 * index, builds the walk-forward regime gate, runs momentum / carry /
 * combined backtests filtered and unfiltered, and computes the crisis-state
 * breakdown. The demo and the golden tests both call
 * {@link #runFullPipeline} so they can never drift apart.
 *
 * <p>Label conventions (pinned): HMM states are sorted by mean ascending, so
 * state 0 has the lowest mean; that need not be the crisis state, which is the
 * highest-variance state (state 2 on this data). The bundled true
 * states use the generator's ordering (0 = calm-bull, 1 = choppy,
 * 2 = crisis); the pinned volatility mapping of the full-sample K=3 fit (calm
 * to the lowest-variance fitted label, crisis to the highest-variance one)
 * aligns them for the Viterbi-accuracy teaching number. The crisis table
 * always uses that K=3 fit, whatever config nStates the walk-forward gate
 * uses.
 */
public final class Pipeline {

    private Pipeline() {
    }

    /**
     * Pinned strategy/backtest configuration (mirrors data/config.json keys).
     *
     * @param nStates HMM state count for the gate and crisis table
     * @param emTol EM convergence tolerance
     * @param emMaxIter EM iteration cap
     * @param varFloor EM variance floor
     * @param momentumLookback trailing momentum window (days)
     * @param volTarget annualized per-asset vol target
     * @param portfolioVolTarget annualized vol target of the combined book
     * @param ewmaLambda EWMA decay for the vol estimate
     * @param leverageCap maximum per-asset leverage
     * @param carryTopN number of carry longs
     * @param carryBottomN number of carry shorts
     * @param rebalanceDays carry rebalance cadence (days)
     * @param trainMinDays observations required before the first gate fit
     * @param refitDays gate refit cadence (days)
     * @param gateMode "prob" or "binary"
     * @param gateThreshold binary-gate probability threshold
     * @param costBps one-way transaction cost in bps of turnover
     * @param tradingDays trading days per year
     * @param gateBand optional no-trade band; absent means 0 (off)
     * @param transitionFloor optional warm-start floor; absent means 0 (off)
     * @param calmGuard optional calm-state guard weight; absent means 0 (off)
     */
    public record Config(int nStates, double emTol, int emMaxIter, double varFloor,
            int momentumLookback, double volTarget, double portfolioVolTarget,
            double ewmaLambda, double leverageCap,
            int carryTopN, int carryBottomN, int rebalanceDays, int trainMinDays, int refitDays,
            String gateMode, double gateThreshold, double costBps, int tradingDays,
            double gateBand, double transitionFloor, double calmGuard) {
    }

    /**
     * Bundled market data.
     *
     * @param dates trading dates, length T
     * @param indexReturns market index daily returns, length T
     * @param trendReturns trend-asset daily returns, T x 6
     * @param fxSpotReturns currency spot returns, T x 6
     * @param fxRateDiffs annualized rate differentials, T x 6
     * @param trueStates generator regime path (teaching only), length T
     * @param config pinned configuration
     */
    public record Dataset(LocalDate[] dates, double[] indexReturns, double[][] trendReturns,
            double[][] fxSpotReturns, double[][] fxRateDiffs, int[] trueStates, Config config) {
    }

    /**
     * Everything the demo and the golden tests need from one full run.
     *
     * @param dataset the bundled data
     * @param config the effective configuration (after any overrides)
     * @param hmm2 full-sample K=2 model
     * @param fit2 its fit result
     * @param hmm3 full-sample K=3 model
     * @param fit3 its fit result
     * @param viterbi3 full-sample K=3 Viterbi path
     * @param smoothed3 full-sample K=3 smoothed probabilities, T x 3
     * @param stationary3 stationary distribution of the K=3 transition matrix
     * @param viterbiAccuracy fraction of days where Viterbi matches the
     *     volatility-mapped true state (teaching value)
     * @param gate walk-forward gate values, length T
     * @param gateModels fitted gate models in refit order
     * @param guardTrips refit days on which the calm-state guard tripped
     * @param results backtests keyed momentum/carry x unfiltered/filtered
     * @param crisisLabel highest-variance state of the K=3 fit (API_SPEC 2.2)
     * @param portfolioScale portfolio vol scale per day (API_SPEC 3.5)
     * @param combinedNet 50/50 combined net returns keyed unfiltered/filtered
     * @param combinedMetrics metrics of the combined portfolios
     * @param crisis per-strategy state-conditional performance tables
     */
    public record Result(Dataset dataset, Config config,
            GaussianHmm hmm2, HmmFitResult fit2, GaussianHmm hmm3, HmmFitResult fit3,
            int[] viterbi3, double[][] smoothed3, double[] stationary3, double viterbiAccuracy,
            double[] gate, List<GaussianHmm> gateModels, int[] guardTrips, int crisisLabel,
            double[] portfolioScale,
            Map<String, Backtest.Result> results,
            Map<String, double[]> combinedNet, Map<String, Backtest.Metrics> combinedMetrics,
            Map<String, Backtest.StateStats[]> crisis) {
    }

    /**
     * Both sleeves at the portfolio vol scale, before any gate (API_SPEC 3.5).
     *
     * @param momentum scaled momentum positions
     * @param carry scaled carry positions
     * @param carryReturns carry total returns
     * @param scale the scale per day
     */
    public record ScaledBook(double[][] momentum, double[][] carry, double[][] carryReturns,
            double[] scale) {
    }

    /**
     * Builds the scaled book. The scale comes from the unscaled, ungated book at
     * the config cost, so every gate multiplies the same book and none is
     * vol-targeted back up.
     *
     * @param ds the dataset
     * @param cfg the configuration
     * @return the scaled book
     */
    public static ScaledBook scaledBook(Dataset ds, Config cfg) {
        double[][] mom = Strategies.momentumPositions(ds.trendReturns(), cfg.momentumLookback(),
                cfg.volTarget(), cfg.ewmaLambda(), cfg.leverageCap());
        double[][] carry = Strategies.carryPositions(ds.fxRateDiffs(), cfg.carryTopN(),
                cfg.carryBottomN(), cfg.rebalanceDays());
        double[][] carryRets = Strategies.carryTotalReturns(ds.fxSpotReturns(), ds.fxRateDiffs());
        double[] rawMom = Backtest.run(mom, ds.trendReturns(), cfg.costBps(), null).netReturns();
        double[] rawCarry = Backtest.run(carry, carryRets, cfg.costBps(), null).netReturns();
        double[] raw = new double[rawMom.length];
        for (int t = 0; t < raw.length; t++) {
            raw[t] = 0.5 * rawMom[t] + 0.5 * rawCarry[t];
        }
        double[] scale = Strategies.portfolioScale(raw, cfg.portfolioVolTarget(), cfg.ewmaLambda(),
                cfg.leverageCap(), cfg.momentumLookback());
        return new ScaledBook(Strategies.applyGate(mom, scale), Strategies.applyGate(carry, scale),
                carryRets, scale);
    }

    /**
     * Loads the bundled CSVs and config from a data directory.
     *
     * @param dataDir directory holding the bundled files
     * @return the dataset
     * @throws IllegalArgumentException if files are missing or the frames
     *     disagree on length
     */
    public static Dataset loadDataset(Path dataDir) {
        for (String name : new String[] {"market_index.csv", "trend_assets.csv", "fx_carry.csv",
                "true_states.csv", "config.json"}) {
            if (!Files.exists(dataDir.resolve(name))) {
                throw new IllegalArgumentException("missing bundled data file: " + dataDir.resolve(name));
            }
        }
        Csv idx = readCsv(dataDir.resolve("market_index.csv"));
        Csv trend = readCsv(dataDir.resolve("trend_assets.csv"));
        Csv fx = readCsv(dataDir.resolve("fx_carry.csv"));
        Csv truth = readCsv(dataDir.resolve("true_states.csv"));
        int t2 = idx.rows.size();
        if (t2 == 0) {
            throw new IllegalArgumentException("bundled CSVs are empty");
        }
        if (trend.rows.size() != t2 || fx.rows.size() != t2 || truth.rows.size() != t2) {
            throw new IllegalArgumentException("bundled CSVs have inconsistent lengths");
        }
        LocalDate[] dates = new LocalDate[t2];
        double[] indexReturns = new double[t2];
        int[] trueStates = new int[t2];
        for (int t = 0; t < t2; t++) {
            dates[t] = LocalDate.parse(idx.rows.get(t)[0]);
            indexReturns[t] = Double.parseDouble(idx.rows.get(t)[idx.col("ret")]);
            trueStates[t] = Integer.parseInt(truth.rows.get(t)[truth.col("state")]);
            if (trueStates[t] < 0 || trueStates[t] > 2) {
                throw new IllegalArgumentException("true_states.csv: state labels must be integers in [0, 2]");
            }
        }
        double[][] trendReturns = numericBlock(trend, "asset_");
        double[][] spot = numericBlock(fx, "spot_ret_");
        double[][] diffs = numericBlock(fx, "rate_diff_");

        Map<?, ?> cfg;
        try {
            cfg = (Map<?, ?>) Json.parse(Files.readString(dataDir.resolve("config.json")));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        Config config = new Config(
                cfgInt(cfg, "n_states"), cfgDouble(cfg, "em_tol"), cfgInt(cfg, "em_max_iter"),
                cfgDouble(cfg, "var_floor"), cfgInt(cfg, "momentum_lookback"),
                cfgDouble(cfg, "vol_target"), cfgDouble(cfg, "portfolio_vol_target"),
                cfgDouble(cfg, "ewma_lambda"),
                cfgDouble(cfg, "leverage_cap"), cfgInt(cfg, "carry_top_n"),
                cfgInt(cfg, "carry_bottom_n"), cfgInt(cfg, "rebalance_days"),
                cfgInt(cfg, "train_min_days"), cfgInt(cfg, "refit_days"),
                (String) cfg.get("gate_mode"), cfgDouble(cfg, "gate_threshold"),
                cfgDouble(cfg, "cost_bps"), cfgInt(cfg, "trading_days"),
                cfgOptional(cfg, "gate_band"), cfgOptional(cfg, "transition_floor"),
                cfgOptional(cfg, "calm_guard"));
        return new Dataset(dates, indexReturns, trendReturns, spot, diffs, trueStates, config);
    }

    /**
     * Runs the entire study on the bundled data with the pinned config.
     *
     * @param dataDir directory holding the bundled CSVs + config.json
     * @return the pipeline result
     */
    public static Result runFullPipeline(Path dataDir) {
        return runFullPipeline(dataDir, null, null);
    }

    /**
     * Runs the entire study, optionally overriding the walk-forward cadence
     * (tests may shorten the loop; golden values always use the bundled
     * pinned config).
     *
     * @param dataDir directory holding the bundled CSVs + config.json
     * @param refitDays optional refit-cadence override (null keeps config)
     * @param trainMinDays optional first-fit override (null keeps config)
     * @return the pipeline result
     */
    public static Result runFullPipeline(Path dataDir, Integer refitDays, Integer trainMinDays) {
        Dataset ds = loadDataset(dataDir);
        Config base = ds.config();
        Config cfg = new Config(base.nStates(), base.emTol(), base.emMaxIter(), base.varFloor(),
                base.momentumLookback(), base.volTarget(), base.portfolioVolTarget(),
                base.ewmaLambda(), base.leverageCap(),
                base.carryTopN(), base.carryBottomN(), base.rebalanceDays(),
                trainMinDays != null ? trainMinDays : base.trainMinDays(),
                refitDays != null ? refitDays : base.refitDays(),
                base.gateMode(), base.gateThreshold(), base.costBps(), base.tradingDays(),
                base.gateBand(), base.transitionFloor(), base.calmGuard());
        double[] rIdx = ds.indexReturns();
        int k = cfg.nStates();

        // Full-sample HMM fits (model selection + crisis attribution).
        GaussianHmm hmm2 = new GaussianHmm(2, cfg.emTol(), cfg.emMaxIter(), cfg.varFloor());
        HmmFitResult fit2 = hmm2.fit(rIdx);
        GaussianHmm hmm3 = new GaussianHmm(3, cfg.emTol(), cfg.emMaxIter(), cfg.varFloor());
        HmmFitResult fit3 = hmm3.fit(rIdx);
        int[] viterbi3 = hmm3.viterbi(rIdx);
        double[][] smoothed3 = hmm3.smoothedProbabilities(rIdx);
        double[] stationary3 = hmm3.stationaryDistribution();

        // Teaching comparison via the pinned volatility mapping of the K=3 fit
        // (independent of the gate's config nStates): calm-bull -> argmin
        // variance, crisis -> argmax variance, choppy -> the rest.
        int k3 = hmm3.nStates();
        int calmLabel = 0;
        int crisisLabel = 0;
        for (int i = 1; i < k3; i++) {
            if (hmm3.params().variances[i][0] < hmm3.params().variances[calmLabel][0]) {
                calmLabel = i;
            }
            if (hmm3.params().variances[i][0] > hmm3.params().variances[crisisLabel][0]) {
                crisisLabel = i;
            }
        }
        int choppyLabel = 3 - calmLabel - crisisLabel;
        int[] map = {calmLabel, choppyLabel, crisisLabel};
        int agree = 0;
        for (int t = 0; t < viterbi3.length; t++) {
            if (viterbi3[t] == map[ds.trueStates()[t]]) {
                agree++;
            }
        }
        double viterbiAccuracy = (double) agree / viterbi3.length;

        // Walk-forward regime gate on the index (no lookahead).
        Strategies.GateResult gr = Strategies.regimeGate(rIdx, k, cfg.trainMinDays(), cfg.refitDays(),
                cfg.gateMode(), cfg.gateThreshold(), cfg.emTol(), cfg.emMaxIter(), cfg.varFloor(),
                cfg.gateBand(), cfg.transitionFloor(), cfg.calmGuard());

        double cost = cfg.costBps();
        LocalDate[] dates = ds.dates();

        ScaledBook book = scaledBook(ds, cfg);
        double[][] momPos = book.momentum();
        double[][] carryPos = book.carry();
        double[][] carryRets = book.carryReturns();

        Map<String, Backtest.Result> results = new LinkedHashMap<>();
        results.put("momentum_unfiltered", Backtest.run(momPos, ds.trendReturns(), cost, dates));
        results.put("momentum_filtered",
                Backtest.run(Strategies.applyGate(momPos, gr.gate()), ds.trendReturns(), cost, dates));
        results.put("carry_unfiltered", Backtest.run(carryPos, carryRets, cost, dates));
        results.put("carry_filtered",
                Backtest.run(Strategies.applyGate(carryPos, gr.gate()), carryRets, cost, dates));

        // Combined portfolio: 50/50 in each strategy's net return stream.
        Map<String, double[]> combinedNet = new LinkedHashMap<>();
        Map<String, Backtest.Metrics> combinedMetrics = new LinkedHashMap<>();
        for (String tag : new String[] {"unfiltered", "filtered"}) {
            double[] mom = results.get("momentum_" + tag).netReturns();
            double[] car = results.get("carry_" + tag).netReturns();
            double[] net = new double[mom.length];
            for (int t = 0; t < net.length; t++) {
                net[t] = 0.5 * mom[t] + 0.5 * car[t];
            }
            combinedNet.put(tag, net);
            combinedMetrics.put(tag, Backtest.computeMetrics(net, dates));
        }

        // Crisis table: attribute daily strategy returns to the full-sample
        // K=3 Viterbi state (crisisLabel = highest variance); the table has k3
        // rows regardless of the gate's nStates.
        Map<String, Backtest.StateStats[]> crisis = new LinkedHashMap<>();
        for (Map.Entry<String, Backtest.Result> e : results.entrySet()) {
            crisis.put(e.getKey(),
                    Backtest.stateConditionalReturns(e.getValue().netReturns(), viterbi3, k3));
        }
        for (String tag : new String[] {"unfiltered", "filtered"}) {
            crisis.put("combined_" + tag,
                    Backtest.stateConditionalReturns(combinedNet.get(tag), viterbi3, k3));
        }

        return new Result(ds, cfg, hmm2, fit2, hmm3, fit3, viterbi3, smoothed3, stationary3,
                viterbiAccuracy, gr.gate(), gr.models(), gr.guardTrips(), crisisLabel, book.scale(), results, combinedNet,
                combinedMetrics, crisis);
    }

    // ------------------------------------------------------------------ //
    // Tiny CSV reader (header + numeric/date columns)
    // ------------------------------------------------------------------ //

    private static final class Csv {
        final String[] header;
        final List<String[]> rows;

        Csv(String[] header, List<String[]> rows) {
            this.header = header;
            this.rows = rows;
        }

        int col(String name) {
            for (int i = 0; i < header.length; i++) {
                if (header[i].equals(name)) {
                    return i;
                }
            }
            throw new IllegalArgumentException("missing CSV column: " + name);
        }
    }

    private static Csv readCsv(Path file) {
        List<String> lines;
        try {
            lines = Files.readAllLines(file);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        if (lines.isEmpty()) {
            throw new IllegalArgumentException("empty CSV: " + file);
        }
        String[] header = lines.get(0).split(",", -1);
        List<String[]> rows = new ArrayList<>(lines.size() - 1);
        for (int i = 1; i < lines.size(); i++) {
            String line = lines.get(i);
            if (!line.isBlank()) {
                rows.add(line.split(",", -1));
            }
        }
        return new Csv(header, rows);
    }

    /** Extracts every column whose header starts with a prefix, in header order. */
    private static double[][] numericBlock(Csv csv, String prefix) {
        List<Integer> cols = new ArrayList<>();
        for (int i = 0; i < csv.header.length; i++) {
            if (csv.header[i].startsWith(prefix)) {
                cols.add(i);
            }
        }
        double[][] out = new double[csv.rows.size()][cols.size()];
        for (int t = 0; t < csv.rows.size(); t++) {
            for (int a = 0; a < cols.size(); a++) {
                out[t][a] = Double.parseDouble(csv.rows.get(t)[cols.get(a)]);
            }
        }
        return out;
    }

    /** Optional numeric config value (API_SPEC 4): absent means 0, which is off. */
    private static double cfgOptional(Map<?, ?> cfg, String key) {
        return cfg.containsKey(key) ? cfgDouble(cfg, key) : 0.0;
    }

    /**
     * Gate with {@code r[injectT]} replaced and the calm-state guard on.
     *
     * @return the {@code calm_guard_injected} golden scalars (API_SPEC 5)
     */
    public static Map<String, Double> calmGuardValues(Result pipe, double calmGuard, int injectT,
            double injectReturn) {
        Config cfg = pipe.config();
        double[] r = pipe.dataset().indexReturns().clone();
        r[injectT] = injectReturn;
        Strategies.GateResult gr = Strategies.regimeGate(r, cfg.nStates(), cfg.trainMinDays(),
                cfg.refitDays(), cfg.gateMode(), cfg.gateThreshold(), cfg.emTol(), cfg.emMaxIter(),
                cfg.varFloor(), 0.0, 0.0, calmGuard);
        double sum = 0.0;
        int start = cfg.trainMinDays() - 1;
        for (int t = start; t < gr.gate().length; t++) {
            sum += gr.gate()[t];
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("mean_gate", sum / (gr.gate().length - start));
        out.put("trip_count", (double) gr.guardTrips().length);
        out.put("first_trip_day", gr.guardTrips().length > 0 ? (double) gr.guardTrips()[0] : -1.0);
        return out;
    }

    /**
     * Filtered backtests with the pipeline's gate banded by {@code band}. The
     * band only post-processes the gate, so no refit is needed.
     *
     * @param pipe a full pipeline run
     * @param band band width in [0, 1]
     * @return the {@code gate_band_0p3} golden scalars (API_SPEC 5)
     */
    public static Map<String, Double> gateBandValues(Result pipe, double band) {
        Dataset ds = pipe.dataset();
        Config cfg = pipe.config();
        double[] gate = Strategies.applyGateBand(pipe.gate(), band);
        ScaledBook book = scaledBook(ds, cfg);
        double[][] momPos = book.momentum();
        double[][] carryPos = book.carry();
        double[][] carryRets = book.carryReturns();
        Backtest.Result mom = Backtest.run(Strategies.applyGate(momPos, gate), ds.trendReturns(),
                cfg.costBps(), ds.dates());
        Backtest.Result carry = Backtest.run(Strategies.applyGate(carryPos, gate), carryRets,
                cfg.costBps(), ds.dates());
        double[] net = new double[mom.netReturns().length];
        for (int t = 0; t < net.length; t++) {
            net[t] = 0.5 * mom.netReturns()[t] + 0.5 * carry.netReturns()[t];
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("mean_turnover_momentum", mean(mom.turnover()));
        out.put("mean_turnover_carry", mean(carry.turnover()));
        out.put("sharpe_combined", Backtest.computeMetrics(net, ds.dates()).sharpe());
        return out;
    }

    private static double mean(double[] v) {
        double s = 0.0;
        for (double x : v) {
            s += x;
        }
        return s / v.length;
    }

    private static double cfgDouble(Map<?, ?> cfg, String key) {
        Object v = cfg.get(key);
        if (!(v instanceof Double d)) {
            throw new IllegalArgumentException("config key missing or not numeric: " + key);
        }
        return d;
    }

    /** Integer config value; a non-integral number (e.g. 3.5) is rejected so the loaders agree. */
    private static int cfgInt(Map<?, ?> cfg, String key) {
        double v = cfgDouble(cfg, key);
        if (!Double.isFinite(v) || v != Math.floor(v) || Math.abs(v) > 1e9) {
            throw new IllegalArgumentException("config.json key '" + key + "' must be an integer, got " + v);
        }
        return (int) v;
    }
}
