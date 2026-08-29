package com.quant.regime;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

/**
 * Shared test fixtures: bundled data paths and one full-pipeline run per JVM.
 *
 * <p>The golden values were produced with the pinned full config, so the
 * golden tests run the identical full pipeline once (lazily, shared) and
 * every HMM-level test reuses those fits instead of refitting.
 */
final class TestData {

    /** Bundled data directory, relative to the java/ working directory. */
    static final Path DATA_DIR = Paths.get("..", "data");

    private static Pipeline.Result pipeline;
    private static double[] indexReturns;
    private static Map<?, ?> golden;

    private TestData() {
    }

    /** Full pinned-config pipeline run (computed once, shared). */
    static synchronized Pipeline.Result pipeline() {
        if (pipeline == null) {
            pipeline = Pipeline.runFullPipeline(DATA_DIR);
        }
        return pipeline;
    }

    /** Market-index daily returns from the bundled CSV. */
    static synchronized double[] indexReturns() {
        if (indexReturns == null) {
            indexReturns = Pipeline.loadDataset(DATA_DIR).indexReturns();
        }
        return indexReturns;
    }

    /** Parsed golden.json (root object). */
    static synchronized Map<?, ?> golden() {
        if (golden == null) {
            try {
                golden = (Map<?, ?>) Json.parse(
                        Files.readString(DATA_DIR.resolve("golden").resolve("golden.json")));
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        return golden;
    }

    /** Golden cases as a list of {name, inputs, expect, tol} objects. */
    static List<?> goldenCases() {
        return (List<?>) golden().get("cases");
    }

    /** First slice of a series (copy). */
    static double[] head(double[] x, int n) {
        double[] out = new double[n];
        System.arraycopy(x, 0, out, 0, n);
        return out;
    }
}
