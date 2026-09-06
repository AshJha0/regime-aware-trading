//! End-to-end demo: regime detection + regime-aware momentum and carry.
//!
//! Run: `cd rust && cargo run --release --bin demo`
//!
//! Prints the fitted regime table, Viterbi-vs-true accuracy, the strategy
//! comparison (filtered vs unfiltered) and the crisis-state breakdown.

use std::path::PathBuf;
use std::process::ExitCode;

use regime::{run_full_pipeline, Matrix, Metrics, PipelineOutput, StateStats};

const LINE: &str = "------------------------------------------------------------------------------";

fn pct(x: f64) -> String {
    format!("{:+.2}%", 100.0 * x)
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("demo failed: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> regime::Result<()> {
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../data");
    println!("Regime-Aware Trading — HMM + momentum + FX carry (bundled synthetic data)");
    println!("{LINE}");
    let pipe = run_full_pipeline(&data_dir, None, None)?;
    let ds = &pipe.dataset;
    let k = pipe.hmm3.n_states; // the K=3 tables are keyed to the full-sample fit
    let t_len = ds.index_returns.len();

    // ---------------- model selection ---------------- //
    println!("Data: {} trading days, {} .. {}", t_len, ds.dates[0], ds.dates[t_len - 1]);
    println!("\nModel selection on the market index (diagonal Gaussian HMM):");
    println!("  {:>2} {:>12} {:>12} {:>12} {:>6} {:>5}", "K", "loglik", "AIC", "BIC", "iters", "conv");
    let r_idx = Matrix::column(&ds.index_returns);
    for (kk, model, fit) in [(2, &pipe.hmm2, &pipe.fit2), (3, &pipe.hmm3, &pipe.fit3)] {
        println!(
            "  {:>2} {:>12.4} {:>12.4} {:>12.4} {:>6} {:>5}",
            kk,
            fit.log_likelihood,
            model.aic(&r_idx)?,
            model.bic(&r_idx)?,
            fit.n_iter,
            fit.converged
        );
    }

    // ---------------- regime table ---------------- //
    let p = pipe.hmm3.params().expect("fitted");
    let variances: Vec<f64> = (0..k).map(|i| p.variances.get(i, 0)).collect();
    let mut names = vec!["choppy"; k];
    let (mut lo, mut hi) = (0usize, 0usize);
    for i in 1..k {
        if variances[i] < variances[lo] {
            lo = i;
        }
        if variances[i] > variances[hi] {
            hi = i;
        }
    }
    names[hi] = "crisis";
    names[lo] = "calm-bull";
    println!("\nFitted K=3 regimes (states sorted by mean; 0 = lowest):");
    println!("  {:>5} {:>10} {:>10} {:>9} {:>11}", "state", "label", "mean(ann)", "vol(ann)", "stationary");
    for s in 0..k {
        println!(
            "  {:>5} {:>10} {:>10} {:>9} {:>11.4}",
            s,
            names[s],
            pct(252.0 * p.means.get(s, 0)),
            format!("{:.2}%", 100.0 * (252.0 * p.variances.get(s, 0)).sqrt()),
            pipe.stationary3[s]
        );
    }
    println!("  transition matrix (rows sum to 1):");
    for i in 0..k {
        let row: Vec<String> = (0..k).map(|j| format!("{:.4}", p.transmat.get(i, j))).collect();
        println!("    {}", row.join("  "));
    }

    let mut counts = vec![0usize; k];
    for &s in &pipe.viterbi3 {
        counts[s] += 1;
    }
    let mut true_counts = vec![0usize; k];
    for &s in &ds.true_states {
        true_counts[s] += 1;
    }
    println!(
        "\nViterbi decoding vs true (generator) states: accuracy = {:.2}%",
        100.0 * pipe.viterbi_accuracy
    );
    println!("  decoded days per state {counts:?}  (true calm/choppy/crisis {true_counts:?})");

    // ---------------- strategy comparison ---------------- //
    let cfg = &pipe.config;
    println!("{LINE}");
    println!(
        "Strategies (cost {:.0} bps, gate mode '{}', refit every {}d on expanding window):",
        cfg.cost_bps, cfg.gate_mode, cfg.refit_days
    );
    println!(
        "  {:<22} {:>8} {:>8} {:>7} {:>8} {:>7} {:>6} {:>9}",
        "strategy", "ann ret", "ann vol", "Sharpe", "maxDD", "Calmar", "hit", "worst mo"
    );
    let row = |label: &str, m: &Metrics| {
        println!(
            "  {:<22} {:>8} {:>8} {:>7.2} {:>8} {:>7.2} {:>5.1}% {:>9}",
            label,
            pct(m.ann_return),
            format!("{:.2}%", 100.0 * m.ann_vol),
            m.sharpe,
            pct(m.max_dd),
            m.calmar,
            100.0 * m.hit_rate,
            pct(m.worst_month)
        );
    };
    row("momentum unfiltered", &pipe.momentum_unfiltered.metrics);
    row("momentum filtered", &pipe.momentum_filtered.metrics);
    row("carry unfiltered", &pipe.carry_unfiltered.metrics);
    row("carry filtered", &pipe.carry_filtered.metrics);
    row("combined unfiltered", &pipe.combined_unfiltered.metrics);
    row("combined filtered", &pipe.combined_filtered.metrics);

    // ---------------- crisis breakdown ---------------- //
    println!("{LINE}");
    println!("Annualized net return by decoded regime (state 0 = lowest mean):");
    println!("  {:<22} {:>9} {:>9} {:>10}", "strategy", names[0], names[1], names[2]);
    let crisis_row = |label: &str, c: &[StateStats]| {
        println!(
            "  {:<22} {:>9} {:>9} {:>10}",
            label,
            pct(c[0].ann_return),
            pct(c[1].ann_return),
            pct(c[2].ann_return)
        );
    };
    crisis_row("momentum unfiltered", &pipe.crisis.momentum_unfiltered);
    crisis_row("momentum filtered", &pipe.crisis.momentum_filtered);
    crisis_row("carry unfiltered", &pipe.crisis.carry_unfiltered);
    crisis_row("carry filtered", &pipe.crisis.carry_filtered);
    crisis_row("combined unfiltered", &pipe.crisis.combined_unfiltered);
    crisis_row("combined filtered", &pipe.crisis.combined_filtered);

    takeaway(&pipe);
    Ok(())
}

fn takeaway(pipe: &PipelineOutput) {
    let mu = &pipe.combined_unfiltered.metrics;
    let mf = &pipe.combined_filtered.metrics;
    println!("{LINE}");
    println!("Takeaway: momentum and carry both lose money in the high-vol crisis");
    println!("regime; scaling by the filtered P(calm) improves the combined Sharpe");
    println!(
        "({:.2} -> {:.2}) and maxDD ({} -> {})",
        mu.sharpe,
        mf.sharpe,
        pct(mu.max_dd),
        pct(mf.max_dd)
    );
    println!(
        "at the cost of total return ({} -> {}) — the filter",
        pct(mu.ann_return),
        pct(mf.ann_return)
    );
    println!("buys risk reduction, not free performance.");
}
