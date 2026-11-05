//! Per-call latency of `forward_step` and `RegimeFilter::update`.
//!
//! Fits K = 3 on the bundled index, then times each call on the bundled
//! returns, cycled.  This measures one CPU routine on warm caches.  The model
//! is daily-frequency: there is no market-data, order or network path here,
//! so these numbers say nothing about a trading system's latency.
//!
//! `cargo run --release --bin bench_filter -- [data_dir] [samples]`

use std::hint::black_box;
use std::time::{Duration, Instant};

use regime::{forward_step, load_dataset, GaussianHmm, Matrix, RegimeFilter};

/// Tick source: fenced TSC on x86-64 (`Instant` is too coarse on Windows).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn ticks() -> u64 {
    use std::arch::x86_64::{_mm_lfence, _rdtsc};
    // SAFETY: x86-64 always has SSE2 (lfence) and rdtsc.
    unsafe {
        _mm_lfence();
        let t = _rdtsc();
        _mm_lfence();
        t
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
fn ticks() -> u64 {
    use std::sync::OnceLock;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

fn ns_per_tick() -> f64 {
    let wall = Instant::now();
    let t0 = ticks();
    while wall.elapsed() < Duration::from_millis(200) {}
    let t1 = ticks();
    wall.elapsed().as_nanos() as f64 / (t1 - t0) as f64
}

fn report(name: &str, s: &mut [u64], scale: f64) {
    let mean = s.iter().map(|&v| v as f64).sum::<f64>() / s.len() as f64 * scale;
    s.sort_unstable();
    let q = |p: f64| s[(p * (s.len() - 1) as f64) as usize] as f64 * scale;
    println!(
        "{name:<22} mean {mean:8.1}  p50 {:8.1}  p90 {:8.1}  p99 {:8.1}  p99.9 {:9.1}  max {:11.1} ns",
        q(0.5),
        q(0.9),
        q(0.99),
        q(0.999),
        s[s.len() - 1] as f64 * scale
    );
}

fn main() -> regime::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).map_or("../data", String::as_str);
    let n: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(2_000_000);
    let warmup = 100_000;
    let r = load_dataset(dir)?.index_returns;
    let mut model = GaussianHmm::with_defaults(3)?;
    model.fit(&Matrix::column(&r), None)?;
    let p = model.params().expect("fit just succeeded");
    let scale = ns_per_tick();
    let mut samples = vec![0u64; n];

    for s in samples.iter_mut() {
        let a = ticks();
        let b = ticks();
        *s = b - a;
    }
    println!("samples per routine: {n}, ns per tick: {scale:.4}, K=3 D=1");
    report("timer floor", &mut samples, scale);

    let mut alpha = p.startprob.clone();
    for i in 0..warmup + n {
        let x = [r[i % r.len()]];
        let a = ticks();
        alpha = forward_step(p, &alpha, &x)?;
        let b = ticks();
        if i >= warmup {
            samples[i - warmup] = b - a;
        }
    }
    black_box(&alpha);
    report("forward_step", &mut samples, scale);

    let mut filt = RegimeFilter::new(p, None)?;
    for i in 0..warmup + n {
        let x = [r[i % r.len()]];
        let a = ticks();
        let out = filt.update(&x)?[0];
        let b = ticks();
        black_box(out);
        if i >= warmup {
            samples[i - warmup] = b - a;
        }
    }
    report("RegimeFilter::update", &mut samples, scale);
    Ok(())
}
