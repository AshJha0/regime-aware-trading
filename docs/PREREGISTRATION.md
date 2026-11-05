# Preregistration: rule for any future gate claim

Fixed on 2026-11-05. This page sets the test a regime gate has to pass before anyone in this project says it works. It is written before the test is run, and it does not change after results are seen.

## Current verdict

On real data the HMM gate lost to the vol rule. Over Kenneth French daily data from 1963-07-01 to 2026-09-30, at 5 bps, with the pinned configuration (`docs/REAL_DATA_EVALUATION.md`):

| Book | HMM gate minus 21-day vol rule, Sharpe | 95% interval |
|---|---|---|
| Market | -0.333 | [-0.56, -0.08] |
| Industry momentum | -0.324 | [-0.54, -0.11] |

A band of 0.3 narrowed the gap but did not close it: -0.293 [-0.51, -0.06] on the market and -0.204 [-0.43, +0.01] on industry momentum. Those two rows compare the banded gate with the unbanded vol rule, so they are not the same-banded test defined below.

**Work on the gate stops unless the test below is passed.** That means no new gate variants, no further tuning and no claim that the gate adds value. The code stays in the repository as a worked example.

## Hypothesis

A regime gate beats the same-banded 21-day vol rule on Sharpe at 5 bps out of sample.

The vol rule is `vol_ratio_gate` with a 21-day window. "Same-banded" means it goes through `apply_gate_band` with the same band as the gate, so a gate cannot win only by trading less.

## Data

Real data only, from the Kenneth R. French Data Library, daily, on a holdout nobody has inspected.

The full 1963-2026 history has now been seen. It informed the verdict above, so it cannot serve as a holdout, and neither can any part of it. A future test therefore needs one of these:

1. New data dated after 2026-09-30, on the same series.
2. A different market that has not been run through any gate in this project.

Earlier data may be used to fit the model, since the walk-forward gate needs a history. Only days in the holdout count towards the result. The holdout length is set in the registration, before any holdout return is read. A few months of new data will not settle this: if the interval comes out wide and includes zero, the test has failed.

## Pass rule

All of these must hold:

- One pinned configuration, written down and committed before the holdout is read: gate settings, band, books, holdout start and end dates, bootstrap seed. There is no choosing among several.
- Paired stationary block bootstrap of the Sharpe difference, gate minus vol rule, using `paired_block_bootstrap`: mean block 21 days, at least 2000 resamples, on net daily returns at 5 bps.
- The lower end of the 95% interval is above zero.
- No tuning after seeing results. If the test fails, the failure stands. A changed configuration is a new hypothesis and needs fresh data that has not been seen.

If more than one book is registered, all of them must pass.

## Reporting

The result is published whatever it shows, on this page, along with the commit of the registered configuration. A failed test is recorded here and the verdict above stays.
