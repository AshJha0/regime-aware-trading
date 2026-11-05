"""Independent correctness check: every expected value was worked out by hand.

The arithmetic is in the comments and the expected values are literals.  No
expected value is computed with library code, so a bug in the code cannot
agree with itself here.  If one of these fails, do not change the number
without redoing the sum on paper.
"""

import numpy as np
import pytest

from regime import (
    CostAssumptions,
    HMMParams,
    apply_gate,
    apply_gate_band,
    carry_positions,
    carry_total_returns,
    forward_step,
    momentum_positions,
    run_backtest,
    trade_cost,
    vol_ratio_gate,
)


def close(x, expected, tol=1e-12):
    assert np.asarray(x, dtype=float) == pytest.approx(expected, abs=tol)


class TestBacktest:
    """Four days, two assets, 100 bps.

    P = [.5, 0], [.5, .5], [.25, .5], [0, 0]
    r = [.10, .20], [.02, .04], [-.10, .02], [.08, -.02]
    """

    @pytest.fixture
    def result(self):
        positions = [[0.5, 0.0], [0.5, 0.5], [0.25, 0.5], [0.0, 0.0]]
        returns = [[0.10, 0.20], [0.02, 0.04], [-0.10, 0.02], [0.08, -0.02]]
        return run_backtest(np.array(positions), np.array(returns), cost_bps=100.0)

    def test_gross_uses_yesterdays_position(self, result):
        # t0: nothing held, so the +10% and +20% are not earned.
        # t1: .5*.02 + 0*.04     =  .01
        # t2: .5*-.10 + .5*.02   = -.05 + .01 = -.04
        # t3: .25*.08 + .5*-.02  =  .02 - .01 =  .01
        close(result.gross_returns, [0.0, 0.01, -0.04, 0.01])

    def test_turnover(self, result):
        # t0: .5 + 0 = .5 (entry from flat)
        # t1: 0 + .5 = .5;  t2: .25 + 0 = .25;  t3: .25 + .5 = .75
        close(result.turnover, [0.5, 0.5, 0.25, 0.75])

    def test_costs_and_net(self, result):
        # 100 bps = .01 of turnover: .005, .005, .0025, .0075
        # net: 0-.005, .01-.005, -.04-.0025, .01-.0075
        close(result.net_returns, [-0.005, 0.005, -0.0425, 0.0025])

    def test_equity(self, result):
        # .995
        # .995 * 1.005        = .999975
        # .999975 * .9575     = .9574760625
        # .9574760625 * 1.0025 = .9574760625 + .00239369015625 = .95986975265625
        close(result.equity, [0.995, 0.999975, 0.9574760625, 0.95986975265625])

    def test_max_drawdown(self, result):
        # Peak .999975 at t1; the trough at t2 is that peak times .9575.
        # t3 sits at .9575*1.0025 of the peak, which is higher.  So -.0425.
        close(result.metrics["max_dd"], -0.0425)

    def test_return_vol_sharpe(self, result):
        # mean = (-.005 + .005 - .0425 + .0025)/4 = -.04/4 = -.01
        # ann_return = 252 * -.01 = -2.52
        # deviations: .005, .015, -.0325, .0125
        # squares: .000025, .000225, .00105625, .00015625; sum .0014625
        # var = .0014625/4 = .000365625
        # ann_vol = sqrt(252 * .000365625) = sqrt(.0921375) = .303542
        # sharpe^2 = .0252/.000365625 = 68.92308, so sharpe = -8.30199
        m = result.metrics
        close(m["ann_return"], -2.52)
        close(m["ann_vol"], 0.303542, tol=1e-5)
        close(m["sharpe"], -8.30199, tol=1e-4)

    def test_other_metrics(self, result):
        # hit: net > 0 on t1 and t3, so 2/4.
        # calmar: -2.52/.0425 = -59.29412
        # worst_month with no dates: one short block, the whole sample,
        # .95986975265625 - 1.
        m = result.metrics
        close(m["hit_rate"], 0.5)
        close(m["calmar"], -59.29412, tol=1e-4)
        close(m["worst_month"], -0.04013024734375)


def test_timing_is_previous_position_times_todays_return():
    # Position opened at the t1 close.  The +50% of t1 happens before it, and
    # only the +10% of t2 is earned.  A lookahead bug would show .5 at t1.
    res = run_backtest(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.5, 0.1]))
    close(res.gross_returns, [0.0, 0.0, 0.1])


def test_momentum_signal_ewma_sizing_and_cap():
    # lookback 2, lambda .5, target .10, cap .5, two assets.
    # A: .01, .01, .01, -.05      B: -.02, 0, -.02, .04
    # Trailing 2-day growth:
    #   t1: A 1.0201 (+)   B .98 (-)
    #   t2: A 1.0201 (+)   B .98 (-)
    #   t3: A 1.01*.95 = .9595 (-)   B .98*1.04 = 1.0192 (+)
    # EWMA variance, seeded at t1 with the mean of squares:
    #   A: t1 (.0001+.0001)/2 = .0001; t2 .00005+.00005 = .0001;
    #      t3 .00005 + .5*.0025 = .0013
    #   B: t1 (.0004+0)/2 = .0002; t2 .0001+.0002 = .0003;
    #      t3 .00015 + .5*.0016 = .00095
    # Leverage = .10/sqrt(252 var) = sqrt(.01/(252 var)):
    #   A t1, t2: sqrt(.01/.0252) = .62994, capped at .5
    #   A t3: sqrt(.01/.3276) = .174714
    #   B t1: sqrt(.01/.0504) = .445435
    #   B t2: sqrt(.01/.0756) = .363696
    #   B t3: sqrt(.01/.2394) = .204379
    # Sign times leverage, halved for two assets:
    r = np.array([[0.01, -0.02], [0.01, 0.0], [0.01, -0.02], [-0.05, 0.04]])
    pos = momentum_positions(r, lookback=2, vol_target=0.10, ewma_lambda=0.5, leverage_cap=0.5)
    expected = [0.0, 0.0, 0.25, -0.2227175, 0.25, -0.181848, -0.087357, 0.1021895]
    close(pos.ravel(), expected, tol=1e-5)


class TestCarry:
    # Three currencies, one long and one short, rebalance every 2 days.
    DIFFS = np.array([
        [0.252, 0.0252, 0.126],   # t0 rebalance: long 0, short 1
        [0.0, 0.252, -0.252],     # t1 not a rebalance: this ranking is ignored
        [0.0252, 0.0252, 0.0],    # t2 rebalance: 0 and 1 tie, lower index long; short 2
        [0.0, 0.0, 0.0],
    ])
    SPOT = np.array([[0.01, 0.02, 0.03], [0.01, -0.01, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    def test_positions_schedule_and_tie_break(self):
        pos = carry_positions(self.DIFFS, top_n=1, bottom_n=1, rebalance_days=2)
        close(pos.ravel(), [1, -1, 0, 1, -1, 0, 1, 0, -1, 1, 0, -1])

    def test_total_return_accrues_yesterdays_differential(self):
        # t0: spot only.
        # t1: spot + t0 diff/252 = .01+.001, -.01+.0001, 0+.0005
        # t2: t1 diff/252 = 0, .001, -.001
        # t3: t2 diff/252 = .0001, .0001, 0
        total = carry_total_returns(self.SPOT, self.DIFFS)
        close(total.ravel(), [
            0.01, 0.02, 0.03,
            0.011, -0.0099, 0.0005,
            0.0, 0.001, -0.001,
            0.0001, 0.0001, 0.0,
        ])

    def test_backtest_of_the_carry_book(self):
        # t1: [1,-1,0].[.011,-.0099,.0005]  = .011 + .0099 = .0209
        # t2: [1,-1,0].[0,.001,-.001]       = -.001
        # t3: [1,0,-1].[.0001,.0001,0]      = .0001
        # turnover: 2, 0, (0 + 1 + 1) = 2, 0
        pos = carry_positions(self.DIFFS, top_n=1, bottom_n=1, rebalance_days=2)
        res = run_backtest(pos, carry_total_returns(self.SPOT, self.DIFFS))
        close(res.gross_returns, [0.0, 0.0209, -0.001, 0.0001])
        close(res.turnover, [2.0, 0.0, 2.0, 0.0])


def test_gate_band_then_backtest():
    # Band .3 from a held value of 1:
    #   .9:  |.9-1| = .1, keep 1
    #   .6:  .4 > .3, move to .6
    #   .55: .05, keep .6
    #   .1:  .5 > .3, move to .1
    banded = apply_gate_band(np.array([0.9, 0.6, 0.55, 0.1]), 0.3)
    close(banded, [1.0, 0.6, 0.6, 0.1])
    # Gated positions: [1,-1], [.6,-.6], [.6,-.6], [.1,-.1]
    # gross: t1 .01 - .03 = -.02
    #        t2 .6*.02 + .6*.01 = .012 + .006 = .018
    #        t3 .6*-.05 - .6*0 = -.03
    # turnover: 2, .4+.4 = .8, 0, .5+.5 = 1
    pos = apply_gate(np.tile([1.0, -1.0], (4, 1)), banded)
    r = np.array([[0.0, 0.0], [0.01, 0.03], [0.02, -0.01], [-0.05, 0.0]])
    res = run_backtest(pos, r)
    close(res.gross_returns, [0.0, -0.02, 0.018, -0.03])
    close(res.turnover, [2.0, 0.8, 0.0, 1.0])


def test_vol_ratio_gate():
    # Window 1, so vol is |r|: .01, .02, .01, .04.
    # Expanding median: .01, .015, .01, .015.
    # t0 is before first_day, so 1.
    # t1 .015/.02 = .75;  t2 .01/.01 = 1;  t3 .015/.04 = .375
    gate = vol_ratio_gate(np.array([0.01, -0.02, 0.01, 0.04]), first_day=1, window=1)
    close(gate, [1.0, 0.75, 1.0, 0.375])


def test_forward_step_with_equal_likelihoods():
    # Two states at means -1 and +1, variance 1, observation 0: the two
    # likelihoods are equal, so the update is the prediction alone.
    # From [.5, .5]: .5*.9 + .5*.2 = .55 and .5*.1 + .5*.8 = .45.
    params = HMMParams(
        startprob=np.array([0.5, 0.5]),
        transmat=np.array([[0.9, 0.1], [0.2, 0.8]]),
        means=np.array([[-1.0], [1.0]]),
        variances=np.array([[1.0], [1.0]]),
    )
    close(forward_step(params, np.array([0.5, 0.5]), np.array([0.0])), [0.55, 0.45])


def test_cost_formula():
    # AUM 1e8, dw .1, so trade = 1e7.  ADV 1e9, participation .01, root .1.
    # spread: 2 bps * 1e7 = 2000
    # impact: 1 * .01 * .1 * 1e7 = 10000
    a = CostAssumptions(half_spread_bps=[2.0], adv_usd=[1e9], sigma_daily=[0.01], impact_y=1.0)
    close(trade_cost(np.array([0.1]), 1e8, a), [12000.0], tol=1e-6)
