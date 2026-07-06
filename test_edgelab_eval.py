"""TDD for edgelab.eval — the honest-evaluation rig.

Headline metric = edge per trade = realized win rate MINUS entry-price
breakeven, with a confidence band AND a minimum-detectable-edge so that
"no edge" is reported with the same force as "edge": it means "no edge larger
than X cents at this sample size", never "we failed to find one".

Threshold sweeps manufacture false edge, so the swept results are deflated:
Deflated Sharpe Ratio (Bailey & Lopez de Prado) and Probability of Backtest
Overfitting (PBO via CSCV).
"""

import numpy as np

from edgelab.eval import (edge_per_trade, min_detectable_edge,
                          deflated_sharpe_ratio, pbo)


# ---- edge per trade ----

def test_edge_per_trade_positive():
    # bought 100 binaries at 0.50, won 60 -> edge per trade = 0.60-0.50 = +0.10
    entry = [0.50] * 100
    won = [1] * 60 + [0] * 40
    res = edge_per_trade(entry, won)
    assert abs(res["edge"] - 0.10) < 1e-9
    assert res["n"] == 100
    assert res["t_stat"] > 1.0
    assert res["ci_lo"] < res["edge"] < res["ci_hi"]


def test_edge_per_trade_zero_when_fairly_priced():
    # bought at exactly the realized frequency -> ~zero edge
    entry = [0.60] * 100
    won = [1] * 60 + [0] * 40
    res = edge_per_trade(entry, won)
    assert abs(res["edge"]) < 1e-9
    assert abs(res["t_stat"]) < 1e-6


# ---- minimum detectable edge ----

def test_mde_shrinks_with_n():
    sd = 0.5
    mde_small = min_detectable_edge(n=100, sd=sd)
    mde_big = min_detectable_edge(n=10_000, sd=sd)
    assert mde_big < mde_small
    # MDE scales ~ 1/sqrt(n): 100x the n -> ~10x tighter
    assert abs(mde_small / mde_big - 10.0) < 0.5


def test_mde_formula_value():
    # (z_a/2 + z_b) * sd / sqrt(n) with z=1.96, 0.842
    val = min_detectable_edge(n=400, sd=0.5)
    expected = (1.959963985 + 0.841621234) * 0.5 / np.sqrt(400)
    assert abs(val - expected) < 1e-6


# ---- deflated sharpe ----

def test_dsr_decreases_with_more_trials():
    # same observed SR, more trials searched -> more deflation -> lower DSR
    kw = dict(sr=0.15, sr_std=0.1, T=1000)
    assert deflated_sharpe_ratio(n_trials=2, **kw) > deflated_sharpe_ratio(n_trials=500, **kw)


def test_dsr_in_unit_interval():
    p = deflated_sharpe_ratio(sr=0.2, sr_std=0.1, n_trials=50, T=1000)
    assert 0.0 <= p <= 1.0


# ---- PBO via CSCV ----

def test_pbo_high_for_pure_noise():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((240, 20))   # 20 strategies, all noise
    p = pbo(M, n_splits=10)
    assert 0.3 < p < 0.7                  # ~0.5 for noise


def test_pbo_low_for_one_dominant_strategy():
    rng = np.random.default_rng(1)
    M = rng.standard_normal((240, 20)) * 0.1
    M[:, 0] += 0.5                        # strategy 0 truly dominates everywhere
    p = pbo(M, n_splits=10)
    assert p < 0.2
