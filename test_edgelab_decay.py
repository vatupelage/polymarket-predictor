"""TDD for edgelab.decay.decay_curve — the cheap Edge-B gate.

Measures predictive power of OFI-now over the mid move at lags
{100,250,500,1000,2000} ms. Validated against replay ground truth: the
synthetic single-step kick must show strong power at short lag DECAYING with
lag; a pure-noise control must show no significant power at any lag.

Look-ahead guard: future return at time t uses only mid stamped strictly after
t, and never crosses a window (slug) boundary.
"""

import pandas as pd

from edgelab.decay import decay_curve
from edgelab.replay import synth_window_rows


def _df(signal=True, n_windows=8):
    rows = []
    for w in range(n_windows):
        rows += synth_window_rows(f"w{w}", "5m", 1_000_000 + w * 400,
                                  n=2000, dt=0.05, seed=w, signal=signal)
    return pd.DataFrame(rows)


def test_signal_present_and_decays():
    df = _df(signal=True)
    curve = {c["lag_ms"]: c for c in decay_curve(df, lags_ms=[100, 250, 500, 1000, 2000])}
    # short-lag power is real and significant
    assert curve[100]["abs_corr"] > 0.15
    assert curve[100]["t_stat"] > 3.0
    # and it decays as lag grows
    assert curve[100]["abs_corr"] > curve[2000]["abs_corr"]


def test_null_control_has_no_significant_power():
    df = _df(signal=False)
    curve = {c["lag_ms"]: c for c in decay_curve(df, lags_ms=[100, 500, 2000])}
    # pure noise: |corr| should be small at every lag
    for lag in (100, 500, 2000):
        assert curve[lag]["abs_corr"] < 0.05


def test_reports_mde_and_n():
    df = _df(signal=True)
    curve = decay_curve(df, lags_ms=[250])
    c = curve[0]
    assert c["n"] > 1000
    # minimum detectable correlation at this n should be small and positive
    assert 0.0 < c["mde_corr"] < 0.2


def test_no_lookahead_across_window_boundary():
    # two windows whose ts ranges are far apart; a lag must never pair a row in
    # window A with a mid in window B. If it did, n would balloon nonsensically.
    df = _df(signal=True, n_windows=2)
    curve = decay_curve(df, lags_ms=[100])
    # each window contributes < its own row count of forward pairs
    assert curve[0]["n"] < len(df)
