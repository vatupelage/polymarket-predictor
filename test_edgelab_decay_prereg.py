"""TDD for the pre-data hardening of edgelab.decay:
  - pre-registered move target (mid vs micro-price), selectable
  - strict t -> t+lag look-ahead guard (no contemporaneous/earlier pairing,
    even with tied timestamps)
  - liquidity-regime segmentation (decay curve per regime, not one blended line)
  - MDE-cited verdict: "dead at RTT" only when surviving |corr| < MDE

These exist so the first real-data run cannot silently cheat on the one degree
of freedom (the move definition) or report a negative result as vibes.
"""

import numpy as np
import pandas as pd

from edgelab.decay import (decay_curve, decay_by_regime, decay_verdict,
                           microprice, _forward_returns)
from edgelab.replay import synth_window_rows


# ---- micro-price ----

def test_microprice_equal_sizes_is_mid():
    assert abs(microprice(0.40, 100, 0.50, 100) - 0.45) < 1e-12


def test_microprice_pulled_toward_ask_when_more_bid_size():
    # heavy bid depth -> buy pressure -> micro-price above mid
    m = microprice(0.40, 300, 0.50, 100)
    assert m > 0.45
    assert abs(m - 0.475) < 1e-9


# ---- move target is selectable / pre-registered ----

def test_target_microprice_runs_and_differs_from_mid():
    rows = []
    for w in range(4):
        rows += synth_window_rows(f"w{w}", "5m", 1_000_000 + w * 400, n=1500,
                                  seed=w, signal=True)
    df = pd.DataFrame(rows)
    # make micro-price meaningfully different by skewing sizes
    df["best_bid_sz"] = 300.0
    df["best_ask_sz"] = 50.0
    c_mid = {x["lag_ms"]: x for x in decay_curve(df, [250], target="mid")}
    c_mic = {x["lag_ms"]: x for x in decay_curve(df, [250], target="microprice")}
    assert c_mid[250]["corr"] != c_mic[250]["corr"]


# ---- look-ahead guard ----

def test_forward_returns_never_pairs_contemporaneous_or_earlier():
    ts = np.array([0.0, 0.0, 0.0, 0.1, 0.1, 0.2])   # ties at 0.0 and 0.1
    mid = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    fr = _forward_returns(ts, mid, 0.05)
    # i=0..2 (ts=0) must pair with the first ts>=0.05 => index 3 (ts 0.1)
    assert fr[0] == 4.0 - 1.0
    assert fr[2] == 4.0 - 3.0
    # last row has no future price at +0.05 => NaN
    assert np.isnan(fr[5])


def test_decay_curve_rejects_nonpositive_lag():
    df = pd.DataFrame(synth_window_rows("w", "5m", 0, n=200))
    try:
        decay_curve(df, [0])
        assert False, "expected ValueError for non-positive lag"
    except ValueError:
        pass


# ---- regime segmentation ----

def test_decay_by_regime_splits_and_can_differ():
    # window A: real signal; window B (different regime label): pure noise
    rows = synth_window_rows("us-1", "5m", 1_000_000, n=1500, seed=1, signal=True)
    rows += synth_window_rows("ovn-1", "5m", 2_000_000, n=1500, seed=2, signal=False)
    df = pd.DataFrame(rows)
    regime = df["slug"].map(lambda s: "liquid" if s.startswith("us") else "thin")
    out = decay_by_regime(df, [250], regime=regime)
    assert set(out) == {"liquid", "thin"}
    liq = out["liquid"][0]["abs_corr"]
    thin = out["thin"][0]["abs_corr"]
    assert liq > 0.15 and thin < 0.05      # pooling would have hidden this


# ---- MDE-cited verdict ----

def test_verdict_alive_when_corr_exceeds_mde():
    curve = [{"lag_ms": 1, "abs_corr": 0.20, "t_stat": 9.0, "mde_corr": 0.03, "n": 9000}]
    v = decay_verdict(curve, rtt_ms=1)
    assert v["alive"] is True
    assert "0.2" in v["statement"] and "0.03" in v["statement"]


def test_verdict_dead_when_corr_below_mde():
    curve = [{"lag_ms": 1, "abs_corr": 0.01, "t_stat": 0.4, "mde_corr": 0.03, "n": 9000}]
    v = decay_verdict(curve, rtt_ms=1)
    assert v["alive"] is False
    # the verdict must be explicit that this is an MDE statement, not vibes
    assert "MDE" in v["statement"] or "detect" in v["statement"].lower()
