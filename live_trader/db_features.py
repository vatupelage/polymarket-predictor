# live_trader/db_features.py
"""Pure feature extraction for the dollar-bar PTB model.

Features (ordered): distance-to-strike, time-left, and Wang's dollar-bar
microstructure on the most recent completed bar plus a rolling realized vol.
"""
from __future__ import annotations

import math

FEATURE_NAMES = ["drift_pct", "secs_to_close", "duration", "ret", "log_ret",
                 "volatility", "mean_price", "rvol"]


def _population_stdev(xs):
    n = len(xs)
    if n == 0:
        return 0.0
    mean = sum(xs) / n
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / n)


def build_features(bars: list, drift_pct: float, secs_to_close: float,
                   vol_window: int = 10):
    """Return the ordered feature dict, or None if there are too few bars.
    `bars` are completed bars (oldest..newest); the latest is used for the
    single-bar features and the last `vol_window` for realized vol.
    Requires at least `vol_window` bars (and >= 2 always)."""
    need = max(2, vol_window)
    if bars is None or len(bars) < need:
        return None
    last = bars[-1]
    o, h, l, c = last["open"], last["high"], last["low"], last["close"]
    window = bars[-vol_window:]
    rets = [(b["close"] - b["open"]) / b["open"] for b in window if b["open"]]
    return {
        "drift_pct": drift_pct,
        "secs_to_close": secs_to_close,
        "duration": last["duration"],
        "ret": (c - o) / o if o else 0.0,
        "log_ret": math.log(c / o) if (o and c > 0) else 0.0,
        "volatility": h - l,
        "mean_price": (o + h + l + c) / 4.0,
        "rvol": _population_stdev(rets),
    }
