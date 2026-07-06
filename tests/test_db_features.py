# tests/test_db_features.py
import math
from live_trader.db_features import build_features, FEATURE_NAMES


def _bar(o, h, l, c, dur=20.0, vol=50.0):
    return {"open": o, "high": h, "low": l, "close": c, "volume": vol,
            "dollar_value": 5_000_000.0, "start_ts": 0, "end_ts": int(dur * 1000),
            "duration": dur}


def test_returns_none_when_too_few_bars():
    assert build_features([], drift_pct=0.1, secs_to_close=180, vol_window=3) is None
    assert build_features([_bar(100, 101, 99, 100)], drift_pct=0.1,
                          secs_to_close=180, vol_window=3) is None


def test_feature_values():
    bars = [_bar(100, 102, 99, 101), _bar(101, 103, 100, 102),
            _bar(102, 104, 101, 103)]
    f = build_features(bars, drift_pct=0.25, secs_to_close=180.0, vol_window=3)
    assert f["drift_pct"] == 0.25
    assert f["secs_to_close"] == 180.0
    # latest bar = (102,104,101,103)
    assert f["duration"] == 20.0
    assert abs(f["ret"] - (103 - 102) / 102) < 1e-9
    assert abs(f["log_ret"] - math.log(103 / 102)) < 1e-9
    assert f["volatility"] == 104 - 101
    assert f["mean_price"] == (102 + 104 + 101 + 103) / 4
    # rvol = stdev of last 3 bar returns
    rets = [(101 - 100) / 100, (102 - 101) / 101, (103 - 102) / 102]
    mean = sum(rets) / 3
    expected_rvol = math.sqrt(sum((r - mean) ** 2 for r in rets) / 3)
    assert abs(f["rvol"] - expected_rvol) < 1e-9


def test_feature_order_matches_names():
    bars = [_bar(100, 102, 99, 101), _bar(101, 103, 100, 102),
            _bar(102, 104, 101, 103)]
    f = build_features(bars, drift_pct=0.25, secs_to_close=180.0, vol_window=3)
    assert list(f.keys()) == FEATURE_NAMES
