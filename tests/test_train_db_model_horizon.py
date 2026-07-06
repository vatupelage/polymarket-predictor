import importlib
import numpy as np
import pandas as pd


def _synth_df(window_s, n_windows=3, trades_per_s=2, base=100.0):
    """Linear-up price so close>strike every window; dense enough to build bars."""
    rows = []
    t0 = window_s * 1000  # first aligned window start (ms)
    total_s = window_s * (n_windows + 1)
    px = base
    for s in range(total_s):
        ts_ms = t0 + s * 1000
        for _ in range(trades_per_s):
            rows.append((ts_ms, px, 1.0))
        px += 0.01  # strictly rising -> close>strike
    return pd.DataFrame(rows, columns=["ts", "price", "qty"])


def _load_trainer(monkeypatch, window_s, monitor_start_s):
    monkeypatch.setenv("DBM_WINDOW_S", str(window_s))
    monkeypatch.setenv("DBM_MONITOR_START_S", str(monitor_start_s))
    import train.train_db_model as T
    importlib.reload(T)
    return T


def test_15m_horizon_sets_offset_and_feature(monkeypatch):
    T = _load_trainer(monkeypatch, 900, 840)
    assert T.WINDOW_S == 900
    assert T.MONITOR_START_S == 840
    assert T.DECISION_OFFSET_S == 60  # decide 60s into the window
    df = _synth_df(900)
    bars = T.build_bars(df, threshold=5000.0)  # small thr -> many bars
    X, y, tss = T.make_dataset(df, bars, threshold=5000.0,
                               window_s=900, monitor_start_s=840)
    assert len(y) >= 1
    # secs_to_close is FEATURE_NAMES[1]; every row must carry the 15m decision time
    sidx = T.FEATURE_NAMES.index("secs_to_close")
    assert all(row[sidx] == 840 for row in X)
    assert all(v == 1 for v in y)  # rising price -> close>strike


def test_5m_defaults_unchanged(monkeypatch):
    monkeypatch.delenv("DBM_WINDOW_S", raising=False)
    monkeypatch.delenv("DBM_MONITOR_START_S", raising=False)
    import train.train_db_model as T
    importlib.reload(T)
    assert T.WINDOW_S == 300
    assert T.MONITOR_START_S == 240
    assert T.DECISION_OFFSET_S == 60


def test_backtest_buylow_handles_no_logged_windows(monkeypatch, capsys):
    monkeypatch.delenv("DBM_WINDOW_S", raising=False)
    monkeypatch.delenv("DBM_MONITOR_START_S", raising=False)
    import train.train_db_model as T
    importlib.reload(T)
    # Empty logged set must not raise and must print a clear skip notice.
    T.backtest_buylow(model=None, calibrator=None, df=None, bars=None, logged=[])
    out = capsys.readouterr().out
    assert "0 logged windows" in out
