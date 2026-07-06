# Dollar-Bar PTB Contrarian Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new live bot mode where a calibrated XGBoost/LightGBM model (Wang dollar-bar microstructure + distance-to-strike) predicts P(BTC close > Price to Beat) per 5-min window, and the bot buys the predicted side only when its PM ask is below ~0.50, working the order to fill cheap. The LSTM is removed from this path entirely.

**Architecture:** A Binance aggTrade websocket feeds a pure dollar-bar builder into a thread-safe ring buffer. A static, offline-trained, calibrated model loaded from `.joblib` predicts the window outcome at ~3 min before close. A pure decision gate compares the calibrated probability to the live PM ask; a work-the-order loop in `bot.py` buys the predicted side cheap. A dedicated runner in `run_live_bot.py` walks every window and never imports `run_baseline` (so TF/LSTM never load).

**Tech Stack:** Python 3, `xgboost`, `lightgbm`, `scikit-learn` (isotonic calibration), `websocket-client` (already used by `arb_ws.py`), `pandas`, `pytest`. Spec: `docs/superpowers/specs/2026-06-04-dbmodel-ptb-contrarian-design.md`.

---

## File Structure

| File | Responsibility |
|---|---|
| `live_trader/dollar_bars.py` | Pure `DollarBarBuilder`, `parse_aggtrade`, thread-safe `BarBuffer`, and `BinanceAggTradeClient` (live WS feed). |
| `live_trader/db_features.py` | Pure `build_features(bars, drift_pct, secs_to_close, ...)`. |
| `live_trader/db_decision.py` | Pure `db_decision(...)` gate. |
| `live_trader/db_model.py` | `DbModel` — loads joblib bundle, `predict_p_up(features)`. |
| `tools/fetch_binance_aggtrades.py` | Offline: download historical BTCUSDT aggTrades to parquet. |
| `train/train_db_model.py` | Offline: build bars+windows, train+calibrate XGB/LGBM, eval, backtest on logged windows, save `.joblib`. |
| `live_trader/config.py` (modify) | Add `BOT_DB_*` fields + loaders. |
| `live_trader/bot.py` (modify) | Add `_execute_dbmodel_trade` + `on_prediction` dispatch. |
| `run_live_bot.py` (modify) | Add `dbmodel` profile, `--dbmodel` flag, `_run_dbmodel_mode` runner. |
| `tests/test_dollar_bars.py`, `tests/test_db_features.py`, `tests/test_db_decision.py` | Unit tests for the pure units. |
| `requirements-live.txt` (modify) | Add `xgboost`, `lightgbm`, `scikit-learn`. |

**Shared signatures (used across tasks — keep identical):**
- Bar dict keys: `open, high, low, close, volume, dollar_value, start_ts, end_ts, duration` (`*_ts` are ms ints; `duration` seconds float).
- Feature dict keys (ordered): `drift_pct, secs_to_close, duration, ret, log_ret, volatility, mean_price, rvol`.
- `DbModel.predict_p_up(features: dict) -> float` (calibrated P(close>PTB)).
- `db_decision(...) -> {side, p_side, conf, ask, edge, eligible, buy_now}`.

---

## Phase A — Pure units (TDD)

### Task 1: DollarBarBuilder + aggTrade parser

**Files:**
- Create: `live_trader/dollar_bars.py`
- Test: `tests/test_dollar_bars.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dollar_bars.py
from live_trader.dollar_bars import DollarBarBuilder, parse_aggtrade


def test_no_bar_until_threshold_crossed():
    b = DollarBarBuilder(threshold_usd=1000.0)
    # 2 trades of $400 each = $800 < $1000 -> no bar yet
    assert b.add_trade(100.0, 4.0, 1_000) is None   # $400
    assert b.add_trade(100.0, 4.0, 2_000) is None   # $800


def test_bar_emitted_on_threshold_cross():
    b = DollarBarBuilder(threshold_usd=1000.0)
    b.add_trade(100.0, 4.0, 1_000)                  # $400
    bar = b.add_trade(101.0, 6.0, 4_000)            # +$606 -> $1006 >= $1000
    assert bar is not None
    assert bar["open"] == 100.0
    assert bar["close"] == 101.0
    assert bar["high"] == 101.0
    assert bar["low"] == 100.0
    assert bar["volume"] == 10.0
    assert abs(bar["dollar_value"] - 1006.0) < 1e-6
    assert bar["start_ts"] == 1_000
    assert bar["end_ts"] == 4_000
    assert abs(bar["duration"] - 3.0) < 1e-6        # (4000-1000)/1000 seconds


def test_builder_resets_after_bar():
    b = DollarBarBuilder(threshold_usd=1000.0)
    b.add_trade(100.0, 11.0, 1_000)                 # $1100 -> emits bar
    # next trade starts a fresh bar
    assert b.add_trade(100.0, 4.0, 5_000) is None
    assert b._dollar == 400.0


def test_parse_aggtrade():
    msg = {"p": "104250.10", "q": "0.005", "T": 1700000000123}
    price, qty, ts = parse_aggtrade(msg)
    assert price == 104250.10
    assert qty == 0.005
    assert ts == 1700000000123
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/vidura/btcpredictor/predictor && python -m pytest tests/test_dollar_bars.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'live_trader.dollar_bars'`

- [ ] **Step 3: Write minimal implementation**

```python
# live_trader/dollar_bars.py
"""Dollar-bar construction from a Binance aggTrade feed.

A dollar bar closes when cumulative traded value (sum of price*qty) crosses a
threshold. Pure builder + parser are unit-tested; the websocket client is a thin
live wrapper around them. See docs/superpowers/specs/2026-06-04-dbmodel-ptb-contrarian-design.md
"""
from __future__ import annotations

import json
import threading
from collections import deque


def parse_aggtrade(msg: dict):
    """Extract (price, qty, ts_ms) from a Binance aggTrade record.
    Works for both the websocket payload and the REST/CSV shape (keys p, q, T)."""
    return float(msg["p"]), float(msg["q"]), int(msg["T"])


class DollarBarBuilder:
    """Accumulates trades and emits a bar dict when dollar volume >= threshold."""

    def __init__(self, threshold_usd: float):
        self.threshold = float(threshold_usd)
        self._reset()

    def _reset(self):
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self._vol = 0.0
        self._dollar = 0.0
        self._start_ts = None
        self._end_ts = None

    def add_trade(self, price: float, qty: float, ts_ms: int):
        """Add one trade. Returns a completed bar dict when the threshold is
        crossed (and resets), else None."""
        if self._open is None:
            self._open = price
            self._high = price
            self._low = price
            self._start_ts = ts_ms
        self._high = max(self._high, price)
        self._low = min(self._low, price)
        self._close = price
        self._vol += qty
        self._dollar += price * qty
        self._end_ts = ts_ms
        if self._dollar >= self.threshold:
            bar = {
                "open": self._open, "high": self._high, "low": self._low,
                "close": self._close, "volume": self._vol,
                "dollar_value": self._dollar,
                "start_ts": self._start_ts, "end_ts": self._end_ts,
                "duration": (self._end_ts - self._start_ts) / 1000.0,
            }
            self._reset()
            return bar
        return None


class BarBuffer:
    """Thread-safe fixed-size ring buffer of recent bars."""

    def __init__(self, maxlen: int = 500):
        self._dq = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, bar: dict):
        with self._lock:
            self._dq.append(bar)

    def snapshot(self) -> list:
        with self._lock:
            return list(self._dq)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/vidura/btcpredictor/predictor && python -m pytest tests/test_dollar_bars.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/vidura/btcpredictor/predictor
git add live_trader/dollar_bars.py tests/test_dollar_bars.py
git commit -m "feat(dbmodel): pure dollar-bar builder + aggTrade parser + bar buffer"
```

---

### Task 2: Binance aggTrade websocket client

**Files:**
- Modify: `live_trader/dollar_bars.py` (append `BinanceAggTradeClient`)

This is a live network wrapper (not unit-tested for I/O; the pure parts it relies on are already tested in Task 1). It follows the `websocket-client` usage already present in `live_trader/arb_ws.py`.

- [ ] **Step 1: Append the client to `live_trader/dollar_bars.py`**

```python
# --- append to live_trader/dollar_bars.py ---
import time
import websocket  # websocket-client, already a dependency (see arb_ws.py)


class BinanceAggTradeClient:
    """Streams BTCUSDT aggTrades, builds dollar bars, exposes last_price + a
    thread-safe BarBuffer. Auto-reconnects. Start with .start() (spawns a daemon
    thread); read .last_price and .bars.snapshot() from other threads."""

    URL = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"

    def __init__(self, threshold_usd: float, buffer_len: int = 500):
        self.builder = DollarBarBuilder(threshold_usd)
        self.bars = BarBuffer(buffer_len)
        self.last_price = None
        self._stop = False
        self._thread = None

    def _on_message(self, _ws, message):
        try:
            price, qty, ts = parse_aggtrade(json.loads(message))
        except Exception:
            return
        self.last_price = price
        bar = self.builder.add_trade(price, qty, ts)
        if bar is not None:
            self.bars.append(bar)

    def _run(self):
        while not self._stop:
            try:
                ws = websocket.WebSocketApp(
                    self.URL, on_message=self._on_message,
                )
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                print(f"  [BINANCE-WS] error {type(e).__name__}: {e} — reconnecting in 3s")
            if not self._stop:
                time.sleep(3)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop = True
```

- [ ] **Step 2: Smoke-test the live feed manually (10 s)**

Run:
```bash
cd /home/vidura/btcpredictor/predictor && python -c "
import time
from live_trader.dollar_bars import BinanceAggTradeClient
c = BinanceAggTradeClient(threshold_usd=5_000_000).start()
time.sleep(10)
print('last_price:', c.last_price, 'bars:', len(c.bars.snapshot()))
assert c.last_price is not None, 'no trades received'
print('OK')
"
```
Expected: prints a BTC price (e.g. `last_price: 10xxxx.x`) and `OK`. (Bars may be 0 in 10 s at a $5M threshold — that's fine; we only assert `last_price`.)

- [ ] **Step 3: Commit**

```bash
cd /home/vidura/btcpredictor/predictor
git add live_trader/dollar_bars.py
git commit -m "feat(dbmodel): Binance aggTrade websocket client feeding dollar bars"
```

---

### Task 3: Feature builder

**Files:**
- Create: `live_trader/db_features.py`
- Test: `tests/test_db_features.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/vidura/btcpredictor/predictor && python -m pytest tests/test_db_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'live_trader.db_features'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/vidura/btcpredictor/predictor && python -m pytest tests/test_db_features.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/vidura/btcpredictor/predictor
git add live_trader/db_features.py tests/test_db_features.py
git commit -m "feat(dbmodel): pure dollar-bar feature builder"
```

---

### Task 4: Decision gate

**Files:**
- Create: `live_trader/db_decision.py`
- Test: `tests/test_db_decision.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db_decision.py
from live_trader.db_decision import db_decision


def _kw(**over):
    base = dict(p_up=0.62, top_ask_up=0.42, top_ask_down=0.60,
                target_ask=0.45, max_ask=0.50, min_conf=0.10, fee_buffer=0.01)
    base.update(over)
    return base


def test_picks_up_when_p_up_above_half():
    d = db_decision(**_kw())
    assert d["side"] == "UP"
    assert abs(d["p_side"] - 0.62) < 1e-9
    assert d["ask"] == 0.42


def test_eligible_and_buy_now_when_below_target():
    d = db_decision(**_kw(top_ask_up=0.44))  # 0.44 <= target 0.45, edge 0.18 > buffer
    assert d["eligible"] is True
    assert d["buy_now"] is True


def test_eligible_not_buy_now_between_target_and_ceiling():
    d = db_decision(**_kw(top_ask_up=0.48))  # 0.45 < 0.48 < 0.50
    assert d["eligible"] is True
    assert d["buy_now"] is False


def test_not_eligible_at_or_above_ceiling():
    d = db_decision(**_kw(top_ask_up=0.50))  # not < max_ask
    assert d["eligible"] is False
    assert d["buy_now"] is False


def test_not_eligible_when_edge_below_buffer():
    # p_up 0.505 -> p_side 0.505, ask 0.50 would be > max anyway; use ask 0.49
    d = db_decision(**_kw(p_up=0.495, top_ask_up=0.49, top_ask_down=0.49))
    # side flips to DOWN (p_up<0.5); p_side=0.505, ask_down=0.49, edge=0.015>buffer,
    # but conf=|0.495-0.5|*2=0.01 < min_conf 0.10 -> not eligible
    assert d["side"] == "DOWN"
    assert d["eligible"] is False


def test_down_side_uses_down_ask():
    d = db_decision(**_kw(p_up=0.30, top_ask_up=0.80, top_ask_down=0.22))
    assert d["side"] == "DOWN"
    assert abs(d["p_side"] - 0.70) < 1e-9
    assert d["ask"] == 0.22
    assert d["eligible"] is True   # 0.22<0.50, edge 0.48>buffer, conf 0.40>min
    assert d["buy_now"] is True    # 0.22<=0.45


def test_missing_ask_not_eligible():
    d = db_decision(**_kw(top_ask_up=None))
    assert d["eligible"] is False
    assert d["buy_now"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/vidura/btcpredictor/predictor && python -m pytest tests/test_db_decision.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'live_trader.db_decision'`

- [ ] **Step 3: Write minimal implementation**

```python
# live_trader/db_decision.py
"""Pure decision gate for the dollar-bar PTB contrarian strategy.

Given the model's calibrated P(close>PTB) and the live PM asks, choose the
predicted side and decide whether to buy. We only fire when the predicted side's
ask is below the ceiling AND the model's probability beats the ask by more than
the fee buffer AND model confidence clears a floor. `buy_now` is True once the
ask has dipped to the aspirational target; the caller adds the deadline rule.
"""
from __future__ import annotations


def db_decision(*, p_up, top_ask_up, top_ask_down, target_ask, max_ask,
                min_conf, fee_buffer):
    side = "UP" if p_up >= 0.5 else "DOWN"
    p_side = p_up if side == "UP" else (1.0 - p_up)
    ask = top_ask_up if side == "UP" else top_ask_down
    conf = abs(p_up - 0.5) * 2.0
    edge = (p_side - ask) if ask is not None else None
    eligible = (
        ask is not None
        and ask < max_ask
        and edge is not None and edge > fee_buffer
        and conf >= min_conf
    )
    buy_now = bool(eligible and ask <= target_ask)
    return {
        "side": side, "p_side": p_side, "conf": conf, "ask": ask,
        "edge": edge, "eligible": bool(eligible), "buy_now": buy_now,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/vidura/btcpredictor/predictor && python -m pytest tests/test_db_decision.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
cd /home/vidura/btcpredictor/predictor
git add live_trader/db_decision.py tests/test_db_decision.py
git commit -m "feat(dbmodel): pure contrarian decision gate"
```

---

### Task 5: Model loader

**Files:**
- Create: `live_trader/db_model.py`

No unit test (it loads a trained artifact that doesn't exist until Task 8). It is exercised end-to-end in Task 8's smoke step and live.

- [ ] **Step 1: Write the implementation**

```python
# live_trader/db_model.py
"""Loads the trained dollar-bar PTB model bundle and returns a calibrated P(up).

The bundle (saved by train/train_db_model.py) is a dict:
  {"model": fitted estimator, "calibrator": isotonic/None,
   "feature_names": [...], "threshold_usd": float, "meta": {...}}
"""
from __future__ import annotations

import joblib
import numpy as np


class DbModel:
    def __init__(self, path: str):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.calibrator = bundle.get("calibrator")
        self.feature_names = bundle["feature_names"]
        self.threshold_usd = bundle["threshold_usd"]
        self.meta = bundle.get("meta", {})

    def predict_p_up(self, features: dict) -> float:
        x = np.array([[features[name] for name in self.feature_names]], dtype=float)
        raw = float(self.model.predict_proba(x)[0, 1])
        if self.calibrator is not None:
            raw = float(self.calibrator.predict([raw])[0])
        return min(max(raw, 0.0), 1.0)
```

- [ ] **Step 2: Verify it imports**

Run: `cd /home/vidura/btcpredictor/predictor && python -c "from live_trader.db_model import DbModel; print('import OK')"`
Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
cd /home/vidura/btcpredictor/predictor
git add live_trader/db_model.py
git commit -m "feat(dbmodel): calibrated model loader"
```

---

## Phase B — Offline data & training

### Task 6: Historical aggTrade fetcher

**Files:**
- Create: `tools/fetch_binance_aggtrades.py`

Pulls Binance's public monthly aggTrade dumps from `data.binance.vision` (no API key, far faster than the REST `/aggTrades` 1000-row pages). Each monthly zip contains a CSV with columns: aggId, price, qty, firstId, lastId, timestamp(ms), isBuyerMaker, isBestMatch.

- [ ] **Step 1: Write the implementation**

```python
# tools/fetch_binance_aggtrades.py
"""Download Binance BTCUSDT monthly aggTrade dumps and store a compact parquet
of (ts_ms, price, qty). Usage:
    python tools/fetch_binance_aggtrades.py 2026-03 2026-06 data/aggtrades.parquet
Downloads months in [start, end] inclusive (YYYY-MM)."""
import io
import sys
import zipfile
import urllib.request

import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT"


def months(start: str, end: str):
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            m, y = 1, y + 1


def fetch_month(ym: str) -> pd.DataFrame:
    url = f"{BASE}/BTCUSDT-aggTrades-{ym}.zip"
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            df = pd.read_csv(
                fh, header=None,
                names=["aggId", "price", "qty", "firstId", "lastId",
                       "ts", "isBuyerMaker", "isBestMatch"],
                usecols=["price", "qty", "ts"],
            )
    # Some 2025+ dumps store ts in microseconds; normalize to ms.
    if df["ts"].iloc[0] > 10**14:
        df["ts"] = df["ts"] // 1000
    return df[["ts", "price", "qty"]]


def main():
    if len(sys.argv) != 4:
        print("usage: fetch_binance_aggtrades.py START_YYYY-MM END_YYYY-MM OUT.parquet")
        sys.exit(1)
    start, end, out = sys.argv[1], sys.argv[2], sys.argv[3]
    frames = [fetch_month(ym) for ym in months(start, end)]
    df = pd.concat(frames, ignore_index=True).sort_values("ts")
    df.to_parquet(out, index=False)
    print(f"  wrote {len(df):,} trades to {out} "
          f"({pd.to_datetime(df['ts'].min(), unit='ms')} .. "
          f"{pd.to_datetime(df['ts'].max(), unit='ms')})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to fetch the training window**

Run (covers our logged data span Apr–Jun plus lead-in):
```bash
cd /home/vidura/btcpredictor/predictor && mkdir -p data && python tools/fetch_binance_aggtrades.py 2026-03 2026-06 data/aggtrades.parquet
```
Expected: prints download lines per month and a final `wrote N,NNN,NNN trades to data/aggtrades.parquet` with a date range spanning 2026-03 .. 2026-06. (If a month 404s because it isn't published yet, fetch only the available months and adjust the end argument.)

- [ ] **Step 3: Commit the fetcher (not the data)**

```bash
cd /home/vidura/btcpredictor/predictor
echo "data/" >> .gitignore
git add tools/fetch_binance_aggtrades.py .gitignore
git commit -m "feat(dbmodel): Binance monthly aggTrade fetcher"
```

---

### Task 7: θ calibration helper (median bar ≈ 20–30 s)

**Files:**
- Create: `train/calibrate_threshold.py`

- [ ] **Step 1: Write the implementation**

```python
# train/calibrate_threshold.py
"""Pick the dollar-bar threshold so the median bar lasts ~20-30s on the data.
    python train/calibrate_threshold.py data/aggtrades.parquet
Prints median bar duration for a grid of thresholds; choose one near 25s."""
import sys
import statistics as st

import pandas as pd

from live_trader.dollar_bars import DollarBarBuilder


def median_duration(trades, threshold):
    b = DollarBarBuilder(threshold)
    durs = []
    for ts, price, qty in trades:
        bar = b.add_trade(price, qty, int(ts))
        if bar:
            durs.append(bar["duration"])
    return st.median(durs) if durs else None, len(durs)


def main():
    df = pd.read_parquet(sys.argv[1])
    # subsample one recent day for speed
    cutoff = df["ts"].max() - 24 * 3600 * 1000
    day = df[df["ts"] >= cutoff]
    trades = list(zip(day["ts"].values, day["price"].values, day["qty"].values))
    print(f"calibrating on {len(trades):,} trades (last 24h)")
    for thr in (1e6, 2e6, 5e6, 1e7, 2e7, 5e7):
        med, n = median_duration(trades, thr)
        print(f"  threshold=${thr:>12,.0f}  median_dur={med}  bars={n}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and record the chosen θ**

Run: `cd /home/vidura/btcpredictor/predictor && python train/calibrate_threshold.py data/aggtrades.parquet`
Expected: a table of thresholds vs median bar duration. **Choose the threshold whose `median_dur` is closest to 25 s** and use it as `THRESHOLD_USD` in Task 8. (Record the value in the commit message.)

- [ ] **Step 3: Commit**

```bash
cd /home/vidura/btcpredictor/predictor
git add train/calibrate_threshold.py
git commit -m "feat(dbmodel): dollar-bar threshold calibration helper"
```

---

### Task 8: Train, calibrate, evaluate, and backtest the model

**Files:**
- Create: `train/train_db_model.py`
- Create: `models/` (output dir)

This is the heart of Phase B. It (1) builds dollar bars over the full history, (2) builds the large *reconstructed* training set (strike=price@window-open, label=price@close>strike, features@decision-time = window-open+120s), (3) trains XGBoost + LightGBM with imbalance handling, (4) isotonic-calibrates on a held-out slice, (5) reports Brier/log-loss/accuracy, and (6) runs the **honest buy-low backtest on our logged windows** (real PM asks `up_price`/`down_price`, real `would_have_won`), netting the verified fee.

- [ ] **Step 1: Write the implementation**

```python
# train/train_db_model.py
"""Train the dollar-bar PTB contrarian model.

  python train/train_db_model.py data/aggtrades.parquet <THRESHOLD_USD> models/db_ptb.joblib

Pipeline: build bars -> reconstructed windows (train) -> XGB+LGBM + isotonic
calibration -> metrics -> buy-low backtest on logged windows -> save winner.
"""
import json
import sys
import math
import glob
import statistics as st

import numpy as np
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from live_trader.dollar_bars import DollarBarBuilder
from live_trader.db_features import build_features, FEATURE_NAMES

WINDOW_S = 300
MONITOR_START_S = 180          # decision @ 120s into the window
DECISION_OFFSET_S = WINDOW_S - MONITOR_START_S  # 120
VOL_WINDOW = 10
FEE_RATE = 0.07                # crypto taker feeRate (verified)


def fee_per_share(p):          # taker fee in $ per share at price p
    return FEE_RATE * p * (1 - p)


def build_bars(df, threshold):
    b = DollarBarBuilder(threshold)
    bars = []
    for ts, price, qty in zip(df["ts"].values, df["price"].values, df["qty"].values):
        bar = b.add_trade(float(price), float(qty), int(ts))
        if bar:
            bars.append(bar)
    return bars


def price_at(df_ts, df_px, t_ms, side):
    """side='first': first trade at/after t_ms; 'last': last trade at/before t_ms."""
    idx = np.searchsorted(df_ts, t_ms)
    if side == "first":
        if idx >= len(df_ts):
            return None
        return float(df_px[idx])
    else:  # last <= t_ms
        if idx == 0:
            return None
        return float(df_px[idx - 1])


def make_dataset(df, bars, threshold):
    """Reconstruct aligned 5-min windows; return (X, y, ts_list)."""
    df_ts = df["ts"].values
    df_px = df["price"].values
    bar_end = np.array([b["end_ts"] for b in bars])
    t0 = (int(df_ts[0]) // (WINDOW_S * 1000) + 1) * (WINDOW_S * 1000)
    t_last = int(df_ts[-1])
    X, y, tss = [], [], []
    ws = t0
    while ws + WINDOW_S * 1000 <= t_last:
        strike = price_at(df_ts, df_px, ws, "first")
        decision_ts = ws + DECISION_OFFSET_S * 1000
        px_dec = price_at(df_ts, df_px, decision_ts, "last")
        close_px = price_at(df_ts, df_px, ws + WINDOW_S * 1000, "last")
        if None in (strike, px_dec, close_px) or strike == 0:
            ws += WINDOW_S * 1000
            continue
        # bars completed by decision time
        hi = np.searchsorted(bar_end, decision_ts, side="right")
        recent = bars[max(0, hi - VOL_WINDOW):hi]
        drift_pct = (px_dec - strike) / strike * 100.0
        feats = build_features(recent, drift_pct=drift_pct,
                               secs_to_close=MONITOR_START_S, vol_window=VOL_WINDOW)
        if feats is not None:
            X.append([feats[n] for n in FEATURE_NAMES])
            y.append(1 if close_px > strike else 0)
            tss.append(ws)
        ws += WINDOW_S * 1000
    return np.array(X, dtype=float), np.array(y, dtype=int), tss


def load_logged_windows():
    """Logged real windows: slug, ws, drift_pct, asks, outcome. From skip+trade history."""
    rows = []
    for path in ("skip_history.jsonl", "trade_history.jsonl", "trade_history_v2.jsonl"):
        for fn in glob.glob(path):
            for line in open(fn):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                won_key = "would_have_won" if "would_have_won" in d else "won"
                if d.get(won_key) is None or not d.get("slug"):
                    continue
                direction = (d.get("direction") or "").upper()
                won = d[won_key]
                up_won = (1 if won else 0) if direction == "UP" else \
                         (0 if won else 1) if direction == "DOWN" else None
                if up_won is None:
                    continue
                try:
                    ws = int(d["slug"].rsplit("-", 1)[1])
                except Exception:
                    continue
                rows.append({"slug": d["slug"], "ws": ws, "up_won": up_won,
                             "ask_up": d.get("up_price"), "ask_down": d.get("down_price")})
    # dedupe by slug
    seen, out = set(), []
    for r in rows:
        if r["slug"] in seen:
            continue
        seen.add(r["slug"])
        out.append(r)
    return out


def feats_for_window(df_ts, df_px, bars, bar_end, ws):
    strike = price_at(df_ts, df_px, ws, "first")
    decision_ts = ws + DECISION_OFFSET_S * 1000
    px_dec = price_at(df_ts, df_px, decision_ts, "last")
    if None in (strike, px_dec) or strike == 0:
        return None
    hi = np.searchsorted(bar_end, decision_ts, side="right")
    recent = bars[max(0, hi - VOL_WINDOW):hi]
    drift_pct = (px_dec - strike) / strike * 100.0
    return build_features(recent, drift_pct=drift_pct,
                          secs_to_close=MONITOR_START_S, vol_window=VOL_WINDOW)


def backtest_buylow(model, calibrator, df, bars, logged,
                    target_ask=0.45, max_ask=0.50, min_conf=0.10, fee_buffer=0.01):
    """Buy-low contrarian backtest on logged windows w/ real PM asks. One ask
    snapshot/window (no dip simulation) => conservative fills at the logged ask."""
    df_ts = df["ts"].values       # ms (normalized by the fetcher)
    df_px = df["price"].values
    bar_end = np.array([b["end_ts"] for b in bars])
    trades = []
    for r in logged:
        feats = feats_for_window(df_ts, df_px, bars, bar_end, r["ws"] * 1000)
        if feats is None:
            continue
        x = np.array([[feats[n] for n in FEATURE_NAMES]], dtype=float)
        p_up = float(model.predict_proba(x)[0, 1])
        if calibrator is not None:
            p_up = float(calibrator.predict([p_up])[0])
        side = "UP" if p_up >= 0.5 else "DOWN"
        p_side = p_up if side == "UP" else 1 - p_up
        ask = r["ask_up"] if side == "UP" else r["ask_down"]
        conf = abs(p_up - 0.5) * 2
        if ask is None or ask >= max_ask or conf < min_conf or (p_side - ask) <= fee_buffer:
            continue
        won = (r["up_won"] == 1) if side == "UP" else (r["up_won"] == 0)
        shares = 1.0 / ask
        pnl = (shares - 1.0 if won else -1.0) - fee_per_share(ask) * shares
        trades.append((pnl, won, ask, side))
    if not trades:
        print("  BACKTEST: 0 trades fired (model never beat a sub-0.50 ask).")
        return
    pnls = [t[0] for t in trades]
    n = len(pnls); wins = sum(1 for t in trades if t[1])
    mu = st.mean(pnls); sd = st.stdev(pnls) if n > 1 else 0.0
    tstat = mu / (sd / math.sqrt(n)) if sd else 0.0
    print(f"  BACKTEST buy-low (logged windows, real asks): trades={n} "
          f"win%={100*wins/n:.1f} total=${sum(pnls):+.2f} mean/trade=${mu:+.3f} "
          f"t={tstat:+.2f}")


def main():
    parquet, threshold, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    df = pd.read_parquet(parquet).sort_values("ts").reset_index(drop=True)
    print(f"loaded {len(df):,} trades; building bars @ ${threshold:,.0f} ...")
    bars = build_bars(df, threshold)
    print(f"  {len(bars):,} dollar bars")
    X, y, tss = make_dataset(df, bars, threshold)
    print(f"  reconstructed windows: {len(y):,}  up-rate={y.mean():.3f}")

    # chronological split 70/10/20 (train / calibrate / test)
    n = len(y); i_tr, i_ca = int(0.70 * n), int(0.80 * n)
    Xtr, ytr = X[:i_tr], y[:i_tr]
    Xca, yca = X[i_tr:i_ca], y[i_tr:i_ca]
    Xte, yte = X[i_ca:], y[i_ca:]
    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())

    candidates = {
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            eval_metric="logloss", n_jobs=4),
        "lightgbm": LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced", n_jobs=4),
    }
    results = {}
    for name, model in candidates.items():
        model.fit(Xtr, ytr)
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(model.predict_proba(Xca)[:, 1], yca)
        p_te = cal.predict(model.predict_proba(Xte)[:, 1])
        brier = brier_score_loss(yte, p_te)
        ll = log_loss(yte, np.clip(p_te, 1e-6, 1 - 1e-6))
        acc = accuracy_score(yte, (p_te >= 0.5).astype(int))
        print(f"  {name:9s} test Brier={brier:.4f} logloss={ll:.4f} acc={acc:.3f}")
        results[name] = (model, cal, brier)

    best = min(results, key=lambda k: results[k][2])
    model, cal, _ = results[best]
    print(f"  WINNER (lowest Brier): {best}")

    logged = load_logged_windows()
    print(f"  logged windows for backtest: {len(logged)}")
    backtest_buylow(model, cal, df, bars, logged)

    bundle = {"model": model, "calibrator": cal, "feature_names": FEATURE_NAMES,
              "threshold_usd": threshold,
              "meta": {"winner": best, "n_train_windows": int(len(y)),
                       "monitor_start_s": MONITOR_START_S, "vol_window": VOL_WINDOW}}
    joblib.dump(bundle, out)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run training (use the θ chosen in Task 7)**

Run (substitute the chosen threshold, e.g. `10000000`):
```bash
cd /home/vidura/btcpredictor/predictor && mkdir -p models && python train/train_db_model.py data/aggtrades.parquet 10000000 models/db_ptb.joblib
```
Expected: prints bar count, reconstructed-window count + up-rate, per-model Brier/log-loss/accuracy, the winner, the **buy-low backtest line** (trades / win% / total / mean / t-stat), and `saved models/db_ptb.joblib`.

- [ ] **Step 3: Decision checkpoint — read the backtest line**

This is the go/no-go evidence the spec calls for. **Report the backtest numbers to the user before any live deploy.** Interpretation: if `total` is negative or `t` ≈ 0, the model does not beat the sub-0.50 price net of fee (the expected outcome per our research) — flag it. If `total` > 0 with `t` ≳ 2 across a non-trivial trade count, it's worth the live test. Either way the artifact is saved so the live wiring (Phase C) can proceed; the live/no-live call is the user's.

- [ ] **Step 4: Smoke-test loading the saved model**

Run:
```bash
cd /home/vidura/btcpredictor/predictor && python -c "
from live_trader.db_model import DbModel
m = DbModel('models/db_ptb.joblib')
f = {'drift_pct':0.2,'secs_to_close':180,'duration':22,'ret':0.0003,
     'log_ret':0.0003,'volatility':15.0,'mean_price':104000,'rvol':0.0004}
print('P(up)=', round(m.predict_p_up(f), 4), '| winner:', m.meta.get('winner'))
"
```
Expected: prints a probability in [0,1] and the winning model name.

- [ ] **Step 5: Commit**

```bash
cd /home/vidura/btcpredictor/predictor
echo "models/" >> .gitignore
git add train/train_db_model.py .gitignore
git commit -m "feat(dbmodel): training + isotonic calibration + buy-low backtest"
```

---

## Phase C — Live integration

### Task 9: Config fields

**Files:**
- Modify: `live_trader/config.py` (add fields to `BotConfig` dataclass + loader in `load_config`)

- [ ] **Step 1: Add the dataclass fields**

In `live_trader/config.py`, after the `fav90_stop_bid: float` field (line ~167), add:

```python
    # Dollar-bar PTB contrarian mode — activated by --dbmodel in run_live_bot.py.
    # Predicts P(close>PTB) from a calibrated XGB/LGBM model on dollar-bar
    # microstructure + distance-to-strike, then buys the predicted side only when
    # its ask is below dbmodel_max_ask, working the order down toward target_ask.
    # No LSTM/run_baseline in this path. See specs/2026-06-04-dbmodel-...
    dbmodel_mode: bool
    dbmodel_model_path: str
    dbmodel_threshold_usd: float    # 0 = use the threshold saved in the model bundle
    dbmodel_monitor_start_s: float  # start polling this many s before close
    dbmodel_target_ask: float       # aspirational fill price
    dbmodel_max_ask: float          # hard entry ceiling
    dbmodel_deadline_s: float       # take best <= ceiling once s2c <= this
    dbmodel_poll_s: float           # poll cadence (s)
    dbmodel_min_conf: float         # min |P-0.5|*2
    dbmodel_fee_buffer: float       # min edge (P_side - ask) over the fee
    dbmodel_daily_stop: float       # UTC-day realized-PnL kill-switch (USD, negative)
```

- [ ] **Step 2: Add the loader block**

In `load_config(...)`, alongside the other `BotConfig(...)` kwargs (near the fav90 loaders), add:

```python
        dbmodel_mode=os.environ.get("BOT_DBMODEL_MODE", "false").lower() in ("1", "true", "yes"),
        dbmodel_model_path=os.environ.get("BOT_DB_MODEL_PATH", "models/db_ptb.joblib"),
        dbmodel_threshold_usd=float(os.environ.get("BOT_DB_THRESHOLD_USD", "0")),
        dbmodel_monitor_start_s=float(os.environ.get("BOT_DB_MONITOR_START_S", "180")),
        dbmodel_target_ask=float(os.environ.get("BOT_DB_TARGET_ASK", "0.45")),
        dbmodel_max_ask=float(os.environ.get("BOT_DB_MAX_ASK", "0.50")),
        dbmodel_deadline_s=float(os.environ.get("BOT_DB_DEADLINE_S", "20")),
        dbmodel_poll_s=float(os.environ.get("BOT_DB_POLL_S", "5")),
        dbmodel_min_conf=float(os.environ.get("BOT_DB_MIN_CONF", "0.10")),
        dbmodel_fee_buffer=float(os.environ.get("BOT_DB_FEE_BUFFER", "0.01")),
        dbmodel_daily_stop=float(os.environ.get("BOT_DB_DAILY_STOP", "-10.0")),
```

- [ ] **Step 3: Verify the config loads**

Run: `cd /home/vidura/btcpredictor/predictor && python -c "from live_trader.config import load_config; c=load_config('.env'); print('dbmodel_mode',c.dbmodel_mode,'max_ask',c.dbmodel_max_ask,'target',c.dbmodel_target_ask)"`
Expected: `dbmodel_mode False max_ask 0.5 target 0.45`

- [ ] **Step 4: Commit**

```bash
cd /home/vidura/btcpredictor/predictor
git add live_trader/config.py
git commit -m "feat(dbmodel): config fields + env loaders"
```

---

### Task 10: Trade executor in bot.py

**Files:**
- Modify: `live_trader/bot.py` (add a `DBMODEL_LOG_PATH` const, an `import`, the `_execute_dbmodel_trade` method, and a dispatch branch in `on_prediction`)

The executor mirrors fav90's structure: poll loop → buy_market → hold to resolution → `_record`/redeem. It reads bars+price from objects the runner attaches to the bot (`self.db_model`, `self.db_client`, `self.db_strikes`).

- [ ] **Step 1: Add the import and log-path constant**

Near the top of `live_trader/bot.py`, after `from .fav90 import fav90_decision` (line 27), add:

```python
from .db_features import build_features, FEATURE_NAMES  # noqa: F401
from .db_decision import db_decision
```

After the `FAV90_TRAJ_LOG_PATH` definition (line ~79), add:

```python
# DBMODEL LOG — one JSONL record per dbmodel poll (fired or not): the model's
# P(up), chosen side, live ask, edge, and gate flags. Mirrors fav90_log.jsonl.
DBMODEL_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dbmodel_log.jsonl",
)
```

- [ ] **Step 2: Add the dispatch branch in `on_prediction`**

In `on_prediction`, mirror the fav90 dispatch (after the daily-reset/hard-stop checks, before the model gate stack). Insert immediately after the `if self.cfg.fav90_mode:` block (ends ~line 345):

```python
        # DBMODEL MODE: bypass the model gate stack. Dispatch to the dollar-bar
        # contrarian executor (predicts P(close>PTB), buys the predicted side
        # cheap). The runner attaches self.db_model / self.db_client / self.db_strikes.
        if self.cfg.dbmodel_mode:
            if self.cfg.dbmodel_daily_stop < 0 and self._daily_pnl <= self.cfg.dbmodel_daily_stop:
                print(f"  [BOT {_fmt_ts()}] DBMODEL kill-switch: daily_pnl="
                      f"${self._daily_pnl:+.2f} <= ${self.cfg.dbmodel_daily_stop:.2f} — skip")
                return
            with self._lock:
                if self._active >= self.cfg.max_concurrent_trades:
                    print(f"  [BOT {_fmt_ts()}] DBMODEL skip: {self._active} positions open")
                    return
                self._active += 1
            threading.Thread(target=self._execute_dbmodel_trade,
                             args=(window,), daemon=True).start()
            return
```

- [ ] **Step 3: Add the `_execute_dbmodel_trade` method**

Add this method to the bot class, right before `_fav90_traj_write` (line ~1108):

```python
    def _execute_dbmodel_trade(self, window):
        """Dollar-bar PTB contrarian executor. Computes the model prediction once
        at monitor-start, then works the order: poll the predicted side's ask and
        buy when it dips to target_ask, or the best price under max_ask by the
        deadline. Holds to resolution. Logs every poll to DBMODEL_LOG_PATH."""
        slug = window["slug"]
        end_ts = window["end_ts"]
        try:
            market = self.client.resolve_market(slug)
            up_token = market["up_token"]
            down_token = market["down_token"]
            condition_id = market.get("condition_id")

            strike = self.db_strikes.get(slug)
            price_now = self.db_client.last_price
            if strike is None or price_now is None:
                print(f"  [BOT {_fmt_ts()}] DBMODEL skip {slug}: no strike/price "
                      f"(strike={strike} price={price_now})")
                return
            drift_pct = (price_now - strike) / strike * 100.0
            bars = self.db_client.bars.snapshot()
            feats = build_features(bars, drift_pct=drift_pct,
                                   secs_to_close=self.cfg.dbmodel_monitor_start_s,
                                   vol_window=self.db_model.meta.get("vol_window", 10))
            if feats is None:
                print(f"  [BOT {_fmt_ts()}] DBMODEL skip {slug}: too few bars "
                      f"({len(bars)})")
                return
            p_up = self.db_model.predict_p_up(feats)

            poll_s = self.cfg.dbmodel_poll_s
            deadline = end_ts - self.cfg.dbmodel_deadline_s
            chosen_side = None
            our_ask = None
            poll_count = 0
            while time.time() < deadline:
                poll_count += 1
                s2c = end_ts - time.time()
                try:
                    top_ask_up = self.client.get_top_ask(up_token)
                    top_ask_down = self.client.get_top_ask(down_token)
                except Exception as e:
                    print(f"  [BOT {_fmt_ts()}] DBMODEL poll {poll_count}: ask fetch failed ({e})")
                    time.sleep(poll_s)
                    continue
                d = db_decision(
                    p_up=p_up, top_ask_up=top_ask_up, top_ask_down=top_ask_down,
                    target_ask=self.cfg.dbmodel_target_ask, max_ask=self.cfg.dbmodel_max_ask,
                    min_conf=self.cfg.dbmodel_min_conf, fee_buffer=self.cfg.dbmodel_fee_buffer,
                )
                ask_s = "None" if d["ask"] is None else f"{d['ask']:.3f}"
                edge_s = "None" if d["edge"] is None else f"{d['edge']:+.3f}"
                print(f"  [BOT {_fmt_ts()}] DBMODEL poll #{poll_count}: s2c={s2c:.0f}s "
                      f"P(up)={p_up:.3f} side={d['side']} ask={ask_s} edge={edge_s} "
                      f"[elig={'Y' if d['eligible'] else 'N'} buy={'Y' if d['buy_now'] else 'N'}]")
                try:
                    with open(DBMODEL_LOG_PATH, "a") as f:
                        f.write(json.dumps({
                            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                            "slug": slug, "secs_to_close": round(s2c, 1),
                            "p_up": round(p_up, 4), "side": d["side"],
                            "ask": d["ask"], "edge": (round(d["edge"], 4) if d["edge"] is not None else None),
                            "drift_pct": round(drift_pct, 4), "strike": strike,
                            "eligible": d["eligible"], "buy_now": d["buy_now"],
                        }) + "\n")
                except Exception as _e:
                    print(f"  [BOT {_fmt_ts()}] dbmodel log error: {_e}")
                # Buy at target dip, or take best under ceiling once past the deadline window.
                take_now = d["buy_now"] or (d["eligible"] and s2c <= self.cfg.dbmodel_deadline_s + poll_s)
                if take_now:
                    chosen_side = d["side"]
                    our_ask = d["ask"]
                    break
                time.sleep(poll_s)

            if chosen_side is None:
                print(f"  [BOT {_fmt_ts()}] SKIP (dbmodel-no-cheap-entry): {poll_count} polls")
                self._record_skip(reason="dbmodel_no_cheap_entry",
                                  slug=slug, end_ts=end_ts, direction=d["side"],
                                  confidence=p_up * 100, ptb=strike, live_price=price_now,
                                  drift_pct=drift_pct, final_up=p_up,
                                  top_ask_up=top_ask_up, top_ask_down=top_ask_down,
                                  signals=None, details=f"polls={poll_count}")
                return

            token_id = up_token if chosen_side == "UP" else down_token
            stake = self.cfg.stake_usdc
            print(f"  [BOT {_fmt_ts()}] DBMODEL BUY {chosen_side} {slug} "
                  f"ask={our_ask:.3f} P(up)={p_up:.3f} stake=${stake:.2f}"
                  + ("  (DRY RUN)" if self.cfg.dry_run else ""))
            order_resp = self.client.buy_market(token_id, stake)
            if isinstance(order_resp, dict) and "skipped" in order_resp:
                print(f"  [BOT {_fmt_ts()}] skipped: {order_resp['skipped']}")
                self._record_skip(reason="book_vanished", slug=slug, end_ts=end_ts,
                                  direction=chosen_side, confidence=p_up * 100,
                                  ptb=strike, live_price=price_now, drift_pct=drift_pct,
                                  final_up=p_up, signals=None,
                                  details=str(order_resp.get("skipped")))
                return
            fill = _parse_fill(order_resp, stake)
            if fill:
                shares, fill_px, actual_filled_usdc = fill
                print(f"  [BOT {_fmt_ts()}] filled: {shares:.2f} shares @ ~${fill_px:.3f}")
            else:
                shares, fill_px, actual_filled_usdc = None, None, None
                print(f"  [BOT {_fmt_ts()}] order resp: {order_resp}")

            extras = dict(ptb=strike, live_price=price_now, drift_pct=drift_pct,
                          final_up=p_up, signals=None, shares=shares, fill_px=fill_px,
                          stake=stake, actual_filled_usdc=actual_filled_usdc)

            resolution = self.client.wait_for_resolution(slug, end_ts + 600)
            if resolution is None:
                print(f"  [BOT {_fmt_ts()}] {slug}: resolution timeout")
                self._record(slug, chosen_side, p_up * 100, None, 0.0, **extras)
                return
            won = (chosen_side == "UP" and resolution["up_won"]) or \
                  (chosen_side == "DOWN" and not resolution["up_won"])
            cost_basis = actual_filled_usdc if actual_filled_usdc is not None else stake
            if won and shares is not None:
                pnl = shares - cost_basis
            elif won:
                pnl = cost_basis
            else:
                pnl = -cost_basis
            self._record(slug, chosen_side, p_up * 100, won, pnl, **extras)
            print(f"  [BOT {_fmt_ts()}] {slug}: {'WIN ' if won else 'LOSS'} pnl=${pnl:+.2f}")
            if won and condition_id:
                tx = self.client.redeem_position(condition_id)
                print(f"  [BOT {_fmt_ts()}] redeemed -> {tx}" if tx else
                      f"  [BOT {_fmt_ts()}] redeem failed or no shares")
            self._print_summary()
        except PolymarketError as e:
            print(f"  [BOT {_fmt_ts()}] polymarket error: {e}")
        except Exception as e:
            print(f"  [BOT {_fmt_ts()}] dbmodel trade error: {type(e).__name__}: {e}")
        finally:
            with self._lock:
                self._active = max(0, self._active - 1)
```

- [ ] **Step 3b: Verify bot.py still imports**

Run: `cd /home/vidura/btcpredictor/predictor && python -c "import live_trader.bot; print('bot import OK')"`
Expected: `bot import OK`

- [ ] **Step 4: Commit**

```bash
cd /home/vidura/btcpredictor/predictor
git add live_trader/bot.py
git commit -m "feat(dbmodel): work-the-order contrarian executor + on_prediction dispatch"
```

---

### Task 11: Runner, profile, and CLI flag

**Files:**
- Modify: `run_live_bot.py` (add `dbmodel` profile, `--dbmodel` arg, `_run_dbmodel_mode`, and an early-return in `main()` before `import run_baseline`)

- [ ] **Step 1: Add the profile**

In the `PROFILES` dict in `run_live_bot.py`, after the `"fav90"` entry, add:

```python
    "dbmodel": {
        "name": "DBMODEL — DOLLAR-BAR PTB CONTRARIAN (LIVE $1)",
        "tagline": "model predicts P(close>PTB); buy predicted side when ask<0.50",
        "env": {
            "BOT_DBMODEL_MODE": "true",
            "BOT_DB_MONITOR_START_S": "180",
            "BOT_DB_TARGET_ASK": "0.45",
            "BOT_DB_MAX_ASK": "0.50",
            "BOT_DB_DEADLINE_S": "20",
            "BOT_DB_POLL_S": "5",
            "BOT_DB_MIN_CONF": "0.10",
            "BOT_DB_FEE_BUFFER": "0.01",
            "BOT_DB_DAILY_STOP": "-10.0",
            "BOT_STAKE_USDC": "1.0",
            "BOT_ENABLE_CASH_OUT": "false",
        },
    },
```

- [ ] **Step 2: Add the CLI flag**

Where `--fav90` is defined (line ~165), add below it:

```python
    parser.add_argument(
        "--dbmodel", dest="profile_dbmodel", action="store_true",
        help=f"Profile DBMODEL: {PROFILES['dbmodel']['tagline']}")
```

- [ ] **Step 3: Add the runner function**

After `_run_fav90_mode` (line ~325), add:

```python
def _run_dbmodel_mode(profile, cfg, bot):
    """Dollar-bar PTB contrarian runner. Starts a Binance aggTrade stream, loads
    the calibrated model, captures each window's strike at open, and dispatches the
    contrarian executor once per window. Never imports run_baseline (no TF/LSTM)."""
    import time
    import datetime
    import threading
    from live_trader.dollar_bars import BinanceAggTradeClient
    from live_trader.db_model import DbModel

    print("=" * 70)
    print("DBMODEL MODE — dollar-bar PTB contrarian on EVERY 5m window (LIVE $1)")
    print("=" * 70)
    model = DbModel(cfg.dbmodel_model_path)
    threshold = cfg.dbmodel_threshold_usd or model.threshold_usd
    print(f"  Model:           {cfg.dbmodel_model_path} (winner={model.meta.get('winner')})")
    print(f"  Dollar-bar θ:    ${threshold:,.0f}")
    print(f"  Rule:            predict P(close>PTB); BUY predicted side when ask in "
          f"(0, {cfg.dbmodel_max_ask:.2f}); work down to {cfg.dbmodel_target_ask:.2f}")
    print(f"  Monitor:         from {cfg.dbmodel_monitor_start_s:.0f}s before close, "
          f"poll {cfg.dbmodel_poll_s:.0f}s, deadline {cfg.dbmodel_deadline_s:.0f}s")
    print(f"  Stake:           ${cfg.stake_usdc:.2f} · hold to resolution")
    print(f"  Kill-switch:     daily realized PnL <= ${cfg.dbmodel_daily_stop:.2f}")
    print(f"  Mode:            {'DRY-RUN' if cfg.dry_run else '*** LIVE — REAL $1 orders ***'}")
    print("=" * 70, flush=True)

    # attach shared state for the executor
    bot.db_model = model
    bot.db_client = BinanceAggTradeClient(threshold_usd=threshold).start()
    bot.db_strikes = {}

    if cfg.reconcile_on_start and not cfg.dry_run:
        try:
            r = bot.client.sweep_orphan_winners()
            print(f"  Reconcile: redeemed {r.get('redeemed', 0)} orphan winner(s)")
        except Exception as e:
            print(f"  Reconcile error ({type(e).__name__}: {e}) — continuing")

    # warm up: wait for the first bars/price before trading
    print("  warming up Binance feed (waiting for first price)...", flush=True)
    for _ in range(30):
        if bot.db_client.last_price is not None:
            break
        time.sleep(1)
    print(f"  feed up. last_price={bot.db_client.last_price}. Ctrl-C to stop.\n", flush=True)

    last_slug = None
    cur_ws = None
    try:
        while True:
            now = time.time()
            ws = (int(now) // 300) * 300
            slug = f"btc-updown-5m-{ws}"
            end_ts = ws + 300
            s2c = end_ts - now
            # capture strike at window open (first tick after rollover)
            if ws != cur_ws:
                cur_ws = ws
                if bot.db_client.last_price is not None:
                    bot.db_strikes[slug] = bot.db_client.last_price
                    print(f"  [DBMODEL {datetime.datetime.now():%H:%M:%S}] window {slug} "
                          f"strike=${bot.db_client.last_price:,.2f}", flush=True)
            # dispatch once per window when it enters the monitor zone
            if slug != last_slug and (cfg.dbmodel_monitor_start_s - 30) < s2c <= cfg.dbmodel_monitor_start_s:
                if slug not in bot.db_strikes and bot.db_client.last_price is not None:
                    bot.db_strikes[slug] = bot.db_client.last_price  # fallback if we missed open
                last_slug = slug
                window = {"slug": slug, "end_ts": end_ts}
                with bot._lock:
                    bot._active += 1
                threading.Thread(target=bot._execute_dbmodel_trade,
                                 args=(window,), daemon=True).start()
                print(f"  [DBMODEL {datetime.datetime.now():%H:%M:%S}] window {slug} "
                      f"dispatched (closes in {s2c:.0f}s)", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n  dbmodel stopped.")
        bot.db_client.stop()
```

- [ ] **Step 4: Wire profile selection + early return in `main()`**

In `main()`, where `--fav90` is handled (the `if args.profile_fav90:` near line 394 and the early-return near line 427), add parallel handling. After the fav90 profile-apply block add:

```python
    if args.profile_dbmodel:
        profile = _apply_profile("dbmodel")
```

And before `import run_baseline` (line ~431), after the fav90 early-return block, add:

```python
    # DBMODEL MODE: dollar-bar contrarian runner on its own per-window clock.
    # Early return BEFORE importing run_baseline so we never load TF/LSTM.
    if args.profile_dbmodel:
        _run_dbmodel_mode(profile, cfg, bot)
        return
```

- [ ] **Step 5: Verify the CLI parses and the runner imports (no network)**

Run: `cd /home/vidura/btcpredictor/predictor && python -c "import ast; ast.parse(open('run_live_bot.py').read()); print('run_live_bot.py parses OK')"`
Expected: `run_live_bot.py parses OK`

Run: `cd /home/vidura/btcpredictor/predictor && python run_live_bot.py --help 2>&1 | grep -A1 dbmodel`
Expected: shows the `--dbmodel` flag and its help line.

- [ ] **Step 6: Commit**

```bash
cd /home/vidura/btcpredictor/predictor
git add run_live_bot.py
git commit -m "feat(dbmodel): runner, profile, and --dbmodel flag (no run_baseline/LSTM)"
```

---

### Task 12: Full local dry-run

**Files:** none (verification)

- [ ] **Step 1: Run the whole pipeline in dry-run for ~6 minutes (one full window)**

Run:
```bash
cd /home/vidura/btcpredictor/predictor && BOT_DRY_RUN=true timeout 380 python run_live_bot.py --dbmodel 2>&1 | tee /tmp/dbmodel_dryrun.log
```
Expected in the log: the DBMODEL banner; `feed up. last_price=...`; at least one `window ... strike=$...` line; a `window ... dispatched` line; then `DBMODEL poll #N` lines showing `P(up)=`, `side=`, `ask=`, `edge=`, gate flags; and either a `DBMODEL BUY ... (DRY RUN)` or `SKIP (dbmodel-no-cheap-entry)`. No `tensorflow`/`run_baseline`/`lstm` text anywhere.

- [ ] **Step 2: Confirm no LSTM/TF loaded**

Run: `grep -ci "tensorflow\|run_baseline\|lstm" /tmp/dbmodel_dryrun.log`
Expected: `0`

- [ ] **Step 3: Confirm the poll log was written**

Run: `tail -2 /home/vidura/btcpredictor/predictor/dbmodel_log.jsonl`
Expected: two JSON lines with `p_up`, `side`, `ask`, `edge`, `eligible`, `buy_now`.

- [ ] **Step 4: Run the full test suite**

Run: `cd /home/vidura/btcpredictor/predictor && python -m pytest tests/test_dollar_bars.py tests/test_db_features.py tests/test_db_decision.py -v`
Expected: all pass.

---

## Phase D — Server deploy (eu-west-1)

> SSH key `~/.ssh/aws_arb.pem`; host `<server-host>`; user `ubuntu`; server path `/home/ubuntu/btcpredictor/predictor/`; venv `/mnt/data/s5venv`. Server is NOT git-tracked — scp files; back up first. Never scp `.env`.

### Task 13: Server dependencies + code + model

**Files:** none (deploy)

- [ ] **Step 1: Confirm/install ML deps in the server venv**

Run:
```bash
ssh -i ~/.ssh/aws_arb.pem ubuntu@<server-host> \
  "/mnt/data/s5venv/bin/python -c 'import xgboost,lightgbm,sklearn,joblib;print(\"deps OK\")' 2>&1 || /mnt/data/s5venv/bin/pip install xgboost lightgbm scikit-learn joblib"
```
Expected: `deps OK`, or a successful install followed by re-running the import to confirm.

- [ ] **Step 2: Add deps to requirements-live.txt and commit (local)**

```bash
cd /home/vidura/btcpredictor/predictor
printf "xgboost\nlightgbm\nscikit-learn\njoblib\n" >> requirements-live.txt
git add requirements-live.txt
git commit -m "chore(dbmodel): add xgboost/lightgbm/sklearn/joblib to live reqs"
```

- [ ] **Step 3: Back up the server's current bot files**

```bash
ssh -i ~/.ssh/aws_arb.pem ubuntu@<server-host> \
  "cd /home/ubuntu/btcpredictor/predictor && mkdir -p _bak/dbmodel_$(date +%Y%m%d) && cp -v live_trader/bot.py live_trader/config.py run_live_bot.py _bak/dbmodel_$(date +%Y%m%d)/ 2>/dev/null; echo backed up"
```
Expected: `backed up`.

- [ ] **Step 4: Copy the new/changed files + the model (dos2unix on text)**

```bash
cd /home/vidura/btcpredictor/predictor
H=ubuntu@<server-host>
K=~/.ssh/aws_arb.pem
scp -i $K live_trader/dollar_bars.py live_trader/db_features.py live_trader/db_decision.py \
      live_trader/db_model.py live_trader/bot.py live_trader/config.py \
      $H:/home/ubuntu/btcpredictor/predictor/live_trader/
scp -i $K run_live_bot.py $H:/home/ubuntu/btcpredictor/predictor/
ssh -i $K $H "mkdir -p /home/ubuntu/btcpredictor/predictor/models"
scp -i $K models/db_ptb.joblib $H:/home/ubuntu/btcpredictor/predictor/models/
ssh -i $K $H "cd /home/ubuntu/btcpredictor/predictor && dos2unix live_trader/dollar_bars.py live_trader/db_features.py live_trader/db_decision.py live_trader/db_model.py live_trader/bot.py live_trader/config.py run_live_bot.py 2>/dev/null; echo synced"
```
Expected: scp progress lines and `synced`.

- [ ] **Step 5: Verify imports + model load on the server**

```bash
ssh -i ~/.ssh/aws_arb.pem ubuntu@<server-host> \
  "cd /home/ubuntu/btcpredictor/predictor && /mnt/data/s5venv/bin/python -c 'import live_trader.bot; from live_trader.db_model import DbModel; m=DbModel(\"models/db_ptb.joblib\"); print(\"server OK winner\", m.meta.get(\"winner\"))'"
```
Expected: `server OK winner <name>`.

---

### Task 14: Launch on server + verify live

**Files:** none (deploy)

- [ ] **Step 1: Confirm no old bot is running, then launch dbmodel in tmux**

```bash
ssh -i ~/.ssh/aws_arb.pem ubuntu@<server-host> \
  "pkill -f 'run_live_bo[t].py' 2>/dev/null; tmux kill-session -t dbmodel 2>/dev/null; \
   cd /home/ubuntu/btcpredictor/predictor && \
   tmux new-session -d -s dbmodel 'BOT_DB_MODEL_PATH=models/db_ptb.joblib /mnt/data/s5venv/bin/python -u run_live_bot.py --dbmodel 2>&1 | tee -a dbmodel_live.log'; \
   sleep 20 && tail -25 dbmodel_live.log"
```
Expected: the DBMODEL banner, `feed up. last_price=...`, and the loop starting. (Confirm `*** LIVE — REAL $1 orders ***` shows since `.env` has no `BOT_DRY_RUN`.)

- [ ] **Step 2: Watch one full window resolve (~6 min)**

```bash
ssh -i ~/.ssh/aws_arb.pem ubuntu@<server-host> \
  "cd /home/ubuntu/btcpredictor/predictor && sleep 360 && tail -40 dbmodel_live.log && echo '--- poll log ---' && tail -5 dbmodel_log.jsonl"
```
Expected: at least one `strike=$...`, one `dispatched`, several `DBMODEL poll #N` lines, and either a `BUY`/`WIN`/`LOSS` or `SKIP (dbmodel-no-cheap-entry)`. Confirm no errors/tracebacks.

- [ ] **Step 3: Report status to the user**

Summarize: feed health (last_price updating), windows dispatched, trades fired vs skipped, any fills + outcomes, and a reminder that fills are rare by design (only when the model beats a sub-0.50 ask). Reference the Task 8 backtest numbers as the prior expectation.

---

## Self-Review

**1. Spec coverage:**
- §1 goal (calibrated XGB/LGBM, P(close>PTB), buy <0.50, no LSTM) → Tasks 8, 4, 10, 11. ✓
- §3.1 target / §3.2 features (`drift_pct`, `secs_to_close`, Wang 5 + rvol) → Task 3 `FEATURE_NAMES`, Task 8 dataset. ✓
- §3.3 dollar bars + θ calibration → Tasks 1, 7. ✓
- §3.4 training: reconstructed Binance windows, chronological split, imbalance, isotonic calibration, validated against logged windows → Task 8. ✓
- §4 live decision + work-the-order buy-low → Tasks 4, 10. ✓
- §5 component files → all created. ✓
- §6 data flow (WS → builder → buffer → features → model → decision → buy) → Tasks 2, 10, 11. ✓
- §7 offline backtest gate → Task 8 `backtest_buylow` + Step 3 checkpoint. ✓
- §8 config env → Task 9. ✓
- §9 risks: PTB source (strike captured at open, validated vs logged ptb) → Task 11/8; deps confirmed → Task 13; LSTM removed (no run_baseline import) → Task 11 + Task 12 Step 2 grep. ✓

**2. Placeholder scan:** No "TBD"/"handle errors"/"similar to". θ is concretely chosen via Task 7's output and passed as an explicit arg. ✓

**3. Type consistency:** `FEATURE_NAMES` order is identical in `db_features.py` (Task 3), training (Task 8), and `DbModel` (Task 5). Bar dict keys identical across Tasks 1/3/8. `db_decision` return keys (`side, p_side, conf, ask, edge, eligible, buy_now`) used consistently in Tasks 4 and 10. `DbModel.predict_p_up` signature matches its callers (Tasks 8 smoke, 10). ✓
