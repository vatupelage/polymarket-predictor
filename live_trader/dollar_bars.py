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


import time
import websocket  # websocket-client, already a dependency (see arb_ws.py)


class BinanceAggTradeClient:
    """Streams a symbol's aggTrades, builds dollar bars, exposes last_price + a
    thread-safe BarBuffer. Auto-reconnects. Start with .start() (spawns a daemon
    thread); read .last_price and .bars.snapshot() from other threads.

    `symbol` is the Binance pair (default BTCUSDT for back-compat); the WS stream
    name is the lowercased pair + @aggTrade."""

    def __init__(self, threshold_usd: float, buffer_len: int = 500,
                 symbol: str = "BTCUSDT"):
        self.symbol = symbol.upper()
        self.URL = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@aggTrade"
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
