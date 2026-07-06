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
