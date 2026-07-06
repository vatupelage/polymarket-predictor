import numpy as np
import pandas as pd

from edgelab.scalp import scalp_window, scalp_trades


def _win(slug, ts, ofi, ask, bid, side="up"):
    return pd.DataFrame({
        "slug": slug, "side": side, "ts": ts,
        "ofi_inc": ofi, "best_ask": ask, "best_bid": bid,
    })


def test_entry_at_ask_exit_at_bid_after_lag():
    # ts 0.0..0.5s, lag 200ms. Triggers at idx0 (ofi5) and idx3 (ofi9).
    g = _win("w", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
             [5, 0, 0, 9, 0, 0],
             [0.52, .99, .99, 0.55, .99, .99],
             [0.50, 0.51, 0.53, 0.54, 0.55, 0.58])
    tr = scalp_window(g, theta=1.0, lag_ms=200)
    assert len(tr) == 2
    # trade 0: enter ask 0.52 @t=0, exit first event >= 0.2 -> idx2 bid 0.53
    assert tr[0]["entry_ask"] == 0.52 and tr[0]["exit_bid"] == 0.53
    assert abs(tr[0]["pnl"] - 0.01) < 1e-9
    # trade 1: enter ask 0.55 @t=0.3, exit first event >= 0.5 -> idx5 bid 0.58
    assert tr[1]["entry_ask"] == 0.55 and tr[1]["exit_bid"] == 0.58
    assert abs(tr[1]["pnl"] - 0.03) < 1e-9


def test_non_overlapping_skips_triggers_inside_open_trade():
    # idx1 also exceeds theta but trade0 occupies idx0..idx2 -> idx1 must be skipped.
    g = _win("w", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
             [5, 8, 0, 9, 0, 0],
             [0.52, 0.60, .99, 0.55, .99, .99],
             [0.50, 0.51, 0.53, 0.54, 0.55, 0.58])
    tr = scalp_window(g, theta=1.0, lag_ms=200)
    assert len(tr) == 2
    assert [t["entry_ask"] for t in tr] == [0.52, 0.55]   # not 0.60


def test_no_lookahead_exit_strictly_after_entry_plus_lag():
    g = _win("w", [0.0, 0.05, 0.1, 0.25, 0.4],
             [9, 0, 0, 0, 0],
             [0.50, .9, .9, .9, .9],
             [0.50, 0.51, 0.52, 0.55, 0.60])
    tr = scalp_window(g, theta=1.0, lag_ms=200)
    assert len(tr) == 1
    # exit must be first event with ts >= 0.0 + 0.2 -> idx3 (t=0.25), not idx2 (0.1)
    assert tr[0]["exit_ts"] >= 0.0 + 0.2 - 1e-12
    assert tr[0]["exit_bid"] == 0.55


def test_trigger_without_exit_in_window_is_dropped():
    # last-event trigger has no event >= t+lag -> no trade
    g = _win("w", [0.0, 0.1, 0.45],
             [0, 0, 9],
             [.9, .9, 0.50],
             [.9, .9, 0.50])
    tr = scalp_window(g, theta=1.0, lag_ms=200)
    assert tr == []


def test_threshold_is_long_only_positive_ofi():
    # negative and sub-theta OFI never trigger; only idx2 (ofi=5) does
    g = _win("w", [0.0, 0.1, 0.2, 0.4],
             [-9, 0.5, 5, 0],
             [.9, .9, 0.50, .9],
             [.9, .9, .9, 0.53])
    tr = scalp_window(g, theta=1.0, lag_ms=200)
    assert len(tr) == 1
    assert tr[0]["entry_ask"] == 0.50 and tr[0]["exit_bid"] == 0.53


def test_scalp_trades_pools_windows_and_filters_side():
    up = _win("a", [0.0, 0.3], [9, 0], [0.50, .9], [.9, 0.55], side="up")
    dn = _win("a", [0.0, 0.3], [9, 0], [0.40, .9], [.9, 0.45], side="down")
    df = pd.concat([up, dn], ignore_index=True)
    tr_up = scalp_trades(df, theta=1.0, lag_ms=200, side="up")
    assert len(tr_up) == 1 and tr_up[0]["entry_ask"] == 0.50
    tr_dn = scalp_trades(df, theta=1.0, lag_ms=200, side="down")
    assert len(tr_dn) == 1 and tr_dn[0]["entry_ask"] == 0.40
