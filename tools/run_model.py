# tools/run_model.py
"""Run the trained dollar-bar PTB model by itself (no bot, no PM, no LSTM).

Two modes:

  REPLAY (instant, uses your historical parquet):
      python3 tools/run_model.py replay [N]
    Reconstructs the last N 5-min windows from data/aggtrades.parquet, and for
    each prints the model's calibrated P(up), its pick, and whether it was right.

  LIVE (connects to Binance aggTrade feed):
      python3 tools/run_model.py live
    Builds dollar bars from the live feed; at 60s into each 5-min window
    (decision time) prints the model's P(up) and pick. Needs a few minutes of
    warm-up so enough bars exist for the rolling-vol feature.
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live_trader.db_model import DbModel
from live_trader.db_features import build_features, FEATURE_NAMES
from live_trader.dollar_bars import DollarBarBuilder

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "db_ptb.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "aggtrades.parquet")

WINDOW_S = 300
MONITOR_START_S = 240            # decision @ 60s into window (matches PREDICT_AT=60)
DECISION_OFFSET_S = WINDOW_S - MONITOR_START_S
VOL_WINDOW = 10


def _price_at(df_ts, df_px, t_ms, side):
    idx = np.searchsorted(df_ts, t_ms)
    if side == "first":
        return float(df_px[idx]) if idx < len(df_ts) else None
    return float(df_px[idx - 1]) if idx > 0 else None


def replay(n_windows):
    import pandas as pd
    model = DbModel(MODEL_PATH)
    print(f"model: {model.meta.get('winner','?')}  threshold=${model.threshold_usd:,.0f}")
    df = pd.read_parquet(DATA_PATH).sort_values("ts").reset_index(drop=True)
    df_ts, df_px, df_qty = df["ts"].values, df["price"].values, df["qty"].values

    # build all bars once
    b = DollarBarBuilder(model.threshold_usd)
    bars = []
    for ts, px, q in zip(df_ts, df_px, df_qty):
        bar = b.add_trade(float(px), float(q), int(ts))
        if bar:
            bars.append(bar)
    bar_end = np.array([bb["end_ts"] for bb in bars])

    t_last = int(df_ts[-1])
    starts = []
    ws = (t_last // (WINDOW_S * 1000)) * (WINDOW_S * 1000) - WINDOW_S * 1000
    while len(starts) < n_windows and ws > int(df_ts[0]):
        starts.append(ws)
        ws -= WINDOW_S * 1000
    starts.reverse()

    print(f"{'window_open(UTC)':>20}  {'P(up)':>6}  pick  ask-side?  {'result':>8}")
    hits = tot = 0
    for ws in starts:
        strike = _price_at(df_ts, df_px, ws, "first")
        dec_ts = ws + DECISION_OFFSET_S * 1000
        px_dec = _price_at(df_ts, df_px, dec_ts, "last")
        close_px = _price_at(df_ts, df_px, ws + WINDOW_S * 1000, "last")
        if None in (strike, px_dec, close_px) or strike == 0:
            continue
        hi = np.searchsorted(bar_end, dec_ts, side="right")
        recent = bars[max(0, hi - VOL_WINDOW):hi]
        drift = (px_dec - strike) / strike * 100.0
        feats = build_features(recent, drift_pct=drift, secs_to_close=MONITOR_START_S,
                               vol_window=VOL_WINDOW)
        if feats is None:
            continue
        p_up = model.predict_p_up(feats)
        pick = "UP" if p_up >= 0.5 else "DOWN"
        up_won = close_px > strike
        correct = (pick == "UP") == up_won
        hits += correct; tot += 1
        t = time.strftime("%m-%d %H:%M:%S", time.gmtime(ws / 1000))
        print(f"{t:>20}  {p_up:6.3f}  {pick:>4}  drift={drift:+.3f}%  "
              f"{'WIN ' if correct else 'loss':>8}")
    if tot:
        print(f"\nreplayed {tot} windows  model accuracy={100*hits/tot:.1f}%")


def live():
    from live_trader.dollar_bars import BinanceAggTradeClient
    model = DbModel(MODEL_PATH)
    print(f"model: {model.meta.get('winner','?')}  threshold=${model.threshold_usd:,.0f}")
    print("connecting to Binance aggTrade feed; warming up bars...")
    feed = BinanceAggTradeClient(model.threshold_usd, buffer_len=500)
    feed.start()
    # strike = Binance price at window open (captured within first 30s of window).
    strike = {"ws": None, "px": None}
    fired = set()
    try:
        while True:
            now = time.time()
            ws = (int(now) // WINDOW_S) * WINDOW_S
            secs_in = now - ws
            last_px = feed.last_price
            if strike["ws"] != ws and secs_in < 30 and last_px:
                strike["ws"], strike["px"] = ws, last_px
            if (secs_in >= DECISION_OFFSET_S and ws not in fired
                    and strike["ws"] == ws and strike["px"]):
                fired.add(ws)
                bars = feed.bars.snapshot()[-VOL_WINDOW:]
                t = time.strftime("%H:%M:%S", time.gmtime(ws))
                if len(bars) < VOL_WINDOW or not last_px:
                    print(f"[{t}] warming up ({len(bars)}/{VOL_WINDOW} bars)")
                    continue
                drift = (last_px - strike["px"]) / strike["px"] * 100.0
                feats = build_features(bars, drift_pct=drift,
                                       secs_to_close=MONITOR_START_S, vol_window=VOL_WINDOW)
                if feats is not None:
                    p_up = model.predict_p_up(feats)
                    pick = "UP" if p_up >= 0.5 else "DOWN"
                    print(f"[{t}] strike={strike['px']:.1f} drift={drift:+.3f}% "
                          f"P(up)={p_up:.3f} -> pick {pick}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nstopped.")
        feed.stop()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "replay"
    if mode == "live":
        live()
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        replay(n)
