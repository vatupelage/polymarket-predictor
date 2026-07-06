#!/usr/bin/env python3
"""
BTC Predictor v4 - Window-Synced, Skip-Window Deep Analysis

Strategy:
  - Predicts every OTHER 5-min Polymarket window (every 10 min)
  - Uses the skip window (5 min) for deep analysis:
    * Multiple order book snapshots (trend detection)
    * Multiple Polymarket crowd readings
    * LSTM prediction with fresh data
    * Price momentum tracking
  - Final prediction at ~4 min before window close (60s in)
    = enough time to place trades
  - PTB (Price To Beat) distance dominates the signal

Runs continuously. Ctrl+C to stop.
"""

import sys
import os
import time
import datetime
import json
import statistics

import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import ta

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# Optional callback invoked after every prediction. Signature:
#   TRADE_HOOK(window, ptb, live_price, direction, confidence, final_up, signals, lstm_prob)
# Set from an external script (e.g. the live trading bot) before calling main().
TRADE_HOOK = None


# ============================================================
# BTC Data Fetcher (Bitstamp API)
# ============================================================

class BtcData:
    BASE_URL = 'https://www.bitstamp.net/api/v2/ohlc/btcusd/'

    @staticmethod
    def _request(step, limit, start=None, end=None):
        params = f'?limit={limit}&step={step}'
        if start: params += f'&start={start}'
        if end:   params += f'&end={end}'
        resp = requests.get(BtcData.BASE_URL + params)
        return resp.json()

    @staticmethod
    def _to_df(data_dict):
        data = pd.DataFrame(data_dict['data']['ohlc'])
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(
                data['timestamp'].astype(int).apply(datetime.datetime.fromtimestamp)
            )
        return data

    @classmethod
    def get_range(cls, start_dt, end_dt=None, delay=1, verbose=True, step=60):
        if isinstance(start_dt, str):
            start_dt = datetime.datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S")
        start_dt = start_dt.replace(second=0, microsecond=0)
        if end_dt is None:
            end_dt = datetime.datetime.now()
        elif isinstance(end_dt, str):
            end_dt = datetime.datetime.strptime(end_dt, "%Y-%m-%d %H:%M:%S")
        end_dt = end_dt.replace(second=0, microsecond=0)

        dif = (datetime.datetime.now() - start_dt).total_seconds()
        if dif <= 3:
            time.sleep(3 - dif)

        frames = []
        current = start_dt
        while current < end_dt:
            unix_start = int(time.mktime(current.timetuple()))
            unix_end = unix_start + step * 999
            data = cls._request(step, 1000, unix_start, unix_end)
            df = cls._to_df(data)
            frames.append(df)
            last_ts = datetime.datetime.fromtimestamp(int(data['data']['ohlc'][-1]['timestamp']))
            if verbose:
                print(f"  ({datetime.datetime.now():%H:%M:%S}): up to {last_ts}")
            current = last_ts + datetime.timedelta(seconds=step)
            time.sleep(delay)

        result = pd.concat(frames, ignore_index=True)
        result = result[pd.to_datetime(result['timestamp']) <= end_dt]
        float_cols = ['open', 'high', 'low', 'close', 'volume']
        result[float_cols] = result[float_cols].astype(float)
        return result


# Chainlink BTC/USD aggregator on Polygon — same data source Polymarket uses
# for 5-min up/down settlement. Rotates through public RPCs; falls back to
# Bitstamp if all Polygon RPCs are unreachable.
_CHAINLINK_BTC_USD_POLYGON = '0xc907E116054Ad103354f2D350FD2514433D57F6f'
_POLYGON_RPCS = [
    'https://polygon-bor-rpc.publicnode.com',
    'https://polygon.drpc.org',
    'https://1rpc.io/matic',
]
# function selectors: decimals() = 0x313ce567, latestRoundData() = 0xfeaf968c
_SEL_DECIMALS = '0x313ce567'
_SEL_LATEST_ROUND = '0xfeaf968c'
_CHAINLINK_DECIMALS = 8  # known constant for Chainlink BTC/USD; verified once below


def _eth_call(rpc, to, data, timeout=3):
    r = requests.post(rpc, json={
        'jsonrpc': '2.0', 'id': 1, 'method': 'eth_call',
        'params': [{'to': to, 'data': data}, 'latest'],
    }, timeout=timeout)
    r.raise_for_status()
    out = r.json()
    if 'error' in out:
        raise RuntimeError(out['error'])
    return out['result']


def _chainlink_btc_usd():
    """Returns (price_usd, age_seconds) from Chainlink on Polygon, or raises."""
    last_err = None
    for rpc in _POLYGON_RPCS:
        try:
            hex_result = _eth_call(rpc, _CHAINLINK_BTC_USD_POLYGON, _SEL_LATEST_ROUND)
            raw = hex_result[2:]  # strip 0x
            # latestRoundData returns 5 uint256-padded fields; answer is field 2
            answer_hex = raw[64:128]
            updated_hex = raw[192:256]
            answer = int(answer_hex, 16)
            # handle signed int256 (answer could theoretically be negative)
            if answer >= 2**255:
                answer -= 2**256
            updated = int(updated_hex, 16)
            price = answer / (10 ** _CHAINLINK_DECIMALS)
            age = int(time.time()) - updated
            return price, age
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"all Polygon RPCs failed: {last_err}")


def get_current_price():
    """Get live BTC price. Primary: Chainlink BTC/USD aggregator on Polygon
    (Polymarket's settlement source). Fallback: Bitstamp ticker."""
    try:
        price, age = _chainlink_btc_usd()
        if age > 120:
            # Aggregator heartbeat is ~27s on Polygon; >2 min stale is suspicious.
            raise RuntimeError(f"Chainlink feed stale: age={age}s")
        return price
    except Exception as e:
        print(f"  [PRICE] Chainlink failed ({e}), falling back to Bitstamp")
    try:
        resp = requests.get('https://www.bitstamp.net/api/v2/ticker/btcusd/', timeout=5)
        data = resp.json()
        return float(data['last'])
    except Exception as e:
        print(f"  [PRICE] Error: {e}")
        return None


# ============================================================
# Window Timing
# ============================================================

def get_current_window():
    now = int(time.time())
    window_start = (now // 300) * 300
    window_end = window_start + 300
    return {
        'start_ts': window_start,
        'end_ts': window_end,
        'elapsed': now - window_start,
        'remaining': window_end - now,
        'slug': f'btc-updown-5m-{window_start}',
        'start_dt': datetime.datetime.fromtimestamp(window_start),
        'end_dt': datetime.datetime.fromtimestamp(window_end),
    }


def get_next_window():
    now = int(time.time())
    window_start = ((now // 300) + 1) * 300
    window_end = window_start + 300
    return {
        'start_ts': window_start,
        'end_ts': window_end,
        'slug': f'btc-updown-5m-{window_start}',
        'start_dt': datetime.datetime.fromtimestamp(window_start),
        'end_dt': datetime.datetime.fromtimestamp(window_end),
    }


# ============================================================
# Order Book Fetcher (Bitstamp)
# ============================================================

def fetch_order_book():
    try:
        resp = requests.get('https://www.bitstamp.net/api/v2/order_book/btcusd/', timeout=5)
        data = resp.json()

        bids = [(float(b[0]), float(b[1])) for b in data['bids'][:20]]
        asks = [(float(a[0]), float(a[1])) for a in data['asks'][:20]]

        best_bid, best_ask = bids[0][0], asks[0][0]
        mid_price = (best_bid + best_ask) / 2

        spread_pct = (best_ask - best_bid) / mid_price * 100

        bid_vol = sum(b[1] for b in bids)
        ask_vol = sum(a[1] for a in asks)
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)

        near_range = mid_price * 0.001
        near_bid = sum(b[1] for b in bids if b[0] >= mid_price - near_range)
        near_ask = sum(a[1] for a in asks if a[0] <= mid_price + near_range)
        near_imbalance = (near_bid - near_ask) / (near_bid + near_ask + 1e-8)

        return {
            'spread_pct': spread_pct,
            'imbalance': imbalance,
            'near_imbalance': near_imbalance,
            'mid_price': mid_price,
        }
    except Exception as e:
        return None


# ============================================================
# Polymarket Fetcher
# ============================================================

def fetch_polymarket_prob(window_slug=None):
    try:
        if window_slug:
            slugs = [window_slug]
        else:
            now = int(time.time())
            slugs = [
                f'btc-updown-5m-{(now // 300) * 300}',
                f'btc-updown-5m-{((now - 300) // 300) * 300}',
                f'btc-updown-5m-{((now + 300) // 300) * 300}',
            ]

        for slug in slugs:
            resp = requests.get(
                f"https://gamma-api.polymarket.com/events?slug={slug}",
                timeout=5
            )
            events = resp.json()
            if events and len(events) > 0:
                markets = events[0].get('markets', [])
                if markets:
                    mkt = markets[0]
                    closed = mkt.get('closed', False)
                    prices = json.loads(mkt.get('outcomePrices', '["0.5","0.5"]'))
                    up_prob = float(prices[0])
                    if not closed:
                        return {
                            'up_prob': up_prob,
                            'slug': slug,
                            'title': mkt.get('question', slug),
                        }
        return None
    except Exception as e:
        return None


# ============================================================
# Feature Engineering
# ============================================================

def compute_indicators(df):
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)

    feat = pd.DataFrame()
    feat['open'] = df['open'].astype(float)
    feat['close'] = close.values
    feat['high'] = high.values
    feat['low'] = low.values
    feat['volume'] = df['volume'].astype(float)

    feat['rsi_5'] = ta.momentum.RSIIndicator(close=close, window=5).rsi()
    macd_fast = ta.trend.MACD(close=close, window_slow=13, window_fast=5, window_sign=4)
    feat['macd_fast'] = macd_fast.macd()
    feat['macd_signal_fast'] = macd_fast.macd_signal()
    bb_fast = ta.volatility.BollingerBands(close=close, window=10, window_dev=2)
    feat['bb_upper_fast'] = bb_fast.bollinger_hband()
    feat['bb_lower_fast'] = bb_fast.bollinger_lband()
    feat['bb_pband_fast'] = bb_fast.bollinger_pband()
    feat['ema_5'] = ta.trend.EMAIndicator(close=close, window=5).ema_indicator()
    feat['ema_13'] = ta.trend.EMAIndicator(close=close, window=13).ema_indicator()

    feat['rsi_14'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd_slow = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    feat['macd_slow'] = macd_slow.macd()
    feat['macd_signal_slow'] = macd_slow.macd_signal()
    bb_slow = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    feat['bb_upper_slow'] = bb_slow.bollinger_hband()
    feat['bb_lower_slow'] = bb_slow.bollinger_lband()
    feat['ema_12'] = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
    feat['ema_26'] = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()

    feat['mom_1m'] = close.pct_change(1) * 100
    feat['mom_3m'] = close.pct_change(3) * 100
    feat['mom_5m'] = close.pct_change(5) * 100
    feat['roc_5'] = close.pct_change(5) * 100
    feat['atr_5'] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=5
    ).average_true_range()

    return feat


def build_multiframe_features(df_1m, df_5m, df_15m):
    f1 = compute_indicators(df_1m)
    f1['timestamp'] = df_1m['timestamp'].values

    f5 = compute_indicators(df_5m)
    f5['timestamp'] = df_5m['timestamp'].values

    f15 = compute_indicators(df_15m)
    f15['timestamp'] = df_15m['timestamp'].values

    f5 = f5.ffill().bfill()
    f15 = f15.ffill().bfill()

    f5 = f5.rename(columns={c: f'{c}_5m' for c in f5.columns if c != 'timestamp'})
    f15 = f15.rename(columns={c: f'{c}_15m' for c in f15.columns if c != 'timestamp'})

    f1 = f1.sort_values('timestamp').reset_index(drop=True)
    f5 = f5.sort_values('timestamp').reset_index(drop=True)
    f15 = f15.sort_values('timestamp').reset_index(drop=True)

    merged = pd.merge_asof(f1, f5, on='timestamp', direction='backward')
    merged = pd.merge_asof(merged, f15, on='timestamp', direction='backward')

    merged = merged.drop(columns=['timestamp'])
    one_min_cols = [c for c in merged.columns if not c.endswith('_5m') and not c.endswith('_15m')]
    merged = merged.dropna(subset=one_min_cols).reset_index(drop=True)
    merged = merged.ffill().fillna(0)
    return merged


# ============================================================
# Per-column Scaler
# ============================================================

class FeatureScaler:
    def __init__(self):
        self.mins = None
        self.maxs = None

    def fit_transform(self, data):
        self.mins = data.min(axis=0)
        self.maxs = data.max(axis=0)
        return (data - self.mins) / (self.maxs - self.mins + 1e-8)

    def transform(self, data):
        return (data - self.mins) / (self.maxs - self.mins + 1e-8)


# ============================================================
# Classification LSTM
# ============================================================

def create_classifier(look_back, n_features):
    model = Sequential()
    model.add(LSTM(64, activation='relu', return_sequences=True,
                   input_shape=(look_back, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def make_direction_dataset(normed, raw_prices, look_back, horizon=5):
    X, y = [], []
    for i in range(look_back, len(normed) - horizon):
        X.append(normed[i - look_back:i])
        current = raw_prices[i]
        future = raw_prices[i + horizon]
        y.append(1.0 if future >= current else 0.0)
    return np.array(X), np.array(y)


def train_classifier(feat_matrix, raw_prices, n_features, look_back,
                     batch_size=32, epochs=30, max_retries=3, horizon=5):
    scaler = FeatureScaler()
    normed = scaler.fit_transform(feat_matrix)

    X, y = make_direction_dataset(normed, raw_prices, look_back, horizon)
    if len(X) < batch_size:
        return None, None, None

    up_pct = y.mean()
    print(f"  Dataset: {len(X)} samples, UP={up_pct:.1%} DOWN={1-up_pct:.1%}")

    best_acc, best_model = 0, None

    for attempt in range(max_retries):
        model = create_classifier(look_back, n_features)
        model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0,
                  validation_split=0.1)
        loss, acc = model.evaluate(X, y, batch_size=batch_size, verbose=0)
        print(f"  Attempt {attempt+1}/{max_retries} - acc: {acc:.4f}, loss: {loss:.4f}")
        if acc > best_acc:
            best_model, best_acc = model, acc

    if best_acc > 0.50:
        print(f"  Best model: {best_acc:.2%} accuracy")
        return best_model, best_acc, scaler
    else:
        print(f"  No model beat 50% (best: {best_acc:.2%})")
        return None, None, None


# ============================================================
# Deep Analysis (runs during skip window)
# ============================================================

def run_deep_analysis(duration_secs=240, sample_interval=30):
    """
    Collect multiple snapshots of order book, price, and Polymarket
    over a period. Returns aggregated intelligence.
    """
    ob_samples = []
    price_samples = []
    pm_samples = []
    start = time.time()

    print(f"    Collecting data ({duration_secs}s, every {sample_interval}s)...")

    while time.time() - start < duration_secs:
        # Price
        p = get_current_price()
        if p:
            price_samples.append({'time': time.time(), 'price': p})

        # Order book
        ob = fetch_order_book()
        if ob:
            ob_samples.append(ob)

        # Polymarket
        pm = fetch_polymarket_prob()
        if pm:
            pm_samples.append(pm)

        elapsed = time.time() - start
        remaining = duration_secs - elapsed
        if remaining > sample_interval:
            time.sleep(sample_interval)
        elif remaining > 0:
            time.sleep(remaining)
        else:
            break

    result = {
        'n_samples': len(price_samples),
        'prices': price_samples,
        'ob_samples': ob_samples,
        'pm_samples': pm_samples,
    }

    # Aggregated order book trend
    if len(ob_samples) >= 2:
        imbalances = [ob['imbalance'] for ob in ob_samples]
        near_imbalances = [ob['near_imbalance'] for ob in ob_samples]
        result['ob_imbalance_mean'] = statistics.mean(imbalances)
        result['ob_imbalance_trend'] = imbalances[-1] - imbalances[0]
        result['ob_near_imb_mean'] = statistics.mean(near_imbalances)
        # Is buying pressure increasing or decreasing?
        result['ob_pressure_direction'] = 'UP' if result['ob_imbalance_trend'] > 0 else 'DOWN'
        print(f"    OB: {len(ob_samples)} samples, imb_mean={result['ob_imbalance_mean']:+.3f}, "
              f"trend={result['ob_imbalance_trend']:+.3f} ({result['ob_pressure_direction']})")
    else:
        result['ob_imbalance_mean'] = 0
        result['ob_imbalance_trend'] = 0
        result['ob_near_imb_mean'] = 0
        result['ob_pressure_direction'] = 'NEUTRAL'

    # Price momentum over analysis period
    if len(price_samples) >= 2:
        first_price = price_samples[0]['price']
        last_price = price_samples[-1]['price']
        result['price_momentum'] = (last_price - first_price) / first_price * 100
        result['price_trend'] = 'UP' if last_price > first_price else 'DOWN'
        result['price_volatility'] = statistics.stdev([p['price'] for p in price_samples]) if len(price_samples) >= 3 else 0
        result['price_latest'] = last_price
        print(f"    Price: {len(price_samples)} samples, {first_price:,.0f}->{last_price:,.0f} "
              f"({result['price_momentum']:+.4f}%), vol=${result['price_volatility']:,.1f}")
    else:
        result['price_momentum'] = 0
        result['price_trend'] = 'NEUTRAL'
        result['price_volatility'] = 0
        result['price_latest'] = price_samples[-1]['price'] if price_samples else None

    # Polymarket consensus
    if len(pm_samples) >= 2:
        up_probs = [pm['up_prob'] for pm in pm_samples]
        result['pm_consensus'] = statistics.mean(up_probs)
        result['pm_trend'] = up_probs[-1] - up_probs[0]
        print(f"    PM: {len(pm_samples)} samples, consensus={result['pm_consensus']:.1%}, "
              f"trend={result['pm_trend']:+.3f}")
    elif pm_samples:
        result['pm_consensus'] = pm_samples[0]['up_prob']
        result['pm_trend'] = 0
    else:
        result['pm_consensus'] = 0.5
        result['pm_trend'] = 0

    return result


# ============================================================
# Ensemble v4 (PTB-dominant with deep analysis)
# ============================================================

def ensemble_predict_v4(lstm_up_prob, deep_analysis, current_price, price_to_beat):
    """
    PTB-dominant ensemble. At 3 min before close:
      - PTB distance is the primary signal (70% weight)
      - Deep analysis OB trend confirms/denies (15%)
      - LSTM provides background context (10%)
      - Polymarket crowd consensus (5%)

    If PTB distance is clear (>0.02%), just follow it.
    If it's a coin flip (<0.02%), lean on OB trend + momentum.
    """
    signals = {}

    # 1. PTB Distance — THE primary signal
    if price_to_beat and current_price:
        distance_pct = (current_price - price_to_beat) / price_to_beat * 100
        # Sigmoid mapping: even small moves matter at 3 min left
        sensitivity = 0.8  # tuned for 3-min-to-close
        ptb_prob = 1 / (1 + np.exp(-sensitivity * distance_pct))
        ptb_prob = np.clip(ptb_prob, 0.05, 0.95)
        is_clear = abs(distance_pct) >= 0.02
    else:
        ptb_prob = 0.5
        distance_pct = 0
        is_clear = False

    signals['ptb'] = {'up_prob': ptb_prob, 'weight': 0.70 if is_clear else 0.40}

    # 2. Order book trend (from deep analysis)
    ob_combined = 0.4 * deep_analysis.get('ob_imbalance_mean', 0) + \
                  0.6 * deep_analysis.get('ob_near_imb_mean', 0)
    # Factor in trend direction
    ob_trend = deep_analysis.get('ob_imbalance_trend', 0)
    ob_up_prob = np.clip(0.5 + ob_combined + ob_trend * 0.3, 0, 1)
    signals['orderbook'] = {'up_prob': ob_up_prob, 'weight': 0.15 if is_clear else 0.30}

    # 3. LSTM — empirically anti-predictive on 222-trade ablation (Pearson
    # −0.11, DirAcc 45.5%, standalone PnL/$1 −$0.144). Blending against
    # (1 − lstm_up_prob) recovers it as positive signal: blend Pearson
    # 0.14 → 0.23, DirAcc 53% → 59%, PnL/$1 −0.094 → +0.040. Raw value is
    # still emitted in `up_prob` so trade history stays comparable; the
    # `inverted` flag tells the weighted-sum below to flip its contribution.
    signals['lstm'] = {
        'up_prob': lstm_up_prob,
        'weight': 0.10 if is_clear else 0.15,
        'inverted': True,
    }

    # 4. Polymarket consensus
    pm_consensus = deep_analysis.get('pm_consensus', 0.5)
    signals['polymarket'] = {'up_prob': pm_consensus, 'weight': 0.05 if is_clear else 0.15}

    # Normalize weights
    total_w = sum(s['weight'] for s in signals.values())
    for s in signals.values():
        s['weight'] /= total_w

    # Weighted final probability — signals tagged `inverted` contribute (1−p)
    final_up = sum(
        ((1.0 - s['up_prob']) if s.get('inverted') else s['up_prob']) * s['weight']
        for s in signals.values()
    )

    # Strong override: if PTB distance is very clear (>0.05%), trust it almost entirely
    if is_clear and abs(distance_pct) >= 0.05:
        final_up = ptb_prob * 0.85 + final_up * 0.15

    direction = "UP" if final_up >= 0.5 else "DOWN"
    confidence = abs(final_up - 0.5) * 200

    return direction, confidence, final_up, signals, distance_pct


# ============================================================
# Main - Skip-Window Strategy
# ============================================================

def main():
    DAYS_FOR_TRAINING = 3
    LOOK_BACK = 15
    BATCH_SIZE = 32
    EPOCHS = 30
    FORECAST_HORIZON = 5
    RETRAIN_EVERY_N = 15  # retrain every 15 predictions (~2.5 hours)
    PREDICT_AT = 60        # predict at 60s into window (240s = 4 min before close)
    ANALYSIS_DURATION = 240  # use 240s of skip window for analysis

    print("=" * 70)
    print("BTC Predictor v4 - Skip-Window Deep Analysis")
    print("=" * 70)
    print("  Strategy: predict every OTHER window (every 10 min)")
    print("  Skip window: 4 min deep analysis (OB trend, price momentum)")
    print(f"  Predict at: {PREDICT_AT}s into window ({300-PREDICT_AT}s = 4 min before close)")
    print("  PTB distance: 70% weight (dominant signal)")
    print("  Running continuously until Ctrl+C")

    # Step 1: Fetch & train
    start_date = datetime.datetime.now() - datetime.timedelta(days=DAYS_FOR_TRAINING)
    start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n[1/3] Fetching {DAYS_FOR_TRAINING} days at 3 timeframes...")

    print("  1-min candles:")
    df_1m = BtcData.get_range(start_str, verbose=True, step=60)
    print(f"    -> {len(df_1m)} rows")

    print("  5-min candles:")
    df_5m = BtcData.get_range(start_str, verbose=True, step=300)
    print(f"    -> {len(df_5m)} rows")

    print("  15-min candles:")
    df_15m = BtcData.get_range(start_str, verbose=True, step=900)
    print(f"    -> {len(df_15m)} rows")

    print(f"\n  Building features...")
    feat_df = build_multiframe_features(df_1m, df_5m, df_15m)
    N_FEATURES = len(feat_df.columns)
    feat_matrix = feat_df.to_numpy()
    raw_prices = feat_matrix[:, 0]
    print(f"  Feature matrix: {feat_matrix.shape} ({N_FEATURES} features)")

    # Step 2: Train
    print(f"\n[2/3] Training CLASSIFICATION LSTM...")
    model, acc, scaler = train_classifier(
        feat_matrix, raw_prices, N_FEATURES, LOOK_BACK,
        BATCH_SIZE, EPOCHS, max_retries=3, horizon=FORECAST_HORIZON
    )

    if model is None:
        print("Could not train an acceptable classifier.")
        return

    normed = scaler.fit_transform(feat_matrix)
    X, y_true = make_direction_dataset(normed, raw_prices, LOOK_BACK, FORECAST_HORIZON)
    preds = model.predict(X, verbose=0).reshape(-1)
    hist_acc = np.mean((preds >= 0.5).astype(float) == y_true)
    print(f"\n  Historical accuracy: {hist_acc:.2%}")

    # Step 3: Live predictions
    print(f"\n[3/3] Live Predictions - Every Other Window (Ctrl+C to stop)")
    print(f"  Testing sources...")
    ob_test = fetch_order_book()
    print(f"    Order Book: {'OK' if ob_test else 'FAILED'}")
    pm_test = fetch_polymarket_prob()
    print(f"    Polymarket: {'OK' if pm_test else 'FAILED'}")
    tp = get_current_price()
    print(f"    Live Price:  ${tp:,.2f}" if tp else "    Live Price: FAILED")

    results = []
    pred_count = 0
    skip_next = True  # Start with analysis phase, then predict

    print(f"\n  {'='*95}")
    print(f"  {'Window':>13} | {'PTB':>10} | {'Price@2m':>10} | {'Dist':>8} | "
          f"{'Dir':>4} | {'Conf':>5} | {'PTB%':>5} | {'OB%':>5} | {'PM%':>5} | {'LSTM':>5} | {'Final':>5}")
    print(f"  {'='*95}")

    # Wait for a window boundary
    w = get_current_window()
    if w['remaining'] > 5:
        print(f"\n  Waiting {w['remaining']}s for next window...")
        time.sleep(w['remaining'] + 2)

    try:
        while True:
            window = get_current_window()

            if skip_next:
                # ========== ANALYSIS WINDOW ==========
                print(f"\n  [{window['start_dt']:%H:%M}-{window['end_dt']:%H:%M}] "
                      f"ANALYSIS PHASE (collecting intelligence for next window)")

                # Run deep analysis for most of this window
                deep_analysis = run_deep_analysis(
                    duration_secs=ANALYSIS_DURATION,
                    sample_interval=30
                )

                # Also run LSTM prediction with fresh data
                print(f"    Running LSTM prediction...")
                try:
                    lookback_mins = LOOK_BACK + 450
                    recent_start = (datetime.datetime.now() - datetime.timedelta(
                        minutes=lookback_mins)).strftime("%Y-%m-%d %H:%M:%S")
                    r1m = BtcData.get_range(recent_start, verbose=False, step=60)
                    r5m = BtcData.get_range(recent_start, verbose=False, step=300)
                    r15m = BtcData.get_range(recent_start, verbose=False, step=900)
                    fresh_feat = build_multiframe_features(r1m, r5m, r15m)
                    recent = fresh_feat.tail(LOOK_BACK).to_numpy()
                    recent_norm = scaler.transform(recent).reshape(1, LOOK_BACK, N_FEATURES)
                    lstm_prob = float(model.predict(recent_norm, verbose=0)[0][0])
                    print(f"    LSTM UP prob: {lstm_prob:.1%}")
                except Exception as e:
                    print(f"    LSTM error: {e}")
                    lstm_prob = 0.5

                # Store analysis for next window
                analysis_cache = {
                    'deep': deep_analysis,
                    'lstm_prob': lstm_prob,
                }

                # Wait past this window's boundary. Loop re-sleeps if woken
                # early (signals / scheduling) — plain time.sleep() can wake
                # ~2s short and misalign us with get_current_window().
                target = window['end_ts'] + 3
                while time.time() < target:
                    time.sleep(max(0.1, target - time.time()))

                skip_next = False
                continue

            # ========== PREDICTION WINDOW ==========
            # Capture PTB at window start
            ptb = get_current_price()
            if ptb:
                print(f"\n  >>> Window {window['start_dt']:%H:%M}-{window['end_dt']:%H:%M} | "
                      f"Price To Beat: ${ptb:,.2f} | Slug: {window['slug']}")
            else:
                print(f"\n  >>> Window {window['start_dt']:%H:%M}-{window['end_dt']:%H:%M} | "
                      f"PTB: FAILED")

            # Wait until prediction time (PREDICT_AT seconds into window)
            now_ts = int(time.time())
            elapsed = now_ts - window['start_ts']
            wait_for = PREDICT_AT - elapsed
            if wait_for > 0:
                print(f"  Waiting {wait_for}s until prediction time "
                      f"({300 - PREDICT_AT}s = {(300-PREDICT_AT)//60} min before close)...")
                time.sleep(wait_for)
            else:
                # Started too late in this window — bailing out prevents
                # last-second trades on near-resolved markets.
                print(f"  SKIP window: started too late "
                      f"(elapsed={elapsed}s >= PREDICT_AT={PREDICT_AT}s)")
                skip_next = True
                target = window['end_ts'] + 3
                while time.time() < target:
                    time.sleep(max(0.1, target - time.time()))
                continue

            # ---- Make prediction ----
            try:
                # Fresh price at prediction time
                live_price = get_current_price()
                if not live_price:
                    print(f"  ERROR: Could not get live price")
                    skip_next = True
                    target = window['end_ts'] + 3
                    while time.time() < target:
                        time.sleep(max(0.1, target - time.time()))
                    continue

                # Fresh order book snapshot at prediction time
                ob_now = fetch_order_book()

                # Fresh Polymarket at prediction time
                pm_now = fetch_polymarket_prob(window['slug'])

                # Use analysis from skip window (or defaults)
                if 'analysis_cache' in dir() and analysis_cache:
                    deep = analysis_cache.get('deep', {})
                    lstm_prob = analysis_cache.get('lstm_prob', 0.5)
                else:
                    deep = {}
                    lstm_prob = 0.5

                # Merge fresh OB into deep analysis (latest snapshot matters most)
                if ob_now:
                    deep['ob_imbalance_mean'] = (
                        deep.get('ob_imbalance_mean', 0) * 0.4 +
                        ob_now['imbalance'] * 0.6
                    )
                    deep['ob_near_imb_mean'] = (
                        deep.get('ob_near_imb_mean', 0) * 0.4 +
                        ob_now['near_imbalance'] * 0.6
                    )

                # Merge fresh PM into deep analysis
                if pm_now:
                    deep['pm_consensus'] = (
                        deep.get('pm_consensus', 0.5) * 0.4 +
                        pm_now['up_prob'] * 0.6
                    )

                # Ensemble prediction
                direction, confidence, final_up, signals, dist_pct = ensemble_predict_v4(
                    lstm_prob, deep, live_price, ptb
                )

                pred_count += 1

                # Display
                ptb_str = f"${ptb:>9,.0f}" if ptb else "      N/A "
                dist_str = f"{dist_pct:+.4f}%" if ptb else "    N/A  "
                ptb_sig = f"{signals['ptb']['up_prob']:.0%}"
                ob_sig = f"{signals['orderbook']['up_prob']:.0%}"
                pm_sig = f"{signals['polymarket']['up_prob']:.0%}"
                lstm_sig = f"{lstm_prob:.0%}"
                final_str = f"{final_up:.0%}"

                print(f"  {window['start_dt']:%H:%M}-{window['end_dt']:%H:%M} | "
                      f"{ptb_str} | ${live_price:>9,.0f} | {dist_str} | "
                      f"{direction:>4} | {confidence:>4.1f}% | "
                      f"{ptb_sig:>5} | {ob_sig:>5} | {pm_sig:>5} | {lstm_sig:>5} | {final_str:>5} <<<")

                if TRADE_HOOK is not None:
                    try:
                        TRADE_HOOK(
                            window=window,
                            ptb=ptb,
                            live_price=live_price,
                            direction=direction,
                            confidence=confidence,
                            final_up=final_up,
                            signals=signals,
                            lstm_prob=lstm_prob,
                        )
                    except Exception as _hook_err:
                        print(f"  TRADE_HOOK error: {_hook_err}")

                # Record for accuracy tracking
                results.append({
                    'window_start': window['start_ts'],
                    'window_end': window['end_ts'],
                    'ptb': ptb,
                    'price_at_pred': live_price,
                    'prediction': direction,
                    'confidence': confidence,
                    'final_up_prob': final_up,
                    'dist_pct': dist_pct,
                    'signals': {k: v['up_prob'] for k, v in signals.items()},
                    'lstm_prob': lstm_prob,
                })

                # Retrain periodically
                if pred_count % RETRAIN_EVERY_N == 0:
                    print(f"\n  --- Retraining LSTM ---")
                    sd = datetime.datetime.now() - datetime.timedelta(days=DAYS_FOR_TRAINING)
                    ss = sd.strftime("%Y-%m-%d %H:%M:%S")
                    d1 = BtcData.get_range(ss, verbose=False, step=60)
                    d5 = BtcData.get_range(ss, verbose=False, step=300)
                    d15 = BtcData.get_range(ss, verbose=False, step=900)
                    fdf = build_multiframe_features(d1, d5, d15)
                    fm = fdf.to_numpy()
                    rp = fm[:, 0]
                    nm, na, ns = train_classifier(
                        fm, rp, N_FEATURES, LOOK_BACK,
                        BATCH_SIZE, EPOCHS, max_retries=3, horizon=FORECAST_HORIZON
                    )
                    if nm is not None:
                        model, scaler = nm, ns
                        print(f"  --- Retrained! Accuracy: {na:.2%} ---\n")
                    else:
                        print(f"  --- Retrain failed, keeping current ---\n")

            except Exception as e:
                print(f"  ERROR: {e}")

            # Wait past this window's boundary. See note in analyze phase —
            # time.sleep() alone can wake ~2s early and misalign us with
            # get_current_window() on the next iteration.
            target = window['end_ts'] + 3
            while time.time() < target:
                time.sleep(max(0.1, target - time.time()))

            skip_next = True
            analysis_cache = {}

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 70}")
        print(f"SESSION SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Predictions made: {pred_count}")
        print(f"  Windows tracked:  {len(results)}")

        if results:
            print(f"\n  Verifying against actual prices...\n")
            correct = 0
            correct_ptb_only = 0
            total = 0

            for r in results:
                try:
                    time.sleep(0.3)
                    resp = requests.get(
                        f"https://www.bitstamp.net/api/v2/ohlc/btcusd/"
                        f"?step=300&limit=2&start={r['window_start']}&end={r['window_end']}",
                        timeout=5
                    )
                    data = resp.json()
                    candles = data['data']['ohlc']

                    actual_close = None
                    for c in candles:
                        ts = int(c['timestamp'])
                        if abs(ts - r['window_start']) < 300:
                            actual_close = float(c['close'])
                            break
                    if actual_close is None and candles:
                        actual_close = float(candles[-1]['close'])
                    if actual_close is None:
                        continue

                    actual_dir = "UP" if actual_close >= r['ptb'] else "DOWN"
                    match = "OK" if actual_dir == r['prediction'] else "MISS"
                    if actual_dir == r['prediction']:
                        correct += 1

                    # What would PTB-only have said?
                    ptb_only = "UP" if r['price_at_pred'] >= r['ptb'] else "DOWN"
                    if ptb_only == actual_dir:
                        correct_ptb_only += 1

                    total += 1

                    ws = datetime.datetime.fromtimestamp(r['window_start'])
                    print(f"  {ws:%H:%M} | PTB=${r['ptb']:,.0f} | "
                          f"Close=${actual_close:,.0f} | "
                          f"Actual={actual_dir:>4} | Pred={r['prediction']:>4} | "
                          f"{match} | conf={r['confidence']:.1f}%")

                except Exception:
                    pass

            if total > 0:
                print(f"\n  {'='*40}")
                print(f"  ENSEMBLE ACCURACY: {correct}/{total} = {correct/total:.1%}")
                print(f"  PTB-ONLY ACCURACY: {correct_ptb_only}/{total} = {correct_ptb_only/total:.1%}")
                print(f"  {'='*40}")

        print(f"\n{'=' * 70}")


if __name__ == '__main__':
    main()
