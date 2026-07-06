#!/usr/bin/env python3
"""
BTC Predictor Dashboard - Polymarket 5-Min Up/Down Edition.
  - Fast + Slow indicators tuned for 5-min prediction
  - Order book data (Bitstamp)
  - Polymarket crowd probability
  - Ensemble: LSTM + Order Book + Polymarket

Run:  python3 run_dashboard.py
Then open http://127.0.0.1:8050 in your browser.
"""

import os
import sys
import time
import json
import queue
import threading
import datetime

import requests
import numpy as np
import pandas as pd
import sklearn.metrics
import tensorflow as tf
import ta

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

import plotly.graph_objects as go
import plotly.express as px
from numpy import radians, cos, sin

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Suppress TF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# ============================================================
# BTC Data Fetcher (from Bitstamp API)
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
        """Get OHLCV data. step=60 for 1-min, 300 for 5-min, 900 for 15-min."""
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
                print(f"  Fetched up to {last_ts}")
            current = last_ts + datetime.timedelta(seconds=step)
            time.sleep(delay)

        result = pd.concat(frames, ignore_index=True)
        result = result[pd.to_datetime(result['timestamp']) <= end_dt]
        float_cols = ['open', 'high', 'low', 'close', 'volume']
        result[float_cols] = result[float_cols].astype(float)
        return result

    @classmethod
    def get_current(cls):
        unix_now = int(time.mktime(datetime.datetime.now().timetuple()))
        return cls._to_df(cls._request(60, 1, unix_now, unix_now))


# ============================================================
# Feature Engineering
# ============================================================

def compute_indicators(df):
    """Compute technical indicators — both fast (5-min tuned) and slow."""
    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)

    feat = pd.DataFrame()
    feat['open'] = df['open'].astype(float)
    feat['close'] = close.values
    feat['high'] = high.values
    feat['low'] = low.values
    feat['volume'] = df['volume'].astype(float)

    # Fast indicators (tuned for 5-min)
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

    # Slow indicators (trend context)
    feat['rsi_14'] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
    macd_slow = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    feat['macd_slow'] = macd_slow.macd()
    feat['macd_signal_slow'] = macd_slow.macd_signal()
    bb_slow = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    feat['bb_upper_slow'] = bb_slow.bollinger_hband()
    feat['bb_lower_slow'] = bb_slow.bollinger_lband()
    feat['ema_12'] = ta.trend.EMAIndicator(close=close, window=12).ema_indicator()
    feat['ema_26'] = ta.trend.EMAIndicator(close=close, window=26).ema_indicator()

    # Momentum
    feat['roc_5'] = close.pct_change(5) * 100
    feat['atr_5'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=5).average_true_range()

    return feat


def build_multiframe_features(df_1m, df_5m, df_15m):
    """Combine 1-min, 5-min, and 15-min indicators into one feature row per minute."""
    f1 = compute_indicators(df_1m)
    f1['timestamp'] = df_1m['timestamp'].values

    f5 = compute_indicators(df_5m)
    f5['timestamp'] = df_5m['timestamp'].values

    f15 = compute_indicators(df_15m)
    f15['timestamp'] = df_15m['timestamp'].values

    # Forward-fill NaN in lower-timeframe indicators (warm-up period)
    f5 = f5.ffill().bfill()
    f15 = f15.ffill().bfill()

    f5 = f5.rename(columns={c: f'{c}_5m' for c in f5.columns if c != 'timestamp'})
    f15 = f15.rename(columns={c: f'{c}_15m' for c in f15.columns if c != 'timestamp'})

    f1 = f1.sort_values('timestamp').reset_index(drop=True)
    f5 = f5.sort_values('timestamp').reset_index(drop=True)
    f15 = f15.sort_values('timestamp').reset_index(drop=True)

    merged = pd.merge_asof(f1, f5, on='timestamp', direction='backward')
    merged = pd.merge_asof(merged, f15, on='timestamp', direction='backward')

    # Only drop rows where 1-min indicators have NaN (warm-up), fill any remaining
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

    def inverse_price(self, normed):
        """Inverse for 'open' column (index 0)."""
        return normed * (self.maxs[0] - self.mins[0]) + self.mins[0]


# ============================================================
# Order Book Fetcher (Bitstamp)
# ============================================================

def fetch_order_book():
    """Fetch Bitstamp BTC/USD order book and compute features."""
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
            'spread_pct': spread_pct, 'imbalance': imbalance,
            'near_imbalance': near_imbalance, 'mid_price': mid_price,
        }
    except:
        return None


# ============================================================
# Polymarket Fetcher
# ============================================================

def fetch_polymarket_prob():
    """Get current Polymarket 'Up' probability for BTC 5-min market."""
    try:
        now = int(time.time())
        for offset in [0, 300]:
            rounded = ((now + offset) // 300) * 300
            slug = f"btc-updown-5m-{rounded}"
            resp = requests.get(
                f"https://gamma-api.polymarket.com/events?slug={slug}", timeout=5
            )
            events = resp.json()
            if events:
                markets = events[0].get('markets', [])
                if markets:
                    prices = json.loads(markets[0].get('outcomePrices', '["0.5","0.5"]'))
                    return {'up_prob': float(prices[0]), 'slug': slug}
        return None
    except:
        return None


def ensemble_predict(lstm_direction, lstm_confidence, ob_data, pm_data,
                     w_lstm=0.50, w_ob=0.30, w_pm=0.20):
    """Combine LSTM + order book + Polymarket into direction + confidence."""
    lstm_score = lstm_confidence if lstm_direction == "UP" else -lstm_confidence

    ob_score = 0.0
    if ob_data:
        ob_score = np.clip(0.4 * ob_data['imbalance'] * 2 + 0.6 * ob_data['near_imbalance'] * 2, -1, 1)
    else:
        w_lstm += w_ob; w_ob = 0

    pm_score = 0.0
    if pm_data:
        pm_score = (pm_data['up_prob'] - 0.5) * 2
    else:
        w_lstm += w_pm; w_pm = 0

    total = w_lstm + w_ob + w_pm
    final = (lstm_score * w_lstm + ob_score * w_ob + pm_score * w_pm) / total
    return ("UP" if final >= 0 else "DOWN"), abs(final) * 100


# ============================================================
# LSTM Predictor (runs in background thread)
# ============================================================

class BaselinePredictor:
    FORECAST_HORIZON = 5  # predict 5 minutes ahead

    def __init__(self, days_for_training=3, look_back=15, batch_size=32,
                 epochs=25, refresh_rate=10, leniency=5, max_retries=3):
        self.days = days_for_training
        self.look_back = look_back
        self.batch_size = batch_size
        self.epochs = epochs
        self.refresh_rate = refresh_rate
        self.leniency = leniency
        self.max_retries = max_retries

        self._model = None
        self._scaler = None
        self._n_features = None
        self._btc_data = None      # 1-min data
        self._btc_data_5m = None   # 5-min data
        self._btc_data_15m = None  # 15-min data
        self._predictions = []
        self._pred_times = []
        self._directions = []       # ensemble direction per prediction
        self._confidences = []      # ensemble confidence per prediction
        self._ob_data = None        # latest order book data
        self._pm_data = None        # latest polymarket data
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._running = False

    def _create_model(self, n_features):
        model = Sequential()
        model.add(LSTM(64, activation='relu', return_sequences=True,
                       input_shape=(self.look_back, n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(32, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='Nadam', loss='mse')
        return model

    def _make_5min_dataset(self, normed):
        """Build X, y arrays where y is the open price 5 minutes ahead."""
        h = self.FORECAST_HORIZON
        lb = self.look_back
        X, y = [], []
        for i in range(lb, len(normed) - h):
            X.append(normed[i - lb:i])
            y.append(normed[i + h, 0])  # target = open price column
        return np.array(X), np.array(y)

    def _train(self, feat_matrix, n_features):
        scaler = FeatureScaler()
        normed = scaler.fit_transform(feat_matrix)

        X, y = self._make_5min_dataset(normed)
        if len(X) < self.batch_size:
            return None, None, None

        h = self.FORECAST_HORIZON
        price_col = normed[:, 0]
        curr_mse = sklearn.metrics.mean_squared_error(price_col[h:], price_col[:-h])
        max_loss = curr_mse + abs(self.leniency) * np.std(price_col)

        best_loss, best_model = float('inf'), None
        print(f"  [Predictor] Training ({n_features} features, 64+32 LSTM, {self.epochs} epochs)... Max loss: {max_loss:.6f}")

        for attempt in range(self.max_retries):
            m = self._create_model(n_features)
            m.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, shuffle=False, verbose=0)
            loss = m.evaluate(X, y, batch_size=self.batch_size, verbose=0)
            print(f"  [Predictor] Attempt {attempt+1}/{self.max_retries} - loss: {loss:.6f}")
            if loss < best_loss:
                best_model, best_loss = m, loss

        if best_loss < max_loss:
            return best_model, best_loss, scaler
        return None, None, None

    def _fetch_all_timeframes(self, start_str, verbose=True):
        """Fetch 1m, 5m, 15m data from start_str."""
        print(f"  [Predictor] Fetching 1-min candles...")
        df_1m = BtcData.get_range(start_str, verbose=verbose, step=60)
        print(f"    -> {len(df_1m)} rows")
        print(f"  [Predictor] Fetching 5-min candles...")
        df_5m = BtcData.get_range(start_str, verbose=verbose, step=300)
        print(f"    -> {len(df_5m)} rows")
        print(f"  [Predictor] Fetching 15-min candles...")
        df_15m = BtcData.get_range(start_str, verbose=verbose, step=900)
        print(f"    -> {len(df_15m)} rows")
        return df_1m, df_5m, df_15m

    def _loop(self):
        print("[Predictor] Fetching historical data at 3 timeframes...")
        start = datetime.datetime.now() - datetime.timedelta(days=self.days)
        start_str = start.strftime("%Y-%m-%d %H:%M:%S")

        df_1m, df_5m, df_15m = self._fetch_all_timeframes(start_str)

        with self._lock:
            self._btc_data = df_1m
            self._btc_data_5m = df_5m
            self._btc_data_15m = df_15m

        trim = self.days * 1440
        last_train = 0

        # Align to minute boundary
        wait = 60 - datetime.datetime.now().second
        if wait > 1:
            time.sleep(wait)

        print("[Predictor] Starting prediction loop...")
        while not self._stop.is_set():
            now = datetime.datetime.now()

            # Refresh 1-min data
            try:
                new = BtcData.get_range(
                    (self._btc_data['timestamp'].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S"),
                    verbose=False, step=60
                )
                with self._lock:
                    old = self._btc_data[self._btc_data['timestamp'] < new['timestamp'].iloc[0]]
                    self._btc_data = pd.concat([old, new], ignore_index=True).tail(trim)
                # Also refresh 5m and 15m
                new5 = BtcData.get_range(
                    (self._btc_data_5m['timestamp'].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S"),
                    verbose=False, step=300
                )
                with self._lock:
                    old5 = self._btc_data_5m[self._btc_data_5m['timestamp'] < new5['timestamp'].iloc[0]]
                    self._btc_data_5m = pd.concat([old5, new5], ignore_index=True).tail(trim // 5)
                new15 = BtcData.get_range(
                    (self._btc_data_15m['timestamp'].iloc[-1]).strftime("%Y-%m-%d %H:%M:%S"),
                    verbose=False, step=900
                )
                with self._lock:
                    old15 = self._btc_data_15m[self._btc_data_15m['timestamp'] < new15['timestamp'].iloc[0]]
                    self._btc_data_15m = pd.concat([old15, new15], ignore_index=True).tail(trim // 15)
            except Exception as e:
                print(f"[Predictor] Data refresh error: {e}")

            # Build multi-timeframe features
            try:
                feat_df = build_multiframe_features(self._btc_data, self._btc_data_5m, self._btc_data_15m)
                feat_matrix = feat_df.to_numpy()
                n_features = feat_matrix.shape[1]
                self._n_features = n_features
            except Exception as e:
                print(f"[Predictor] Feature build error: {e}")
                time.sleep(60)
                continue

            # Train / retrain
            if self._model is None or (time.time() - last_train) / 60 >= self.refresh_rate:
                print(f"[Predictor] Training model...")
                model, loss, scaler = self._train(feat_matrix, n_features)
                if model is not None:
                    with self._lock:
                        self._model = model
                        self._scaler = scaler
                    last_train = time.time()
                    print(f"[Predictor] Model trained (loss={loss:.6f})")
                else:
                    print(f"[Predictor] Training failed, keeping old model")

            # Predict 5 minutes ahead with ensemble
            direction, confidence = "UP", 50.0
            if self._model is not None and self._scaler is not None:
                try:
                    lookback_mins = self.look_back + 450
                    recent_start = (datetime.datetime.now() - datetime.timedelta(minutes=lookback_mins)).strftime("%Y-%m-%d %H:%M:%S")
                    r1m = BtcData.get_range(recent_start, verbose=False, step=60)
                    r5m = BtcData.get_range(recent_start, verbose=False, step=300)
                    r15m = BtcData.get_range(recent_start, verbose=False, step=900)

                    fresh_feat = build_multiframe_features(r1m, r5m, r15m)
                    current_price = float(fresh_feat['open'].iloc[-1])

                    recent = fresh_feat.tail(self.look_back).to_numpy()
                    recent_norm = self._scaler.transform(recent).reshape(1, self.look_back, self._n_features)
                    pred_norm = self._model.predict(recent_norm, verbose=0)[0][0]
                    pred = self._scaler.inverse_price(pred_norm)

                    # Ensemble
                    pct = (pred - current_price) / abs(current_price)
                    lstm_dir = "UP" if pct > 0 else "DOWN"
                    lstm_conf = min(abs(pct) * 500, 1.0)

                    ob = fetch_order_book()
                    pm = fetch_polymarket_prob()
                    direction, confidence = ensemble_predict(lstm_dir, lstm_conf, ob, pm)

                    with self._lock:
                        self._ob_data = ob
                        self._pm_data = pm
                except Exception as e:
                    print(f"[Predictor] Prediction error: {e}")
                    current_price = float(self._btc_data['open'].iloc[-1])
                    pred = current_price
            else:
                current_price = float(self._btc_data['open'].iloc[-1])
                pred = current_price

            with self._lock:
                self._predictions.append(float(pred))
                ts = self._btc_data['timestamp'].iloc[-1] + datetime.timedelta(minutes=self.FORECAST_HORIZON)
                self._pred_times.append(str(ts))
                self._directions.append(direction)
                self._confidences.append(confidence)

            self._ready.set()
            print(f"[Predictor] {direction} ({confidence:.1f}%) | Forecast: ${pred:,.2f} (current ${current_price:,.2f})")

            # Wait 5 minutes before next prediction cycle
            elapsed = (datetime.datetime.now() - now).seconds
            wait_secs = self.FORECAST_HORIZON * 60 - elapsed
            if wait_secs > 0:
                self._stop.wait(timeout=wait_secs)

    def run(self):
        if not self._running:
            self._running = True
            self._stop.clear()
            t = threading.Thread(target=self._loop, daemon=True)
            t.start()

    def stop(self):
        self._stop.set()
        self._running = False

    def has_model(self):
        return self._model is not None

    def predict(self):
        if self._model is not None and self._predictions:
            return self._predictions[-1]
        return None

    def latest_direction(self):
        with self._lock:
            if self._directions:
                return self._directions[-1], self._confidences[-1]
        return None, 0

    def latest_ob(self):
        with self._lock:
            return self._ob_data

    def latest_pm(self):
        with self._lock:
            return self._pm_data

    def predict_pct(self):
        if self._model is not None and self._btc_data is not None and self._predictions:
            last = self._btc_data['open'].iloc[-1]
            return 100 * ((self._predictions[-1] - last) / abs(last))
        return None

    def history(self, window=None):
        self._ready.wait(timeout=300)
        with self._lock:
            if self._model is None or self._scaler is None or self._btc_data is None:
                return None
            try:
                feat_df = build_multiframe_features(self._btc_data, self._btc_data_5m, self._btc_data_15m)
                feat_matrix = feat_df.to_numpy()
                normed = self._scaler.fit_transform(feat_matrix)
                h = self.FORECAST_HORIZON
                X, y_true = self._make_5min_dataset(normed)
                preds_norm = self._model.predict(X, verbose=0).reshape(-1)
                predictions = self._scaler.inverse_price(preds_norm)
                actuals = self._scaler.inverse_price(y_true)

                # Use 1-min timestamps aligned to predictions
                timestamps = self._btc_data['timestamp'].to_numpy()
                # After feature build + dropna, the timestamps may be shorter
                # Use the tail of timestamps matching feat_matrix length
                ts_tail = timestamps[-len(feat_matrix):]
                offset = self.look_back + h
                n = min(len(predictions), len(ts_tail) - offset)
                df = pd.DataFrame({
                    'time': ts_tail[offset:offset + n],
                    'actual': actuals[:n],
                    'prediction': predictions[:n]
                })
                if window:
                    return df.tail(window)
                return df
            except Exception as e:
                print(f"[Predictor] History error: {e}")
                return None

    def predictions_so_far(self):
        with self._lock:
            if not self._predictions:
                return None
            return pd.DataFrame({
                'time': self._pred_times,
                'prediction': self._predictions,
            })

    def accuracy(self, step=15, offset=0):
        h = self.history()
        if h is not None and len(h) > 1:
            h = h.iloc[-offset::step] if offset else h.iloc[::step]
            tc = h['actual'].diff().shift(-1)
            pc = h['prediction'].shift(-1) - h['actual']
            return float(np.mean((tc < 0) == (pc < 0)))
        return 0

    def smape(self, offset=0):
        h = self.history()
        if h is not None and len(h) > 0:
            if offset:
                h = h.iloc[-offset:]
            n = np.abs(h['prediction'] - h['actual'])
            d = np.abs(h['actual']) + np.abs(h['prediction'])
            return float(100 * np.mean(n / d))
        return 0

    def noise(self, offset=0):
        h = self.history()
        if h is not None and len(h) > 1:
            if offset:
                h = h.iloc[-offset:]
            return float(h['prediction'].diff().abs().mean())
        return 0

    def get_data(self):
        with self._lock:
            if self._btc_data is not None:
                return self._btc_data.copy()
        return None


# ============================================================
# Technical Analysis / Trading Signals
# ============================================================

def fill_trends(df):
    """Compute EMA and Ichimoku indicators."""
    close = df['close'].astype(float)
    df['ema_5']   = ta.trend.EMAIndicator(close=close, window=5).ema_indicator()
    df['ema_13']  = ta.trend.EMAIndicator(close=close, window=13).ema_indicator()
    df['ema_21']  = ta.trend.EMAIndicator(close=close, window=21).ema_indicator()
    df['ema_55']  = ta.trend.EMAIndicator(close=close, window=55).ema_indicator()
    df['ema_100'] = ta.trend.EMAIndicator(close=close, window=100).ema_indicator()
    df['ema_200'] = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()

    ich = ta.trend.IchimokuIndicator(
        high=df['high'].astype(float),
        low=df['low'].astype(float),
        window1=9, window2=26, window3=52
    )
    df['ichimoku_conversion'] = ich.ichimoku_conversion_line()
    df['ichimoku_base']       = ich.ichimoku_base_line()
    df['ichimoku_a']          = ich.ichimoku_a()
    df['ichimoku_b']          = ich.ichimoku_b()
    return df


def ma_crossover(df, close, slow, fast):
    df['close'] = close
    df['slow'] = slow
    df['fast'] = fast
    df['signal'] = 0.0
    df.loc[df['slow'] > df['fast'], 'signal'] = 1.0
    df['position'] = df['signal'].diff()
    return df


def gauge(avg_pos):
    if 0.6 <= avg_pos <= 1.0:
        return 'Strong Buy'
    elif 0.2 <= avg_pos < 0.6:
        return 'Buy'
    elif -0.2 <= avg_pos < 0.2:
        return 'Neutral'
    elif -0.6 <= avg_pos < -0.2:
        return 'Sell'
    elif -1.0 <= avg_pos < -0.6:
        return 'Strong Sell'
    return 'Neutral'


def ema_strategy(data):
    pairs = [
        ('ema_5', 'ema_200'), ('ema_13', 'ema_100'), ('ema_13', 'ema_200'),
        ('ema_21', 'ema_55'), ('ema_21', 'ema_100'), ('ema_21', 'ema_200'),
        ('ema_55', 'ema_100'), ('ema_55', 'ema_200'), ('ema_100', 'ema_200'),
        ('ema_5', 'ema_13'),
    ]
    actions = []
    for slow_col, fast_col in pairs:
        df = data['timestamp'].to_frame()
        ma_crossover(df, data['close'], data[slow_col], data[fast_col])
        pos = df['position'].iloc[-1]
        if pos == 1.0:
            actions.append('Buy')
        elif pos == -1.0:
            actions.append('Sell')
        else:
            actions.append('Neutral')
    return actions


def ichimoku_strategy(data):
    actions = []
    close = data['close'].astype(float)

    # Conversion-Base signal
    conv = data['ichimoku_conversion']
    base = data['ichimoku_base']
    if conv.iloc[-1] > base.iloc[-1] and close.iloc[-1] > data['ichimoku_a'].iloc[-1]:
        actions.append('Buy')
    elif conv.iloc[-1] < base.iloc[-1]:
        actions.append('Sell')
    else:
        actions.append('Neutral')

    # Price-Base signal
    if close.iloc[-1] > base.iloc[-1]:
        actions.append('Buy')
    elif close.iloc[-1] < base.iloc[-1]:
        actions.append('Sell')
    else:
        actions.append('Neutral')

    return actions


def compute_trading_signal(btc_data):
    """Get the overall trading signal from EMA + Ichimoku strategies."""
    if btc_data is None or len(btc_data) < 200:
        return 'Neutral', 0.0

    data = btc_data.tail(250).copy().reset_index(drop=True)
    data = fill_trends(data)

    all_actions = ema_strategy(data) + ichimoku_strategy(data)
    action_scores = []
    for a in all_actions:
        if a == 'Buy':
            action_scores.append(1)
        elif a == 'Sell':
            action_scores.append(-1)
        else:
            action_scores.append(0)

    avg_pos = np.mean(action_scores) if action_scores else 0
    return gauge(avg_pos), avg_pos


def make_gauge_figure(action, avg_pos):
    """Create a gauge chart for the trading signal."""
    values = [50, 10, 10, 10, 10, 10]
    labels = [" ", "STRONG SELL", "SELL", "NEUTRAL", "BUY", "STRONG BUY"]
    marker_colors = [
        'rgb(255,255,255)', 'rgb(255,0,0)', 'rgb(255,123,138)',
        'rgb(209,211,220)', 'rgb(84,189,254)', 'rgb(0,140,251)'
    ]

    # Map action to angle on gauge
    action_angles = {
        'Strong Sell': 162, 'Sell': 126, 'Neutral': 90, 'Buy': 54, 'Strong Buy': 18
    }
    angle = action_angles.get(action, 90)

    fig = go.Figure()
    fig.add_trace(go.Pie(
        values=values, labels=labels,
        marker=dict(colors=marker_colors, line=dict(width=0)),
        pull=0.05, domain=dict(x=[0, 1], y=[0, 1]),
        name="Gauge", hole=0.3, direction="clockwise",
        rotation=90, showlegend=False, textinfo='label',
        textposition='inside', textfont_size=11,
        hoverinfo='label'
    ))

    # Add needle using a shape line (compatible with newer plotly)
    r = 0.35
    x_head = 0.5 + r * cos(radians(angle))
    y_head = 0.5 + r * sin(radians(angle))
    fig.add_shape(
        type='line', x0=0.5, y0=0.5, x1=x_head, y1=y_head,
        xref='paper', yref='paper',
        line=dict(color='rgb(46,60,88)', width=3),
    )
    # Needle center dot
    fig.add_shape(
        type='circle', x0=0.48, y0=0.48, x1=0.52, y1=0.52,
        xref='paper', yref='paper',
        fillcolor='rgb(46,60,88)', line=dict(color='rgb(46,60,88)'),
    )

    fig.update_layout(
        title=dict(text=f"Trading Signal: {action}", x=0.5),
        height=350, margin=dict(t=50, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


# ============================================================
# Dashboard
# ============================================================

colors = {"background": 'rgb(46, 60, 88)', 'text': '#fafafa'}

# Start the predictor
predictor = BaselinePredictor(days_for_training=3, look_back=15, batch_size=32, epochs=25)

print("=" * 60)
print("BTC Predictor - Polymarket 5-Min Up/Down Dashboard")
print("=" * 60)
print("  LSTM + Order Book + Polymarket Ensemble")
print("  Fast: RSI(5), MACD(5,13,4), BB(10), EMA(5,13)")
print("  Slow: RSI(14), MACD(12,26,9), BB(20), EMA(12,26)")
print("\nStarting (fetching 3 days at 3 timeframes + training)...")
print("Dashboard: http://127.0.0.1:8050 (ready in 3-5 minutes)\n")

predictor.run()

# Build Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    # Header
    dbc.Row(dbc.Col(html.H1(
        "BTC 5-Min Up/Down Predictor (Polymarket)",
        style={'color': colors['text'], 'textAlign': 'center', 'padding': '20px'}
    )), style={'backgroundColor': colors['background'], 'marginBottom': '20px'}),

    # Row 1: Price chart + direction prediction
    dbc.Row([
        dbc.Col([
            html.H4("Live BTC Price & Predictions"),
            dcc.Graph(id='btc-graph'),
            dcc.Interval(id='btc-update', interval=30 * 1000, n_intervals=0),
        ], width=8),
        dbc.Col([
            html.H4("Ensemble Prediction"),
            html.Div(id='direction-display', style={
                'textAlign': 'center', 'padding': '20px', 'borderRadius': '10px',
                'marginBottom': '15px'
            }),
            html.H5("Price Forecast"),
            dcc.Graph(id='percent-change', style={'height': '200px'}),
            html.Hr(),
            html.H5("Signal Sources"),
            html.Div(id='signal-sources'),
        ], width=4),
    ], className='mb-4'),

    # Row 2: Metrics + Gauge
    dbc.Row([
        dbc.Col([
            html.H4("Past 5 Hours"),
            html.Div(id='accuracy-recent', className='mb-2'),
            html.Div(id='smape-recent', className='mb-2'),
            html.Div(id='noise-recent', className='mb-2'),
        ], width=3),
        dbc.Col([
            html.H4("Past 3 Days"),
            html.Div(id='accuracy-training', className='mb-2'),
            html.Div(id='smape-training', className='mb-2'),
            html.Div(id='noise-training', className='mb-2'),
        ], width=3),
        dbc.Col([
            html.H4("Model Status"),
            html.Div(id='model-status'),
        ], width=2),
        dbc.Col([
            html.H4("Trading Signal"),
            dcc.Graph(id='trading-gauge', style={'height': '350px'}),
        ], width=4),
    ], className='mb-4'),
    dcc.Interval(id='metrics-update', interval=60 * 1000, n_intervals=0),

], fluid=True)


# --- Callbacks ---

@app.callback(
    Output('btc-graph', 'figure'),
    [Input('btc-update', 'n_intervals')]
)
def update_btc_graph(n):
    fig = go.Figure()
    if predictor.has_model():
        df = predictor.history(120)
        if df is not None:
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['actual'],
                name='Actual', line=dict(color='#636EFA')
            ))
            fig.add_trace(go.Scatter(
                x=df['time'], y=df['prediction'],
                name='Prediction', line=dict(color='#EF553B', dash='dash')
            ))

            # Add live predictions
            preds = predictor.predictions_so_far()
            if preds is not None and len(preds) > 0:
                fig.add_trace(go.Scatter(
                    x=preds['time'], y=preds['prediction'],
                    name='Live Predictions', mode='markers',
                    marker=dict(color='#00CC96', size=8)
                ))

    fig.update_layout(
        xaxis_title='Time', yaxis_title='Price (USD)',
        template='plotly_white',
        margin=dict(t=30, b=30),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        yaxis=dict(tickformat='$,.0f'),
    )
    return fig


@app.callback(
    Output('percent-change', 'figure'),
    [Input('metrics-update', 'n_intervals')]
)
def update_pct_change(n):
    pred = predictor.predict()
    if pred is None:
        fig = go.Figure(go.Indicator(mode="number", value=0, number={'prefix': "$", 'valueformat': ",.2f"}))
        fig.update_layout(height=230, margin=dict(t=30, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
        return fig

    h = predictor.history(1)
    ref = h['actual'].values[0] if h is not None and len(h) > 0 else pred

    fig = go.Figure(go.Indicator(
        mode="number+delta",
        value=pred,
        number={'prefix': "$", 'valueformat': ",.2f"},
        delta={'position': "top", 'reference': float(ref), 'relative': False, 'valueformat': ",.2f"},
    ))
    fig.update_layout(
        height=180, margin=dict(t=30, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


@app.callback(
    Output('direction-display', 'children'),
    [Input('metrics-update', 'n_intervals')]
)
def update_direction(n):
    direction, confidence = predictor.latest_direction()
    if direction is None:
        return html.Div([
            html.H2("WAITING...", style={'color': 'gray'}),
            html.P("Model training in progress")
        ])

    color = '#00CC96' if direction == "UP" else '#EF553B'
    arrow = "^" if direction == "UP" else "v"
    return html.Div([
        html.H1(f"{arrow} {direction}", style={
            'color': color, 'fontSize': '48px', 'fontWeight': 'bold', 'margin': '0'
        }),
        html.H3(f"{confidence:.1f}% confidence", style={'color': color, 'margin': '5px 0'}),
        html.P("5-minute ensemble prediction", style={'color': '#666', 'fontSize': '12px'}),
    ])


@app.callback(
    Output('signal-sources', 'children'),
    [Input('metrics-update', 'n_intervals')]
)
def update_signals(n):
    direction, confidence = predictor.latest_direction()
    if direction is None:
        return html.P("Waiting for first prediction...", style={'color': 'gray'})

    items = []
    pred = predictor.predict()
    if pred:
        data = predictor.get_data()
        if data is not None:
            curr = float(data['open'].iloc[-1])
            pct = (pred - curr) / abs(curr) * 100
            lstm_dir = "UP" if pct > 0 else "DOWN"
            items.append(html.P(f"LSTM: {lstm_dir} ({pct:+.3f}%)",
                        style={'color': '#00CC96' if lstm_dir == 'UP' else '#EF553B'}))

    ob = predictor.latest_ob()
    if ob:
        ob_dir = "UP" if ob['imbalance'] > 0 else "DOWN"
        items.append(html.P(f"Order Book: {ob_dir} (imb={ob['imbalance']:+.3f})",
                    style={'color': '#00CC96' if ob_dir == 'UP' else '#EF553B'}))
    else:
        items.append(html.P("Order Book: N/A", style={'color': 'gray'}))

    pm = predictor.latest_pm()
    if pm:
        items.append(html.P(f"Polymarket: Up={pm['up_prob']:.1%}",
                    style={'color': '#00CC96' if pm['up_prob'] > 0.5 else '#EF553B'}))
    else:
        items.append(html.P("Polymarket: N/A", style={'color': 'gray'}))

    return html.Div(items)


@app.callback(
    Output('model-status', 'children'),
    [Input('metrics-update', 'n_intervals')]
)
def update_status(n):
    if predictor.has_model():
        preds = predictor.predictions_so_far()
        n_preds = len(preds) if preds is not None else 0
        return html.Div([
            html.P(f"Model: Active", style={'color': 'green', 'fontWeight': 'bold'}),
            html.P(f"Predictions made: {n_preds}"),
        ])
    return html.P("Model: Training...", style={'color': 'orange', 'fontWeight': 'bold'})


# Metrics callbacks
@app.callback(
    [Output('accuracy-recent', 'children'),
     Output('smape-recent', 'children'),
     Output('noise-recent', 'children')],
    [Input('metrics-update', 'n_intervals')]
)
def update_recent_metrics(n):
    style = {'fontSize': '22px', 'color': colors['background']}
    acc = predictor.accuracy(step=15, offset=300)
    smp = predictor.smape(offset=300)
    nse = predictor.noise(offset=300)
    return (
        html.Span(f'Accuracy: {acc:.2%}', style=style),
        html.Span(f'SMAPE: {smp:.4f}%', style=style),
        html.Span(f'Noise: ${nse:.2f}', style=style),
    )


@app.callback(
    [Output('accuracy-training', 'children'),
     Output('smape-training', 'children'),
     Output('noise-training', 'children')],
    [Input('metrics-update', 'n_intervals')]
)
def update_training_metrics(n):
    style = {'fontSize': '22px', 'color': colors['background']}
    acc = predictor.accuracy(step=15)
    smp = predictor.smape()
    nse = predictor.noise()
    return (
        html.Span(f'Accuracy: {acc:.2%}', style=style),
        html.Span(f'SMAPE: {smp:.4f}%', style=style),
        html.Span(f'Noise: ${nse:.2f}', style=style),
    )


@app.callback(
    Output('trading-gauge', 'figure'),
    [Input('metrics-update', 'n_intervals')]
)
def update_trading_signal(n):
    data = predictor.get_data()
    action, avg_pos = compute_trading_signal(data)
    return make_gauge_figure(action, avg_pos)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)
