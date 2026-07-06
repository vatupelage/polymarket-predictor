# Dollar-Bar PTB Contrarian Model — Design Spec

**Date:** 2026-06-04
**Status:** Approved for implementation planning
**Target:** eu-west-1 server bot (`/home/ubuntu/btcpredictor/predictor/`)
**Supersedes for this work:** the fav90 strategy (not reused), and the legacy LSTM directional signal (removed from the live path entirely)

---

## 1. Goal

A new standalone live mode in which a **calibrated gradient-boosting model (XGBoost / LightGBM)** predicts, for every Polymarket BTC 5-min window, whether BTC will close **above or below the window's Price to Beat (PTB)**. The bot then trades **contrarian value**: it buys the model's predicted side **only when that side's PM ask is below ~0.50** (the market has priced our predicted winner as the underdog), working the order to fill as cheaply as possible. Buying a true winner cheap yields large payouts (+150% at a 0.40 fill).

The model based on Wang Dongsheng's thesis (Ch. 6): **Dollar Bars** + microstructure features + gradient boosting + `scale_pos_weight` for imbalance. The horizon and label are **retargeted** from the thesis's ~daily horizon to the 5-min PM window outcome relative to PTB.

**The LSTM is removed from the signal completely** — the new mode never imports `run_baseline`, so TensorFlow/Keras/`lstm_prob` never load and contribute exactly zero.

---

## 2. Why this is worth testing (honest framing)

Our own research (`directional_mm_no_edge`) concluded the PM price is the best predictor and naive value-betting is null. This strategy is still worth a live test for one specific, evidence-backed reason:

- **5-min adverse moves revert** (`s5_stop_loss_dead`: stop-losses killed winners 2:1 because mid-window adverse moves revert by resolution). This strategy *is* a late-reversion bet — when our side is cheap (< 0.50) mid-window, price is currently on the "wrong" side of PTB; if 5-min moves revert, it can cross back by close.
- **Profit asymmetry lowers the bar:** at a 0.40 entry you need only win **>~41%** of the time (incl. fee) to be +EV. The model does not need to beat the market everywhere — only to be **better calibrated than the price on sub-0.50 entries**.

**Expected behavior:** fires rarely (only on model/market disagreement). If the model is merely as good as the price, it will not clear the fee. The offline backtest (§7) is the go/no-go evidence.

---

## 3. The model

### 3.1 Prediction target
Binary, per window: `y = 1 if close_price > PTB else 0` (UP wins vs DOWN wins). Output is a **calibrated probability** `P_up = P(close > PTB)`.

### 3.2 Features (computed at decision time, ~s2c = 180)
- `drift_pct = (current_price − PTB) / PTB` — signed distance to strike (the dominant feature; mirrors the current production `ptb` signal)
- `secs_to_close` — time left for the price to cross/hold the strike
- **Wang dollar-bar microstructure** over the most recent completed bars:
  - `return = (close − open) / open`
  - `log_return = ln(close / open)`
  - `volatility = high − low` (intra-bar range)
  - `duration` — seconds to fill the bar (activity velocity)
  - `mean_price = (open+high+low+close)/4`
  - rolling realized volatility over the last *N* bars (scales crossing probability in the remaining time)
- Optional last-*N*-bar lags of the microstructure features behind a config knob (trees see sequence only via manual lags).

### 3.3 Dollar bars
A bar closes when cumulative traded value `Σ price·qty ≥ θ`. **θ calibrated from Binance BTCUSDT history so median bar ≈ 20–30 s** (Binance does ~$20–40B/day → θ likely ~$5–10M; Wang's $1M would close many times/sec here and is unusable). θ is a config value, fixed after calibration.

### 3.4 Training
- **Data:** reconstruct windows from Binance aggTrade history — `PTB = price at window-open ts`, `outcome = price 5 min later`. This yields thousands of windows. **Validate against our logged real windows** (`skip_history.jsonl` / `trade_history.jsonl`: `ptb`, `drift_pct`, `would_have_won`) to confirm the reconstructed labels match PM reality.
- **Imbalance:** `scale_pos_weight` (XGBoost), `class_weight="balanced"` (LightGBM).
- **Split:** strict chronological 80/20 (no shuffle, no look-ahead). Last slice of train reserved for calibration.
- **Calibration (mandatory):** isotonic (or Platt) regression fit on the held-out calibration slice so `P_up` is a true probability — required because we compare it to the PM ask-as-probability.
- **Selection:** train XGBoost and LightGBM; pick the winner by **held-out Brier score / calibrated log-loss** *and* the §7 backtest PnL. Ship one model (ensembling is future work).
- **Artifact:** `models/db_ptb.joblib` containing the model + calibrator + metadata (θ, feature list, decision-time s2c, training span, metrics).

---

## 4. Live decision & entry

### 4.1 Per-window flow (new mode, own runner)
1. Window opens at `ws` (slug `btc-updown-5m-{ws}`), closes at `ws+300`.
2. Capture **PTB** for the window (see §6 — fetch PM's published Price to Beat; fallback Binance price at `ws`).
3. At **s2c ≤ 180** (monitor start), compute the model prediction once: `P_up` → `side = UP if P_up ≥ 0.5 else DOWN`; `conf = |P_up − 0.5|·2`. Lock `side` for the window.
4. If `conf < min_conf` → **skip** (record skip, reason `dbmodel_low_conf`).

### 4.2 Work-the-order (buy low)
Poll the chosen side's ask every **`poll_s` (5 s)** from s2c=180 down to the deadline:
- **Eligible** when `ask < max_entry_ask` (0.50) **and** `edge = P_side − ask > fee_buffer`.
- **Buy (market $1)** when `ask ≤ target_entry_ask` (0.45) **or** when `s2c ≤ deadline_s` (~20 s) and still eligible (take best available under the ceiling).
- If the ask **never** drops below `max_entry_ask` before the deadline → **skip** the window (record skip, reason `dbmodel_no_cheap_entry`).
- Log every poll (ask, P_side, edge, eligible) to `dbmodel_log.jsonl`.

`P_side` is fixed at monitor start; `ask` is re-read live, so `edge` updates each poll. Positive edge is structurally guaranteed once `ask < 0.50 < P_side`.

### 4.3 Hold & risk
- Hold to resolution (no stop-loss — optional stopping can't add edge; harness leaves room to add one later).
- `−$10 / UTC-day` realized-PnL kill-switch (reuse `_daily_pnl`).
- One trade attempt per window.

---

## 5. Components / files

All new live units are pure and unit-tested where possible; network and model I/O isolated behind thin wrappers.

| File | Responsibility |
|---|---|
| `live_trader/dollar_bars.py` | Pure `DollarBarBuilder.add_trade(price, qty, ts) -> bar?` (emits a bar dict on θ crossing) **+** a separate Binance aggTrade WS client that feeds it and pushes bars into a thread-safe ring buffer. |
| `live_trader/db_features.py` | Pure: `build_features(bars, drift_pct, secs_to_close) -> feature_vector` (Wang's 5 + rolling vol + optional lags + drift + s2c). |
| `live_trader/db_model.py` | Loads `db_ptb.joblib`; `predict(features) -> P_up` (calibrated). |
| `live_trader/db_decision.py` | Pure gate (mirrors `fav90_decision`): given `P_up`, side asks, `target/max_entry_ask`, `min_conf`, `fee_buffer` → `{side, ask, edge, conf, eligible, buy_now}`. |
| `train/train_db_model.py` | Offline: build bars from Binance history, label vs reconstructed PTB, train XGB+LGBM + calibrate, chronological eval, **backtest the buy-low strategy (§7)**, save the winning `.joblib`. |
| `tools/fetch_binance_aggtrades.py` | Offline: download historical BTCUSDT aggTrades for training. |
| `live_trader/config.py` (edit) | New `BOT_DB_*` fields + loaders. |
| `live_trader/bot.py` (edit) | New `_execute_dbmodel_trade` branch (work-the-order buy-low loop, record, hold to resolution); dispatch in `on_prediction`. |
| `run_live_bot.py` (edit) | New `dbmodel` profile + `--dbmodel` flag + `_run_dbmodel_mode` runner that starts the Binance WS thread, loads the model, walks every window, and **early-returns before `import run_baseline`** (no LSTM/TF). |
| `tests/test_dollar_bars.py`, `tests/test_db_features.py`, `tests/test_db_decision.py` | Unit tests for the pure units. |

---

## 6. Data flow (live)

```
Binance aggTrade WS (btcusdt@aggTrade)
   │  {p, q, T} per trade
   ▼
DollarBarBuilder  ── emits bar on Σ p·q ≥ θ ──►  thread-safe ring buffer (last M bars)
                                                       │
per window @ s2c≤180:  PTB (PM published / Binance@ws) │ + live price (last aggTrade)
                                                       ▼
                            db_features ─► db_model (P_up) ─► db_decision (vs PM asks)
                                                       │ buy_now?
                                                       ▼
                            work-the-order $1 market buy ─► hold to resolution
```

---

## 7. Offline validation gate (go/no-go before/while live)

`train_db_model.py` ends by **backtesting the actual buy-low strategy** on the held-out chronological slice: for each test window, if the model's side ask was < 0.50, simulate a fill at the realized low (bounded by target/ceiling), settle vs PTB, net the dynamic taker fee (`0.07·p·(1−p)` per share; selling not applicable — hold to resolution). Report: trades fired, win%, total PnL, mean/trade, t-stat, and a permutation null. This is the evidence on whether the contrarian edge clears the fee. (User has chosen to wire to live directly; this backtest still runs as part of training and is the honest scoreboard.)

---

## 8. Config (env vars, all overridable)

| Env | Default | Meaning |
|---|---|---|
| `BOT_DBMODEL_MODE` | `false` | enable the mode |
| `BOT_DB_MODEL_PATH` | `models/db_ptb.joblib` | model artifact |
| `BOT_DB_THRESHOLD_USD` | *(calibrated)* | dollar-bar size θ |
| `BOT_DB_MONITOR_START_S` | `180` | start polling (s before close) |
| `BOT_DB_TARGET_ASK` | `0.45` | aspirational fill price |
| `BOT_DB_MAX_ASK` | `0.50` | hard entry ceiling |
| `BOT_DB_DEADLINE_S` | `20` | take best ≤ ceiling by here |
| `BOT_DB_POLL_S` | `5` | poll cadence |
| `BOT_DB_MIN_CONF` | `0.10` | min `|P−0.5|·2` (≈ P≥0.55) |
| `BOT_DB_FEE_BUFFER` | `0.01` | min edge over ask |
| `BOT_DB_STAKE` | `1.0` | $ per trade |
| `BOT_DB_DAILY_STOP` | `-10.0` | UTC-day kill-switch |

---

## 9. Known risks & verification items

1. **Efficiency / fee floor** — a value gate can only filter to +EV windows; if the model isn't better-calibrated than the price on sub-0.50 entries, it loses. The §7 backtest is the arbiter.
2. **PTB source / basis risk** — *verify before training*: what reference does PM's BTC-5m market settle on, and does its Price to Beat equal the Binance open price? Use PM's **published PTB** for the live strike and for label fidelity; reconstructed Binance labels are validated against logged `ptb`/`would_have_won`. Document any feed mismatch.
3. **Regime drift** (`directional_mm_no_edge`: the LSTM edge flipped sign in a week) — static model; refresh via a deliberate retrain, not an autonomous loop.
4. **Fill risk** — buying the underdog (< 0.50) is generally *easier* to fill than the fav90 favorite (the cheap side has resting asks), reducing the FAK run-away problem; still log fills/kills.
5. **Dependency** — confirm `xgboost` + `lightgbm` installed in the server venv `/mnt/data/s5venv`; add to `requirements-live.txt`.

---

## 10. Out of scope (YAGNI)

- Model ensembling, online retraining, sentiment/on-chain features (Wang found them null), LSTM/sequence models, stop-loss, variable sizing. The legacy directional/LSTM code stays on disk (recoverable) but is never executed in this mode.
