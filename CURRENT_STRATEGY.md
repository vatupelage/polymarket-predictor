# BTC Predictor — Current Strategy

**Market:** Polymarket BTC up/down, 5-minute binary windows
**Bot entrypoint:** `predictor/run_live_bot.py` → `predictor/live_trader/bot.py`
**Strategy version:** Hybrid V3 (deployed 2026-05-22)
**Last audit:** 2026-05-22 (n=1057 pooled decisions, window 2026-04-18 → 2026-05-22)

---

## 1. How the bot works

Every 5 minutes the bot opens at most one position on the next BTC up/down window. The decision pipeline is:

1. **Signal generation** — local LSTM produces a directional probability `up_prob ∈ [0, 1]` from recent Binance BTC candles. A secondary "blend" combines it with order-book features (PTB, drift, mid-price) to produce a `confidence` score and a chosen `direction ∈ {UP, DOWN}`.
2. **Gate stack** — the candidate (direction, confidence, entry ask, market state) is run through ~15 skip gates (described in §3). If any gate fires, the trade is skipped and a counterfactual record is written to `skip_history.jsonl`.
3. **Order** — surviving candidates submit a FOK limit order on Polymarket V2 (pUSD collateral, USDC.e CTF settlement) sized to `BOT_STAKE_USDC` (currently $20).
4. **Settlement** — the 5-min window resolves on-chain. The bot redeems shares and appends to `trade_history.jsonl`.
5. **Pre-trade reconciliation** — at startup the bot queries Polymarket positions and redeems orphan winners from prior crashes (`BOT_RECONCILE_ON_START=true`).

### Counterfactual logging

Every skipped trade writes `would_have_won` and `would_have_pnl` to `skip_history.jsonl`. This is what lets us audit each gate's $-impact retroactively (§3) and is also what enabled the LSTM-inv shadow-flip analysis (§4).

---

## 2. Core parameters (as of 2026-05-22)

| Parameter | Value | Notes |
|---|---|---|
| `BOT_STAKE_USDC` | $20 | Quarter-Kelly on $200 bankroll. User reduced from $30 → $10 → $20 during V3 validation. |
| `BOT_HARD_STOP_LOSS` | $90 | Hard cumulative session stop. P90 of weekly worst losing streak. Does not auto-restart. |
| `BOT_MAX_CONCURRENT` | 1 | One open position at a time. |
| `BOT_UP_MIN_ASK / MAX_ASK` | 0.50 / 0.75 | Entry-band on UP ask. |
| `BOT_MAX_SLIPPAGE_BPS` | 200 | Order-level fill-price protection. |
| `BOT_USE_LLM` | false | See [LLM-filter anti-aligned](../../.claude/projects/-home-vidura-btcpredictor/memory/llm_filter_anti_aligned.md). Opus 4.7 backtest 2026-05-20 showed LLM-as-filter loses money. |

---

## 3. Active gate stack

All gates evaluated in source order in `bot.py`. ✅ = active, ❌ = disabled (with audit reasoning).

### ✅ `conf_too_low` — confidence floor
- **Threshold:** `BOT_STRICT_CONF=15`
- **Rule:** skip if blended confidence < 15%.
- **Status:** active since bot inception. Low-conf trades are noise.

### ✅ `lstm_inv_contra` — LSTM anti-predictive bucket
- **Threshold:** `BOT_LSTM_INV_GATE=true`, fires when chosen direction agrees with LSTM at `up_prob ≥ 0.5` (or `≤ 0.5` for DOWN).
- **Rule:** skip when bot direction agrees with the LSTM call. The LSTM is empirically anti-predictive on this market — the production blend already inverts its weight; this gate enforces the inversion at the decision boundary.
- **Quant evidence (OOS, pre-deploy):** agree-with-LSTM bucket meanR = **-$0.090/$1 at 52% W (n=58)**; disagree-with-LSTM = **+$0.245/$1 at 74.5% W (n=102)**, 95% CI [+$0.100, +$0.392] — does not cross zero. Strongest surviving conditional edge after IS→OOS decay.
- **Counterfactual on pre-gate window:** would have saved **+$273** in losses.
- **2026-05-22 audit:** regime-change test pre-vs-post gate cells essentially crossed in live window (pre p=0.018 SIG → post p=0.81 NS), but the two-sample regime-change test was NS (p=0.37). **Decision: keep gate + tripwire at n=200** for re-audit.
- **2026-05-22 shadow-flip implementation:** every `lstm_inv_contra` skip also writes `flip_direction / flip_entry_price / flip_would_have_won / flip_would_have_pnl` to `skip_history.jsonl` so we can evaluate "skip vs. flip" without trading it. Shadow analysis on 89 live skips: meanR **-$0.029**, total **$-51** → flip strategy abandoned, skip-only kept.
- **Live confirmation:** observed firing after 2026-05-22 restart, e.g. `SKIP (lstm-inv-contra): DOWN agrees with LSTM(up=0.460)`.

### ✅ `drift_noise` — micro-drift floor
- **Threshold:** `BOT_NOISE_DRIFT_PCT=0.0005`, `BOT_NOISE_CONF_FLOOR=10.0`
- **Rule:** skip when |drift| below noise floor and conf is also low. Filters trades where there is no real directional information.

### ✅ `position_open` — single-position guard
- **Threshold:** `BOT_MAX_CONCURRENT=1`
- **Rule:** never open a second concurrent position.

### ✅ `entry_out_of_band` — entry ask band
- **Threshold:** `BOT_UP_MIN_ASK=0.50`, `BOT_UP_MAX_ASK=0.75`
- **Rule:** skip when the chosen-side ask is outside the band. Outside the band trades have asymmetric payoff or are too close to 50/50 to clear costs.

### ✅ `high_entry_low_conf` — high-entry + mid-conf trap
- **Threshold:** `BOT_HIGH_ENTRY_CAP=0.70`, `BOT_HIGH_ENTRY_CONF_FLOOR=12.0`
- **Rule:** skip when chosen-side ask ≥ 0.70 AND confidence < 12.
- **Quant evidence (2026-05-15):** caught both 2026-05-15 $70 losses (entries 0.72/0.77 at conf 7.9/8.3). Gate added 2026-05-15.

### ✅ `contra_book` — order-book disagreement
- **Threshold:** `BOT_CONTRA_BOOK_MAX_CONF=7.0`
- **Rule:** skip when book contradicts our direction and confidence is below 7. Prevents low-conviction trades into a hostile book.

### ✅ `up_too_expensive` — expensive UP fill
- **Threshold:** `BOT_EXPENSIVE_FILL_THRESHOLD=0.75`
- **Rule:** skip when UP ask ≥ 0.75 (payoff doesn't justify cost).

### ✅ `crowd_indecision_contra` — crowd-flip rule
- **Threshold:** `BOT_UP_FILTER_CROWD_INDECISION=true`, flip-conf band `[7, 12)`
- **Rule:** when crowd is indecisive (mid-price near 0.5) and confidence sits in `[7, 12)`, the bot flips its direction. Historical: at conf [7, 12) the flip was **8/9 = 88.9% W (+$0.62/$1)**. Outside the band the original signal proceeds untouched.

### ✅ `expensive_fill` — runtime expensive-fill
- **Rule:** post-quote check; skip if the actual fill price exceeds expensive-fill threshold.

### ✅ `book_vanished` — safety
- **Rule:** book disappeared between decision and order. Trade aborted.

---

## 4. Disabled gates (with audit history)

### ❌ `contra_drift` — **DISABLED 2026-05-22**
- **Was:** skip when direction is contra to drift.
- **Audit (2026-05-22, n=1057 pooled):** gate skipped 123 trades with meanR **+$0.045/$1**, costing **$167** in foregone PnL. Largest BAD gate by count.
- **Re-enable criterion:** audit at n≥200 shows negative meanR.
- **Env:** `BOT_CONTRA_DRIFT_ENABLED=false`. Code-gated via `if not self.cfg.use_llm and self.cfg.contra_drift_enabled:` in `bot.py`.

### ❌ `mid_price_high_conf` — **DISABLED 2026-05-22**
- **Was:** skip when mid-price ≥ 0.55 AND conf ≥ 20.
- **Audit (2026-05-22, n=1057 pooled):** gate skipped 19 trades with meanR **+$0.46/$1**, costing **$263** in PnL — worst BAD gate by $ impact.
- **Re-enable criterion:** audit at n≥200 shows negative meanR.
- **Env:** `BOT_MID_PRICE_CAP=0`.

### ❌ `hour_blackout` — **DISABLED 2026-05-22**
- **Was:** skip during hours 18–24 local.
- **Audit (2026-05-22, n=1057 pooled):** gate skipped 19 trades with meanR **+$0.29/$1**, costing **$164** in PnL. Hours 21 and 23 inside the blackout were actively +EV.
- **Re-enable criterion:** audit at n≥200 shows negative meanR.
- **Env:** `BOT_BLACKOUT_HOURS=` (empty).

### ❌ `up_conf_too_low` — **DISABLED 2026-05-15**
- **Was:** UP-direction confidence floor at 7.
- **Audit (2026-05-15):** empirically blocked 5/5 windows in a single session and produced zero trades. Lifetime evidence weak — 95% CI for UP-conf<7 EV spanned [-$0.30, +$0.16], crossing zero. Point estimate -$0.07/$1 but not significant.
- **Re-enable criterion:** losses cluster in low-conf UP trades going forward.
- **Env:** `BOT_UP_HIGH_CONF=0`.

### ❌ `ask_moved_against` — **DISABLED 2026-05-15**
- **Was:** T+8s adverse-move re-check.
- **Audit (2026-05-15):** 26 trades blocked historically; 22 (84.6%) would have won. Even at realistic fill prices the gate cost **+$0.19/$1** forgone per blocked trade. The 2% `MAX_SLIPPAGE_BPS` at the FOK order layer already protects against meaningfully worse fills.
- **Env:** `BOT_ASK_RECHECK_ENABLED=false`.

### ❌ `llm_skip` — **DISABLED (default)**
- **Was:** LLM filter on candidate trade.
- **Audit (2026-05-20, Opus 4.7 backtest):** LLM-as-filter is **anti-aligned with the bot's edge** — using it as a filter loses money. See memory entry [`llm_filter_anti_aligned`](../../.claude/projects/-home-vidura-btcpredictor/memory/llm_filter_anti_aligned.md).
- **Env:** `BOT_USE_LLM=false`. Keep false.

### ❌ `conf_too_high`, `up_drift_negative`, `up_no_ptb_support`
- Disabled by setting thresholds to 0 / false. Either superseded by other gates or never significant.

### ❌ `BOT_DAILY_PROFIT_TARGET`
- **Was:** daily upside cap.
- **Audit:** at $30 stake / $200 bankroll, the bot's edge is roughly time-of-day constant after gates filter bad hours. Capping daily upside only forfeits +$5.85 EV per remaining trade. Hard stop ($90) caps downside; no need to cap upside on a positive-EV strategy. See `predictor/bankroll_quant_backtest.py`.
- **Env:** `BOT_DAILY_PROFIT_TARGET=0`.

---

## 5. Strategy V3 vs prior version (2026-05-22 walk-forward)

| Metric | CURRENT (pre-V3) | V3 (hybrid, deployed) |
|---|---|---|
| Walk-forward positive folds | 2 / 4 | **4 / 4** |
| meanR per $1 | +$0.050 | **+$0.139** |
| Total PnL on test window | comparable | comparable, **fewer trades** |
| Half-Kelly stake ($200 BR) | ~$7 | **~$19** |
| Skip rate | ~65% (over-filtered) | reduced |

V3 = kill 3 worst BAD gates (`contra_drift`, `hour_blackout`, `mid_price_high_conf`) + keep `lstm_inv_contra` (with shadow-flip logging). The 4-of-4 positive walk-forward folds were the decision criterion.

---

## 6. Tripwires (when to re-audit)

- **`lstm_inv_contra`:** when post-gate n ≥ 200 — re-run `gate_health_check.py` and `shadow_flip_analysis.py`. If post-gate edge has decayed to NS *and* the shadow flip flips positive, kill the gate.
- **Any re-enable:** the disabled BAD gates only come back if a fresh n≥200 audit shows negative meanR for the trades they would have blocked.
- **Stake bumps:** do not increase `BOT_STAKE_USDC` past $30 until V3 has its own 50-trade live track record.
- **LLM gate:** stays off unless a new offline backtest contradicts the 2026-05-20 finding.

---

## 7. Audit scripts (in `predictor/`)

| Script | Purpose |
|---|---|
| `quant_reeval_2026_05_22.py` | Full gate audit + univariate edge analysis + strategy comparison. Source of the V3 gate-kill list. |
| `quant_strategy_v3.py` | V3 walk-forward comparison vs current. |
| `gate_health_check.py` | Pre/post-gate agree-vs-contra cell comparison. Run this on `lstm_inv_contra` at n≥200. |
| `shadow_flip_analysis.py` | Wilson + bootstrap CI + sign-perm p-value on the lstm-inv shadow-flip fields. |
| `flip_agree_backtest.py` | The original pre-gate flip backtest that motivated shadow logging. |
| `bankroll_quant_backtest.py` | Kelly sizing + ruin-risk MC. |
