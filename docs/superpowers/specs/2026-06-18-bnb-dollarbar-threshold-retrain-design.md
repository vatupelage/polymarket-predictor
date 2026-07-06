# bnb_15m dollar-bar threshold recalibration + retrain

**Date:** 2026-06-18
**Status:** Approved (spec); bnb_15m paused live until t2 validated

## Background / finding

Live audit of dollar-bar cadence (measured from Binance volume) showed bnb_15m is
the lone miscalibrated bot:

| bot | threshold | sec/bar (now) | 10-bar lookback | vs window |
|-----|-----------|---------------|-----------------|-----------|
| btc_5m  | $250k | ~19s  | ~3.2 min  | 0.6x (300s)  — OK |
| eth_15m | $150k | ~21s  | ~3.5 min  | 0.2x (900s)  — OK (same cadence as btc) |
| bnb_15m | $125k | ~105s | ~17.5 min | 1.2x (900s)  — MISCALIBRATED |

**Drift hypothesis was REFUTED.** BNB 24h $-volume actually *rose* +39% from the
training window ($0.07B -> $0.10B median daily); bars form *faster* now (105s) than
in training (147s). So there is **no train/serve OOD skew from volume drift** — the
cadence is consistent (slightly better now).

**The real issue:** the $125k threshold was too coarse for BNB's liquidity *from the
start*. Even in training it produced ~147s bars -> the 10-bar feature window spanned
~24 min (1.6x the 15-min window). At decision time (60s into the 900s window) <1 bar
forms in-window, so bnb's features are almost entirely pre-window. This is a baked-in
feature-resolution flaw, not a drift bug — train and serve are consistent, so the
retrain is an **experiment** to test whether finer bars improve the representation,
NOT a fix for a confirmed skew.

## Goal

Re-fit bnb_15m on a dollar-bar threshold sized to BNB liquidity so bar cadence
matches btc/eth (~20s/bar) and features capture in-window dynamics. Deploy ONLY if
it beats the incumbent ($125k) out-of-sample.

## 1. Threshold selection (volume-targeted rule)

- Principle: `threshold = current_$volume_per_sec * target_sec_per_bar`,
  target ~= 20s/bar (match btc/eth).
- BNB now: ~$1,193/s * 20s ~= **$24k** (vs $125k, ~5x finer).
- At $24k: ~10 bars ~= 3.4 min lookback, ~3 bars forming inside the first 60s —
  mirrors btc's healthy in-window + run-up blend.
- Use a rolling median of $-vol (BNB dailies swing $0.04B-$0.12B, ~3x) — don't pin
  to a single day.

## 2. Data + features

- Rebuild BNB dollar bars from historical aggTrades/klines at $24k over the same
  ~61-day training span (~5,850 15m windows).
- Regenerate the SAME 8 features: drift_pct, secs_to_close, duration, ret, log_ret,
  volatility, mean_price, rvol. Same window_s=900, monitor_start_s=840 (1-min-in).
- Sanity: new `duration` feature should center ~20s (was ~105-147s).

## 3. Train

- Same pipeline: XGBoost + isotonic calibration (matches incumbent).
- Output: `models/db_ptb_bnb_15m_t2.joblib`.

## 4. Validation gate (decision criterion)

- Purged walk-forward OOS split (train early -> test later; embargo around window
  boundaries to prevent leakage).
- Head-to-head new vs incumbent ($125k) on identical held-out windows:
  - calibration: log-loss / Brier
  - decision quality: net-of-fee per-$1 edge `((1/ask-1) | -1) - 0.07*(1-ask)` + win rate
  - bootstrap CI / DSR on the edge DIFFERENCE — must clear zero, not just point-beat.
- **Deploy only if new beats incumbent OOS by a meaningful, CI-backed margin.**
  Tie -> keep incumbent (simpler).

## 5. Prevent recurrence (all symbols)

- Weekly calibration audit: compute live sec/bar; flag retrain if it strays >~2x
  from trained cadence.
- Re-run the threshold rule monthly. btc/eth currently pass; only bnb retrains now.
- Run the same audit on sol/xrp/doge before any future deployment.

## 6. Deploy

- Hot-swap bundle path in `live_env/bnb_15m.env`, dry-run smoke (bars should warm in
  <~3 min now, not ~17), then go live. Keep incumbent as A/B control until proven.

## Caveats

- Sharpens representation; does NOT manufacture edge. If bnb's true edge is ~0 after
  fees, finer bars won't save it — the validation gate exists to catch exactly that.
- Train/serve already consistent -> real chance t2 does NOT beat incumbent; that's an
  acceptable outcome (keep incumbent, conclude threshold wasn't the bottleneck).
- Underlying edge still statistically unconfirmed (see [[multibot-live-deployment]],
  [[polymarket-crypto-taker-fee]]).

## Validation result (2026-06-18) — GATE FAILED, KEEP INCUMBENT

Trained t2 at **$24,000** (LightGBM winner, isotonic) on the first 70% of the span
(train/OOS cutoff ms = 1778869788799). Scored t2 vs incumbent on the held-out OOS slice
via `train/validate_threshold.py` (paired bootstrap on per-window log-loss, windows aligned
by ws_id intersection, 1553 common windows).

**Threshold note (correcting §"Background"):** the spec said $24k from *current* June volume.
On the **full 61-day training span**, $24k gives per-day median bar duration ~21s (target);
$125k gives ~126s — so $24k is the right finer threshold and the coarse-feature flaw at $125k
is confirmed. (`calibrate_threshold.py` samples only the last 24h, which were two anomalous
5-12x volume-spike days — May 30/31 — so it misleadingly showed $24k≈3s there; the full-span
analysis is authoritative.)

| comparison | t2 $24k logloss | inc $125k logloss | acc t2 / inc | per-window diff (t2−inc) | verdict |
|---|---|---|---|---|---|
| vs deployed t1 (full-span, **in-sample** on OOS) | 0.790 | 0.683 | 0.560 / 0.580 | +0.108, CI [+0.053,+0.171] | FAIL |
| **vs fair t1 (same 70% slice, truly OOS)** | 0.790 | 0.720 | 0.560 / 0.573 | **+0.071, CI [+0.004,+0.141]** | **FAIL** |

The fair control (re-fit $125k on the identical 70% train slice) removes the deployed t1's
in-sample advantage: the gap narrows from +0.108 to +0.071 but the CI still clears zero on the
wrong side — **finer $24k bars do NOT improve and marginally *degrade* OOS prediction quality.**

**No-look-ahead:** train slice is strictly `ts <= cutoff`, OOS is strictly `ts > cutoff`, and the
OOS harness rebuilds dollar bars cold from the cutoff — so no train-side bar state crosses the
split. The plan's "≥1-window embargo" is thus satisfied structurally (cold bar rebuild) rather than
by an explicit time gap.

**Conclusion:** hypothesis refuted. The coarse-feature-resolution concern was real in principle
but was NOT the bottleneck. For a ~14-min-ahead window prediction (decision at 1-min-in), the
longer-horizon context in the coarse ~126s bars carries more signal than fine ~20s-bar recent
microstructure. **Keep the incumbent $125k (`db_ptb_bnb_15m_t1.joblib`). Do not deploy t2.**
Task 8 (deploy) is skipped. The threshold was not the lever; bnb_15m's weak live edge must be
addressed (if at all) by a different mechanism, not bar granularity.
