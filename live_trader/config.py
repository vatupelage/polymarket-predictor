"""Runtime config for the live Polymarket bot.

Secrets are loaded from environment variables (or a .env file sitting next to
run_live_bot.py). Nothing is hardcoded — rotate any key that ever hits disk in
plaintext elsewhere.
"""

import os
from dataclasses import dataclass


@dataclass
class BotConfig:
    clob_host: str
    chain_id: int
    private_key: str
    funder_address: str
    api_key: str
    api_secret: str
    api_passphrase: str
    signature_type: int
    stake_usdc: float
    high_conf_threshold: float
    max_conf_threshold: float
    strict_conf_threshold: float
    min_drift_pct: float
    contra_drift_enabled: bool
    extreme_price_low: float
    extreme_price_high: float
    max_market_disagreement: float
    max_slippage_bps: int
    dry_run: bool
    use_llm: bool
    llm_model: str
    llm_min_conf: float
    llm_min_edge: float
    anthropic_api_key: str | None
    enable_cash_out: bool
    cash_out_delay_sec: int
    cash_out_drop: float
    cash_out_floor: float
    direction_filter: str
    daily_profit_target: float
    hard_stop_loss: float
    max_concurrent_trades: int
    up_max_ask: float
    up_filter_crowd_indecision: bool
    up_stake_usdc: float
    up_min_ask: float
    expensive_fill_threshold: float
    strong_ptb_up_prob: float
    strong_drift_pct: float
    require_ptb_support_up: bool
    require_drift_positive_up: bool
    # Drift-noise gate: skip when |drift_pct| below the threshold AND
    # confidence is at or above the floor. Catches "high-conf trade in a
    # dead market" — the failure mode behind the 2026-05-06 $30 stake bomb
    # (drift = -0.00028%). Set noise_drift_pct = 0 to disable.
    noise_drift_pct: float
    noise_conf_floor: float
    # Mid-price gate: at high confidence, skip when the chosen side's ask
    # is below the cap. Replaces the old hardcoded `ask<0.50 & conf>12`
    # `overconfident_contra` rule with configurable thresholds. Backtest:
    # blocks 13 historical trades for net +$16.97 saved. Disable by
    # setting mid_price_cap = 0.
    mid_price_cap: float
    mid_price_conf_floor: float
    # Contra-book gate confidence ceiling. The `contra_book` rule (ask<0.40)
    # only fires when confidence is BELOW this threshold. At conf>=7 the
    # gate is anti-predictive (+$0.10-0.21/$1 EV per skip — bot beats the
    # book disagreement). Set to a large value (e.g. 100) to restore the
    # old always-fire behavior. Set to 0 to disable contra_book entirely.
    contra_book_max_conf: float
    # UP-direction-specific confidence floor (independent of high_conf_threshold).
    # Post-gate window (53 trades): UP at conf<7 has per-$1 EV of -$0.05 to
    # -$0.10; DOWN at the same conf is profitable. Asymmetric gate fires for
    # direction=="UP" AND confidence < threshold. Set to 0 to disable.
    up_high_conf_threshold: float
    # High-entry + mid-conf gate. Both 2026-05-15 $70 losses fit this bucket:
    # entry>=0.70 AND conf 7-12 -> per-$1 EV -$0.34 across 4 trades in the
    # post-gate window. Skip when chosen-side ask >= high_entry_cap AND conf
    # < high_entry_conf_floor. Set high_entry_cap=0 to disable.
    high_entry_cap: float
    high_entry_conf_floor: float
    # Entry-price band — when active, fires `entry_out_of_band` skip for any
    # trade whose chosen-side ask falls outside [entry_min, entry_max). Used
    # by the --a / --b profile flags in run_live_bot.py to enforce filter:
    #   --a : entry in [0.55, 0.70) — 80%+ W historical bucket
    #   --b : entry in [0.45, 0.55) — higher-RR bucket
    # Defaults (0, 1.0) leave the band wide open = effectively disabled.
    entry_min: float
    entry_max: float
    # Crowd-indecision flip confidence band. The flip rule (invert direction
    # when crowd ~50/50 AND chosen-side ask <= 0.50) only fires when confidence
    # is in [crowd_flip_min_conf, crowd_flip_max_conf). Outside the band the
    # crowd-indecision trigger is ignored and the original signal proceeds.
    # Backtest: at conf [7, 12) the flip is 88.9% W (+$0.62/$1); at conf <7
    # signal beats flip by +$0.30/$1; at conf >=12 sample too small.
    crowd_flip_min_conf: float
    crowd_flip_max_conf: float
    # Ask-recheck (T+8s adverse-move) gate. DISABLED by default after 2026-05-15
    # quant audit: 26 blocked trades historically, 22 (84.6%) would have won.
    # Even at realistic fill prices the gate cost +$0.19/$1 forgone per blocked
    # trade. In a binary up/down market, ask jumping up after a BUY confirms
    # the bot's prediction — gate was aborting on its own validation signal.
    # The 2% MAX_SLIPPAGE_BPS at the FOK layer already protects against fills
    # at significantly worse prices. Set ask_recheck_enabled=true to re-enable.
    ask_recheck_enabled: bool
    ask_recheck_delay_s: float
    ask_recheck_tolerance: float
    ask_recheck_timeout_s: float
    ask_recheck_max_ref_ask: float
    # Pre-trade reconciliation — at bot startup, sweep Polymarket positions API
    # for "orphan winners" (winning shares that were never recorded in
    # trade_history.jsonl, e.g. from a previous crash) and redeem them.
    # Prevents the failure mode discovered on 2026-05-18 where two winning
    # positions worth $56.73 sat unclaimed on-chain. Idempotent — re-runs on
    # every start. Disable with BOT_RECONCILE_ON_START=false.
    reconcile_on_start: bool
    # LSTM-inverted alignment gate. The bot treats LSTM as anti-predictive
    # (weight is inverted in the blend), and OOS-conditional backtest showed
    # the cleanest surviving edge is "direction disagrees with LSTM": n=102,
    # 74.5% W, E[$1]=+$0.245 (boot CI [+$0.100, +$0.392] — does not cross 0).
    # The opposite slice (direction agrees with LSTM) is -$0.090/$1 at 52% W.
    # Gate fires when direction MATCHES the LSTM call, killing the bad bucket.
    # Set BOT_LSTM_INV_GATE=true to enable.
    lstm_inv_gate_enabled: bool
    # Hour-of-day blackout. Comma-separated list of half-open ranges in the
    # bot's local-time hour (matches trade_history timestamps). Backtest:
    # hour band [18, 24) had E[$1] = -$0.020 over n=33 trades — the only
    # losing slot of the four 6-hour buckets. Format: "18-24" or
    # "0-6,18-24". Empty string disables.
    blackout_hours: str
    # Volatility regime gate — skip the whole trade when realized vol over the
    # last `vol_gate_window` price samples exceeds the rolling baseline by a
    # factor of `vol_gate_threshold`. Addresses the failure mode where the
    # bot's entire signal stack (PTB+drift+orderbook+LSTM) degrades together
    # during crash/breakout regimes — all four signals become momentum-
    # following and point the same wrong way during chop after a directional
    # move. Set vol_gate_threshold=0 to disable.
    vol_gate_threshold: float
    vol_gate_window: int           # samples for "now" realized vol (12 ≈ 60 min)
    vol_gate_baseline_window: int  # samples for rolling baseline (~3 days)
    # S5 strategy mode — activated by --s5 in run_live_bot.py. When enabled:
    #   1) Entry must be in [0.40, 0.50) ∪ (0.58, 0.75]  (excludes middle 0.50-0.58)
    #   2) Side-aligned orderbook prob must be >= 0.85
    #   3) Max 1 trade per UTC day (first match wins)
    #   4) vol_regime_break gate is BYPASSED (S5 keeps the crash-day wins)
    # Backtest (n=13/23d): 85% W, meanR +$0.92/$1, t=+3.05, perm-p=0.0004 (Bonferroni-safe),
    # TRAIN +$0.83 / TEST +$0.98, all 6 regime buckets positive, slippage -10% still +$0.82.
    s5_mode: bool
    # If true, disable the LOWER S5 band [0.40, 0.50). Only fires in (0.58, 0.75].
    # Temporary switch — first 10 live fires showed lower band ~flat while upper went 3W/0W.
    s5_disable_lower: bool
    # Fav90 late-confirm mode — activated by --fav90 in run_live_bot.py. Bypasses
    # the model entirely: each window, poll late and BUY the FAVORITE side (higher
    # top ask) when its ask is in [fav90_min_ask, fav90_max_ask] AND the PM book
    # shows >= fav90_min_ask_depth shares resting within 1 tick of the ask, AND
    # secs_to_close <= fav90_entry_max_s. Hold to resolution. $1 stake. This is a
    # deliberate -EV live test of late-favorite entry (see docs spec 2026-06-04).
    fav90_mode: bool
    fav90_entry_max_s: float    # only enter when secs_to_close <= this (late gate)
    fav90_min_ask: float        # favorite-ask band low
    fav90_max_ask: float        # favorite-ask band high
    fav90_min_ask_depth: float  # shares resting at/within 1 tick of the ask
    fav90_daily_stop: float     # daily realized-PnL kill-switch (USD, negative)
    fav90_stop_bid: float       # stop-loss: sell the held position if its bid <= this
    # Dollar-bar PTB model mode — activated by --dbmodel in run_live_bot.py. Each
    # 5-min window, a calibrated XGB/LightGBM model (Wang-thesis dollar-bar feats +
    # distance-to-strike) predicts P(BTC close > Price-to-Beat) at 60s into the
    # window (s2c=240, the leak-free decision time it was trained for) and BUYS the
    # predicted side at market, $1, hold to resolution. No price/conf gates. This is
    # a deliberate -EV live test (backtest: ~63% W but -$0.029/trade fee bleed).
    dbmodel_mode: bool
    dbmodel_path: str           # path to the trained model bundle (.joblib)
    # Asset/timeframe the dbmodel runner trades. `dbmodel_symbol` is the Binance
    # pair base (btc/eth/sol/xrp/doge/bnb) — it drives both the Binance aggTrade
    # WS stream and the Polymarket slug prefix ({symbol}-updown-{tf}-{ws}). The
    # window length and decision instant come from the model bundle (window_s /
    # monitor_start_s): 5m -> 300/240 (matches the live BTC bot), 15m -> 900/420
    # (enter 7 min before close). Defaults preserve the BTC-5m behaviour.
    dbmodel_symbol: str
    # Multi-market shared-wallet mode: when true the dbmodel live path does NOT
    # redeem winners inline. A single centralized redeemer daemon (redeemer_daemon.py)
    # owns ALL on-chain settlement for the cohort, so the per-process _tx_lock can't
    # serialize across the N bot processes sharing one wallet — one daemon = one
    # tx-sender = no "in-flight transaction limit" collisions. Default false keeps
    # the standalone BTC bot redeeming its own winners.
    dbmodel_delegate_redeem: bool
    # Cheap-entry gate (btc_5m_cheap variant). max_ask < 1.0 turns on the
    # decide-then-poll-for-a-dip path; >=1.0 keeps immediate entry-at-decision.
    dbmodel_max_ask: float
    dbmodel_entry_deadline_s: float
    # Market-buy slippage headroom. Polymarket has no true market order; buy_market
    # submits a FAK *limit* priced just above the best ask by this much, so it still
    # crosses when the (fast-flickering) BTC-5m book ticks up between read and post.
    # FAK fills at the resting maker prices up to this cap, so it is NOT overpayment.
    market_max_slippage: float
    # Market-making data collection (read-only — see MM_DATA_SPEC.md). When true,
    # every prediction spins up an MMLogger thread that snapshots full book depth,
    # the trade tape, hypothetical maker quotes, and the reward schedule for the
    # window. Places NO orders. Safe to run alongside or instead of the trader.
    mm_log_enabled: bool
    mm_log_cadence_s: float
    # Cross-leg arbitrage executor (arb_executor.py). When ask_up+ask_down < $1,
    # buy both sides for guaranteed profit. arb_dry_run=true (default) only
    # detects + logs (no orders) so you can measure before risking capital.
    # Live mode uses FOK limit orders, atomic both-legs-or-unwind, depth-capped.
    arb_enabled: bool
    arb_dry_run: bool
    arb_min_edge: float        # min (1 - ask_up - ask_down) to act, clears fees/safety
    arb_min_size: float        # min fillable shares to bother
    arb_max_size: float        # hard cap on shares per leg (risk limit)
    arb_max_usdc_per_leg: float  # dollar budget per leg — the "$1 trade" knob
    arb_poll_s: float          # scan cadence within a window
    arb_max_per_window: int    # max arb executions per 5-min window
    arb_deadline_buffer_s: float  # stop scanning this many secs before close
    arb_ws: bool               # use WebSocket push feed (fast) vs REST poll


def _load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val


def load_config(dotenv_path: str | None = None) -> BotConfig:
    if dotenv_path:
        _load_dotenv(dotenv_path)

    return BotConfig(
        clob_host=os.environ.get("POLY_CLOB_HOST", "https://clob.polymarket.com"),
        chain_id=int(os.environ.get("POLY_CHAIN_ID", "137")),
        private_key=_require("POLY_PRIVATE_KEY"),
        funder_address=_require("POLY_FUNDER_ADDRESS"),
        api_key=_require("POLY_API_KEY"),
        api_secret=_require("POLY_API_SECRET"),
        api_passphrase=_require("POLY_API_PASSPHRASE"),
        signature_type=int(os.environ.get("POLY_SIGNATURE_TYPE", "0")),
        stake_usdc=float(os.environ.get("BOT_STAKE_USDC", "1.0")),
        high_conf_threshold=float(os.environ.get("BOT_HIGH_CONF", "7.0")),
        max_conf_threshold=float(os.environ.get("BOT_MAX_CONF", "0")),
        strict_conf_threshold=float(os.environ.get("BOT_STRICT_CONF", "15.0")),
        min_drift_pct=float(os.environ.get("BOT_MIN_DRIFT_PCT", "0.05")),
        contra_drift_enabled=os.environ.get("BOT_CONTRA_DRIFT_ENABLED", "true").lower() == "true",
        extreme_price_low=float(os.environ.get("BOT_EXTREME_PRICE_LOW", "0.15")),
        extreme_price_high=float(os.environ.get("BOT_EXTREME_PRICE_HIGH", "0.85")),
        max_market_disagreement=float(os.environ.get("BOT_MAX_DISAGREE", "0.20")),
        max_slippage_bps=int(os.environ.get("BOT_MAX_SLIPPAGE_BPS", "200")),
        dry_run=os.environ.get("BOT_DRY_RUN", "false").lower() in ("1", "true", "yes"),
        use_llm=os.environ.get("BOT_USE_LLM", "false").lower() in ("1", "true", "yes"),
        llm_model=os.environ.get("BOT_LLM_MODEL", "claude-haiku-4-5"),
        llm_min_conf=float(os.environ.get("BOT_LLM_MIN_CONF", "60")),
        llm_min_edge=float(os.environ.get("BOT_LLM_MIN_EDGE", "0.03")),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        enable_cash_out=os.environ.get("BOT_ENABLE_CASH_OUT", "true").lower() in ("1", "true", "yes"),
        cash_out_delay_sec=int(os.environ.get("BOT_CASH_OUT_DELAY_SEC", "120")),
        cash_out_drop=float(os.environ.get("BOT_CASH_OUT_DROP", "0.30")),
        cash_out_floor=float(os.environ.get("BOT_CASH_OUT_FLOOR", "0.10")),
        direction_filter=os.environ.get("BOT_DIRECTION_FILTER", "").strip().upper(),
        daily_profit_target=float(os.environ.get("BOT_DAILY_PROFIT_TARGET", "0")),
        hard_stop_loss=float(os.environ.get("BOT_HARD_STOP_LOSS", "0")),
        max_concurrent_trades=int(os.environ.get("BOT_MAX_CONCURRENT", "1")),
        up_max_ask=float(os.environ.get("BOT_UP_MAX_ASK", "0.75")),
        up_filter_crowd_indecision=os.environ.get("BOT_UP_FILTER_CROWD_INDECISION", "true").lower() in ("1", "true", "yes"),
        up_stake_usdc=float(os.environ.get("BOT_UP_STAKE_USDC", os.environ.get("BOT_STAKE_USDC", "1.0"))),
        up_min_ask=float(os.environ.get("BOT_UP_MIN_ASK", "0.50")),
        expensive_fill_threshold=float(os.environ.get("BOT_EXPENSIVE_FILL_THRESHOLD", "0.85")),
        strong_ptb_up_prob=float(os.environ.get("BOT_STRONG_PTB_UP_PROB", "0.60")),
        strong_drift_pct=float(os.environ.get("BOT_STRONG_DRIFT_PCT", "0.05")),
        require_ptb_support_up=os.environ.get("BOT_REQUIRE_PTB_SUPPORT_UP", "true").lower() in ("1", "true", "yes"),
        require_drift_positive_up=os.environ.get("BOT_REQUIRE_DRIFT_POSITIVE_UP", "true").lower() in ("1", "true", "yes"),
        noise_drift_pct=float(os.environ.get("BOT_NOISE_DRIFT_PCT", "0.0005")),
        noise_conf_floor=float(os.environ.get("BOT_NOISE_CONF_FLOOR", "10.0")),
        mid_price_cap=float(os.environ.get("BOT_MID_PRICE_CAP", "0.55")),
        mid_price_conf_floor=float(os.environ.get("BOT_MID_PRICE_CONF_FLOOR", "12.0")),
        contra_book_max_conf=float(os.environ.get("BOT_CONTRA_BOOK_MAX_CONF", "7.0")),
        up_high_conf_threshold=float(os.environ.get("BOT_UP_HIGH_CONF", "7.0")),
        high_entry_cap=float(os.environ.get("BOT_HIGH_ENTRY_CAP", "0.70")),
        high_entry_conf_floor=float(os.environ.get("BOT_HIGH_ENTRY_CONF_FLOOR", "12.0")),
        entry_min=float(os.environ.get("BOT_ENTRY_MIN", "0")),
        entry_max=float(os.environ.get("BOT_ENTRY_MAX", "1.0")),
        crowd_flip_min_conf=float(os.environ.get("BOT_CROWD_FLIP_MIN_CONF", "7.0")),
        crowd_flip_max_conf=float(os.environ.get("BOT_CROWD_FLIP_MAX_CONF", "12.0")),
        ask_recheck_enabled=os.environ.get("BOT_ASK_RECHECK_ENABLED", "false").lower() in ("1", "true", "yes"),
        ask_recheck_delay_s=float(os.environ.get("BOT_ASK_RECHECK_DELAY_S", "8.0")),
        ask_recheck_tolerance=float(os.environ.get("BOT_ASK_RECHECK_TOLERANCE", "0.05")),
        ask_recheck_timeout_s=float(os.environ.get("BOT_ASK_RECHECK_TIMEOUT_S", "3.0")),
        ask_recheck_max_ref_ask=float(os.environ.get("BOT_ASK_RECHECK_MAX_REF_ASK", "0.70")),
        reconcile_on_start=os.environ.get("BOT_RECONCILE_ON_START", "true").lower() in ("1", "true", "yes"),
        lstm_inv_gate_enabled=os.environ.get("BOT_LSTM_INV_GATE", "false").lower() in ("1", "true", "yes"),
        blackout_hours=os.environ.get("BOT_BLACKOUT_HOURS", "").strip(),
        vol_gate_threshold=float(os.environ.get("BOT_VOL_GATE_THRESHOLD", "0")),
        vol_gate_window=int(os.environ.get("BOT_VOL_GATE_WINDOW", "12")),
        vol_gate_baseline_window=int(os.environ.get("BOT_VOL_GATE_BASELINE_WINDOW", "864")),
        s5_mode=os.environ.get("BOT_S5_MODE", "false").lower() in ("1", "true", "yes"),
        s5_disable_lower=os.environ.get("BOT_S5_DISABLE_LOWER", "false").lower() in ("1", "true", "yes"),
        fav90_mode=os.environ.get("BOT_FAV90_MODE", "false").lower() in ("1", "true", "yes"),
        fav90_entry_max_s=float(os.environ.get("BOT_FAV90_ENTRY_MAX_S", "100")),
        fav90_min_ask=float(os.environ.get("BOT_FAV90_MIN_ASK", "0.88")),
        fav90_max_ask=float(os.environ.get("BOT_FAV90_MAX_ASK", "0.92")),
        fav90_min_ask_depth=float(os.environ.get("BOT_FAV90_MIN_ASK_DEPTH", "10")),
        fav90_daily_stop=float(os.environ.get("BOT_FAV90_DAILY_STOP", "-10.0")),
        fav90_stop_bid=float(os.environ.get("BOT_FAV90_STOP_BID", "0.70")),
        dbmodel_mode=os.environ.get("BOT_DBMODEL_MODE", "false").lower() in ("1", "true", "yes"),
        dbmodel_path=os.environ.get("BOT_DBMODEL_PATH", "models/db_ptb.joblib"),
        dbmodel_symbol=os.environ.get("BOT_DBMODEL_SYMBOL", "btc").lower(),
        dbmodel_delegate_redeem=os.environ.get("BOT_DBMODEL_DELEGATE_REDEEM", "false").lower() in ("1", "true", "yes"),
        dbmodel_max_ask=float(os.environ.get("BOT_DBMODEL_MAX_ASK", "1.0")),
        dbmodel_entry_deadline_s=float(os.environ.get("BOT_DBMODEL_ENTRY_DEADLINE_S", "0")),
        market_max_slippage=float(os.environ.get("BOT_MARKET_MAX_SLIPPAGE", "0.05")),
        mm_log_enabled=os.environ.get("BOT_MM_LOG", "false").lower() in ("1", "true", "yes"),
        mm_log_cadence_s=float(os.environ.get("BOT_MM_LOG_CADENCE_S", "2.0")),
        arb_enabled=os.environ.get("BOT_ARB_ENABLED", "false").lower() in ("1", "true", "yes"),
        arb_dry_run=os.environ.get("BOT_ARB_DRY_RUN", "true").lower() in ("1", "true", "yes"),
        arb_min_edge=float(os.environ.get("BOT_ARB_MIN_EDGE", "0.02")),
        arb_min_size=float(os.environ.get("BOT_ARB_MIN_SIZE", "5")),
        arb_max_size=float(os.environ.get("BOT_ARB_MAX_SIZE", "50")),
        arb_max_usdc_per_leg=float(os.environ.get("BOT_ARB_MAX_USDC_PER_LEG", "1.0")),
        arb_poll_s=float(os.environ.get("BOT_ARB_POLL_S", "1.0")),
        arb_max_per_window=int(os.environ.get("BOT_ARB_MAX_PER_WINDOW", "3")),
        arb_deadline_buffer_s=float(os.environ.get("BOT_ARB_DEADLINE_BUFFER_S", "30")),
        arb_ws=os.environ.get("BOT_ARB_WS", "true").lower() in ("1", "true", "yes"),
    )
