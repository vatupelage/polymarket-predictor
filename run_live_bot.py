#!/usr/bin/env python3
"""Entrypoint: run the v4 predictor with a live Polymarket HIGH-same bot.

Usage:
    python3 -u predictor/run_live_bot.py            # default, all .env settings
    python3 -u predictor/run_live_bot.py --a        # PROFILE A (combined entry filter)

Profile flag overrides these env vars before .env is loaded:
    --a : BOT_ENTRY_MIN=0.50, BOT_ENTRY_MAX=0.75,
          BOT_HIGH_CONF=0, BOT_MAX_CONF=0  (no conf gate — entry band only)

Env (via predictor/.env or shell):
    POLY_PRIVATE_KEY         EOA private key used for order signing
    POLY_FUNDER_ADDRESS      Wallet address that holds USDC / receives shares
    POLY_API_KEY             CLOB L2 api key
    POLY_API_SECRET          CLOB L2 api secret
    POLY_API_PASSPHRASE      CLOB L2 api passphrase
    POLY_SIGNATURE_TYPE      0 for EOA (default), 1/2 for proxy wallets
    BOT_STAKE_USDC           notional USDC per trade (default 1.0)
    BOT_HIGH_CONF            HIGH confidence threshold, in % (default 7.0)
    BOT_DRY_RUN              "true" to skip real order submission
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Profile A — unified entry-band filter (no conf gate).
# Backtest on 165 lifetime trades / 15 days, with full current gate stack applied:
#   104 trades / 15 days = 6.9 trades/day
#   68.3% W (Wilson 95% CI [58.7%, 76.5%])
#   E[$1] = +$0.162 (bootstrap 95% CI [+$0.005, +$0.311])
#   Daily EV per $1 staked: +$1.12
#   Walk-forward: 3 of 4 expanding folds positive; 2 of 3 3-fold CV positive
#   97.8% bootstrap probability EV > 0
# Earlier Profile B (conf [7,12) × entry [0.45,0.55)) was dropped:
#   entry [.45,.55) alone is +$0.012/$1 (break-even); the 70% W headline came
#   from the conf overlay being a 1-of-4-buckets cherry-pick, not real edge.
PROFILES = {
    "a": {
        "name": "A — UNIFIED ENTRY BAND",
        "tagline": "entry [0.50, 0.75), no conf gate — robust edge",
        "backtest": "68.3pct W, +$0.162/$1 (n=104 post-gate, ~6.9 trades/day, daily EV +$1.12/$1)",
        "env": {
            "BOT_HIGH_CONF": "0",
            "BOT_MAX_CONF": "0",
            "BOT_ENTRY_MIN": "0.50",
            "BOT_ENTRY_MAX": "0.75",
        },
    },
    # S5 — OB-85 EXTREME-ENTRY. 1 trade per UTC day, sequential first-match.
    # Excludes middle entry band [0.50, 0.58] (the historically-losing tier).
    # Requires side-aligned orderbook >= 0.85 (strong same-side book consensus).
    # Bypasses vol_regime_break gate (S5 keeps crash-day wins like 05-19, 05-20, 05-22).
    # All other gates (contra_book, drift_noise, etc.) remain active.
    # ARB — cross-leg arbitrage. Dry-run MEASUREMENT by default (detects + logs,
    # places NO orders) so you can measure the real fillable edge before risking
    # capital. To go live: set BOT_ARB_DRY_RUN=false in the env (then it uses FOK
    # limit orders, atomic both-legs-or-unwind, depth-capped — the safeguards that
    # the May-22 market-order loss lacked). Directional trading is left OFF.
    "arb": {
        "name": "ARB — CROSS-LEG ARBITRAGE (LIVE, 5-share min)",
        "tagline": "scan ask_up+ask_down<1; FOK atomic both-legs; LIVE at exchange-min 5 sh/leg",
        "backtest": "77 episodes/6d shadow; real but execution-critical (see May-22 -$25)",
        "env": {
            "BOT_ARB_ENABLED": "true",
            "BOT_ARB_DRY_RUN": "false",         # LIVE — places real FOK orders
            # Polymarket min order = 5 shares/leg. Trade exactly that (smallest
            # valid trade): cap size at 5 sh and fund $5/leg so 5 sh is affordable
            # at any price. Cost ~5*(ask_up+ask_down) ≈ $5/pair.
            "BOT_ARB_MAX_SIZE": "5",            # cap at 5 shares = exchange minimum
            "BOT_ARB_MAX_USDC_PER_LEG": "5.0",  # afford 5 sh up to ~$1 price
            "BOT_ARB_MIN_SIZE": "5",            # exchange min (also read from book)
            "BOT_ARB_MAX_PER_WINDOW": "1",      # 1 arb per 5-min window for now
            # WS push is now PRIMARY: arb_ws applies price_change deltas so
            # top-of-book stays live between snapshots (the freeze that forced
            # REST poll is fixed). Real-time book = no 1s staleness. REST poll
            # remains the automatic fallback if websockets is unavailable.
            "BOT_ARB_WS": "true",
            "BOT_ARB_POLL_S": "1.0",            # fallback poll cadence only
            "BOT_DRY_RUN": "true",              # keep directional trades OFF
            "BOT_MM_LOG": "true",               # collect book/tape alongside
        },
    },
    # MM — read-only market-making data collection (MM_DATA_SPEC.md). Logs full
    # book depth, the trade tape, hypothetical maker quotes, and the reward
    # schedule for every window. Forces dry-run so NO orders are ever placed.
    "mm": {
        "name": "MM — DATA COLLECTION (read-only, no trades)",
        "tagline": "logs book/tape/sim-quotes/rewards per window; dry-run forced",
        "backtest": "n/a — collecting data to backtest liquidity provision",
        "env": {
            "BOT_MM_LOG": "true",
            "BOT_DRY_RUN": "true",       # never place an order in MM mode
            "BOT_MM_LOG_CADENCE_S": "2.0",
        },
    },
    "s5": {
        "name": "S5 — OB-85 EXTREME-ENTRY (TEST MODE, UPPER-ONLY)",
        "tagline": "entry (0.58,0.75] only, lower band [0.40,0.50) DISABLED, ob_side>=0.85, $1 stake",
        "backtest": "85pct W, +$0.923/$1 (n=13/23d, t=+3.05, perm-p=0.0004, TR/TE both +$0.8+)",
        "env": {
            "BOT_S5_MODE": "true",
            "BOT_S5_DISABLE_LOWER": "true",  # TEMP: lower band off — live n=7 was flat
            "BOT_HIGH_CONF": "0",
            "BOT_MAX_CONF": "0",
            "BOT_UP_MIN_ASK": "0.40",   # gate is a no-op for upper band; left as backup
            "BOT_STAKE_USDC": "1.0",    # test mode: $1 stake until live-validated
            "BOT_UP_STAKE_USDC": "1.0",
        },
    },
    "fav90": {
        "name": "FAV90 — LATE-FAVORITE DEPTH-CONFIRM (LIVE $1 TEST)",
        "tagline": "buy favorite at ask [0.88,0.92] when secs_to_close<=100 AND PM book depth>=10, hold to resolution, $1",
        "backtest": "-EV by construction (EV=-fee); deliberate live test of late entry + late book depth (spec 2026-06-04)",
        "env": {
            "BOT_FAV90_MODE": "true",
            "BOT_FAV90_ENTRY_MAX_S": "100",
            "BOT_FAV90_MIN_ASK": "0.88",
            "BOT_FAV90_MAX_ASK": "0.92",
            "BOT_FAV90_MIN_ASK_DEPTH": "10",
            "BOT_FAV90_DAILY_STOP": "-10.0",
            "BOT_FAV90_STOP_BID": "0.70",
            "BOT_STAKE_USDC": "1.0",
            "BOT_UP_STAKE_USDC": "1.0",
            "BOT_ENABLE_CASH_OUT": "false",   # hold to resolution (no early exit)
            "BOT_MAX_CONCURRENT": "1",
            "BOT_HIGH_CONF": "0",
            "BOT_MAX_CONF": "0",
            # Neutralize the model gate stack — fav90 trades the favorite at ~0.90,
            # which would otherwise trip the UP/expensive/mid gates designed for the
            # 0.40-0.75 model strategy. fav90 dispatch already bypasses on_prediction.
            "BOT_UP_MAX_ASK": "1.0",                 # disable up_too_expensive
            "BOT_EXPENSIVE_FILL_THRESHOLD": "1.0",   # disable expensive_fill
            "BOT_REQUIRE_DRIFT_POSITIVE_UP": "false",
            "BOT_REQUIRE_PTB_SUPPORT_UP": "false",
            "BOT_HIGH_ENTRY_CAP": "0",               # disable high_entry_low_conf
            "BOT_MID_PRICE_CAP": "0",                # disable mid_price gate
            "BOT_UP_FILTER_CROWD_INDECISION": "false",
            "BOT_USE_LLM": "false",
            "BOT_VOL_GATE_THRESHOLD": "0",
        },
    },
    "dbmodel": {
        "name": "DBMODEL — DOLLAR-BAR PTB MODEL (LIVE $5, REAL-BALANCE STOP-LOSS)",
        "tagline": "every 5m window, model predicts P(close>PTB) at s2c=240 and BUYS its side at market, $5, hold to resolution; halts on real -$BOT_HARD_STOP_LOSS drawdown",
        "backtest": "-EV by construction (~63% W but -$0.029/trade fee bleed, t=-0.79); deliberate live test (spec 2026-06-04)",
        "env": {
            "BOT_DBMODEL_MODE": "true",
            "BOT_STAKE_USDC": "5.0",
            "BOT_UP_STAKE_USDC": "5.0",
            "BOT_ENABLE_CASH_OUT": "false",   # hold to resolution (no early exit)
            "BOT_MAX_CONCURRENT": "1",
            "BOT_USE_LLM": "false",
            # The dbmodel runner dispatches its own gate-free executor
            # (_execute_dbmodel_trade), so the directional gate stack is never
            # reached. No neutralization needed.
        },
    },
}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run the live BTC up/down trading bot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--a", "-a", dest="profile_a", action="store_true",
        help=f"Profile A: {PROFILES['a']['tagline']}",
    )
    parser.add_argument(
        "--s5", dest="profile_s5", action="store_true",
        help=f"Profile S5: {PROFILES['s5']['tagline']}",
    )
    parser.add_argument(
        "--fav90", dest="profile_fav90", action="store_true",
        help=f"Profile FAV90: {PROFILES['fav90']['tagline']}",
    )
    parser.add_argument(
        "--mm", dest="profile_mm", action="store_true",
        help="MM data collection: read-only book/tape/quotes/rewards logging, NO trades (dry-run)",
    )
    parser.add_argument(
        "--arb", dest="profile_arb", action="store_true",
        help="Arbitrage executor: LIVE at $1/leg, FOK atomic both-legs (override BOT_ARB_DRY_RUN=true to measure-only)",
    )
    parser.add_argument(
        "--dbmodel", dest="profile_dbmodel", action="store_true",
        help=f"Profile DBMODEL: {PROFILES['dbmodel']['tagline']}",
    )
    return parser.parse_args()


def _apply_profile(key):
    # setdefault (not assignment) so an explicit command-line env var wins over
    # the profile default. Precedence: CLI env > profile > .env. This is what
    # lets `BOT_ARB_DRY_RUN=false ... --arb` actually go live despite the
    # profile defaulting it to "true".
    profile = PROFILES[key]
    for env_key, env_val in profile["env"].items():
        os.environ.setdefault(env_key, env_val)
    return profile


def _run_arb_mode(profile, cfg, bot):
    """Dedicated arbitrage runner: scans EVERY 5-min window on its own clock and
    prints arb-relevant output. Does not run the directional prediction loop."""
    import time
    import datetime

    print("=" * 70)
    print("ARB MODE — cross-leg arbitrage on Polymarket BTC 5m (EVERY window)")
    print("=" * 70)
    print(f"  Profile:         {profile['name']}")
    mode = ("DRY-RUN (measure only — NO orders placed)" if cfg.arb_dry_run
            else "*** LIVE — placing REAL FOK orders ***")
    print(f"  Mode:            {mode}")
    print(f"  Trigger:         ask_up + ask_down < {1 - cfg.arb_min_edge:.3f}  "
          f"(edge >= {cfg.arb_min_edge:.3f})")
    print(f"  Size:            {cfg.arb_max_size:.0f} sh/leg (exchange min), "
          f"${cfg.arb_max_usdc_per_leg:.2f}/leg budget · ~${5*0.5*2:.0f}/pair typical · depth-capped")
    feed = "WebSocket push (fast)" if cfg.arb_ws else f"REST poll @{cfg.arb_poll_s:.0f}s"
    print(f"  Feed:            {feed}, up to {cfg.arb_max_per_window}/window, "
          f"stop {cfg.arb_deadline_buffer_s:.0f}s before close")
    print(f"  Safeguards:      FOK limit (no slippage) · atomic both-legs-or-unwind · depth-capped")
    print(f"  Log:             predictor/arb_history.jsonl  "
          f"(analyze: python3 predictor/analyze_arb.py)")
    print(f"  MM data logging: {'on' if cfg.mm_log_enabled else 'off'}")
    print(f"  CLOB host:       {cfg.clob_host}")
    print(f"  Funder:          {cfg.funder_address}")
    print("=" * 70)

    # Only in LIVE arb: redeem any arb winners orphaned by a prior run.
    if not cfg.arb_dry_run and cfg.reconcile_on_start:
        try:
            # force=True: arb keeps BOT_DRY_RUN=true for the directional bot, but
            # arb winners are real and must actually be claimed (else they strand).
            r = bot.client.sweep_orphan_winners(force=True)
            print(f"  Reconcile: redeemed {r.get('redeemed', 0)} orphan winner(s)")
        except Exception as e:
            print(f"  Reconcile error ({type(e).__name__}: {e}) — continuing")

    bot.arb_executor.console = True          # stream arb events to stdout
    if cfg.mm_log_enabled:
        bot.mm_logger.start_continuous()
    bot.arb_executor.start_continuous()

    print("  Scanner live. Heartbeat every 60s. Ctrl-C to stop.\n", flush=True)
    ticks = 0
    try:
        while True:
            time.sleep(60)
            ticks += 1
            s = bot.arb_executor.stats
            lat = f" lat={s['last_lat_ms']}ms" if s.get('last_lat_ms') else ""
            print(f"  [ARB ❤ {datetime.datetime.now().strftime('%H:%M:%S')}] "
                  f"windows={s['windows']} detect={s['detections']} "
                  f"fillable={s['acted']} thin={s['thin']} locked={s['locked']} "
                  f"unwound={s['unwound']} gross=${s['gross_profit']:.2f} "
                  f"best_edge={s['best_edge']:.1%}{lat} | last: {s['last_event']}", flush=True)
            # LIVE: every ~5 min, redeem resolved arb winners (claim the $1/share
            # the winning leg pays). Without this the profit never materializes
            # and winning shares pile up unclaimed.
            if not cfg.arb_dry_run and ticks % 5 == 0:
                try:
                    # Primary: claim our tracked positions directly by condition_id
                    # (no data-api row cap). Backstop: sweep for anything orphaned.
                    r = bot.arb_executor.redeem_resolved()
                    if r.get("redeemed", 0) > 0:
                        print(f"  [ARB] redeemed {r['redeemed']} winner(s) "
                              f"({r.get('pending', 0)} still pending)", flush=True)
                except Exception as e:
                    print(f"  [ARB] redeem error ({type(e).__name__}: {e})", flush=True)
    except KeyboardInterrupt:
        print("\n  Arb scanner stopped.")


def _run_fav90_mode(profile, cfg, bot):
    """Dedicated fav90 runner: walks EVERY consecutive 5-min window on its own
    clock and dispatches the late-favorite depth-confirm executor. Does NOT run
    the directional prediction loop and never imports run_baseline (TF/LSTM) —
    fav90 ignores the model entirely (it picks the favorite from the PM book), so
    the ML stack is pure waste here and the predict-every-OTHER-window cadence
    would also miss half the 5-min windows."""
    import time
    import datetime
    import threading

    print("=" * 70)
    print("FAV90 MODE — late-favorite depth-confirm on EVERY 5m window (LIVE $1)")
    print("=" * 70)
    print(f"  Profile:         {profile['name']}")
    print(f"  Rule:            BUY favorite when secs_to_close <= {cfg.fav90_entry_max_s:.0f}s "
          f"AND ask in [{cfg.fav90_min_ask:.2f}, {cfg.fav90_max_ask:.2f}] "
          f"AND ask-depth >= {cfg.fav90_min_ask_depth:.0f} sh within 1 tick")
    print(f"  Stake:           ${cfg.stake_usdc:.2f} · hold to resolution UNLESS bid <= ${cfg.fav90_stop_bid:.2f}")
    print(f"  Stop-loss:       if held bid <= ${cfg.fav90_stop_bid:.2f}, SELL immediately, retry till flat")
    print(f"  Kill-switch:     daily realized PnL <= ${cfg.fav90_daily_stop:.2f} pauses trading for the UTC day")
    print(f"  Poll log:        predictor/fav90_log.jsonl  (one record per poll, fired or not)")
    print(f"  Mode:            {'DRY-RUN (no orders)' if cfg.dry_run else '*** LIVE — REAL $1 orders ***'}")
    print(f"  CLOB host:       {cfg.clob_host}")
    print(f"  Funder:          {cfg.funder_address}")
    print("=" * 70, flush=True)

    if cfg.reconcile_on_start and not cfg.dry_run:
        try:
            r = bot.client.sweep_orphan_winners()
            print(f"  Reconcile: redeemed {r.get('redeemed', 0)} orphan winner(s)")
        except Exception as e:
            print(f"  Reconcile error ({type(e).__name__}: {e}) — continuing")

    print("  fav90 live. One trade attempt per 5m window (dispatched ~150s before "
          "close). Ctrl-C to stop.\n", flush=True)
    last_slug = None
    try:
        while True:
            now = time.time()
            ws = (int(now) // 300) * 300
            end_ts = ws + 300
            slug = f"btc-updown-5m-{ws}"
            secs_to_close = end_ts - now
            # Dispatch each window exactly once, when it enters the late zone with
            # enough time left to poll + fire + fill. The executor's fav90 branch
            # then polls every 10s until close-60, firing the first qualifying poll.
            if slug != last_slug and 65 < secs_to_close <= 150:
                last_slug = slug
                window = {"slug": slug, "end_ts": end_ts}
                with bot._lock:
                    bot._active += 1
                threading.Thread(
                    target=bot._execute_trade,
                    args=(window, "UP", 0.0, None, None, None, None, None),
                    daemon=True,
                ).start()
                print(f"  [FAV90 {datetime.datetime.now().strftime('%H:%M:%S')}] "
                      f"window {slug} dispatched (closes in {secs_to_close:.0f}s)", flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n  fav90 stopped.")


def _run_dbmodel_mode(profile, cfg, bot):
    """Dedicated dollar-bar PTB model runner. Streams Binance aggTrades into a
    DollarBarBuilder; once per 5-min window, at 60s into the window (s2c=240, the
    leak-free decision time the model was trained for), it builds features, gets
    the model's calibrated P(close>PTB), and dispatches a gate-free $1 market buy
    on the predicted side (hold to resolution). Never imports run_baseline — the
    only signal is the XGB/LightGBM model fed by Binance, so TF/LSTM never load."""
    import time
    import datetime
    import threading

    from live_trader.db_model import DbModel
    from live_trader.db_features import build_features, FEATURE_NAMES
    from live_trader.dollar_bars import BinanceAggTradeClient

    VOL_WINDOW = 10

    model_path = cfg.dbmodel_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(HERE, model_path)
    model = DbModel(model_path)

    # Window length + decision instant come from the model bundle: 5m -> 300/240
    # (matches the original BTC bot), 15m -> 900/420 (enter 7 min before close).
    WINDOW_S = model.window_s
    MONITOR_START_S = model.monitor_start_s
    from live_trader.db_decision import entry_decision
    MAX_ASK = cfg.dbmodel_max_ask
    DEADLINE_S = cfg.dbmodel_entry_deadline_s
    CHEAP = MAX_ASK < 1.0
    symbol = cfg.dbmodel_symbol.lower()
    binance_pair = f"{symbol.upper()}USDT"
    tf = "15m" if WINDOW_S == 900 else "5m" if WINDOW_S == 300 else f"{WINDOW_S}s"
    slug_prefix = f"{symbol}-updown-{tf}"

    print("=" * 70)
    print(f"DBMODEL MODE — dollar-bar PTB model bets EVERY {tf} {symbol.upper()} "
          f"window (stake ${cfg.stake_usdc:.0f})")
    print("=" * 70)
    print(f"  Profile:         {profile['name']}")
    print(f"  Model:           {model_path}")
    print(f"                   winner={model.meta.get('winner','?')} "
          f"threshold=${model.threshold_usd:,.0f} feats={len(FEATURE_NAMES)}")
    print(f"  Market:          {slug_prefix}-{{ws}}  (window={WINDOW_S}s)")
    print(f"  Rule:            at s2c={MONITOR_START_S}s, BUY argmax P(up) side at market")
    print(f"  Stake:           ${cfg.stake_usdc:.2f} · hold to resolution (no gates, no cap)")
    if CHEAP:
        print(f"  Entry gate:      ask <= {MAX_ASK:.2f}, poll 2s, skip if none by s2c={DEADLINE_S:.0f}s", flush=True)
    print(f"  Feed:            Binance {binance_pair.lower()}@aggTrade -> dollar bars")
    print(f"  Log:             predictor/dbmodel_log.jsonl  (one record per fire)")
    print(f"  Mode:            {'DRY-RUN (no orders)' if cfg.dry_run else f'*** LIVE — REAL ${cfg.stake_usdc:.0f} orders ***'}")
    print(f"  CLOB host:       {cfg.clob_host}")
    print(f"  Funder:          {cfg.funder_address}")
    print("=" * 70, flush=True)

    if cfg.reconcile_on_start and not cfg.dry_run:
        try:
            r = bot.client.sweep_orphan_winners()
            print(f"  Reconcile: redeemed {r.get('redeemed', 0)} orphan winner(s)")
        except Exception as e:
            print(f"  Reconcile error ({type(e).__name__}: {e}) — continuing")

    feed = BinanceAggTradeClient(model.threshold_usd, buffer_len=500, symbol=binance_pair)
    feed.start()
    print("  Binance feed started; warming up dollar bars "
          f"(need >= {VOL_WINDOW}). Ctrl-C to stop.\n", flush=True)

    # Real-balance stop-loss. The dbmodel path bypasses on_prediction's stop, and
    # that stop reads an optimistic counter; the only trustworthy kill-switch is
    # the wallet. Snapshot pUSD+USDC.e now and halt when the real drawdown hits
    # cfg.hard_stop_loss. If we can't even read the balance, refuse to trade live
    # rather than run uncapped.
    from live_trader import risk
    start_stable = None
    stop_check_ws = None
    if not cfg.dry_run and cfg.hard_stop_loss > 0:
        try:
            start_stable = risk.fetch_stable_balance(cfg.funder_address)
            print(f"  Stop-loss:       REAL-balance halt at -${cfg.hard_stop_loss:.2f} "
                  f"(start stable ${start_stable:.2f})", flush=True)
        except Exception as e:
            print(f"  Stop-loss:       FATAL — could not read start balance "
                  f"({type(e).__name__}: {e}). Refusing to trade live without a "
                  f"working stop-loss. Exiting.", flush=True)
            feed.stop()
            return
    elif not cfg.dry_run:
        print("  Stop-loss:       *** DISABLED (BOT_HARD_STOP_LOSS=0) — live with "
              "NO stop-loss ***", flush=True)

    # BOT_DBMODEL_MAX_TRADES > 0 caps how many fires the runner dispatches, then
    # waits for the in-flight trade(s) to settle and exits. 0 = unlimited (normal
    # operation). Used for a controlled "place exactly one live trade" test.
    max_trades = int(os.environ.get("BOT_DBMODEL_MAX_TRADES", "0"))
    trades_dispatched = 0

    # strike = Binance price at window open (captured within the first 30s of the
    # window). A window is only traded if we observed its open, so a mid-window
    # start skips the current window and resumes at the next full one.
    strike = {"ws": None, "px": None}
    fired = set()
    pending = {}   # ws -> dict(slug,end_ts,direction,p_up,drift,last_px,strike,token_id,window)
    try:
        while True:
            now = time.time()
            ws = (int(now) // WINDOW_S) * WINDOW_S
            end_ts = ws + WINDOW_S
            secs_to_close = end_ts - now
            last_px = feed.last_price

            if strike["ws"] != ws and secs_to_close > (WINDOW_S - 30) and last_px:
                strike["ws"], strike["px"] = ws, last_px

            # Real-balance stop-loss: read the wallet once per window. Adds back
            # capital locked in open trades (cost debited, payout not yet landed)
            # so an in-flight position doesn't false-trigger the halt.
            if start_stable is not None and stop_check_ws != ws:
                stop_check_ws = ws
                try:
                    cur = risk.fetch_stable_balance(cfg.funder_address)
                    with bot._lock:
                        open_n = bot._active
                    tripped, rpnl = risk.stop_loss_tripped(
                        start_stable, cur, open_n, cfg.stake_usdc, cfg.hard_stop_loss)
                    print(f"  [DBMODEL {datetime.datetime.now():%H:%M:%S}] risk: "
                          f"real_pnl=${rpnl:+.2f} (stable ${cur:.2f}, {open_n} open) "
                          f"limit -${cfg.hard_stop_loss:.0f}", flush=True)
                    if tripped:
                        print(f"  [DBMODEL] *** STOP-LOSS HIT: real_pnl=${rpnl:+.2f} "
                              f"<= -${cfg.hard_stop_loss:.2f} — no new trades, settling "
                              f"{open_n} open, exiting ***", flush=True)
                        while True:
                            with bot._lock:
                                if bot._active == 0:
                                    break
                            time.sleep(3)
                        feed.stop()
                        return
                except Exception as e:
                    print(f"  [DBMODEL] risk read error ({type(e).__name__}: {e}) "
                          f"— will retry next window", flush=True)

            if (ws not in fired and secs_to_close <= MONITOR_START_S
                    and strike["ws"] == ws and strike["px"]):
                fired.add(ws)                       # one decision per window, period
                slug = f"{slug_prefix}-{ws}"
                bars = feed.bars.snapshot()[-VOL_WINDOW:]
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                if len(bars) < VOL_WINDOW or not last_px:
                    print(f"  [DBMODEL {ts}] {slug}: warming up "
                          f"({len(bars)}/{VOL_WINDOW} bars) — skip", flush=True)
                    continue
                drift_pct = (last_px - strike["px"]) / strike["px"] * 100.0
                feats = build_features(bars, drift_pct=drift_pct,
                                       secs_to_close=MONITOR_START_S, vol_window=VOL_WINDOW)
                if feats is None:
                    print(f"  [DBMODEL {ts}] {slug}: features unavailable — skip", flush=True)
                    continue
                detail = model.predict_detailed(feats)
                p_up = detail["p_up"]
                direction = "UP" if p_up >= 0.5 else "DOWN"
                print(f"  [DBMODEL {ts}] {slug}: P(up)={p_up:.3f} drift={drift_pct:+.3f}% "
                      f"-> BUY {direction} (s2c={secs_to_close:.0f}s)", flush=True)
                # window carries the paper-log context (features, raw proba, window
                # open) so the dry-run settler can write a rich record. Ignored by
                # the live path.
                window = {"slug": slug, "end_ts": end_ts, "ws": ws,
                          "features": dict(feats), "raw_proba": detail["raw"],
                          "strike": strike["px"],
                          "monitor_start_s": MONITOR_START_S, "window_s": WINDOW_S}
                if not CHEAP:
                    with bot._lock:
                        bot._active += 1
                    threading.Thread(
                        target=bot._execute_dbmodel_trade,
                        args=(window, direction, p_up, drift_pct, last_px, strike["px"]),
                        daemon=True,
                    ).start()
                    trades_dispatched += 1
                    if max_trades and trades_dispatched >= max_trades:
                        print(f"  [DBMODEL {ts}] max_trades={max_trades} reached — "
                              f"waiting for the trade to settle (resolution + redeem)...",
                              flush=True)
                        while True:
                            with bot._lock:
                                active = bot._active
                            if active == 0:
                                break
                            time.sleep(3)
                        print(f"  [DBMODEL {datetime.datetime.now().strftime('%H:%M:%S')}] "
                              f"single live test complete — stopping.", flush=True)
                        feed.stop()
                        return
                else:
                    # max_trades intentionally not wired in CHEAP mode — it only
                    # applies to the standard immediate-entry path's controlled
                    # single-trade test; CHEAP entries are deferred and polled.
                    try:
                        mkt = bot.client.resolve_market(slug)
                        token_id = mkt["up_token"] if direction == "UP" else mkt["down_token"]
                        pending[ws] = {"slug": slug, "end_ts": end_ts, "direction": direction,
                                       "p_up": p_up, "drift": drift_pct, "last_px": last_px,
                                       "strike": strike["px"], "token_id": token_id, "window": window}
                        print(f"  [DBMODEL {ts}] {slug}: decided BUY {direction}; "
                              f"waiting for ask <= {MAX_ASK:.2f}", flush=True)
                    except Exception as e:
                        print(f"  [DBMODEL {ts}] {slug}: resolve failed ({type(e).__name__}: {e}) — skip", flush=True)
                        bot._record_skip(reason="resolve_failed", details=str(e),
                                         slug=slug, end_ts=end_ts, direction=direction,
                                         confidence=abs(p_up - 0.5) * 200.0, ptb=strike["px"],
                                         live_price=last_px, drift_pct=drift_pct,
                                         final_up=(p_up >= 0.5), top_ask_up=None,
                                         top_ask_down=None, signals=None)
            if CHEAP and pending:
                for ws_p, pe in list(pending.items()):
                    s2c_p = pe["end_ts"] - time.time()
                    try:
                        ask = bot.client.get_top_ask(pe["token_id"])
                    except Exception:
                        ask = None
                    act = entry_decision(ask, s2c_p, MAX_ASK, DEADLINE_S)
                    tsp = datetime.datetime.now().strftime("%H:%M:%S")
                    if act == "enter":
                        ask_fmt = f"{ask:.3f}" if ask is not None else "?"
                        print(f"  [DBMODEL {tsp}] {pe['slug']}: ask={ask_fmt} <= {MAX_ASK:.2f} "
                              f"-> ENTER (FOK limit)", flush=True)
                        with bot._lock:
                            bot._active += 1
                        threading.Thread(
                            target=bot._execute_dbmodel_trade,
                            args=(pe["window"], pe["direction"], pe["p_up"], pe["drift"],
                                  pe["last_px"], pe["strike"]),
                            kwargs={"limit_price": MAX_ASK},
                            daemon=True,
                        ).start()
                        del pending[ws_p]
                    elif act == "skip":
                        print(f"  [DBMODEL {tsp}] {pe['slug']}: no ask <= {MAX_ASK:.2f} by "
                              f"s2c={DEADLINE_S:.0f}s (last ask={ask}) -> SKIP", flush=True)
                        bot._record_skip(reason="no_cheap_entry", details=f"last_ask={ask}",
                                         slug=pe["slug"], end_ts=pe["end_ts"], direction=pe["direction"],
                                         confidence=abs(pe["p_up"] - 0.5) * 200.0, ptb=pe["strike"],
                                         live_price=pe["last_px"], drift_pct=pe["drift"],
                                         final_up=(pe["p_up"] >= 0.5), top_ask_up=None,
                                         top_ask_down=None, signals=None)
                        del pending[ws_p]
                    # else "wait": leave pending for the next tick
            time.sleep(2)
    except KeyboardInterrupt:
        print("\n  dbmodel stopped.")
        feed.stop()


def _start_memtrace():
    """Gated memory-leak diagnostic (BOT_MEMTRACE=1). Every 2 min, append to
    memdiag.log: a gc object-type census + tracemalloc top allocations and the
    TOP GROWTH since the previous snapshot (the leaking line shows up here).
    Off by default; pure diagnostic, never touches trading."""
    import tracemalloc, threading as _th, time as _t
    # Proper use this time: read the FILTERED traceback-growth diff (size_diff by
    # allocation site), excluding tracemalloc's own frames + frozen importlib +
    # the diagnostic itself. That names the real leaking line. (Earlier failures:
    # reading the raw object census counts tracemalloc's own tuples; gc.get_objects
    # + get_referrers retained their own snapshots. Both avoided here.)
    tracemalloc.start(12)
    path = os.path.join(HERE, "memdiag.log")
    _filt = (
        tracemalloc.Filter(False, "<frozen *>"),
        tracemalloc.Filter(False, "*tracemalloc*"),
        tracemalloc.Filter(False, "*run_live_bot.py"),   # exclude this probe
        tracemalloc.Filter(False, "<unknown>"),
    )

    def _rss_mb():
        try:
            with open("/proc/self/status") as sf:
                for ln in sf:
                    if ln.startswith("VmRSS"):
                        return int(ln.split()[1]) // 1024
        except Exception:
            return 0
        return 0

    def _loop():
        prev = None
        while True:
            _t.sleep(120)
            try:
                snap = tracemalloc.take_snapshot().filter_traces(_filt)
                with open(path, "a") as f:
                    f.write(f"\n==== {_t.strftime('%H:%M:%S')} rss={_rss_mb()}MB "
                            f"threads={_th.active_count()} ====\n")
                    if prev is not None:
                        f.write("TOP GROWTH since last (by alloc site, real code only):\n")
                        for st in snap.compare_to(prev, "traceback")[:10]:
                            if st.size_diff <= 0:
                                continue
                            f.write(f"  +{st.size_diff/1e6:.1f}MB (now {st.size/1e6:.1f}MB, "
                                    f"{st.count_diff:+d} blocks)\n")
                            for line in st.traceback.format()[-4:]:
                                f.write(f"      {line.strip()}\n")
                    f.write("TOP TOTAL (real code only):\n")
                    for st in snap.statistics("lineno")[:6]:
                        f.write(f"  {st.size/1e6:.1f}MB {st.count} blocks {st.traceback.format()[-1].strip()}\n")
                prev = snap
            except Exception:
                pass

    _th.Thread(target=_loop, name="memtrace", daemon=True).start()


def main():
    args = _parse_args()
    if os.environ.get("BOT_MEMTRACE"):
        _start_memtrace()
    profile = None
    if args.profile_a:
        profile = _apply_profile("a")
    if args.profile_s5:
        profile = _apply_profile("s5")
    if args.profile_fav90:
        profile = _apply_profile("fav90")
    if args.profile_mm:
        profile = _apply_profile("mm")
    if args.profile_arb:
        profile = _apply_profile("arb")
    if args.profile_dbmodel:
        profile = _apply_profile("dbmodel")

    # load_config calls _load_dotenv which uses os.environ.setdefault, so any
    # profile overrides set above will win over the values in .env.
    from live_trader.config import load_config  # noqa: E402
    from live_trader.polymarket import PolymarketBotClient  # noqa: E402
    from live_trader.bot import HighSameBot  # noqa: E402

    dotenv = os.path.join(HERE, ".env")
    cfg = load_config(dotenv_path=dotenv)

    client = PolymarketBotClient(cfg)
    bot = HighSameBot(cfg, client)

    # ARB MODE: dedicated arbitrage runner. Does NOT run the directional
    # prediction loop (which only covers every other window and floods the
    # console). Scans EVERY 5-min window on its own clock and prints arb data.
    # NOTE: run_baseline (TensorFlow/pandas) is imported lazily *after* this
    # early return — the arb path never needs the ML stack, so a co-located arb
    # box can skip those heavy deps entirely. bot.py imports fetch_order_book
    # lazily for the S5 directional path, so this doesn't break that mode.
    if args.profile_arb:
        _run_arb_mode(profile, cfg, bot)
        return

    # FAV90 MODE: model-free late-favorite runner on its own per-window clock.
    # Early return BEFORE importing run_baseline so we never load the TF/LSTM
    # stack — fav90 doesn't use any model signal.
    if args.profile_fav90:
        _run_fav90_mode(profile, cfg, bot)
        return

    # DBMODEL MODE: dollar-bar PTB model runner on its own per-window clock. Early
    # return BEFORE importing run_baseline so the TF/LSTM stack never loads — this
    # mode's only signal is the calibrated XGB/LightGBM model fed by Binance.
    if args.profile_dbmodel:
        _run_dbmodel_mode(profile, cfg, bot)
        return

    import run_baseline  # noqa: E402,F401  (directional path only; loads TF)

    print("=" * 70)
    print("LIVE BOT — HIGH-same on Polymarket BTC 5m up/down")
    print("=" * 70)

    if profile is not None:
        print(f"  PROFILE:         {profile['name']}")
        print(f"                   ({profile['tagline']})")
        print(f"  Backtest:        {profile['backtest']}")
        conf_label = ("off" if cfg.max_conf_threshold == 0
                      else f"[{cfg.high_conf_threshold:.1f}%, {cfg.max_conf_threshold:.1f}%)")
        print(f"  Conf band:       {conf_label}")
        if cfg.s5_mode:
            if cfg.s5_disable_lower:
                print(f"  S5 entry band:   (0.58, 0.75]  ← UPPER-ONLY (lower [0.40,0.50) TEMP DISABLED)")
            else:
                print(f"  S5 entry band:   [0.40, 0.50) U (0.58, 0.75]  (excludes 0.50-0.58)")
            print(f"  S5 ob_side:      >= 0.85 side-aligned (Bitstamp BTC book)")
            print(f"  S5 daily cap:    DISABLED (test mode — trades continuously)")
            print(f"  S5 vol-gate:     BYPASSED (vol_gate_threshold={cfg.vol_gate_threshold:.2f} ignored in S5)")
            print(f"  S5 shadow log:   predictor/s5_shadow.jsonl (PM-implied side prob logged, NOT gated)")
        else:
            print(f"  Entry band:      [{cfg.entry_min:.2f}, {cfg.entry_max:.2f})")
        print(f"  Other gates:     stay active (contra_book, contra_drift,")
        print(f"                   high_entry_low_conf, drift_noise, ask_moved, etc.)")
        print("-" * 70)

    print(f"  CLOB host:       {cfg.clob_host}")
    print(f"  Chain id:        {cfg.chain_id}")
    print(f"  Funder:          {cfg.funder_address}")
    print(f"  Stake DOWN:      ${cfg.stake_usdc:.2f} USDC")
    print(f"  Stake UP:        ${cfg.up_stake_usdc:.2f} USDC (asymmetric)")
    cap = f"{cfg.max_conf_threshold:.1f}%" if cfg.max_conf_threshold > 0 else "off"
    print(f"  Conf band:       {cfg.high_conf_threshold:.1f}% to {cap}")
    band = (f"[{cfg.entry_min:.2f}, {cfg.entry_max:.2f})"
            if cfg.entry_min > 0 or cfg.entry_max < 1.0 else "off")
    print(f"  Entry band:      {band}")
    print(f"  Max concurrent:  {cfg.max_concurrent_trades} trade(s)")
    up_cap = f"{cfg.up_max_ask:.2f}" if cfg.up_max_ask > 0 else "off"
    print(f"  UP ask range:    [{cfg.up_min_ask:.2f}, {up_cap}] (override >cap if PTB+drift strong)")
    print(f"  UP gates:        ptb_support={cfg.require_ptb_support_up} "
          f"drift_positive={cfg.require_drift_positive_up}")
    print(f"  Crowd-indec:     {'on' if cfg.up_filter_crowd_indecision else 'off'} "
          f"(both dirs, ask<=0.50)")
    print(f"  Expensive fill:  ask>{cfg.expensive_fill_threshold:.2f} "
          f"(override: PTB>={cfg.strong_ptb_up_prob:.2f} & |drift|>={cfg.strong_drift_pct:.2f}%)")
    print(f"  Dry run:         {cfg.dry_run}")
    print("=" * 70)

    # Pre-trade reconciliation: redeem any orphan winners from prior runs.
    if cfg.reconcile_on_start and not cfg.dry_run:
        print("-" * 70)
        print("Pre-trade reconciliation — scanning for orphan winning positions...")
        try:
            result = client.sweep_orphan_winners()
            if result["redeemed"] > 0:
                print(f"  ✓ Reconcile: redeemed {result['redeemed']} orphan position(s), "
                      f"~${result['value']:.2f} recovered")
            elif result["winners"] > 0:
                print(f"  Reconcile: {result['winners']} candidate(s), "
                      f"{result['skipped']} already-clean, {result['failed']} failed")
            else:
                print(f"  ✓ Reconcile: no orphan winners ({result['scanned']} positions scanned)")
        except Exception as e:
            print(f"  Reconcile error ({type(e).__name__}: {e}) — continuing anyway")
        print("-" * 70)

    # MM data collection runs on its OWN clock — captures EVERY 5-min window,
    # unlike the trading loop which only predicts every other window.
    if cfg.mm_log_enabled:
        bot.mm_logger.start_continuous()

    # Arbitrage executor — own clock, scans every window. Dry-run by default.
    if cfg.arb_enabled:
        bot.arb_executor.start_continuous()

    run_baseline.TRADE_HOOK = bot.on_prediction
    run_baseline.main()


if __name__ == "__main__":
    main()
