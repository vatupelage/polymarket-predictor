# Cheap-entry dbmodel bot (btc_5m_cheap)

**Date:** 2026-06-19
**Status:** Approved (design)
**Branch:** predictfun-port

## Goal

Run a second live btc 5-minute dbmodel bot that bets the model's chosen side
**only when it can enter at an ask ≤ 0.50** — operationalizing the "buy below
fair value" thesis (the only structural way to have a positive edge). It runs
ALONGSIDE the existing standard `btc_5m` bot as a clean A/B experiment, at $1
stake, on the same wallet.

## Background / honest framing

- The model's accuracy is real (~65% paper, ~64% live) but profit-neutral
  because the price already equals the odds; net edge after the ~3% taker fee is
  ≈0 (see memories `dbmodel_no_net_edge_confirmed`, `polymarket_crypto_taker_fee`).
- The ONLY way to profit is to buy a side for less than it is truly worth. The
  ask-band analysis on 2,813 paper trades showed `ask ≤ 0.50` entries net
  **+1.56%/$1** (vs +0.61% for >0.50), driven by a thin `[0.40,0.45)` band
  (+18%, n=112, t=1.64). **Directionally right, NOT statistically confirmed.**
- **Validity caveat (the experiment exists to resolve this):** the backtest
  measured windows where the ask *happened to be* ≤0.50 at decision time. This
  bot actively *waits* for the ask to *drop* to ≤0.50 during the window. An ask
  falling below 0.50 can mean the market just turned against that side (adverse
  selection / catching a falling knife), so the live sample may behave worse than
  the backtest. $1/stake makes this a cheap probe to find out.

## Approach: parametrize the existing dbmodel codepath (do not fork)

Add two env vars to the dbmodel path. At their defaults the standard bot behaves
EXACTLY as today, so the running `btc_5m` bot is untouched.

- `BOT_DBMODEL_MAX_ASK` (float, default `1.0`): enter only when the chosen
  side's ask ≤ this value. `1.0` ⇒ always true on first poll ⇒ immediate
  entry-at-decision = current behavior.
- `BOT_DBMODEL_ENTRY_DEADLINE_S` (int, default `0`): once `secs_to_close ≤` this,
  stop trying and skip the window. `0` is irrelevant when MAX_ASK=1.0 (entry is
  immediate); for the cheap bot it is `30`.

Alternatives rejected: a separate `--dbmodel-cheap` profile or a standalone
script — both duplicate the well-tested loop and risk drift.

## Loop change: decouple "decide" from "enter"

Current dbmodel loop (`run_live_bot.py`): at `s2c ≤ MONITOR_START_S` (240 for
5m), decide the argmax side once and immediately dispatch a market buy.

New behavior when `MAX_ASK < 1.0`:

1. At `s2c ≤ MONITOR_START_S`, decide the side once (unchanged feature/predict
   logic). Resolve the market to obtain the chosen `token_id`. Store a
   `pending[ws]` record `{slug, end_ts, ws, direction, p_up, drift_pct, strike,
   last_px, token_id, window}`. Do NOT trade yet. The `fired` set still
   guarantees one DECISION per window.
2. On each 2s loop tick, for the single pending window:
   - read `client.get_top_ask(token_id)`;
   - if `ask ≤ MAX_ASK` → **enter** (fill method below); clear `pending[ws]`;
   - elif `secs_to_close ≤ DEADLINE_S` → **skip**: write a skip record
     (`reason="no_cheap_entry"`) to the bot's log so trigger frequency is
     measurable; clear `pending[ws]`;
   - else → keep polling next tick.

When `MAX_ASK == 1.0`: on the first poll after decision `ask ≤ 1.0` is always
true ⇒ immediate entry ⇒ behavior identical to today (regression-protected by a
unit test).

### Entry decision is a pure function (the TDD seam)

```
entry_decision(ask, secs_to_close, max_ask, deadline_s) -> "enter" | "wait" | "skip"
  ask is None                        -> "wait"   (no book yet; try again)
  ask <= max_ask                     -> "enter"
  secs_to_close <= deadline_s        -> "skip"
  otherwise                          -> "wait"
```

### Fill method: FOK limit at MAX_ASK (not market)

When entering, place a **fill-or-kill limit buy at `MAX_ASK`** via the existing
`buy_limit_fok(token_id, price=MAX_ASK, size=stake/MAX_ASK)` rather than
`buy_market`. This GUARANTEES we never pay > 0.50 even if the ask ticks up
between observing it and ordering. If the FOK cannot fill at ≤ MAX_ASK it
cancels; the loop keeps polling until fill or deadline. The standard bot
(MAX_ASK=1.0) continues to use the existing market-buy path unchanged.

## Components / files

- `live_trader/db_decision.py` (or a small new helper module): `entry_decision()`
  pure function + unit tests. Keep it dependency-free.
- `run_live_bot.py` dbmodel loop: read the two env vars; add the
  decide→poll→enter state machine guarded by `MAX_ASK < 1.0`.
- `live_trader/config.py`: parse `BOT_DBMODEL_MAX_ASK`, `BOT_DBMODEL_ENTRY_DEADLINE_S`.
- `live_trader/bot.py`: a cheap-entry executor path that uses `buy_limit_fok` at
  MAX_ASK (or parametrize `_execute_dbmodel_trade` with an optional limit price);
  reused logging.
- `live_env/btc_5m_cheap.env` (server): symbol=btc, model=models/db_ptb.joblib,
  stake=1.0, MAX_ASK=0.50, DEADLINE_S=30, log=live_btc_5m_cheap.jsonl.

## Deployment

- Reuse the existing `dblive@.service` template (loads `live_env/%i.env`):
  `sudo systemctl start dblive@btc_5m_cheap`. No new unit file.
- Runs alongside the standard `btc_5m` bot. On a window where ask≤0.50 BOTH bots
  may buy the same side (~$2 total that window); each logs its own fills so the
  data stays separable.
- Stop-loss disabled (`BOT_HARD_STOP_LOSS=0`, consistent with the other $1 bots),
  hold to resolution, shared `0xFUNDER` wallet, existing redeemer covers winners.
- Rollout: scp code to the (untracked) server, create the env file, dry-run smoke
  to confirm decide→poll→would-enter with NO orders, then start live.

## Testing

- Unit-test `entry_decision()`: cheap+early→enter; expensive+early→wait;
  past-deadline (no fill)→skip; `ask=None`→wait; `max_ask=1.0`→enter on first
  call (proves the standard bot is unchanged).
- Dry-run smoke on the server: observe one full window decide→poll cadence and a
  simulated entry/skip, with no live orders placed.

## Success criteria

- Standard `btc_5m` bot behavior provably unchanged (unit test + dry-run).
- New bot only ever fills at ≤ 0.50, skips windows with no dip (logged), and runs
  live at $1 alongside the baseline.
- After enough live windows, compare cheap-bot net PnL vs the baseline to judge
  whether the patient cheap-entry actually beats buy-at-market — resolving the
  adverse-selection question with real money.

## Caveats

- Unconfirmed edge; adverse-selection risk is the central unknown (above).
- Low trade frequency (~15% of windows historically had ask≤0.50, and the
  *waiting* variant may differ), so live validation is slow.
- Fees are highest on cheap entries (`0.07*(1-ask)` ≈ 3.5% at 0.50), already
  reflected in the +1.56% net figure.
