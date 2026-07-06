# Cheap-entry dbmodel bot (btc_5m_cheap) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live btc 5-minute dbmodel bot that bets the model's side only when it can enter at an ask ≤ 0.50, running alongside the standard bot on the all-markets server at $1 stake.

**Architecture:** Parametrize the existing `run_live_bot.py --dbmodel` loop with two env vars (`BOT_DBMODEL_MAX_ASK`, `BOT_DBMODEL_ENTRY_DEADLINE_S`). When `MAX_ASK < 1.0`, the loop decouples "decide side" (s2c=240, unchanged) from "enter" (poll the chosen token's ask every 2s; enter via a FOK limit at MAX_ASK on the first dip; skip at the deadline). Defaults (`MAX_ASK=1.0`) leave the standard bot's behavior identical.

**Tech Stack:** Python 3, existing `live_trader` package (DbModel, PolymarketBotClient, BinanceAggTradeClient), pytest, systemd `dblive@.service` template.

## Global Constraints

- Standard `btc_5m` bot behavior MUST be unchanged when the new env vars are unset (`MAX_ASK` default `1.0`, `DEADLINE_S` default `0`).
- New bot: stake `$1.0`, `MAX_ASK=0.50`, `ENTRY_DEADLINE_S=30`, model `models/db_ptb.joblib`, own log `live_btc_5m_cheap.jsonl`.
- Entry MUST never pay > MAX_ASK: use `buy_limit_fok(token_id, price=MAX_ASK, size=stake/MAX_ASK)`, not `buy_market`.
- Decision instant stays the model's trained instant (s2c = `model.monitor_start_s` = 240 for 5m). One DECISION per window.
- Server is NOT git-tracked; deploy by scp. Funder wallet `0xYOUR_FUNDER_ADDRESS`. Stop-loss disabled (`BOT_HARD_STOP_LOSS=0`). NEVER `git add .`/`-A`; explicit paths only.
- Run on the all-markets server `<server-host>` (DNS changes on stop/start), key `~/.ssh/all_markets.pem`, venv `/home/ubuntu/btcpredictor/.venv`.

---

### Task 1: `entry_decision()` pure function

**Files:**
- Modify: `live_trader/db_decision.py` (append a new function; leave existing `db_decision` untouched)
- Test: `test_entry_decision.py`

**Interfaces:**
- Produces: `entry_decision(ask: float | None, secs_to_close: float, max_ask: float, deadline_s: float) -> str` returning one of `"enter" | "wait" | "skip"`.

- [ ] **Step 1: Write the failing test**

```python
# test_entry_decision.py
from live_trader.db_decision import entry_decision


def test_enter_when_cheap_and_before_deadline():
    assert entry_decision(0.48, secs_to_close=120, max_ask=0.50, deadline_s=30) == "enter"

def test_enter_at_exactly_max_ask():
    assert entry_decision(0.50, secs_to_close=120, max_ask=0.50, deadline_s=30) == "enter"

def test_wait_when_expensive_and_before_deadline():
    assert entry_decision(0.62, secs_to_close=120, max_ask=0.50, deadline_s=30) == "wait"

def test_skip_when_past_deadline_and_not_cheap():
    assert entry_decision(0.62, secs_to_close=20, max_ask=0.50, deadline_s=30) == "skip"

def test_enter_takes_priority_at_deadline_if_cheap():
    # cheap wins even if we are at/under the deadline
    assert entry_decision(0.49, secs_to_close=20, max_ask=0.50, deadline_s=30) == "enter"

def test_wait_when_ask_missing():
    assert entry_decision(None, secs_to_close=120, max_ask=0.50, deadline_s=30) == "wait"

def test_skip_when_ask_missing_past_deadline():
    assert entry_decision(None, secs_to_close=20, max_ask=0.50, deadline_s=30) == "skip"

def test_standard_bot_enters_on_first_poll_when_max_ask_one():
    # MAX_ASK=1.0 -> any real ask <= 1.0 -> immediate enter (standard behavior)
    assert entry_decision(0.87, secs_to_close=240, max_ask=1.0, deadline_s=0) == "enter"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_entry_decision.py -q`
Expected: FAIL with `ImportError: cannot import name 'entry_decision'`

- [ ] **Step 3: Write minimal implementation**

Append to `live_trader/db_decision.py`:

```python
def entry_decision(ask, secs_to_close, max_ask, deadline_s):
    """Patient cheap-entry gate for the dbmodel bot. Side is already chosen by the
    model; this decides WHEN to act on the chosen token's current `ask`.

    Returns:
      "enter" - ask is known and <= max_ask (buy now, via FOK limit at max_ask)
      "skip"  - past the deadline with no cheap entry (give up this window)
      "wait"  - otherwise (keep polling)

    With max_ask >= 1.0 any real ask triggers "enter" on the first poll, so the
    standard bot's immediate entry-at-decision behavior is preserved.
    """
    if ask is not None and ask <= max_ask:
        return "enter"
    if secs_to_close <= deadline_s:
        return "skip"
    return "wait"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_entry_decision.py -q`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add live_trader/db_decision.py test_entry_decision.py
git commit -m "feat(cheap-entry): pure entry_decision gate (ask<=max + deadline)"
```

---

### Task 2: Config — `BOT_DBMODEL_MAX_ASK` and `BOT_DBMODEL_ENTRY_DEADLINE_S`

**Files:**
- Modify: `live_trader/config.py` (BotConfig dataclass near line 189; `load_config` near line 319)
- Test: `test_dbmodel_cheap_config.py`

**Interfaces:**
- Produces: `cfg.dbmodel_max_ask: float` (default `1.0`), `cfg.dbmodel_entry_deadline_s: float` (default `0.0`).

- [ ] **Step 1: Write the failing test**

```python
# test_dbmodel_cheap_config.py
import os
from live_trader.config import load_config

def _base_env(monkeypatch):
    # minimal env so load_config builds without external calls
    for k in ("BOT_DBMODEL_MAX_ASK", "BOT_DBMODEL_ENTRY_DEADLINE_S"):
        monkeypatch.delenv(k, raising=False)

def test_defaults_preserve_standard_bot(monkeypatch):
    _base_env(monkeypatch)
    cfg = load_config()
    assert cfg.dbmodel_max_ask == 1.0
    assert cfg.dbmodel_entry_deadline_s == 0.0

def test_cheap_values_parsed(monkeypatch):
    monkeypatch.setenv("BOT_DBMODEL_MAX_ASK", "0.50")
    monkeypatch.setenv("BOT_DBMODEL_ENTRY_DEADLINE_S", "30")
    cfg = load_config()
    assert cfg.dbmodel_max_ask == 0.50
    assert cfg.dbmodel_entry_deadline_s == 30.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_dbmodel_cheap_config.py -q`
Expected: FAIL with `AttributeError: 'BotConfig' object has no attribute 'dbmodel_max_ask'`

- [ ] **Step 3: Write minimal implementation**

In `live_trader/config.py`, add two fields to the `BotConfig` dataclass immediately after `dbmodel_delegate_redeem: bool` (line ~189):

```python
    # Cheap-entry gate (btc_5m_cheap variant). max_ask < 1.0 turns on the
    # decide-then-poll-for-a-dip path; >=1.0 keeps immediate entry-at-decision.
    dbmodel_max_ask: float
    dbmodel_entry_deadline_s: float
```

And in `load_config(...)`, immediately after the `dbmodel_delegate_redeem=...` line (~319):

```python
        dbmodel_max_ask=float(os.environ.get("BOT_DBMODEL_MAX_ASK", "1.0")),
        dbmodel_entry_deadline_s=float(os.environ.get("BOT_DBMODEL_ENTRY_DEADLINE_S", "0")),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_dbmodel_cheap_config.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add live_trader/config.py test_dbmodel_cheap_config.py
git commit -m "feat(cheap-entry): config BOT_DBMODEL_MAX_ASK + ENTRY_DEADLINE_S"
```

---

### Task 3: Executor — optional FOK-limit entry in `_execute_dbmodel_trade`

**Files:**
- Modify: `live_trader/bot.py` (`_execute_dbmodel_trade`, signature near line 1123)
- Test: `test_dbmodel_limit_entry.py`

**Interfaces:**
- Consumes: `entry_decision` (Task 1) is used by the caller (Task 4), not here.
- Produces: `_execute_dbmodel_trade(self, window, direction, p_up, drift_pct=None, live_price=None, ptb=None, limit_price=None)`. When `limit_price` is set and not dry-run, the buy goes through `self.client.buy_limit_fok(token_id, limit_price, stake/limit_price)` instead of `buy_market`. When dry-run, the hypothetical fill uses `limit_price` (if set) as the entry ask.

- [ ] **Step 1: Write the failing test**

```python
# test_dbmodel_limit_entry.py
# Verifies the dry-run hypothetical-fill entry price honors limit_price, so the
# cheap-entry path books fills at the limit, not the market snapshot ask.
import types
from live_trader import bot as botmod

def test_dryrun_hypo_entry_uses_limit_price(monkeypatch):
    # Build a HighSameBot-like stub exposing only what _execute_dbmodel_trade needs.
    b = botmod.HighSameBot.__new__(botmod.HighSameBot)
    b.cfg = types.SimpleNamespace(dry_run=True, stake_usdc=1.0, dbmodel_delegate_redeem=False)
    b._lock = __import__("threading").Lock()
    b._active = 1
    class FakeClient:
        def resolve_market(self, slug):
            return {"up_token": "UP", "down_token": "DN", "condition_id": "c"}
        def get_top_ask(self, t):
            return 0.62 if t == "UP" else 0.40   # market ask on chosen side = 0.62
        def wait_for_resolution(self, slug, deadline):
            return {"up_won": True}
    b.client = FakeClient()
    captured = {}
    def fake_log(slug, p_up, direction, our_ask, shares, fill_px, stake, won, pnl, **kw):
        captured.update(entry_ask=our_ask, shares=shares, won=won)
    monkeypatch.setattr(b, "_dbmodel_log", fake_log, raising=False)
    monkeypatch.setattr(b, "_sample_book_path", lambda *a, **k: None, raising=False)

    window = {"slug": "btc-updown-5m-1", "end_ts": 0, "ws": 0,
              "features": {}, "raw_proba": 0.7, "strike": 100.0}
    b._execute_dbmodel_trade(window, "UP", 0.7, limit_price=0.50)

    # hypothetical fill must use the 0.50 limit, NOT the 0.62 market ask
    assert captured["entry_ask"] == 0.50
    assert abs(captured["shares"] - (1.0 / 0.50)) < 1e-9
```

> Note for the implementer: if `_execute_dbmodel_trade`'s real signature/log call differs from the stub above, adapt the test to the actual `_dbmodel_log` signature you see in `bot.py` — the assertion that matters is **entry ask == limit_price (0.50), not the market ask (0.62)** in dry-run, and shares == stake/limit_price.

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_dbmodel_limit_entry.py -q`
Expected: FAIL — `_execute_dbmodel_trade()` got an unexpected keyword argument `limit_price` (or hypo entry == 0.62).

- [ ] **Step 3: Write minimal implementation**

In `live_trader/bot.py`, change the signature:

```python
    def _execute_dbmodel_trade(self, window, direction, p_up, drift_pct=None,
                               live_price=None, ptb=None, limit_price=None):
```

In the LIVE branch (where it currently calls `self.client.buy_market(token_id, stake)`), branch on `limit_price`:

```python
            if not self.cfg.dry_run:
                if limit_price is not None:
                    size = round(stake / limit_price, 2)
                    order_resp = self.client.buy_limit_fok(token_id, limit_price, size)
                else:
                    order_resp = self.client.buy_market(token_id, stake)
```

In the DRY-RUN branch, make the hypothetical entry honor `limit_price`:

```python
                hypo_ask = limit_price if limit_price is not None else our_ask
                hypo_shares = (stake / hypo_ask) if hypo_ask else None
```

and use `hypo_ask` wherever the dry-run record's entry ask is logged (replace the `our_ask` passed to `_dbmodel_log` / settlement entry with `hypo_ask`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_dbmodel_limit_entry.py -q`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full live_trader test set to confirm no regression**

Run: `python3 -m pytest test_entry_decision.py test_dbmodel_cheap_config.py test_dbmodel_limit_entry.py -q`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add live_trader/bot.py test_dbmodel_limit_entry.py
git commit -m "feat(cheap-entry): optional FOK-limit entry in _execute_dbmodel_trade"
```

---

### Task 4: Loop wiring — decide-then-poll in the dbmodel runner

**Files:**
- Modify: `run_live_bot.py` (dbmodel loop, the decision block near `if (ws not in fired and secs_to_close <= MONITOR_START_S ...)` and the surrounding `while True`)

**Interfaces:**
- Consumes: `entry_decision` (Task 1), `cfg.dbmodel_max_ask`, `cfg.dbmodel_entry_deadline_s` (Task 2), `_execute_dbmodel_trade(..., limit_price=)` (Task 3).
- Produces: live behavior — when `MAX_ASK < 1.0`, decision and entry are decoupled; otherwise unchanged.

- [ ] **Step 1: Add the import + read config + pending map**

Near the top of the dbmodel section (after `model = DbModel(model_path)` and where `WINDOW_S` is set), add:

```python
    from live_trader.db_decision import entry_decision
    MAX_ASK = cfg.dbmodel_max_ask
    DEADLINE_S = cfg.dbmodel_entry_deadline_s
    CHEAP = MAX_ASK < 1.0
```

Near `fired = set()` add:

```python
    pending = {}   # ws -> dict(slug,end_ts,direction,p_up,drift,last_px,strike,token_id,window)
```

Update the startup banner Rule/Stake lines to reflect the gate when `CHEAP`:

```python
    if CHEAP:
        print(f"  Entry gate:      ask <= {MAX_ASK:.2f}, poll 2s, skip if none by s2c={DEADLINE_S:.0f}s", flush=True)
```

- [ ] **Step 2: In the decision block, branch immediate vs pending**

The existing decision block ends by dispatching the trade in a thread. Replace the dispatch tail so that:
- when `not CHEAP`: keep the existing immediate dispatch (NO behavior change);
- when `CHEAP`: resolve the market, store a pending entry, and DO NOT dispatch yet.

```python
                # ... after `direction = "UP" if p_up >= 0.5 else "DOWN"` and the print ...
                window = {"slug": slug, "end_ts": end_ts, "ws": ws,
                          "features": dict(feats), "raw_proba": detail["raw"],
                          "strike": strike["px"]}
                if not CHEAP:
                    with bot._lock:
                        bot._active += 1
                    threading.Thread(
                        target=bot._execute_dbmodel_trade,
                        args=(window, direction, p_up, drift_pct, last_px, strike["px"]),
                        daemon=True,
                    ).start()
                    trades_dispatched += 1
                else:
                    try:
                        mkt = bot.client.resolve_market(slug)
                        token_id = mkt["up_token"] if direction == "UP" else mkt["down_token"]
                        pending[ws] = {"slug": slug, "end_ts": end_ts, "direction": direction,
                                       "p_up": p_up, "drift": drift_pct, "last_px": last_px,
                                       "strike": strike["px"], "token_id": token_id, "window": window}
                        print(f"  [DBMODEL {ts}] {slug}: decided BUY {direction}; "
                              f"waiting for ask<= {MAX_ASK:.2f}", flush=True)
                    except Exception as e:
                        print(f"  [DBMODEL {ts}] {slug}: resolve failed ({type(e).__name__}: {e}) — skip", flush=True)
                # NOTE: keep the existing max_trades handling only in the `not CHEAP` branch.
```

- [ ] **Step 3: Add the poll-and-enter handler before `time.sleep(2)`**

```python
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
                        print(f"  [DBMODEL {tsp}] {pe['slug']}: ask={ask:.3f} <= {MAX_ASK:.2f} "
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
                        print(f"  [DBMODEL {tsp}] {pe['slug']}: no ask<= {MAX_ASK:.2f} by "
                              f"s2c={DEADLINE_S:.0f}s (last ask={ask}) -> SKIP", flush=True)
                        bot._record_skip(reason="no_cheap_entry", details=f"last_ask={ask}",
                                         slug=pe["slug"], end_ts=pe["end_ts"], direction=pe["direction"],
                                         confidence=abs(pe["p_up"] - 0.5) * 200.0, ptb=pe["strike"],
                                         live_price=pe["last_px"], drift_pct=pe["drift"],
                                         final_up=(pe["p_up"] >= 0.5), top_ask_up=None,
                                         top_ask_down=None, signals=None)
                        del pending[ws_p]
                    # else "wait": leave pending for the next tick
```

> Implementer note: confirm `bot._record_skip(...)` accepts these kwargs (it is called the same way in `_execute_dbmodel_trade`'s skip path — mirror that call exactly). If a kwarg differs, match the real signature.

- [ ] **Step 4: Manual syntax + import check (no live calls)**

Run: `python3 -c "import ast; ast.parse(open('run_live_bot.py').read()); print('syntax ok')"`
Expected: `syntax ok`

- [ ] **Step 5: Commit**

```bash
git add run_live_bot.py
git commit -m "feat(cheap-entry): decide-then-poll loop wiring (gated by MAX_ASK<1)"
```

---

### Task 5: Deploy to the all-markets server + dry-run smoke + go live

**Files:**
- Create (server): `live_env/btc_5m_cheap.env`
- Copy (server): `live_trader/db_decision.py`, `live_trader/config.py`, `live_trader/bot.py`, `run_live_bot.py`

- [ ] **Step 1: Copy the changed code to the server**

```bash
H=ubuntu@<server-host>
K=~/.ssh/all_markets.pem
cd /home/vidura/btcpredictor/predictor
for f in live_trader/db_decision.py live_trader/config.py live_trader/bot.py run_live_bot.py; do
  scp -o StrictHostKeyChecking=no -i $K "$f" "$H:~/btcpredictor/predictor/$f"
done
```

- [ ] **Step 2: Create the cheap env file on the server**

```bash
ssh -o StrictHostKeyChecking=no -i $K $H 'cat > ~/btcpredictor/predictor/live_env/btc_5m_cheap.env <<EOF
BOT_DBMODEL_SYMBOL=btc
BOT_DBMODEL_PATH=models/db_ptb.joblib
BOT_STAKE_USDC=1.0
BOT_DBMODEL_MAX_ASK=0.50
BOT_DBMODEL_ENTRY_DEADLINE_S=30
BOT_DBMODEL_LOG=/home/ubuntu/btcpredictor/predictor/live_btc_5m_cheap.jsonl
EOF
echo written; cat ~/btcpredictor/predictor/live_env/btc_5m_cheap.env'
```

- [ ] **Step 3: Dry-run smoke (NO orders) — verify decide→poll→enter/skip**

```bash
ssh -o StrictHostKeyChecking=no -i $K $H 'cd ~/btcpredictor/predictor &&
  set -a && . ./.env && . ./live_env/btc_5m_cheap.env && export BOT_DRY_RUN=true && set +a &&
  timeout 360 /home/ubuntu/btcpredictor/.venv/bin/python3 -u run_live_bot.py --dbmodel 2>&1 | head -40'
```
Expected: banner shows `Entry gate: ask <= 0.50 ...`, a `decided BUY <side>; waiting for ask<= 0.50` line, then either an `ENTER (FOK limit)` (DRY RUN) or a `SKIP` line within ~5 min. NO real order placed (dry-run).

- [ ] **Step 4: Start live**

```bash
ssh -o StrictHostKeyChecking=no -i $K $H 'sudo systemctl start dblive@btc_5m_cheap && sleep 6 && systemctl is-active dblive@btc_5m_cheap'
```
Expected: `active`.

- [ ] **Step 5: Verify the live banner shows the gate + LIVE $1**

```bash
ssh -o StrictHostKeyChecking=no -i $K $H 'sudo journalctl -u dblive@btc_5m_cheap --no-pager -n 25 | grep -ivE "secret|key|passphrase"'
```
Expected: `*** LIVE — REAL $1 orders ***`, `Entry gate: ask <= 0.50`, funder `0xFUNDER...`. Confirm the standard `dblive@btc_5m` is still `active` and unaffected.

---

## Self-Review

- **Spec coverage:** parametrized path (Task 2) ✓; decide-then-poll loop + deadline skip (Task 4) ✓; FOK-limit at MAX_ASK (Task 3) ✓; pure `entry_decision` + tests (Task 1) ✓; separate `dblive@btc_5m_cheap` instance, own log, $1, server deploy + dry-run smoke (Task 5) ✓; standard bot unchanged (Task 1 test `max_ask=1.0`, Task 2 defaults test, Task 4 `not CHEAP` branch) ✓.
- **Placeholder scan:** none — all steps carry concrete code/commands. Two implementer notes flag where a real signature must be matched (intentional, with the invariant to preserve stated).
- **Type consistency:** `entry_decision(ask, secs_to_close, max_ask, deadline_s)` identical across Tasks 1 and 4; `limit_price` kwarg identical across Tasks 3 and 4; `cfg.dbmodel_max_ask` / `cfg.dbmodel_entry_deadline_s` identical across Tasks 2 and 4.
