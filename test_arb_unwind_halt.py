#!/usr/bin/env python3
"""Verify the two arb safety fixes without touching the live market:

  1. sell_market trusts the KNOWN fill qty when the balance API lags (0.0),
     instead of skipping the unwind (the 19:10 'balance too low (have 0.0)' bug).
  2. arb_executor flattens a one-leg fill with known_qty AND trips the
     kill-switch, refusing further live trades until restart.

Run:  python3 test_arb_unwind_halt.py
"""
import os, sys, time, tempfile, types
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

import live_trader.polymarket as pm
from live_trader.polymarket import PolymarketBotClient
from live_trader.arb_executor import ArbExecutor

pm.time.sleep = lambda *_a, **_k: None   # don't actually wait out the retry backoff
PASS, FAIL = [], []
def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(("  ✓ " if cond else "  ✗ ") + name)


# ---------- Fix 1: sell_market known_qty fallback / retry ----------
def make_sellable_self(balance_seq):
    seq = list(balance_seq)
    placed = {"posted": False}
    def get_bal(_tok):
        return seq.pop(0) if len(seq) > 1 else seq[0]
    fake_client = types.SimpleNamespace(
        get_order_book=lambda _t: {"bids": [[0.34, 100]]},
        create_market_order=lambda _a: "signed",
        post_order=lambda *_a, **_k: placed.__setitem__("posted", True) or {"status": "sold"},
    )
    s = object.__new__(PolymarketBotClient)
    s.cfg = types.SimpleNamespace(dry_run=True, signature_type=0)
    s._client = fake_client
    s.get_conditional_balance = get_bal
    return s, placed

print("Fix 1 — sell_market unwind vs lagging balance:")
# (a) balance stuck at 0.0 but we KNOW 5 shares filled -> must still sell
s, placed = make_sellable_self([0.0])
r = PolymarketBotClient.sell_market(s, "DOWN", 5.0, force=True, known_qty=5.0)
check("lagging 0.0 balance + known_qty -> sell is PLACED (not skipped)",
      placed["posted"] and r == {"status": "sold"})

# (b) balance catches up on a retry -> uses real balance, sells
s, placed = make_sellable_self([0.0, 5.0])
r = PolymarketBotClient.sell_market(s, "DOWN", 5.0, force=True, known_qty=5.0)
check("balance catches up on retry -> sell is PLACED", placed["posted"])

# (c) regression: NO known_qty + 0.0 balance -> still safely skips (unchanged)
s, placed = make_sellable_self([0.0])
r = PolymarketBotClient.sell_market(s, "DOWN", 5.0, force=True)
check("no known_qty + 0.0 balance -> skips (existing behavior preserved)",
      (not placed["posted"]) and "skipped" in r)


# ---------- Fix 2/3/4: executor unwind + kill-switch ----------
print("\nFix 2 — executor one-leg fill -> flatten(known_qty) + halt:")
class FakeClient:
    def __init__(self):
        self.buy_calls = 0
        self.sell_calls = []
        self.sign_order = []          # order legs were signed in
        self.post_order = []          # order legs were posted in
    # Tier-1 split path: executor signs both, then posts both concurrently.
    def sign_limit_fok(self, token, px, size):
        self.sign_order.append(token)
        return {"signed": ("ord", token, px, size)}
    def post_signed_fok(self, signed):
        self.buy_calls += 1
        token = signed[1]
        self.post_order.append(token)
        if token == "UP":               # cheap thin leg gets sniped -> killed
            return {"killed": True}
        return {"takingAmount": signed[3]}    # DOWN leg fills
    def sell_market(self, tok, sh, force=False, known_qty=None):
        self.sell_calls.append((tok, sh, force, known_qty))
        return {"status": "sold"}

cfg = types.SimpleNamespace(
    arb_enabled=True, arb_dry_run=False, dry_run=True,
    arb_max_usdc_per_leg=5.0, arb_max_size=5, arb_min_size=5,
    arb_max_per_window=1, arb_min_edge=0.01,
)
tmp = tempfile.mkdtemp()
fc = FakeClient()
ex = ArbExecutor(fc, cfg, tmp, console=False)

# au_px+ad_px = 0.95 -> gross_edge 0.05; depth 50 each -> size floors to 5 (max)
acted = ex._execute_arb("btc-updown-5m-test", "UP", "DOWN",
                        au_px=0.50, au_sz=50, ad_px=0.45, ad_sz=50,
                        gross_edge=0.05, exch_min=5)
time.sleep(0.2)  # let async log writer drain

check("one-leg fill is detected as an action", acted is True)
check("Tier-1: BOTH legs signed BEFORE either is posted",
      fc.sign_order == ["UP", "DOWN"] and fc.post_order == ["UP", "DOWN"])
check("unwind called sell_market with known_qty == filled shares (5.0)",
      fc.sell_calls == [("DOWN", 5.0, True, 5.0)])
check("kill-switch tripped (_halted = True)", ex._halted is True)
check("last_event shows HALTED", "HALTED" in ex.stats["last_event"])

# second live attempt must be refused without firing any new legs
buys_before = fc.buy_calls
acted2 = ex._execute_arb("btc-updown-5m-test2", "UP", "DOWN",
                         au_px=0.50, au_sz=50, ad_px=0.45, ad_sz=50,
                         gross_edge=0.05, exch_min=5)
check("after halt, next live arb is refused (returns False)", acted2 is False)
check("after halt, NO new legs fired", fc.buy_calls == buys_before)

# ---------- Fix 3: redeem_position no-ops under dry_run unless forced ----------
print("\nFix 3 — redeem_position must claim in arb mode (BOT_DRY_RUN=true):")
def make_redeem_self(dry_run):
    calls = {"built": False}
    # patch web3 so no network: redeem_position imports web3 lazily
    import live_trader.polymarket as _pm
    class _FakeCall:
        def call(self, *_a, **_k): return True
        def build_transaction(self, *_a, **_k):
            calls["built"] = True
            raise RuntimeError("stop-before-network")  # we only assert it got this far
    class _FakeFns:
        def redeemPositions(self, *_a, **_k): return _FakeCall()
    class _FakeContract:
        functions = _FakeFns()
    class _FakeEth:
        def __init__(s): s.account = _pm  # unused path
    s = object.__new__(PolymarketBotClient)
    s.cfg = types.SimpleNamespace(dry_run=dry_run, private_key="0x"+"11"*32)
    s._tx_lock = __import__("threading").Lock()
    return s, calls, _FakeContract()

# (a) dry_run + NOT forced -> returns None immediately, builds nothing
import live_trader.polymarket as _pm
s, calls, _ = make_redeem_self(dry_run=True)
r = PolymarketBotClient.redeem_position(s, "0x"+"00"*32)  # force defaults False
check("dry_run + not forced -> returns None (no claim)", r is None and not calls["built"])

# (b) dry_run + forced -> bypasses the gate (proceeds past the dry_run guard).
# We patch web3 to confirm it gets past the guard into the build path.
import sys as _sys, types as _t
fake_web3 = _t.ModuleType("web3")
class _W3:
    def __init__(s,*a,**k):
        s.eth = _t.SimpleNamespace(
            account=_t.SimpleNamespace(from_key=lambda k: _t.SimpleNamespace(address="0xabc")),
            get_transaction_count=lambda a: 0, gas_price=1, )
    @staticmethod
    def HTTPProvider(*a,**k): return None
    @staticmethod
    def to_checksum_address(a): return a
    def eth_contract(s): pass
fake_web3.Web3 = _W3
_real = _sys.modules.get("web3")
_sys.modules["web3"] = fake_web3
try:
    s2 = object.__new__(PolymarketBotClient)
    s2.cfg = types.SimpleNamespace(dry_run=True, private_key="0x"+"11"*32)
    s2._tx_lock = __import__("threading").Lock()
    passed_guard = {"v": False}
    # monkeypatch contract creation to flag we got past the dry_run return
    orig_contract = _W3.__init__
    # call: it will build a contract via w3.eth.contract -> attribute missing -> raises,
    # but only AFTER passing the dry_run guard, which is what we assert.
    try:
        PolymarketBotClient.redeem_position(s2, "0x"+"00"*32, force=True)
    except Exception:
        passed_guard["v"] = True  # got past guard into web3 path (then failed on fake)
    check("dry_run + force=True -> bypasses guard, enters claim path",
          passed_guard["v"] is True)
finally:
    if _real is not None: _sys.modules["web3"] = _real
    else: _sys.modules.pop("web3", None)


# ---------- Fix 4: direct-by-condition_id redemption ----------
print("\nFix 4 — direct-by-condition_id redemption (no data-api cap):")

class RedeemClient:
    """Fake client: one leg fills (locked), records redeem calls, can defer the
    oracle (return None) before paying out (return a tx hash)."""
    def __init__(self, oracle_ready=True):
        self.oracle_ready = oracle_ready
        self.redeemed = []
        self.resolve_calls = 0
    def resolve_market(self, slug):
        self.resolve_calls += 1
        return {"up_token": "UP", "down_token": "DOWN",
                "condition_id": "0xCOND_" + slug.rsplit("-", 1)[1]}
    def sign_limit_fok(self, token, px, size):
        return {"signed": ("ord", token, px, size)}
    def post_signed_fok(self, signed):
        return {"takingAmount": signed[3]}       # BOTH legs fill -> ARB_LOCKED
    def get_conditional_balance(self, tok):
        return 5.0                               # locked arb holds both legs
    def sell_market(self, *a, **k):
        return {"status": "sold"}
    def redeem_position(self, cond, force=False):
        if not self.oracle_ready:
            return None                          # oracle not ready -> stays pending
        self.redeemed.append((cond, force))
        return "0xtxhash_" + cond[-4:]

# a locked arb in a window that has already resolved (slug ts in the past)
past_ws = int(time.time()) - 1000               # window start 1000s ago -> resolved
slug = f"btc-updown-5m-{past_ws}"
tmp2 = tempfile.mkdtemp()
rc = RedeemClient(oracle_ready=True)
cfg2 = types.SimpleNamespace(
    arb_enabled=True, arb_dry_run=False, dry_run=True,
    arb_max_usdc_per_leg=5.0, arb_max_size=5, arb_min_size=5,
    arb_max_per_window=1, arb_min_edge=0.01, arb_deadline_buffer_s=0)
ex2 = ArbExecutor(rc, cfg2, tmp2, console=False)
ex2._execute_arb(slug, "UP", "DOWN", au_px=0.50, au_sz=50, ad_px=0.45, ad_sz=50,
                 gross_edge=0.05, exch_min=5)
time.sleep(0.1)
check("locked arb is tracked for redemption (condition_id recorded)",
      len(ex2._pending) == 1 and any("0xCOND_" in c for c in ex2._pending))

# pending list survives a restart (persisted to disk)
ex2b = ArbExecutor(rc, cfg2, tmp2, console=False)
check("pending redemptions persist across restart", len(ex2b._pending) == 1)

# resolved window -> redeem_resolved claims it directly with force=True
res = ex2b.redeem_resolved()
check("redeem_resolved claims the resolved window (force=True)",
      res["redeemed"] == 1 and rc.redeemed and rc.redeemed[0][1] is True)
check("claimed condition is removed from pending", len(ex2b._pending) == 0)

# oracle-not-ready -> stays pending, retried later (no data loss)
rc2 = RedeemClient(oracle_ready=False)
tmp3 = tempfile.mkdtemp()
ex3 = ArbExecutor(rc2, cfg2, tmp3, console=False)
ex3._mark_for_redeem(slug)
res3 = ex3.redeem_resolved()
check("oracle-not-ready -> nothing redeemed, kept pending",
      res3["redeemed"] == 0 and res3["pending"] == 1)

# a window that has NOT resolved yet (future end_ts) is skipped
tmp4 = tempfile.mkdtemp()
ex4 = ArbExecutor(RedeemClient(True), cfg2, tmp4, console=False)
future_slug = f"btc-updown-5m-{int(time.time())+600}"
ex4._mark_for_redeem(future_slug)
res4 = ex4.redeem_resolved()
check("unresolved (future) window is NOT redeemed yet", res4["redeemed"] == 0)


# ---------- Tier-0: WebSocket price_change delta handling ----------
# Uses the REAL protocol captured from the live socket 2026-05-30: deltas live
# under "price_changes" (plural), each entry carrying its OWN asset_id.
print("\nTier-0 — WS keeps top-of-book live via price_change deltas:")
from live_trader.arb_ws import _load_asks, _apply_price_changes, _best_ask

UP, DN = "tokUP", "tokDN"
books = {UP: {}, DN: {}}
_load_asks(books[UP], [{"price": "0.40", "size": "100"}, {"price": "0.42", "size": "50"}])
_load_asks(books[DN], [{"price": "0.55", "size": "80"}])
check("book snapshot loads asks; best = lowest price", _best_ask(books[UP]) == (0.40, 100.0))

# a real price_change event bundles BOTH tokens; each routed by its asset_id
ev = {"event_type": "price_change", "market": "0xcond", "price_changes": [
    {"asset_id": UP, "price": "0.38", "side": "SELL", "size": "20", "best_ask": "0.38"},
    {"asset_id": DN, "price": "0.53", "side": "SELL", "size": "30", "best_ask": "0.53"},
]}
check("price_changes routed per asset_id -> both books move",
      _apply_price_changes(books, ev) is True
      and _best_ask(books[UP]) == (0.38, 20.0) and _best_ask(books[DN]) == (0.53, 30.0))

# the cheap UP level is taken (size 0) -> removed, best reverts to 0.40
_apply_price_changes(books, {"price_changes": [
    {"asset_id": UP, "price": "0.38", "side": "SELL", "size": "0"}]})
check("size 0 removes the level -> UP best reverts to 0.40", _best_ask(books[UP]) == (0.40, 100.0))

# BUY-side (bid) changes are ignored — we only trade the ask
before = dict(books[UP])
_apply_price_changes(books, {"price_changes": [
    {"asset_id": UP, "price": "0.10", "side": "BUY", "size": "999"}]})
check("bid-side change ignored (asks unchanged)", books[UP] == before)

# changes for an untracked asset_id are ignored (no crash, no effect)
before_all = (dict(books[UP]), dict(books[DN]))
moved = _apply_price_changes(books, {"price_changes": [
    {"asset_id": "SOMEONE_ELSE", "price": "0.01", "side": "SELL", "size": "5"}]})
check("untracked asset_id ignored", moved is False
      and (dict(books[UP]), dict(books[DN])) == before_all)


# ---------- Tier-0: token meta cache feeds create_order (no hot-path GETs) ----------
print("\nTier-0 — warm_token caches tick/neg_risk; order uses it (no network):")
import live_trader.polymarket as pmmod

class MetaClient:
    """Fake CLOB client recording how often tick/neg_risk are fetched and what
    options reach create_order."""
    def __init__(self):
        self.tick_calls = 0
        self.neg_calls = 0
        self.create_opts = []
    def get_tick_size(self, t): self.tick_calls += 1; return "0.001"
    def get_neg_risk(self, t): self.neg_calls += 1; return False
    def create_order(self, args, options=None):
        self.create_opts.append(options)
        return {"signed": True}
    def post_order(self, signed, otype): return {"takingAmount": 5.0}

s = object.__new__(PolymarketBotClient)
s.cfg = types.SimpleNamespace(dry_run=True)
s._tok_meta = {}
s._client = MetaClient()

# warm once, then place two orders -> tick/neg_risk fetched ONCE total, not per order
s.warm_token("TOK")
PolymarketBotClient.buy_limit_fok(s, "TOK", 0.40, 5)
PolymarketBotClient.buy_limit_fok(s, "TOK", 0.41, 5)
check("warm_token fetched tick/neg_risk exactly once (cached)",
      s._client.tick_calls == 1 and s._client.neg_calls == 1)
opts = s._client.create_opts
check("create_order received PartialCreateOrderOptions both times (no hot-path lookup)",
      len(opts) == 2 and all(o is not None and o.neg_risk is False and o.tick_size == "0.001" for o in opts))


# ---------- QA: no hot-path get_full_book; window prune ----------
print("\nQA — execute_from_ws makes NO hot-path network call; prune works:")
class WSClient:
    # deliberately NO get_full_book: if the hot path called it -> AttributeError
    def warm_token(self, t): return ("0.001", False)
    def sign_limit_fok(self, *a): return {"signed": ("o",) + a}
    def post_signed_fok(self, s): return {"takingAmount": s[3]}
    def resolve_market(self, slug): return None

cfgws = types.SimpleNamespace(
    arb_enabled=True, arb_dry_run=True, dry_run=True, arb_max_usdc_per_leg=5.0,
    arb_max_size=5, arb_min_size=5, arb_max_per_window=1, arb_min_edge=0.01,
    arb_deadline_buffer_s=0)
tmpq = tempfile.mkdtemp()
exq = ArbExecutor(WSClient(), cfgws, tmpq, console=False)
# no cached min for this slug -> must fall back to arb_min_size WITHOUT get_full_book
exq.execute_from_ws(f"btc-updown-5m-{int(time.time())}", "UP", "DOWN",
                    0.50, 50, 0.45, 50, time.time())
time.sleep(0.1)
check("execute_from_ws ran with no get_full_book (dry detection logged)",
      exq.stats["detections"] == 1)

old = f"btc-updown-5m-{int(time.time()) - 100000}"
new = f"btc-updown-5m-{int(time.time())}"
exq._win_min_size[old] = 5; exq._win_min_size[new] = 5
exq._noted.update({old, new})
exq._prune_old_windows(int(time.time()) - 1800)
check("prune drops old window, keeps recent",
      old not in exq._win_min_size and new in exq._win_min_size
      and old not in exq._noted and new in exq._noted)


# ---------- QA round 2: ambiguous-skip reconciliation + residual gate ----------
print("\nQA2 — ambiguous post-failure reconciled to a real naked leg:")

class AmbiguousClient:
    """UP leg's POST times out AMBIGUOUSLY (skipped, but actually FILLED on-chain);
    DOWN leg is cleanly killed. Naively this looks like both_killed -> flat, but
    reconciliation against the on-chain balance must catch the naked UP leg."""
    def __init__(self):
        self.sell_calls = []
        self.balances = {"UP": 5.0, "DOWN": 0.0}   # UP actually filled despite the timeout
    def sign_limit_fok(self, token, px, size): return {"signed": ("o", token, px, size)}
    def post_signed_fok(self, signed):
        return ({"skipped": "post_order failed: timeout"} if signed[1] == "UP"
                else {"killed": True})
    def get_conditional_balance(self, tok): return self.balances.get(tok, 0.0)
    def sell_market(self, tok, sh, force=False, known_qty=None):
        self.sell_calls.append((tok, sh)); self.balances[tok] = 0.0
        return {"status": "sold"}
    def resolve_market(self, slug):
        return {"condition_id": "0xAMB", "up_token": "UP", "down_token": "DOWN"}

ac = AmbiguousClient()
tmpa = tempfile.mkdtemp()
exa = ArbExecutor(ac, cfg2, tmpa, console=False)
acted = exa._execute_arb("btc-updown-5m-test-amb", "UP", "DOWN",
                         au_px=0.50, au_sz=50, ad_px=0.45, ad_sz=50,
                         gross_edge=0.05, exch_min=5)
time.sleep(0.1)
check("ambiguous skip + on-chain balance -> detected as naked leg, UNWOUND",
      ac.sell_calls == [("UP", 5.0)])
check("kill-switch tripped after reconciled naked leg", exa._halted is True)

print("\nQA2 — redeem_resolved balance gate skips flat conditions (no gas):")
class GateClient:
    def __init__(self, held): self.held = held; self.redeemed = []
    def get_conditional_balance(self, tok): return self.held.get(tok, 0.0)
    def redeem_position(self, cond, force=False):
        self.redeemed.append(cond); return "0xtx_" + cond[-3:]

# flat after a clean unwind: both tokens 0 -> gate drops, no redeem tx
past = int(time.time()) - 1000
gc_flat = GateClient({"U": 0.0, "D": 0.0})
exg = ArbExecutor(gc_flat, cfg2, tempfile.mkdtemp(), console=False)
exg._pending = {"0xFLAT": {"slug": f"btc-updown-5m-{past}", "end_ts": past + 300,
                           "tokens": ["U", "D"]}}
res = exg.redeem_resolved()
check("flat condition -> NOT redeemed (gas saved), dropped from pending",
      gc_flat.redeemed == [] and len(exg._pending) == 0)

# residual on the winning token -> gate passes, redeem happens
gc_res = GateClient({"U": 4.9, "D": 0.0})
exg2 = ArbExecutor(gc_res, cfg2, tempfile.mkdtemp(), console=False)
exg2._pending = {"0xRES": {"slug": f"btc-updown-5m-{past}", "end_ts": past + 300,
                           "tokens": ["U", "D"]}}
res2 = exg2.redeem_resolved()
check("residual position -> redeemed (claimed)", gc_res.redeemed == ["0xRES"])


# ---------- QA round 3: invalid-price guard, no crash ----------
print("\nQA3 — zero/invalid price is rejected, no ZeroDivisionError:")
class CrashClient:
    def sign_limit_fok(self, *a): raise AssertionError("should never reach signing")
    def post_signed_fok(self, *a): raise AssertionError("should never post")
tmpz = tempfile.mkdtemp()
exz = ArbExecutor(CrashClient(), cfg2, tmpz, console=False)
# au_px=0 would be a div-by-zero in sizing; must be rejected before that
r_bad = exz._execute_arb("btc-updown-5m-bad", "U", "D",
                         au_px=0.0, au_sz=50, ad_px=0.45, ad_sz=50,
                         gross_edge=0.55, exch_min=5)
check("zero ask price -> rejected (no crash, no order)", r_bad is False)
r_bad2 = exz._execute_arb("btc-updown-5m-bad2", "U", "D",
                          au_px=0.50, au_sz=50, ad_px=1.0, ad_sz=50,
                          gross_edge=-0.5, exch_min=5)
check("price >= 1.0 -> rejected", r_bad2 is False)

# prune handles a missing client _tok_meta gracefully (no AttributeError)
exz._prune_old_windows(int(time.time()))
check("prune runs clean even when client has no _tok_meta", True)


print(f"\n{'='*50}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL); sys.exit(1)
print("ALL GREEN")
