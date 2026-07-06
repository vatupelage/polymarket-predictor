"""MMLogger — read-only market-making data collector for Polymarket BTC 5m.

Spec: predictor/MM_DATA_SPEC.md. Collects the four streams needed to backtest
whether providing liquidity (resting limit orders) is profitable, WITHOUT placing
any orders or risking capital:

  mm_book.jsonl       full depth (price+size, both UP & DOWN), one record per
                      token per snapshot.
  mm_tape.jsonl       executed trades (the tape) — dedup'd by trade id.
  mm_sim_quotes.jsonl hypothetical maker quotes we WOULD post + queue-ahead size,
                      so fills can be reconstructed offline from the tape.
  mm_rewards.jsonl    liquidity-mining reward schedule, once per window.

Usage (from the bot, additive — never gates trading):
    self.mm = MMLogger(client, log_dir, cadence_s=2.0, enabled=cfg.mm_log_enabled)
    self.mm.start_window(slug, up_token, down_token, condition_id, end_ts)

Each window runs in its own daemon thread until end_ts. Re-calling start_window
for an already-active slug is a no-op, so it's safe to call on every prediction.
"""

import datetime
import json
import os
import threading
import time


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="milliseconds")


def _microprice(bids, asks):
    """Size-weighted mid: (best_bid*ask_sz + best_ask*bid_sz)/(bid_sz+ask_sz)."""
    if not bids or not asks:
        return None
    bp, bs = bids[0]
    ap, as_ = asks[0]
    denom = bs + as_
    if denom <= 0:
        return (bp + ap) / 2.0
    return (bp * as_ + ap * bs) / denom


def _depth_within(levels, ref, cents):
    """Total size resting within `cents` of the reference price."""
    band = cents / 100.0
    return sum(s for p, s in levels if abs(p - ref) <= band + 1e-9)


def _size_ahead(levels, price):
    """Size already resting at `price` (our queue position if we joined it)."""
    return sum(s for p, s in levels if abs(p - price) < 1e-9)


class MMLogger:
    def __init__(self, client, log_dir, cadence_s: float = 2.0,
                 quote_dists_c=(1, 2, 3), enabled: bool = True):
        self.client = client
        self.log_dir = log_dir
        self.cadence_s = cadence_s
        self.quote_dists_c = quote_dists_c
        self.enabled = enabled
        self._active = set()
        self._lock = threading.Lock()
        self._book_path = os.path.join(log_dir, "mm_book.jsonl")
        self._tape_path = os.path.join(log_dir, "mm_tape.jsonl")
        self._quote_path = os.path.join(log_dir, "mm_sim_quotes.jsonl")
        self._reward_path = os.path.join(log_dir, "mm_rewards.jsonl")
        self._write_lock = threading.Lock()

    def start_window(self, slug, up_token, down_token, condition_id, end_ts):
        if not self.enabled:
            return
        with self._lock:
            if slug in self._active:
                return
            self._active.add(slug)
        threading.Thread(
            target=self._run,
            args=(slug, up_token, down_token, condition_id, end_ts),
            name=f"mmlog-{slug[-10:]}",
            daemon=True,
        ).start()

    def start_continuous(self):
        """Self-clocked loop: log EVERY consecutive 5-min window, independent of
        the trading loop (which only predicts every *other* window). Resolves the
        next window's tokens, logs it start-to-close, then rolls to the next.
        Runs forever in its own daemon thread."""
        if not self.enabled:
            return
        threading.Thread(target=self._continuous_loop, name="mmlog-continuous",
                         daemon=True).start()
        print(f"  [MM] continuous logger started — capturing every 5-min window "
              f"@ {self.cadence_s:.0f}s cadence -> {os.path.basename(self.log_dir)}/mm_*.jsonl")

    def _continuous_loop(self):
        while True:
            try:
                now = int(time.time())
                window_start = (now // 300) * 300
                end_ts = window_start + 300
                slug = f"btc-updown-5m-{window_start}"
                with self._lock:
                    already = slug in self._active
                if not already:
                    mkt = self._resolve(slug)
                    # Gamma often hasn't listed a just-opened window yet (we wake
                    # ~0.5s into it). Retry within the window instead of skipping
                    # the whole 300s — a brief listing delay shouldn't cost a full
                    # window of data/rebate. Give up only if <30s remain (not worth
                    # logging a near-closed window).
                    while mkt is None and time.time() < end_ts - 30:
                        time.sleep(2.0)
                        mkt = self._resolve(slug)
                    if mkt is not None:
                        self.start_window(slug, mkt["up_token"], mkt["down_token"],
                                          mkt.get("condition_id"), end_ts)
                # wake shortly after the next window opens
                time.sleep(max(1.0, end_ts - time.time() + 0.5))
            except Exception as e:
                self._append(self._book_path, {
                    "ts": _now_iso(), "continuous_loop_error": f"{type(e).__name__}: {e}",
                })
                time.sleep(5.0)

    def _resolve(self, slug):
        """Resolve slug -> market dict via the client, tolerating markets that
        aren't listed yet (next window often isn't on Gamma until it opens)."""
        try:
            return self.client.resolve_market(slug)
        except Exception:
            return None

    # --- internal ---------------------------------------------------------

    def _append(self, path, record):
        line = json.dumps(record, separators=(",", ":"))
        with self._write_lock:
            with open(path, "a") as f:
                f.write(line + "\n")

    def _run(self, slug, up_token, down_token, condition_id, end_ts):
        try:
            self._log_rewards(slug, condition_id)
            tick_up = self.client.get_tick_size(up_token)
            tick_dn = self.client.get_tick_size(down_token)
            seen_trades = set()
            last_book_hash = {"UP": None, "DOWN": None}
            while time.time() < float(end_ts):
                snap_ts = _now_iso()
                secs_to_close = int(float(end_ts) - time.time())
                for token, token_id, tick in (
                    ("UP", up_token, tick_up), ("DOWN", down_token, tick_dn)
                ):
                    book = self.client.get_full_book(token_id)
                    if book is None:
                        continue
                    self._log_book(slug, snap_ts, secs_to_close, token, token_id,
                                   tick, book, last_book_hash)
                    self._log_sim_quotes(slug, snap_ts, secs_to_close, token, book, tick)
                self._log_tape(slug, condition_id, snap_ts, secs_to_close, seen_trades)
                # sleep the remainder of the cadence
                time.sleep(self.cadence_s)
        except Exception as e:  # never let a logging error touch trading
            self._append(self._book_path, {
                "ts": _now_iso(), "slug": slug, "error": f"{type(e).__name__}: {e}",
            })
        finally:
            with self._lock:
                self._active.discard(slug)

    def _log_rewards(self, slug, condition_id):
        if not condition_id:
            return
        rewards = self.client.get_market_rewards(condition_id)
        self._append(self._reward_path, {
            "ts": _now_iso(), "slug": slug, "condition_id": condition_id,
            "rewards": rewards,
        })

    def _log_book(self, slug, ts, secs, token, token_id, tick, book, last_hash):
        bids, asks = book["bids"], book["asks"]
        bhash = book.get("hash")
        # dedupe identical consecutive snapshots (book unchanged)
        if bhash is not None and bhash == last_hash.get(token):
            return
        last_hash[token] = bhash
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        mid = ((best_bid + best_ask) / 2.0) if (best_bid is not None and best_ask is not None) else None
        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None
        self._append(self._book_path, {
            "ts": ts, "slug": slug, "secs_to_close": secs,
            "token": token, "token_id": token_id, "tick_size": tick,
            "bids": bids, "asks": asks, "book_hash": bhash,
            "best_bid": best_bid, "best_ask": best_ask,
            "mid": mid, "spread": spread,
            "bid_depth_2c": _depth_within(bids, mid, 2) if mid is not None else None,
            "ask_depth_2c": _depth_within(asks, mid, 2) if mid is not None else None,
            "microprice": _microprice(bids, asks),
        })

    def _log_sim_quotes(self, slug, ts, secs, token, book, tick):
        bids, asks = book["bids"], book["asks"]
        if not bids or not asks:
            return
        mid = (bids[0][0] + asks[0][0]) / 2.0
        step = tick if (tick and tick > 0) else 0.01
        quotes = []
        for d in self.quote_dists_c:
            bid_px = round(mid - d * step, 6)
            ask_px = round(mid + d * step, 6)
            quotes.append({"side": "BID", "price": bid_px, "dist_c": d,
                           "size_ahead": _size_ahead(bids, bid_px)})
            quotes.append({"side": "ASK", "price": ask_px, "dist_c": d,
                           "size_ahead": _size_ahead(asks, ask_px)})
        self._append(self._quote_path, {
            "ts": ts, "slug": slug, "token": token,
            "mid_at_quote": mid, "secs_to_close": secs, "tick_size": tick,
            "quotes": quotes,
        })

    def _log_tape(self, slug, condition_id, ts, secs, seen_trades):
        if not condition_id:
            return
        tape = self.client.get_tape(condition_id)
        if not tape:
            return
        for t in tape:
            # stable dedup key: one tx can have multiple legs (different assets)
            tid = f"{t.get('tx')}-{t.get('asset')}-{t.get('size')}-{t.get('price')}"
            if tid in seen_trades:
                continue
            seen_trades.add(tid)
            self._append(self._tape_path, {
                "ts": ts, "slug": slug, "secs_to_close": secs,
                "trade_id": tid, "trade": t,
            })
