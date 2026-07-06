"""WebSocket fast-path for the arb executor (Tier-1 latency cut).

Replaces the 1-second REST poll with Polymarket's push stream
(wss://ws-subscriptions-clob.polymarket.com/ws/market). You see an arb the
instant the book changes instead of up to 1s later — the single biggest source
of staleness. Protocol:
  subscribe: {"type":"market","assets_ids":[...]}
  events (verified against the live socket 2026-05-30):
    "book"         = full snapshot, TOP-LEVEL asset_id, {bids:[...],asks:[{price,size}]}.
    "price_change" = incremental deltas under key "price_changes" (NOT "changes");
                     each entry carries its OWN asset_id + {price, side, size}
                     (side SELL = ask). size 0 removes the level. We APPLY these
                     to a per-token ask book so top-of-book stays live BETWEEN
                     snapshots — earlier versions ignored deltas (and assumed the
                     wrong field name) and froze on the snapshot, which is why WS
                     was disabled. Now matched to the real protocol.

Runs its own asyncio loop in a daemon thread. On each update it refreshes the
ask book, recomputes best ask, and evaluates the arb condition; when it fires,
execution is offloaded to a threadpool so the receive loop keeps draining (never
blocks on the ~hundreds-ms REST order placement).
"""

import asyncio
import json
import threading
import time

try:
    import websockets
    _HAVE_WS = True
except Exception:
    _HAVE_WS = False

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

_ASK_SIDES = ("sell", "ask")   # price_change side values that mean the ask side


def _load_asks(store: dict, levels):
    """Replace an ask book from a full snapshot. store = {price: size}."""
    store.clear()
    for lv in levels or []:
        try:
            p = float(lv["price"]); s = float(lv["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if s > 0:
            store[p] = s


def _apply_price_changes(books: dict, msg: dict) -> bool:
    """Apply a price_change event to per-token ask books. Returns True if any
    tracked token's asks moved.

    Real protocol: msg["price_changes"] is a list where EACH entry carries its
    own asset_id + {price, side, size} (side SELL = ask, size 0 removes the
    level). `books` maps token_id -> {price: size}; each change is routed to its
    own asset_id's book. Changes for untracked assets / bid side are ignored.
    """
    touched = False
    for ch in msg.get("price_changes") or []:
        store = books.get(ch.get("asset_id"))
        if store is None:
            continue
        if str(ch.get("side", "")).lower() not in _ASK_SIDES:
            continue
        try:
            p = float(ch["price"]); s = float(ch["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if s <= 0:
            if store.pop(p, None) is not None:
                touched = True
        elif store.get(p) != s:
            store[p] = s
            touched = True
    return touched


def _best_ask(store: dict):
    """(price, size) of the lowest ask with positive size, or (None, 0)."""
    best_p, best_s = None, 0.0
    for p, s in store.items():
        if s > 0 and (best_p is None or p < best_p):
            best_p, best_s = p, s
    return best_p, best_s


class ArbWsFeed:
    """Drives ArbExecutor from the WebSocket stream."""

    def __init__(self, executor):
        self.ex = executor
        self.available = _HAVE_WS

    def start(self):
        if not self.available:
            self.ex._say("websockets not installed — falling back to REST poll")
            return False
        threading.Thread(target=self._run, name="arb-ws", daemon=True).start()
        self.ex._say("WebSocket fast-path active (push book + price_change deltas)")
        return True

    def _run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:
            self.ex._log({"ts": time.strftime("%H:%M:%S"), "ws_fatal": f"{type(e).__name__}: {e}"})

    async def _main(self):
        # Reconnect/roll every window: resolve the current window's two tokens,
        # subscribe, stream until the window closes, then loop to the next one.
        while True:
            win = self.ex.resolve_current_window()
            if win is None:
                await asyncio.sleep(2.0)
                continue
            slug, up_token, down_token, end_ts = win
            try:
                await self._stream_window(slug, up_token, down_token, end_ts)
            except Exception as e:
                self.ex._log({"ts": time.strftime("%H:%M:%S"), "slug": slug,
                              "ws_error": f"{type(e).__name__}: {e}"})
                await asyncio.sleep(1.0)

    async def _stream_window(self, slug, up_token, down_token, end_ts):
        asks = {up_token: {}, down_token: {}}   # token -> {price: size}
        self.ex.note_window(slug, end_ts)
        stop_at = float(end_ts) - self.ex.cfg.arb_deadline_buffer_s
        # We reconnect once per 5-min window. Two settings keep that from leaking
        # memory across windows (confirmed via external RSS stepping ~30-95MB at
        # every window roll, with a tracemalloc hint pointing at
        # decompressor.decompress):
        #   compression=None  -> no permessage-deflate; drops the per-connection
        #     zlib decompressor + its accumulated buffers (the leak). We're
        #     co-located (~0.9ms to the CF edge) so the extra bandwidth is free.
        #   max_size/max_queue -> bound a single connection's receive buffers so
        #     a burst can't balloon RSS (the likely cause of run #1's 667MB OOM).
        async with websockets.connect(WS_URL, ping_interval=10, close_timeout=3,
                                      compression=None, max_size=2**20,
                                      max_queue=32) as ws:
            await ws.send(json.dumps({"type": "market",
                                      "assets_ids": [up_token, down_token]}))
            while time.time() < stop_at:
                timeout = max(0.5, stop_at - time.time())
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError:
                    break  # window over
                msgs = json.loads(raw)
                if isinstance(msgs, dict):
                    msgs = [msgs]
                changed = False
                for m in msgs:
                    et = m.get("event_type")
                    if et == "book":
                        tok = m.get("asset_id")        # book has a top-level asset_id
                        if tok in asks:
                            _load_asks(asks[tok], m.get("asks"))
                            changed = True
                    elif et == "price_change":
                        # routes each change by its OWN asset_id (no top-level one)
                        if _apply_price_changes(asks, m):
                            changed = True
                if not changed:
                    continue
                au, au_sz = _best_ask(asks[up_token])
                ad, ad_sz = _best_ask(asks[down_token])
                if au is None or ad is None:
                    continue
                # hot path: cheap check, offload execution so recv keeps draining
                if (1.0 - (au + ad)) >= self.ex.cfg.arb_min_edge:
                    detect_ts = time.time()
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None, self.ex.execute_from_ws,
                        slug, up_token, down_token,
                        au, au_sz, ad, ad_sz, detect_ts)
