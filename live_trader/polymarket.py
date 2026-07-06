"""Thin wrapper around py-clob-client for the BTC up/down 5m bot.

Responsibilities:
  - Resolve Polymarket slug -> (up_token_id, down_token_id, end_ts) via Gamma API.
  - Submit a BUY market order (FOK) for a given token at a target USDC notional.
  - Query market resolution after settlement.

We deliberately keep this narrow. Anything fancier (order book depth, cancels,
GTC resting orders) is out of scope for the 5-minute in/out strategy.
"""

import json
import math
import threading
import time

import requests

from live_trader._pagination import paginate_all
from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import (
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    MarketOrderArgs,
    OrderArgs,
    OrderType,
    PartialCreateOrderOptions,
)


_HTTP_TUNED = False


def _tune_http_client():
    """Replace py-clob-client's shared httpx client with a warm, HTTP/2-pooled one.

    The library already uses a persistent httpx.Client (good — keep-alive), but
    its default keepalive_expiry is 5s. Arb windows are 5 min apart, so between
    opportunities the connection goes COLD and the order we're racing to send
    pays a fresh TCP+TLS handshake (1-3 extra RTTs). We:
      - raise keepalive_expiry to 10 min so the socket stays hot across windows;
      - enable HTTP/2 (h2 is installed) to multiplex the two concurrent FOK legs
        over one connection instead of contending for sockets.
    Idempotent and best-effort — never blocks startup.
    """
    global _HTTP_TUNED
    if _HTTP_TUNED:
        return
    try:
        import httpx
        import py_clob_client_v2.http_helpers.helpers as _h
        _h._http_client = httpx.Client(
            http2=True,
            timeout=httpx.Timeout(5.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50,
                                keepalive_expiry=600.0),
        )
        _HTTP_TUNED = True
    except Exception:
        pass


GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
# Public trade tape (no auth). The CLOB client's get_market_trades_events returns
# market *metadata*, not trades — this data-api endpoint is the real tape.
DATA_API_TRADES_URL = "https://data-api.polymarket.com/trades"


class PolymarketError(RuntimeError):
    pass


_CONN_ERR_MARKERS = ("server disconnected", "connection", "timeout", "timed out",
                     "reset", "remote", "temporarily unavailable", "eof occurred")


def _is_conn_err(e) -> bool:
    """True if the exception looks like a transient connection drop (stale
    keep-alive, network blip) rather than a real API rejection — safe to retry."""
    return any(m in str(e).lower() for m in _CONN_ERR_MARKERS)


class PolymarketBotClient:
    def __init__(self, cfg):
        self.cfg = cfg
        _tune_http_client()                      # warm HTTP/2 pool before any call
        self._tok_meta = {}                      # token_id -> (tick_size, neg_risk)
        self._client = ClobClient(
            host=cfg.clob_host,
            key=cfg.private_key,
            chain_id=cfg.chain_id,
            signature_type=cfg.signature_type,
            funder=cfg.funder_address,
        )
        self._client.set_api_creds(ApiCreds(
            api_key=cfg.api_key,
            api_secret=cfg.api_secret,
            api_passphrase=cfg.api_passphrase,
        ))
        # Serialize on-chain writes (redeem + auto-wrap) so two concurrent
        # trades don't race for the same EOA nonce.
        self._tx_lock = threading.Lock()

    def measure_rtt(self, n: int = 5) -> dict | None:
        """Median round-trip latency to the CLOB host — NETWORK only, no order.

        Times GET /time (cheap, unauthenticated). This isolates the network leg
        of latency (what co-location fixes) from in-process detect->fill time
        (what code fixes). Run at startup so local vs eu-west-1 is an apples-to-
        apples comparison. Returns None if every probe failed.
        """
        import statistics
        url = self.cfg.clob_host.rstrip("/") + "/time"
        samples = []
        for _ in range(n):
            t = time.time()
            try:
                requests.get(url, timeout=5)
            except Exception:
                continue
            samples.append((time.time() - t) * 1000.0)
        if not samples:
            return None
        return {"median_ms": round(statistics.median(samples)),
                "min_ms": round(min(samples)), "max_ms": round(max(samples)),
                "n": len(samples)}

    def resolve_market(self, slug: str) -> dict:
        """Resolve slug -> {up_token, down_token, end_ts, closed, condition_id}."""
        resp = requests.get(f"{GAMMA_EVENTS_URL}?slug={slug}", timeout=6)
        resp.raise_for_status()
        events = resp.json()
        if not events:
            raise PolymarketError(f"Slug not found: {slug}")
        markets = events[0].get("markets", [])
        if not markets:
            raise PolymarketError(f"No markets in event: {slug}")
        mkt = markets[0]

        raw_token_ids = mkt.get("clobTokenIds")
        if not raw_token_ids:
            raise PolymarketError(f"Market missing clobTokenIds: {slug}")
        token_ids = json.loads(raw_token_ids) if isinstance(raw_token_ids, str) else raw_token_ids

        raw_outcomes = mkt.get("outcomes", '["Up","Down"]')
        outcomes = json.loads(raw_outcomes) if isinstance(raw_outcomes, str) else raw_outcomes

        up_idx = 0
        for i, o in enumerate(outcomes):
            if str(o).strip().lower() in ("up", "yes"):
                up_idx = i
                break
        down_idx = 1 - up_idx

        return {
            "up_token": token_ids[up_idx],
            "down_token": token_ids[down_idx],
            "closed": bool(mkt.get("closed", False)),
            "condition_id": mkt.get("conditionId"),
            "question": mkt.get("question", slug),
            "end_date_iso": mkt.get("endDate"),
            "slug": slug,
        }

    def get_top_ask(self, token_id: str) -> float | None:
        """Lowest ask price on the book — what we'd pay per share for a BUY.
        Returns None if the book is empty or the query fails."""
        try:
            book = self._client.get_order_book(token_id)
            asks = (book.get("asks") if isinstance(book, dict) else getattr(book, "asks", None)) or []
            prices: list[float] = []
            for a in asks:
                raw = getattr(a, "price", None)
                if raw is None and isinstance(a, dict):
                    raw = a.get("price")
                if raw is None:
                    continue
                try:
                    prices.append(float(raw))
                except (TypeError, ValueError):
                    continue
            return min(prices) if prices else None
        except Exception:
            return None

    def get_top_bid(self, token_id: str) -> float | None:
        """Highest bid price on the book — what we'd receive per share for a SELL."""
        try:
            book = self._client.get_order_book(token_id)
            bids = (book.get("bids") if isinstance(book, dict) else getattr(book, "bids", None)) or []
            prices: list[float] = []
            for b in bids:
                raw = getattr(b, "price", None)
                if raw is None and isinstance(b, dict):
                    raw = b.get("price")
                if raw is None:
                    continue
                try:
                    prices.append(float(raw))
                except (TypeError, ValueError):
                    continue
            return max(prices) if prices else None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Read-only market-data helpers for market-making research (MM_DATA_SPEC.md).
    # These never place orders — they only read books/tape/tick/rewards so we can
    # backtest whether providing liquidity is viable before risking capital.
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_levels(side) -> list[list[float]]:
        """Normalize a book side (list of {price,size} dicts or objects) to
        a list of [price, size] floats. Order preserved as returned by the API."""
        out: list[list[float]] = []
        for lvl in side or []:
            p = getattr(lvl, "price", None)
            s = getattr(lvl, "size", None)
            if p is None and isinstance(lvl, dict):
                p, s = lvl.get("price"), lvl.get("size")
            if p is None:
                continue
            try:
                out.append([float(p), float(s) if s is not None else 0.0])
            except (TypeError, ValueError):
                continue
        return out

    def get_full_book(self, token_id: str) -> dict | None:
        """Full depth for one token: {bids:[[p,s]...], asks:[[p,s]...], hash}.
        bids sorted high->low, asks sorted low->high. None on failure.
        Retries on transient connection drops so a stale keep-alive doesn't
        cause a blind poll (idempotent GET — always safe to retry)."""
        book = None
        for attempt in range(3):
            try:
                book = self._client.get_order_book(token_id)
                break
            except Exception as e:
                if _is_conn_err(e) and attempt < 2:
                    time.sleep(0.1)
                    continue
                return None
        if isinstance(book, dict):
            bids_raw, asks_raw = book.get("bids"), book.get("asks")
            bhash = book.get("hash")
            min_sz = book.get("min_order_size")
        else:
            bids_raw, asks_raw = getattr(book, "bids", None), getattr(book, "asks", None)
            bhash = getattr(book, "hash", None)
            min_sz = getattr(book, "min_order_size", None)
        bids = sorted(self._parse_levels(bids_raw), key=lambda x: -x[0])
        asks = sorted(self._parse_levels(asks_raw), key=lambda x: x[0])
        try:
            min_sz = float(min_sz) if min_sz is not None else None
        except (TypeError, ValueError):
            min_sz = None
        return {"bids": bids, "asks": asks, "hash": bhash, "min_order_size": min_sz}

    def get_tape(self, condition_id: str, limit: int = 200) -> list | None:
        """Public executed-trade tape for a market via the Polymarket data-api.
        Returns a list of trimmed trade dicts (newest first), or None on failure.

        Each dict: {tx, side (taker BUY/SELL), outcome (Up/Down), price, size, ts}.
        NOTE: the CLOB client's get_market_trades_events returns market metadata,
        NOT trades — do not use it here."""
        try:
            resp = requests.get(
                DATA_API_TRADES_URL,
                params={"market": condition_id, "limit": limit},
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return None
        if not isinstance(data, list):
            return None
        out = []
        for t in data:
            if not isinstance(t, dict):
                continue
            out.append({
                "tx": t.get("transactionHash"),
                "asset": t.get("asset"),
                "side": t.get("side"),               # taker side
                "outcome": t.get("outcome"),         # "Up" / "Down"
                "price": t.get("price"),
                "size": t.get("size"),
                "ts": t.get("timestamp"),
            })
        return out

    def get_tick_size(self, token_id: str) -> float | None:
        try:
            raw = self._client.get_tick_size(token_id)
            return float(raw) if raw is not None else None
        except Exception:
            return None

    def get_market_rewards(self, condition_id: str):
        """Liquidity-mining reward schedule for a market. Raw payload. None on
        failure (e.g. market not in any reward program)."""
        try:
            return self._client.get_raw_rewards_for_market(condition_id)
        except Exception:
            return None

    def warm_token(self, token_id: str) -> tuple:
        """Pre-fetch tick_size + neg_risk so buy_limit_fok places orders with ZERO
        network lookups on the hot path.

        create_order() otherwise calls get_neg_risk() (and resolves tick size)
        over the wire on EVERY order — extra round-trips paid mid-race. Call this
        once when a window opens (off the hot path); the result is cached and fed
        to create_order via PartialCreateOrderOptions. Best-effort.
        """
        cached = self._tok_meta.get(token_id)
        if cached is not None:
            return cached
        ts = nr = None
        try:
            ts = self._client.get_tick_size(token_id)
        except Exception:
            pass
        try:
            nr = self._client.get_neg_risk(token_id)
        except Exception:
            pass
        self._tok_meta[token_id] = (ts, nr)
        return (ts, nr)

    def warm_signer(self, token_id: str) -> None:
        """JIT-warm the EIP-712 signing path so the FIRST real order isn't ~180ms.

        The very first create_order() in a process pays a one-time setup cost
        (~180ms measured) before steady-state ~0.4ms signing. Arbs cluster at
        window open, so that cold hit would land on the most likely arb. Build
        (never post) one throwaway order at startup to pay it up front.
        """
        try:
            ts, nr = self.warm_token(token_id)
            if ts is None:
                return
            args = OrderArgs(token_id=token_id, price=float(ts), size=5, side="BUY")
            self._client.create_order(
                args, PartialCreateOrderOptions(tick_size=ts, neg_risk=nr))
        except Exception:
            pass

    def sign_limit_fok(self, token_id: str, price: float, size: float) -> dict:
        """Build + LOCALLY SIGN a FOK limit BUY (no network if token is warmed).
        Returns {'signed': order} or {'skipped': reason}.

        Split from posting so the arb executor can sign BOTH legs first, then fire
        both POSTs concurrently — minimizing the gap between the two legs hitting
        the book (a staggered gap is what leaves one leg filled = naked).
        """
        args = OrderArgs(token_id=token_id, price=round(float(price), 3),
                         size=round(float(size), 2), side="BUY")
        # Feed cached tick_size/neg_risk so create_order does NO network lookups
        # on the hot path (warm_token pre-fetches them). Falls back to the
        # library's own (networked) resolution if the token wasn't warmed.
        ts, nr = self._tok_meta.get(token_id, (None, None))
        opts = (PartialCreateOrderOptions(tick_size=ts, neg_risk=nr)
                if ts is not None and nr is not None else None)
        # local signing (+ lookups only if not pre-warmed) — safe to RETRY on a
        # dropped connection (the #1 cause of missed arbs on a flaky link).
        for attempt in range(3):
            try:
                return {"signed": self._client.create_order(args, opts)}
            except Exception as e:
                if _is_conn_err(e) and attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
                    continue
                return {"skipped": f"create_order failed: {e}"}

    def post_signed_fok(self, signed) -> dict:
        """Submit a pre-signed FOK order. The NETWORK leg — fire concurrently.
        Returns the raw resp, {'killed': True} if unfilled, or {'skipped': reason}.

        Does NOT retry on a connection drop (ambiguous: the order may have landed,
        a retry could double-fill).
        """
        try:
            return self._client.post_order(signed, OrderType.FOK)
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("killed", "no orders found", "not enough",
                                      "fully filled or killed")):
                return {"killed": True, "error": str(e)}
            return {"skipped": f"post_order failed: {e}"}

    def buy_limit_fok(self, token_id: str, price: float, size: float) -> dict:
        """Fill-Or-Kill limit BUY (sign + post). Convenience wrapper; the arb hot
        path uses sign_limit_fok + post_signed_fok directly for tighter leg timing.
        NOTE: deliberately does NOT short-circuit on cfg.dry_run — the ArbExecutor
        gates dry/live via cfg.arb_dry_run and only calls this in live mode."""
        s = self.sign_limit_fok(token_id, price, size)
        if "signed" not in s:
            return s
        return self.post_signed_fok(s["signed"])

    def buy_market(self, token_id: str, usdc_amount: float) -> dict:
        """Submit a FAK marketable-limit BUY (partial fills OK).

        Polymarket has NO true market order: create_market_order computes a limit
        price = the level that just fills `amount` (≈ the current best ask) and
        submits a FAK *limit*. On the fast-flickering BTC-5m book the best ask
        ticks up cents/second, so a limit at the snapshot best ask no longer
        crosses by the time it posts -> "no orders found to match" (systematic).

        Fix: pass an explicit limit = best_ask + cfg.market_max_slippage (capped at
        1 - tick). FAK fills at the resting maker prices UP TO this cap, so it is a
        real market buy with slippage protection, not overpayment.

        `amount` is USDC notional for a BUY. Returns the raw CLOB response, or a
        {'skipped': reason} if we bailed before sending.
        """
        if self.cfg.dry_run:
            return {"dry_run": True, "token_id": token_id, "amount": usdc_amount}

        try:
            book = self._client.get_order_book(token_id)
            asks = (book.get("asks") if isinstance(book, dict) else getattr(book, "asks", None)) or []
            prices = []
            for a in asks:
                raw = a.get("price") if isinstance(a, dict) else getattr(a, "price", None)
                if raw is not None:
                    try:
                        prices.append(float(raw))
                    except (TypeError, ValueError):
                        pass
            if not prices:
                return {"skipped": "empty ask side"}
            best_ask = min(prices)
        except Exception as e:
            return {"skipped": f"book query failed: {e}"}

        # Build a marketable limit above the best ask so we still cross if the book
        # ticks up before the order lands.
        try:
            tick = float(self._client.get_tick_size(token_id))
        except Exception:
            tick = 0.01
        if tick <= 0:
            tick = 0.01
        ndig = max(0, round(-math.log10(tick)))
        limit = min(1.0 - tick, best_ask + self.cfg.market_max_slippage)
        limit = round(round(limit / tick) * tick, ndig)
        if limit < best_ask:                 # never price below the ask we saw
            limit = round(best_ask, ndig)

        args = MarketOrderArgs(token_id=token_id, amount=usdc_amount,
                               side="BUY", price=limit)
        signed = self._client.create_market_order(args)
        try:
            return self._client.post_order(signed, OrderType.FAK)
        except Exception as e:
            msg = str(e).lower()
            if "no orders found" in msg or "fully filled or killed" in msg:
                return {"skipped": "book vanished at submission time"}
            raise

    def get_conditional_balance(self, token_id: str) -> float | None:
        """On-chain share balance for a CTF outcome token.

        takingAmount from a BUY response is pre-fee; Polymarket skims ~2% so
        the real balance is always a bit less. Query the chain via CLOB before
        any SELL to avoid "balance not enough" reverts.
        """
        try:
            resp = self._client.get_balance_allowance(BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=token_id,
                signature_type=self.cfg.signature_type,
            ))
            raw = resp.get("balance") if isinstance(resp, dict) else getattr(resp, "balance", None)
            if raw is None:
                return None
            val = float(raw)
            return val / 1e6 if val >= 1e5 else val
        except Exception:
            return None

    def sell_market(self, token_id: str, shares: float, force: bool = False,
                    known_qty: float | None = None) -> dict:
        """Submit a FAK market SELL: dump `shares` at the current best bid.

        Mirror of buy_market — amount field in MarketOrderArgs carries the
        share count (not USDC) when side is SELL.

        force=True bypasses the global dry_run gate. The arb executor MUST pass
        force=True when unwinding a one-leg fill: the arb profile keeps
        BOT_DRY_RUN=true (to suppress the directional bot), but a live arb that
        bought a real leg has to really flatten it — otherwise the "unwind" is a
        no-op and you're left holding a naked directional position.

        known_qty: when the caller already KNOWS how many shares it just bought
        (an arb leg unwind passes the FOK fill quantity), we won't let a lagging
        balance read zero out the sell. The CLOB balance endpoint can trail a
        just-filled buy by a second or two; without this the `min()` clamp below
        would skip the unwind on a 0.0 read, leaving a naked directional leg
        (exactly the 19:10 event: 'balance too low to sell (have 0.0)').
        """
        if self.cfg.dry_run and not force:
            return {"dry_run": True, "token_id": token_id, "shares": shares}

        actual = self.get_conditional_balance(token_id)
        if known_qty is not None and (actual is None or actual < known_qty * 0.99):
            # retry the balance read — it usually propagates within 1-2s
            for _ in range(4):
                time.sleep(0.5)
                actual = self.get_conditional_balance(token_id)
                if actual is not None and actual >= known_qty * 0.99:
                    break
            # still not caught up: the tokens exist on-chain, trust the fill qty
            if actual is None or actual < known_qty * 0.99:
                actual = known_qty
        if actual is not None:
            shares = min(shares, actual * 0.99)
        if shares <= 0.01:
            return {"skipped": f"balance too low to sell (have {actual})"}

        try:
            book = self._client.get_order_book(token_id)
            bids = (book.get("bids") if isinstance(book, dict) else getattr(book, "bids", None)) or []
            if not bids:
                return {"skipped": "empty bid side"}
        except Exception as e:
            return {"skipped": f"book query failed: {e}"}

        args = MarketOrderArgs(token_id=token_id, amount=shares, side="SELL")
        signed = self._client.create_market_order(args)
        try:
            return self._client.post_order(signed, OrderType.FAK)
        except Exception as e:
            msg = str(e).lower()
            if "no orders found" in msg or "fully filled or killed" in msg:
                return {"skipped": "book vanished at submission time"}
            raise

    def fetch_resolution(self, slug: str) -> dict | None:
        """After window close, return {'up_won': bool, 'up_price': float, 'down_price': float}
        or None if not yet resolved."""
        try:
            resp = requests.get(f"{GAMMA_EVENTS_URL}?slug={slug}", timeout=6)
            resp.raise_for_status()
            events = resp.json()
            if not events:
                return None
            mkt = events[0].get("markets", [{}])[0]
            if not mkt.get("closed", False):
                return None
            raw_prices = mkt.get("outcomePrices", '["0","0"]')
            prices = json.loads(raw_prices) if isinstance(raw_prices, str) else raw_prices
            up_price = float(prices[0])
            down_price = float(prices[1])
            return {
                "up_won": up_price >= 0.99,
                "up_price": up_price,
                "down_price": down_price,
            }
        except Exception:
            return None

    def wait_for_resolution(self, slug: str, deadline_ts: int, poll_sec: int = 5) -> dict | None:
        while int(time.time()) < deadline_ts:
            res = self.fetch_resolution(slug)
            if res:
                return res
            time.sleep(poll_sec)
        return self.fetch_resolution(slug)

    @staticmethod
    def binance_window_prices(ws: int, symbol: str = "BTCUSDT",
                              interval: str = "5m") -> dict | None:
        """Independent cross-check for the paper log: the Binance candle that
        opens at window-start `ws` (epoch seconds). `symbol`/`interval` default to
        the BTC-5m market; pass the asset pair + tf for other markets (Binance
        kline intervals "5m"/"15m" match our timeframe strings). Returns
        {open, close, move_pct} or None on failure (geo-block/ratelimit/parse) —
        callers degrade gracefully and keep gamma as ground truth."""
        try:
            url = ("https://api.binance.com/api/v3/klines"
                   f"?symbol={symbol}&interval={interval}&startTime={ws * 1000}&limit=1")
            r = requests.get(url, timeout=6)
            r.raise_for_status()
            k = r.json()
            if not k:
                return None
            o = float(k[0][1])  # open
            c = float(k[0][4])  # close
            return {"open": o, "close": c,
                    "move_pct": (c - o) / o * 100.0 if o else 0.0}
        except Exception:
            return None

    def sweep_orphan_winners(self, min_size: float = 0.1, win_price: float = 0.99,
                             force: bool = False, wrap_each: bool = True) -> dict:
        """Scan Polymarket positions API for orphan winners and redeem them.

        Catches the failure mode where a trade is placed but its record never
        lands in trade_history.jsonl (e.g. process crash between BUY and the
        JSONL write). Without a record the bot's normal post-settlement
        auto-redeem path never runs, and winning shares sit on-chain unclaimed.

        Strategy:
          1. Query data-api.polymarket.com/positions?user=ADDR&redeemable=true.
          2. Filter to positions where curPrice >= win_price AND size >= min_size
             (these are real-money winners; loser shares have curPrice=0).
          3. For each, verify on-chain balance > 0 (skip if already redeemed
             but API hadn't caught up).
          4. Call redeem_position() and report results.

        Returns dict with counts: {scanned, winners, redeemed, skipped, failed}.
        Idempotent — safe to call on every bot start.
        """
        result = {"scanned": 0, "winners": 0, "redeemed": 0, "skipped": 0, "failed": 0, "value": 0.0}
        # Paginate ALL redeemable positions. A single uncapped request is truncated
        # by the data-api (~100 rows); a wallet with a large old-loser backlog would
        # otherwise hide fresh small winners past the cap (the bug that left $12+ of
        # winnings unclaimed on 2026-06-18).
        PAGE = 500
        base = (f"https://data-api.polymarket.com/positions?user={self.cfg.funder_address}"
                f"&redeemable=true&sizeThreshold={min_size}&limit={PAGE}")

        def _fetch(offset):
            r = requests.get(f"{base}&offset={offset}", timeout=15)
            r.raise_for_status()
            return r.json()

        try:
            positions = paginate_all(_fetch, page_size=PAGE)
        except Exception as e:
            print(f"  [RECONCILE] Polymarket API failed ({type(e).__name__}: {e}) — skipping sweep")
            return result

        result["scanned"] = len(positions)
        # Focus on winners — these are positions where the side we hold resolved to 1.0
        winners = [p for p in positions
                   if float(p.get("curPrice", 0)) >= win_price
                   and float(p.get("size", 0)) >= min_size]
        result["winners"] = len(winners)
        if not winners:
            return result

        print(f"  [RECONCILE] {len(winners)} potential orphan winner(s) — verifying on-chain")
        for p in winners:
            slug = p.get("eventSlug") or p.get("slug") or "?"
            outcome = (p.get("outcome") or "").upper()
            size = float(p.get("size", 0))
            cur = float(p.get("curPrice", 0))
            cond_id = p.get("conditionId")
            asset = str(p.get("asset") or "")  # CTF positionId / token_id

            if not cond_id or not asset:
                print(f"    [skip] {slug}: missing conditionId or asset")
                result["skipped"] += 1
                continue

            try:
                bal = self.get_conditional_balance(asset) or 0.0
            except Exception as e:
                print(f"    [skip] {slug}: bal check failed ({type(e).__name__})")
                result["skipped"] += 1
                continue

            if bal < 0.01:
                # Already redeemed; API is stale
                result["skipped"] += 1
                continue

            value_est = size * cur
            print(f"    redeeming {slug} ({outcome}) bal={bal:.4f} ≈${value_est:.2f}")
            try:
                tx = self.redeem_position(cond_id, force=force, wrap=wrap_each)
                if tx:
                    print(f"      ✓ tx={tx[:14]}...")
                    result["redeemed"] += 1
                    result["value"] += value_est
                else:
                    print(f"      ✗ redeem returned None (oracle not ready?)")
                    result["failed"] += 1
            except Exception as e:
                print(f"      ✗ redeem error: {type(e).__name__}: {e}")
                result["failed"] += 1

        return result

    def redeem_position(self, condition_id: str, force: bool = False,
                        wrap: bool = True) -> str | None:
        """Claim winning CTF shares → USDC.e on the EOA.

        Even though V2 trading is denominated in pUSD, the underlying CTF
        positions are USDC.e-collateralized — pUSD lives at the exchange/onramp
        layer, not inside the CTF. Pass USDC.e to redeemPositions or it'll
        compute the wrong positionId and burn nothing.

        The oracle sometimes reports payouts a few seconds *after* Gamma flags
        the market as closed — simulating the call detects that early and lets
        us back off and retry instead of burning a revert tx. Returns the tx
        hash on success, None after exhausting retries.

        force=True bypasses the global dry_run gate, same as sell_market: the arb
        profile keeps BOT_DRY_RUN=true (to suppress the directional bot), but arb
        winners are REAL and must actually be claimed — otherwise the auto-redeem
        sweep silently no-ops and winnings strand on-chain (the exact reason the
        5.18 DOWN winner had to be redeemed by hand on 2026-05-30).
        """
        if self.cfg.dry_run and not force:
            return None

        from web3 import Web3  # local import keeps web3 optional at module load

        # Polygon RPC with fallback. drpc 400s on rapid eth_call / sendRawTransaction
        # and was silently stranding winners (redeem txs failed, winnings sat as CTF
        # tokens). publicnode is reliable for both redeem and the wrap; drpc stays as
        # a last resort. Same fallback ordering as risk._RPCS.
        w3 = None
        for _rpc in ("https://polygon-bor-rpc.publicnode.com",
                     "https://polygon-rpc.com", "https://rpc.ankr.com/polygon",
                     "https://polygon.drpc.org"):
            try:
                _w = Web3(Web3.HTTPProvider(_rpc, request_kwargs={"timeout": 20}))
                if _w.eth.chain_id == 137:
                    w3 = _w
                    break
            except Exception:
                continue
        if w3 is None:
            return None
        acct = w3.eth.account.from_key(self.cfg.private_key)
        ctf_addr = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
        usdce = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
        abi = [{
            "name": "redeemPositions", "type": "function", "stateMutability": "nonpayable",
            "inputs": [
                {"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "indexSets", "type": "uint256[]"},
            ], "outputs": []}]
        ctf = w3.eth.contract(address=ctf_addr, abi=abi)
        call = ctf.functions.redeemPositions(usdce, "0x" + "00" * 32, condition_id, [1, 2])

        with self._tx_lock:
            for attempt in range(6):
                try:
                    call.call({"from": acct.address})  # dry-run: raises if oracle not ready
                    tx = call.build_transaction({
                        "from": acct.address,
                        "nonce": w3.eth.get_transaction_count(acct.address),
                        "chainId": 137,
                        "gas": 250_000,
                        "gasPrice": w3.eth.gas_price,
                    })
                    signed = acct.sign_transaction(tx)
                    h = w3.eth.send_raw_transaction(signed.raw_transaction)
                    rcpt = w3.eth.wait_for_transaction_receipt(h, timeout=180)
                    if rcpt.status == 1:
                        # wrap=False lets the centralized redeemer redeem many
                        # winners first and sweep USDC.e->pUSD once at the end
                        # (fewer txs) instead of wrapping after every redeem.
                        if wrap:
                            self._wrap_usdce_to_pusd(w3, acct)
                        return h.hex()
                    return None
                except Exception:
                    time.sleep(10)
            return None

    def _wrap_usdce_to_pusd(self, w3, acct) -> None:
        """Sweep redeemed USDC.e back to pUSD via Polymarket's CollateralOnramp.

        CTF redemptions pay out USDC.e (the underlying collateral), but the
        bot trades in pUSD. Without this sweep, USDC.e accumulates and pUSD
        bleeds down until a manual wrap. Approval to the Onramp is set MAX
        once during V2 setup, so this is a single tx.
        """
        usdce = w3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
        onramp_addr = w3.to_checksum_address("0x93070a847efEf7F70739046A929D47a521F5B8ee")
        erc20_abi = [{"name": "balanceOf", "type": "function", "stateMutability": "view",
                      "inputs": [{"name": "a", "type": "address"}],
                      "outputs": [{"name": "", "type": "uint256"}]}]
        onramp_abi = [{"name": "wrap", "type": "function", "stateMutability": "nonpayable",
                       "inputs": [{"name": "_asset", "type": "address"},
                                  {"name": "_to", "type": "address"},
                                  {"name": "_amount", "type": "uint256"}],
                       "outputs": []}]
        # USDC.e sits on the funder (== EOA for sig-type 0). Use the funder for the
        # balance read AND the pUSD destination so this is correct under proxy wallets.
        holder = w3.to_checksum_address(self.cfg.funder_address)
        ts = time.strftime("%H:%M:%S")
        try:
            bal = w3.eth.contract(address=usdce, abi=erc20_abi).functions.balanceOf(holder).call()
            if bal < 10_000:  # ignore <$0.01 dust
                return
            onramp = w3.eth.contract(address=onramp_addr, abi=onramp_abi)
            fn = onramp.functions.wrap(usdce, holder, bal)
            # Right-size the gas LIMIT. A fixed 400k limit reserves gas_limit*gas_price
            # up front, so on a low-POL wallet send_raw_transaction throws
            # "insufficient funds for gas" even though the wrap only uses ~140k.
            # estimate_gas also surfaces a would-be revert (e.g. missing approval).
            try:
                gas_limit = int(fn.estimate_gas({"from": acct.address}) * 1.25)
            except Exception as e:
                print(f"  [WRAP {ts}] gas-estimate failed ({type(e).__name__}: {e}) — "
                      f"${bal/1e6:.2f} USDC.e left UNWRAPPED (approval/revert?)")
                return
            gas_price = w3.eth.gas_price
            pol = w3.eth.get_balance(acct.address)
            reserve = gas_limit * gas_price
            if reserve >= pol:
                print(f"  [WRAP {ts}] SKIP: not enough POL for gas (need "
                      f"{reserve/1e18:.4f}, have {pol/1e18:.4f}). "
                      f"${bal/1e6:.2f} USDC.e left UNWRAPPED — TOP UP POL.")
                return
            tx = fn.build_transaction({
                "from": acct.address,
                "nonce": w3.eth.get_transaction_count(acct.address, "pending"),
                "chainId": 137,
                "gas": gas_limit,
                "gasPrice": gas_price,
            })
            signed = acct.sign_transaction(tx)
            h = w3.eth.send_raw_transaction(signed.raw_transaction)
            rcpt = w3.eth.wait_for_transaction_receipt(h, timeout=120)
            if rcpt.status != 1:
                print(f"  [WRAP {ts}] wrap REVERTED (status=0) 0x{h.hex().lstrip('0x')} — "
                      f"${bal/1e6:.2f} USDC.e still unwrapped")
                return
            print(f"  [WRAP {ts}] wrapped ${bal/1e6:.2f} USDC.e -> pUSD "
                  f"(gas {rcpt.gasUsed}, 0x{h.hex().lstrip('0x')})")
            # refresh CLOB cache so the next order sees the freshly-wrapped pUSD
            try:
                self._client.update_balance_allowance(BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,
                    signature_type=self.cfg.signature_type,
                ))
            except Exception:
                pass
        except Exception as e:
            print(f"  [WRAP {ts}] FAILED ({type(e).__name__}: {e}) — "
                  f"USDC.e left unwrapped; check POL gas balance")
            return

    def wrap_usdce(self) -> None:
        """Public one-shot USDC.e->pUSD sweep for the centralized redeemer daemon.
        Builds its own w3 (same RPC fallback as redeem_position) and runs the
        single lazy wrap. Serialized via _tx_lock so it never overlaps a redeem
        in the same process. No-op under dry_run."""
        if self.cfg.dry_run:
            return
        from web3 import Web3
        w3 = None
        for _rpc in ("https://polygon-bor-rpc.publicnode.com",
                     "https://polygon-rpc.com", "https://rpc.ankr.com/polygon",
                     "https://polygon.drpc.org"):
            try:
                _w = Web3(Web3.HTTPProvider(_rpc, request_kwargs={"timeout": 20}))
                if _w.eth.chain_id == 137:
                    w3 = _w
                    break
            except Exception:
                continue
        if w3 is None:
            print("  [WRAP] no working RPC — skipping wrap")
            return
        acct = w3.eth.account.from_key(self.cfg.private_key)
        with self._tx_lock:
            self._wrap_usdce_to_pusd(w3, acct)
