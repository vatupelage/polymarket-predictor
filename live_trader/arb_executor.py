"""ArbExecutor — risk-managed cross-leg arbitrage for Polymarket BTC 5m.

Captures the only real, efficiency-proof edge in the data: when
top_ask(UP) + top_ask(DOWN) < $1, buying BOTH sides is guaranteed profit
(one resolves to $1, you paid less). Direction and informed flow are irrelevant.

The May-22 arb trade lost $25 because the old code used MARKET orders on a thin
book — the cheap quotes vanished and it filled deeper, paying >$1 for the pair.
This executor bakes in the four safeguards that prevent that:

  1. LIMIT FOK orders only — fill the whole leg at the quoted price or nothing.
     No market orders, so no slippage / no sweeping the book.
  2. ATOMIC + CONCURRENT — both FOK legs are submitted simultaneously (Tier-2),
     so neither waits for the other and both hit the book at ~the same instant,
     minimizing the leg-risk gap. If only one leg fills, we immediately UNWIND it
     (flatten) rather than hold a naked directional position.
  3. DEPTH-CAPPED — size is capped at the displayed top-of-book size on BOTH
     legs (and a hard max). We never try to fill more than the book offers at
     the good price — that was the root of the May-22 overpay.
  4. DRY-RUN MEASUREMENT first — by default we only DETECT + LOG (detected edge
     vs fillable size), placing no orders, so you can measure the real
     opportunity and the detected-vs-fillable gap before risking a cent.

All activity is logged to arb_history.jsonl.
"""

import concurrent.futures
import datetime
import json
import os
import queue
import threading
import time


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="milliseconds")


def _parse_buy(resp, intended_size):
    """Return filled shares from a FOK buy resp, or 0.0 if killed/none.
    FOK fills fully or not at all, so a fill == intended_size."""
    if not isinstance(resp, dict):
        return 0.0
    if resp.get("dry_run"):
        return float(intended_size)
    if resp.get("killed") or "skipped" in resp:
        return 0.0
    for k in ("takingAmount", "filledSize", "size_matched"):
        raw = resp.get(k)
        if raw is None:
            continue
        try:
            v = float(raw)
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        return v / 1e6 if v >= 1e5 else v
    return 0.0


class ArbExecutor:
    def __init__(self, client, cfg, log_dir, console=False):
        self.client = client
        self.cfg = cfg
        self.log_path = os.path.join(log_dir, "arb_history.jsonl")
        self.console = console      # print live arb events to stdout
        self._active = set()
        self._lock = threading.Lock()
        self._wlock = threading.Lock()
        self._exec_lock = threading.Lock()      # serialize executions (WS fires fast)
        self._win_exec_count = {}               # slug -> acts done this window
        self._win_min_size = {}                 # slug -> cached exchange min_order_size
        self._noted = set()                     # slugs already counted for stats
        self._halted = False                    # kill-switch: set after any leg-miss
        self._win_cond = {}                     # slug -> condition_id (for direct redeem)
        # Direct-by-condition_id redemption: every window where we hold shares to
        # resolution is tracked here so we can redeem it straight from its known
        # condition_id — no dependence on the data-api positions sweep, which caps
        # at 100 rows and loses small arb winners under a pile of loser tokens.
        self._pending_path = os.path.join(log_dir, "arb_pending_redeem.json")
        self._pending_lock = threading.Lock()
        self._pending = self._load_pending()    # condition_id -> {slug, end_ts}
        self.use_ws = getattr(cfg, "arb_ws", True)
        # Tier-2: 2-worker pool to fire both legs concurrently (one per leg).
        self._leg_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="arb-leg")
        # Tier-2: async logging — keep disk I/O off the execution hot path.
        self._log_q: "queue.Queue" = queue.Queue()
        threading.Thread(target=self._log_writer, name="arb-logw", daemon=True).start()
        # live counters (for the --arb heartbeat)
        self.stats = {"windows": 0, "scans": 0, "detections": 0, "thin": 0,
                      "acted": 0, "locked": 0, "leg1_killed": 0, "unwound": 0,
                      "gross_profit": 0.0, "best_edge": 0.0, "last_lat_ms": 0,
                      "last_event": "—"}

    def _say(self, msg):
        if self.console:
            print(f"  [ARB {datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    # --- public ----------------------------------------------------------

    def start_continuous(self):
        """Start scanning every 5-min window for arb on its own clock. Prefers the
        WebSocket push fast-path (Tier-1 latency); falls back to REST polling if
        websockets is unavailable."""
        if not self.cfg.arb_enabled:
            return
        mode = "DRY-RUN (measure only, no orders)" if self.cfg.arb_dry_run else "LIVE"
        feed_label = "REST poll @1s"
        started_ws = False
        if self.use_ws:
            from .arb_ws import ArbWsFeed
            feed = ArbWsFeed(self)
            started_ws = feed.start()
            if started_ws:
                feed_label = "WebSocket push (fast)"
        if not started_ws:
            threading.Thread(target=self._continuous_loop, name="arb-continuous",
                             daemon=True).start()
        print(f"  [ARB] executor started — mode={mode}  feed={feed_label}  "
              f"min_edge={self.cfg.arb_min_edge:.3f}  max_size={self.cfg.arb_max_size}sh  "
              f"-> arb_history.jsonl")
        # Network RTT baseline — the leg race is unwinnable if this is high.
        # Logged so analyze_arb.py can compare local vs co-located deployments.
        try:
            rtt = self.client.measure_rtt()
        except Exception:
            rtt = None
        if rtt:
            self.stats["rtt_ms"] = rtt["median_ms"]
            print(f"  [ARB] network RTT to CLOB: median={rtt['median_ms']}ms "
                  f"(min={rtt['min_ms']} max={rtt['max_ms']}, n={rtt['n']})  "
                  f"<- co-locate to eu-west-1 to cut this")
            self._log({"ts": _now_iso(), "rtt_ms_median": rtt["median_ms"],
                       "rtt_ms_min": rtt["min_ms"], "rtt_ms_max": rtt["max_ms"]})
        # Tier-1 pre-positioning: keep the NEXT window's tokens (+ the signer)
        # warm so the first arb of each window — when arbs cluster but caches are
        # cold — doesn't pay ~870ms of tick/neg_risk lookup + cold-sign.
        threading.Thread(target=self._prewarm_loop, name="arb-prewarm",
                         daemon=True).start()

    def _prewarm_loop(self):
        """Resolve + warm the NEXT 5-min window's tokens ahead of its open, and
        JIT-warm the signer once. Off the hot path; best-effort."""
        signer_warmed = False
        while True:
            try:
                now = int(time.time())
                nxt = ((now // 300) + 1) * 300
                slug = f"btc-updown-5m-{nxt}"
                mkt = self._resolve(slug)
                if mkt is not None:
                    if mkt.get("condition_id"):
                        self._win_cond[slug] = mkt["condition_id"]
                    self._prewarm_window(slug, mkt["up_token"], mkt["down_token"])
                    if not signer_warmed:
                        warm_sig = getattr(self.client, "warm_signer", None)
                        if warm_sig:
                            warm_sig(mkt["up_token"])
                            signer_warmed = True
                self._prune_old_windows(now - 1800)   # drop windows closed >30m ago
            except Exception:
                pass
            time.sleep(30)

    # --- WebSocket fast-path support (called by arb_ws.ArbWsFeed) ---------

    def resolve_current_window(self):
        """(slug, up_token, down_token, end_ts) for the current 5-min window, or
        None if it can't be resolved yet (next window not listed on Gamma)."""
        now = int(time.time())
        ws = (now // 300) * 300
        slug = f"btc-updown-5m-{ws}"
        mkt = self._resolve(slug)
        if mkt is None:
            return None
        if mkt.get("condition_id"):
            self._win_cond[slug] = mkt["condition_id"]
        self._prewarm_window(slug, mkt["up_token"], mkt["down_token"])
        return (slug, mkt["up_token"], mkt["down_token"], ws + 300)

    def note_window(self, slug, end_ts):
        """Count a window once for stats (WS path calls this per window)."""
        if slug in self._noted:
            return
        self._noted.add(slug)
        self.stats["windows"] += 1
        self._say(f"scanning window {slug[-10:]} (closes in {int(end_ts-time.time())}s)")

    def execute_from_ws(self, slug, up_token, down_token,
                        au_px, au_sz, ad_px, ad_sz, detect_ts):
        """Called (offloaded) when the WS stream sees ask_up+ask_down below the
        edge threshold. min_order_size is pre-warmed off the hot path by
        _prewarm_window; if not yet cached we fall back to the configured min
        (NO network call here — the old get_full_book on the first arb of a
        window was a ~RTT hot-path stall, exactly when arbs cluster)."""
        gross_edge = 1.0 - (au_px + ad_px)
        exch_min = self._win_min_size.get(slug) or self.cfg.arb_min_size
        self._execute_arb(slug, up_token, down_token, au_px, au_sz, ad_px, ad_sz,
                          gross_edge, exch_min, detect_ts=detect_ts)

    def _continuous_loop(self):
        while True:
            try:
                now = int(time.time())
                ws = (now // 300) * 300
                end_ts = ws + 300
                slug = f"btc-updown-5m-{ws}"
                with self._lock:
                    fresh = slug not in self._active
                if fresh:
                    mkt = self._resolve(slug)
                    if mkt is not None:
                        if mkt.get("condition_id"):
                            self._win_cond[slug] = mkt["condition_id"]
                        self._prewarm_window(slug, mkt["up_token"], mkt["down_token"])
                        self.start_window(slug, mkt["up_token"], mkt["down_token"], end_ts)
                time.sleep(max(1.0, end_ts - time.time() + 0.5))
            except Exception as e:
                self._log({"ts": _now_iso(), "loop_error": f"{type(e).__name__}: {e}"})
                time.sleep(5.0)

    def start_window(self, slug, up_token, down_token, end_ts):
        with self._lock:
            if slug in self._active:
                return
            self._active.add(slug)
        self.stats["windows"] += 1
        self._say(f"scanning window {slug[-10:]} (closes in {int(end_ts-time.time())}s)")
        threading.Thread(target=self._scan_window, name=f"arb-{slug[-10:]}",
                         args=(slug, up_token, down_token, end_ts), daemon=True).start()

    # --- internals -------------------------------------------------------

    def _resolve(self, slug):
        try:
            return self.client.resolve_market(slug)
        except Exception:
            return None

    def _prewarm(self, *tokens):
        """Pre-fetch tick_size/neg_risk for the window's tokens off the hot path
        so order placement does zero network lookups when an arb fires."""
        warm = getattr(self.client, "warm_token", None)
        if warm is None:
            return
        for tok in tokens:
            try:
                warm(tok)
            except Exception:
                pass

    def _position_qty(self, token, retries=4):
        """Actual on-chain share balance for a token, retrying through the CLOB
        balance API's post-fill lag (a just-landed order can read 0 for ~1-2s).
        Returns float shares, 0.0 if none/unknown. Used to reconcile our assumed
        position against reality — catches naked legs from ambiguous order
        responses (a POST that timed out but actually filled) and partial unwinds.
        """
        get = getattr(self.client, "get_conditional_balance", None)
        if get is None:
            return 0.0
        for i in range(retries):
            try:
                bal = get(token)
            except Exception:
                bal = None
            if bal:
                return float(bal)
            if i < retries - 1:
                time.sleep(0.4)
        return 0.0

    def _prewarm_window(self, slug, up_token, down_token):
        """Off-hot-path warm of EVERYTHING the order path needs for a window:
        tick_size + neg_risk (per token) AND the exchange min_order_size (per
        slug). Run at window open / pre-open so the hot path makes no lookups."""
        self._prewarm(up_token, down_token)
        if slug not in self._win_min_size:
            try:
                bu = self.client.get_full_book(up_token)
                self._win_min_size[slug] = max(
                    self.cfg.arb_min_size, (bu or {}).get("min_order_size") or 0)
            except Exception:
                pass

    def _prune_old_windows(self, keep_after_ts):
        """Drop per-window bookkeeping for windows that closed long ago so the
        24/7 bot doesn't accumulate dicts/sets unboundedly."""
        for d in (self._win_min_size, self._win_cond, self._win_exec_count):
            for k in [k for k in d if self._end_ts_from_slug(k) < keep_after_ts]:
                d.pop(k, None)
        self._noted = {s for s in self._noted
                       if self._end_ts_from_slug(s) >= keep_after_ts}
        # client's token-keyed meta can't be aged by slug; cap it (re-warms on demand)
        tok_meta = getattr(self.client, "_tok_meta", None)
        if isinstance(tok_meta, dict) and len(tok_meta) > 4000:
            tok_meta.clear()

    # --- direct-by-condition_id redemption -------------------------------

    def _load_pending(self) -> dict:
        try:
            with open(self._pending_path) as f:
                d = json.load(f)
                return d if isinstance(d, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return {}

    def _save_pending(self):
        try:
            tmp = self._pending_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._pending, f)
            os.replace(tmp, self._pending_path)   # atomic — never a half-written file
        except OSError as e:
            self._log({"ts": _now_iso(), "redeem_persist_error": f"{type(e).__name__}: {e}"})

    @staticmethod
    def _end_ts_from_slug(slug):
        """Window resolves 300s after its start ts (the slug's trailing integer)."""
        try:
            return int(slug.rsplit("-", 1)[1]) + 300
        except (IndexError, ValueError):
            return int(time.time())

    def _mark_for_redeem(self, slug, tokens=None):
        """Record a window we (might) hold shares in so redeem_resolved() can claim
        it by condition_id after the market resolves. `tokens` = (up, down) lets
        the resolution-time balance gate skip conditions we turned out to be flat
        in (e.g. a fully-successful unwind) without wasting a redeem tx."""
        cond = self._win_cond.get(slug)
        if not cond:
            mkt = self._resolve(slug)
            cond = (mkt or {}).get("condition_id")
            if cond:
                self._win_cond[slug] = cond
        if not cond:
            self._log({"ts": _now_iso(), "slug": slug,
                       "redeem_track_error": "no condition_id — cannot schedule redeem"})
            return
        with self._pending_lock:
            entry = {"slug": slug, "end_ts": self._end_ts_from_slug(slug)}
            if tokens:
                entry["tokens"] = [t for t in tokens if t]
            self._pending[cond] = entry
            self._save_pending()

    def redeem_resolved(self) -> dict:
        """Claim every tracked position whose window has resolved, straight from
        its condition_id. Call periodically (the --arb heartbeat does, every 5m).

        redeem_position simulates first and returns None if the oracle isn't
        ready yet, so an unresolved/late condition simply stays pending and is
        retried next cycle. Persisted across restarts via arb_pending_redeem.json.
        """
        with self._pending_lock:
            items = list(self._pending.items())
        if not items:
            return {"redeemed": 0, "pending": 0}
        now = int(time.time())
        buffer_s = getattr(self.cfg, "arb_deadline_buffer_s", 0)
        done, redeemed = [], 0
        dropped = 0
        for cond, info in items:
            if now < info.get("end_ts", 0) + buffer_s:
                continue                              # not resolved yet
            # Balance gate (lag-free at resolution time): if we hold nothing in
            # either token, the position was fully unwound/never held — skip the
            # redeem so a clean flatten doesn't burn gas on a 0-value claim.
            toks = info.get("tokens")
            if toks and not any(self._position_qty(t, retries=1) > 0.01 for t in toks):
                done.append(cond); dropped += 1
                continue
            try:
                tx = self.client.redeem_position(cond, force=True)
            except Exception as e:
                self._log({"ts": _now_iso(), "slug": info.get("slug"),
                           "redeem_error": f"{type(e).__name__}: {e}"})
                continue
            if tx:                                    # None == oracle not ready, keep pending
                redeemed += 1
                self._say(f"✓ redeemed {str(info.get('slug',''))[-10:]} tx={tx[:12]}…")
                self._log({"ts": _now_iso(), "slug": info.get("slug"),
                           "result": "REDEEMED", "tx": tx, "condition_id": cond})
                done.append(cond)
        if done:
            with self._pending_lock:
                for c in done:
                    self._pending.pop(c, None)
                self._save_pending()
        return {"redeemed": redeemed, "pending": len(self._pending)}

    def _scan_window(self, slug, up_token, down_token, end_ts):
        executed = 0
        try:
            # stop scanning a bit before close to leave time to flatten if needed
            stop_at = float(end_ts) - self.cfg.arb_deadline_buffer_s
            while time.time() < stop_at:
                if executed >= self.cfg.arb_max_per_window:
                    break
                acted = self._check_and_act(slug, up_token, down_token)
                if acted:
                    executed += 1
                time.sleep(self.cfg.arb_poll_s)
        except Exception as e:
            self._log({"ts": _now_iso(), "slug": slug, "scan_error": f"{type(e).__name__}: {e}"})
        finally:
            with self._lock:
                self._active.discard(slug)

    def _check_and_act(self, slug, up_token, down_token) -> bool:
        """REST-poll detector (fallback path). Fetches both books, prechecks the
        edge, then hands off to the shared execution core."""
        self.stats["scans"] += 1
        # detect_ts marks when this scan started looking — measuring detect->fill
        # on the REST path too (it was only ever set on the WS path, so every
        # live trade ran with latency unrecorded). This is the number that drops
        # when co-located and tells us whether the leg race is winnable.
        detect_ts = time.time()
        bu = self.client.get_full_book(up_token)
        bd = self.client.get_full_book(down_token)
        if not bu or not bd or not bu["asks"] or not bd["asks"]:
            return False
        au_px, au_sz = bu["asks"][0]
        ad_px, ad_sz = bd["asks"][0]
        gross_edge = 1.0 - (au_px + ad_px)
        if gross_edge < self.cfg.arb_min_edge:
            return False
        exch_min = max(self.cfg.arb_min_size, bu.get("min_order_size") or 0,
                       bd.get("min_order_size") or 0)
        return self._execute_arb(slug, up_token, down_token,
                                 au_px, au_sz, ad_px, ad_sz, gross_edge, exch_min,
                                 detect_ts=detect_ts)

    def _execute_arb(self, slug, up_token, down_token,
                     au_px, au_sz, ad_px, ad_sz, gross_edge, exch_min,
                     detect_ts=None) -> bool:
        """Shared execution core (REST + WS paths). Sizing, safeguards, dry/live
        FOK atomic execution with unwind. Serialized + per-window capped."""
        # KILL-SWITCH: a prior leg-miss already left (and flattened) a naked leg.
        # A leg-miss means we're losing the leg race; continuing only accumulates
        # more naked directional bets. Stop trading until manual restart.
        if self._halted and not self.cfg.arb_dry_run:
            self.stats["last_event"] = "⛔ HALTED after leg-miss — restart to resume"
            return False
        self.stats["detections"] += 1
        self.stats["best_edge"] = max(self.stats["best_edge"], gross_edge)

        # Guard against a malformed / zero-price book level: prices must be in
        # (0,1). A 0 ask would ZeroDivisionError in the sizing below and kill the
        # scan thread; a >=1 ask makes no sense for a binary outcome.
        if not (0.0 < au_px < 1.0) or not (0.0 < ad_px < 1.0):
            self._log({"ts": _now_iso(), "slug": slug, "result": "bad_price",
                       "ask_up": au_px, "ask_dn": ad_px})
            return False

        # SAFEGUARD 3: cap size at displayed depth on BOTH legs, the per-leg
        # dollar budget, and a hard share max; floor to whole shares so the
        # USDC amount (price*size) stays <=2 decimals (Polymarket requirement).
        budget = self.cfg.arb_max_usdc_per_leg
        size = float(int(min(au_sz, ad_sz, budget / au_px, budget / ad_px,
                             float(self.cfg.arb_max_size))))
        if size < exch_min:
            self.stats["thin"] += 1
            self.stats["last_event"] = (f"edge {gross_edge:.1%} size {size:.0f}"
                                        f"<min {exch_min:.0f}sh — skip")
            self._say(f"DETECT edge={gross_edge:.1%} → {size:.0f}sh < min {exch_min:.0f}sh, skip")
            if self.cfg.arb_dry_run:
                self._log({"ts": _now_iso(), "slug": slug, "mode": "DRY", "acted": False,
                           "reason": "below_min_order_size", "edge": round(gross_edge, 4),
                           "ask_up": au_px, "ask_dn": ad_px, "depth_up": au_sz,
                           "depth_dn": ad_sz, "fillable_size": size, "exch_min": exch_min})
            return False

        cost = size * (au_px + ad_px)
        profit = size * gross_edge
        base = {"ts": _now_iso(), "slug": slug, "edge": round(gross_edge, 4),
                "ask_up": au_px, "ask_dn": ad_px, "depth_up": au_sz, "depth_dn": ad_sz,
                "fillable_size": round(size, 2), "cost": round(cost, 2),
                "expected_profit": round(profit, 4)}

        # Serialize acts + enforce per-window cap (WS fires many times per second
        # on a persistent arb — without this we'd fire repeatedly on one window).
        with self._exec_lock:
            if self._win_exec_count.get(slug, 0) >= self.cfg.arb_max_per_window:
                return False
            self._win_exec_count[slug] = self._win_exec_count.get(slug, 0) + 1

            # SAFEGUARD 4: dry-run measurement — detect & log, place nothing.
            if self.cfg.arb_dry_run:
                self.stats["acted"] += 1
                self.stats["gross_profit"] += profit
                self.stats["last_event"] = f"FILLABLE edge {gross_edge:.1%} x {size:.0f}sh = +${profit:.2f} (dry)"
                self._say(f"FILLABLE edge={gross_edge:.1%} {au_px:.2f}+{ad_px:.2f} "
                          f"{size:.0f}sh → +${profit:.2f}  (DRY, no order)")
                self._log({**base, "mode": "DRY", "acted": True})
                return True

            # --- LIVE atomic + CONCURRENT execution (safeguards 1 + 2) ---
            # Tier-1: SIGN both legs first (local, ~0.4ms each when warmed), THEN
            # fire both POSTs concurrently. Signing inside each thread would
            # stagger the two POSTs by the signing time; signing up front shrinks
            # the inter-leg gap to ~the threadpool dispatch (µs). If EITHER leg
            # fails to sign, we place NEITHER — no naked-leg risk.
            self._say(f"LIVE edge={gross_edge:.1%} → signing BOTH legs {size:.0f}sh (FOK)…")
            su = self.client.sign_limit_fok(up_token, au_px, size)
            sd = self.client.sign_limit_fok(down_token, ad_px, size)
            if "signed" not in su or "signed" not in sd:
                self.stats["leg1_killed"] += 1
                self.stats["last_event"] = f"sign failed — placed neither (edge {gross_edge:.1%})"
                self._say("✗ one leg failed to sign — placing NEITHER (no naked risk)")
                self._log({**base, "mode": "LIVE", "acted": False,
                           "result": "both_killed", "leg1_resp": str(su)[:120],
                           "leg2_resp": str(sd)[:120]})
                return False
            fut_up = self._leg_pool.submit(self.client.post_signed_fok, su["signed"])
            fut_dn = self._leg_pool.submit(self.client.post_signed_fok, sd["signed"])
            r1, r2 = fut_up.result(), fut_dn.result()
            f1, f2 = _parse_buy(r1, size), _parse_buy(r2, size)
            lat = {"latency_ms": round((time.time() - detect_ts) * 1000)} if detect_ts else {}
            if lat:
                self.stats["last_lat_ms"] = lat["latency_ms"]

            # RECONCILE on ambiguity: a 'skipped' (post_order failed/timeout) leg
            # MAY actually have landed — trusting _parse_buy=0 there would leave a
            # naked leg untracked. When either response is ambiguous (not a clean
            # 'killed'), re-derive both fills from the actual on-chain balance so
            # the locked/killed/unwind branches below act on the TRUTH.
            def _ambiguous(resp):
                return isinstance(resp, dict) and "skipped" in resp and not resp.get("killed")
            if _ambiguous(r1) or _ambiguous(r2):
                self._say("⚠ ambiguous post response — reconciling positions on-chain")
                f1 = self._position_qty(up_token) or f1
                f2 = self._position_qty(down_token) or f2
                self._log({**base, **lat, "mode": "LIVE",
                           "result": "reconcile", "reconciled_up": f1, "reconciled_dn": f2,
                           "leg1_resp": str(r1)[:120], "leg2_resp": str(r2)[:120]})

            if f1 > 0 and f2 > 0:
                locked = profit  # f1==f2==size when both FOK fill -> size*edge
                self.stats["acted"] += 1; self.stats["locked"] += 1
                self.stats["gross_profit"] += locked
                self.stats["last_event"] = (f"ARB LOCKED +${locked:.2f} ({size:.0f}sh)"
                                            + (f" {lat['latency_ms']}ms" if lat else ""))
                self._say(f"✓ ARB LOCKED: {size:.0f}sh both legs → +${locked:.2f}"
                          + (f"  ({lat['latency_ms']}ms detect→fill)" if lat else ""))
                self._log({**base, **lat, "mode": "LIVE", "acted": True,
                           "result": "ARB_LOCKED", "filled_up": f1, "filled_dn": f2,
                           "locked_profit": round(locked, 4)})
                # both legs held to resolution -> claim the winning side directly.
                self._mark_for_redeem(slug, (up_token, down_token))
                return True

            if f1 <= 0 and f2 <= 0:
                self.stats["leg1_killed"] += 1
                self.stats["last_event"] = f"both legs killed (edge {gross_edge:.1%})"
                self._say("both FOK legs killed — no position, no loss")
                self._log({**base, **lat, "mode": "LIVE", "acted": False,
                           "result": "both_killed", "leg1_resp": str(r1)[:120],
                           "leg2_resp": str(r2)[:120]})
                return False

            # SAFEGUARD 2: exactly one leg filled -> naked directional. UNWIND it.
            if f1 > 0:
                tok, sh, side = up_token, f1, "UP"
            else:
                tok, sh, side = down_token, f2, "DOWN"
            self.stats["acted"] += 1; self.stats["unwound"] += 1
            self.stats["last_event"] = f"⚠ one leg filled ({side}), unwound {sh:.0f}sh"
            self._say(f"⚠ only {side} leg filled — UNWINDING {sh:.0f}sh to flatten")
            # force=True: the arb profile keeps global dry_run on for the
            # directional bot, but this is a REAL filled leg that must actually sell.
            # known_qty=sh: don't let a lagging balance read skip the flatten.
            unwind = self.client.sell_market(tok, sh, force=True, known_qty=sh)
            # KILL-SWITCH: trip after any leg-miss so we don't bleed more naked legs.
            self._halted = True
            self.stats["last_event"] = f"⛔ HALTED — leg-miss ({side} {sh:.0f}sh), unwound; restart to resume"
            self._say(f"⛔ KILL-SWITCH TRIPPED after leg-miss — no further trades until restart")
            self._log({**base, **lat, "mode": "LIVE", "acted": True,
                       "result": "ONE_LEG_UNWOUND", "filled_side": side, "filled_shares": sh,
                       "unwind_resp": str(unwind)[:200], "halted": True,
                       "note": "only one FOK leg filled; flattened, then halted to avoid naked directional risk"})
            # ALWAYS track for redemption. sell_market is FAK (partial fills) and
            # can also skip/fail, so the flatten may leave a residual. Rather than
            # trust the response, mark it and let redeem_resolved's resolution-time
            # balance gate decide: residual -> claimed if it wins; clean flatten ->
            # gate sees 0 and drops it for free. Fixes both the partial-unwind
            # residual and the 2026-05-30 00:43 strand.
            self._mark_for_redeem(slug, (up_token, down_token))
            return True

    def _log(self, record):
        # Non-blocking: hand off to the writer thread so disk I/O never sits on
        # the execution hot path. (Tier-2 minimal-hot-path.)
        self._log_q.put(record)

    def _log_writer(self):
        while True:
            record = self._log_q.get()
            try:
                line = json.dumps(record, separators=(",", ":"))
                with open(self.log_path, "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass
