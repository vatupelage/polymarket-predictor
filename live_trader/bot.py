"""HIGH-same live trading bot for Polymarket BTC up/down 5m.

On every v4 prediction:
  - Arbitrage scan: if top_ask(UP) + top_ask(DOWN) < 1.00 - min_edge,
    buy both sides — guaranteed profit, direction irrelevant.
  - Else: flat-stake directional trade gated on confidence + drift.
  - After fill, monitor the position and cash-out if the price drops
    enough (buy_price - drop, floor) before resolution.

One position at a time, enforced by a lock.
"""

import concurrent.futures as cf
import collections
import json
import os
import statistics
import threading
import time
import datetime

from .config import BotConfig
from .polymarket import PolymarketBotClient, PolymarketError
from .mm_logger import MMLogger
from .arb_executor import ArbExecutor
from . import llm_reviewer
from .fav90 import fav90_decision

# S5 polling needs the same Bitstamp book fetch that drives ensemble_predict_v4's
# orderbook signal — without it, the live "ob_side" would diverge from backtest.
# Imported lazily inside the poll loop to avoid cyclic-import surprises.
def _s5_fetch_btc_book():
    try:
        from run_baseline import fetch_order_book  # type: ignore
        return fetch_order_book()
    except Exception:
        return None


# Ask-recheck gate parameters are configured via BotConfig fields:
#   ask_recheck_enabled, ask_recheck_delay_s, ask_recheck_tolerance,
#   ask_recheck_timeout_s, ask_recheck_max_ref_ask.
# Disabled by default after 2026-05-15 audit — see config.py for rationale.


TRADE_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "trade_history.jsonl",
)
SKIP_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "skip_history.jsonl",
)
# S5 SHADOW LOG — one JSONL record per S5 poll. Records BOTH Bitstamp ob_side
# (the value the bot actually gates on) AND Polymarket-implied side prob (logged
# but NOT gated). Lets us later determine which signal is more predictive of
# realized outcome. Schema:
#   {ts, slug, poll_n, direction, our_ask, top_ask_up, top_ask_down,
#    bitstamp_imb, bitstamp_near_imb, bitstamp_ob_up_prob, bitstamp_ob_side,
#    pm_implied_side_prob, in_s5_band, bitstamp_passes_85, pm_passes_85, matched}
S5_SHADOW_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "s5_shadow.jsonl",
)
# FAV90 LOG — one JSONL record per fav90 poll (fired or not). Gives us the late
# (60-100s to close) book snapshots our historical files never captured. Schema:
#   {ts, slug, secs_to_close, fav_side, fav_ask, other_ask, overround,
#    ask_depth, live_price, ptb, btc_gap_pct, timing_ok, price_ok, depth_ok, fired}
FAV90_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fav90_log.jsonl",
)
# FAV90 POST-ENTRY TRAJECTORY — after a fill, log the held position's bid every
# 10s until close (read-only, NO orders). Lets us later evaluate a stop-loss:
# "bought at ~0.90, did it dip to 0.70, and did it recover to win?" Schema:
#   {ts, slug, direction, fill_px, secs_to_close, bid}
FAV90_TRAJ_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "fav90_trajectory.jsonl",
)
# DBMODEL LOG — one JSONL record per dbmodel fire (the model bets every window).
# Schema: {ts, slug, p_up, direction, ask, shares, fill_px, stake, won, pnl}
_PRED_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Both log paths are env-overridable so multiple per-market dbmodel processes
# (Stage B multi-market paper-logging) each write their own file instead of
# racing on a shared append — concurrent writes of large records (full feature
# vector + book path) can exceed the atomic-write size and interleave.
DBMODEL_LOG_PATH = os.environ.get(
    "BOT_DBMODEL_LOG", os.path.join(_PRED_DIR, "dbmodel_log.jsonl"))
# DBMODEL PAPER LOG — rich JSONL written only in dry-run, one record per window,
# with the full feature vector + raw/calibrated proba + both asks + a hypothetical
# fill + the realized outcome settled by gamma (ground truth) and cross-checked
# against Binance. Doubles as a reusable backtest dataset (see analyze_dbmodel_paper.py).
DBMODEL_PAPER_LOG_PATH = os.environ.get(
    "BOT_DBMODEL_PAPER_LOG", os.path.join(_PRED_DIR, "dbmodel_paper.jsonl"))


def _fmt_ts():
    return datetime.datetime.now().strftime("%H:%M:%S")


def _hour_in_blackout(hour: int, spec: str) -> bool:
    """Match `hour` against a comma-separated list of half-open ranges.

    Format: "18-24" or "0-6,18-24". A range "a-b" matches a <= hour < b.
    Invalid tokens are silently skipped — the gate is best-effort and an
    empty / malformed spec means "no blackout".
    """
    if not spec:
        return False
    for part in spec.split(","):
        part = part.strip()
        if not part or "-" not in part:
            continue
        try:
            a_s, b_s = part.split("-", 1)
            a, b = int(a_s), int(b_s)
        except ValueError:
            continue
        if a <= hour < b:
            return True
    return False


def _flatten_signals(signals):
    """Compress {ptb,orderbook,lstm,polymarket}->{up_prob,weight} into a flat dict."""
    if not signals:
        return None
    out = {}
    for name, raw in signals.items():
        if not isinstance(raw, dict):
            continue
        p = raw.get("up_prob")
        w = raw.get("weight")
        if p is not None:
            out[f"{name}_up_prob"] = round(float(p), 4)
        if w is not None:
            out[f"{name}_weight"] = round(float(w), 4)
    return out or None


def _parse_fill(order_resp: dict, stake_usdc: float):
    """BUY: extract (shares_received, avg_fill_price, actual_filled_usdc).

    `makingAmount` (USDC spent) is the source of truth — using `stake_usdc`
    overstates cost on partial fills.
    """
    if not isinstance(order_resp, dict):
        return None
    if order_resp.get("dry_run"):
        return (stake_usdc / 0.5, 0.5, stake_usdc)
    shares = None
    for shares_key in ("takingAmount", "filledSize", "size_matched"):
        raw = order_resp.get(shares_key)
        if raw is None:
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if val <= 0:
            continue
        shares = val / 1e6 if val >= 1e5 else val
        if shares > 0:
            break
        shares = None
    if shares is None:
        return None
    actual_usdc = None
    raw_making = order_resp.get("makingAmount")
    if raw_making is not None:
        try:
            mv = float(raw_making)
            if mv >= 1e5:
                mv /= 1e6
            if mv > 0:
                actual_usdc = mv
        except (TypeError, ValueError):
            pass
    if actual_usdc is None:
        actual_usdc = stake_usdc  # fall back to intent if response is missing field
    px = actual_usdc / shares
    return (shares, px, actual_usdc)


def _parse_sell_fill(order_resp: dict, shares_attempted: float) -> tuple[float, float, float] | None:
    """SELL: extract (usdc_received, shares_filled, avg_sell_price)."""
    if not isinstance(order_resp, dict):
        return None
    if order_resp.get("dry_run"):
        return (shares_attempted * 0.5, shares_attempted, 0.5)
    raw_usdc = order_resp.get("takingAmount")
    raw_shares = order_resp.get("makingAmount")
    if raw_usdc is None or raw_shares is None:
        return None
    try:
        usdc = float(raw_usdc)
        shares_f = float(raw_shares)
    except (TypeError, ValueError):
        return None
    if usdc >= 1e5:
        usdc /= 1e6
    if shares_f >= 1e5:
        shares_f /= 1e6
    if shares_f <= 0:
        return None
    return (usdc, shares_f, usdc / shares_f)


class HighSameBot:
    def __init__(self, cfg: BotConfig, client: PolymarketBotClient):
        self.cfg = cfg
        self.client = client
        self._lock = threading.Lock()
        self._active = 0
        self._wins = 0
        self._losses = 0
        self._pnl = 0.0
        self._trades: list[dict] = []
        self._session_pnl = 0.0
        self._daily_pnl = 0.0
        self._daily_date = datetime.date.today()
        self._daily_target_hit = False
        # Volatility-gate price history: ring buffer of recent live_price samples
        # sized to vol_gate_baseline_window. Seeded from disk on startup so the
        # gate has a meaningful baseline before fresh samples accumulate.
        self._price_history: collections.deque[float] = collections.deque(
            maxlen=max(self.cfg.vol_gate_baseline_window, self.cfg.vol_gate_window + 1)
        )
        if self.cfg.vol_gate_threshold > 0:
            self._seed_price_history()
        # S5 mode: track trades-today for the 1/day cap. Resets at UTC date change.
        self._s5_trades_today = 0
        self._s5_date = datetime.datetime.utcnow().date()
        # Market-making data collector (read-only — MM_DATA_SPEC.md). No orders.
        self.mm_logger = MMLogger(
            client,
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            cadence_s=self.cfg.mm_log_cadence_s,
            enabled=self.cfg.mm_log_enabled,
        )
        # Arbitrage executor (read-only/dry-run by default — see arb_executor.py).
        self.arb_executor = ArbExecutor(
            client, self.cfg,
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

    def _seed_price_history(self) -> None:
        """Populate the price-history deque from trade_history.jsonl and
        skip_history.jsonl on startup, in chronological order. Without this,
        the vol gate has no baseline for ~3 days after a cold start.
        """
        rows = []
        for path in (TRADE_LOG_PATH, SKIP_LOG_PATH):
            if not os.path.exists(path):
                continue
            try:
                with open(path) as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                            px = r.get("live_price")
                            ts = r.get("ts")
                            if px is None or ts is None:
                                continue
                            rows.append((ts, float(px)))
                        except Exception:
                            continue
            except Exception:
                continue
        rows.sort(key=lambda t: t[0])
        cap = self._price_history.maxlen or 0
        for _, px in rows[-cap:]:
            self._price_history.append(px)
        print(f"  [BOT {_fmt_ts()}] vol-gate seeded with {len(self._price_history)} "
              f"historical prices (cap={cap})")

    def _vol_gate_should_skip(self, live_price: float) -> tuple[bool, float, float, float]:
        """Return (should_skip, vol_now, vol_baseline, ratio).

        vol_now      = stdev of pct returns over the last vol_gate_window samples.
        vol_baseline = median of rolling vol over the full deque.
        ratio        = vol_now / vol_baseline.

        Skip if ratio > cfg.vol_gate_threshold.
        """
        self._price_history.append(live_price)
        prices = list(self._price_history)
        win = self.cfg.vol_gate_window
        if len(prices) < win + 1:
            return (False, 0.0, 0.0, 0.0)
        recent = prices[-(win + 1):]
        rets_now = [recent[i + 1] / recent[i] - 1 for i in range(len(recent) - 1)]
        vol_now = statistics.pstdev(rets_now) if len(rets_now) >= 2 else 0.0
        # Rolling baseline: compute stdev over every overlapping window in deque
        rolling_vols = []
        for start in range(0, len(prices) - win):
            seg = prices[start:start + win + 1]
            rs = [seg[i + 1] / seg[i] - 1 for i in range(len(seg) - 1)]
            if len(rs) >= 2:
                rolling_vols.append(statistics.pstdev(rs))
        if len(rolling_vols) < 10:
            # Not enough baseline data — don't gate during warmup
            return (False, vol_now, 0.0, 0.0)
        vol_baseline = statistics.median(rolling_vols)
        if vol_baseline <= 0:
            return (False, vol_now, vol_baseline, 0.0)
        ratio = vol_now / vol_baseline
        return (ratio > self.cfg.vol_gate_threshold, vol_now, vol_baseline, ratio)

    def on_prediction(self, *, window, ptb, live_price, direction, confidence,
                      final_up, signals, lstm_prob):
        today = datetime.date.today()
        if today != self._daily_date:
            print(f"  [BOT {_fmt_ts()}] new day {today} — resetting daily PnL "
                  f"(was ${self._daily_pnl:+.2f} on {self._daily_date})")
            self._daily_date = today
            self._daily_pnl = 0.0
            self._daily_target_hit = False

        # S5 TEST MODE: 1/day cap REMOVED — bot trades continuously, every window
        # treated as its own "day". `_s5_trades_today` still increments per fire so
        # it shows up in logs, but is never checked against a cap. Re-enable the
        # cap by restoring the `if self._s5_trades_today >= 1: return` block.
        if self.cfg.s5_mode:
            utc_today = datetime.datetime.utcnow().date()
            if utc_today != self._s5_date:
                self._s5_date = utc_today
                self._s5_trades_today = 0

        if self.cfg.hard_stop_loss > 0 and self._session_pnl <= -self.cfg.hard_stop_loss:
            print(f"  [BOT {_fmt_ts()}] HARD STOP LOSS hit: session_pnl="
                  f"${self._session_pnl:+.2f} <= -${self.cfg.hard_stop_loss:.2f} — EXITING")
            os._exit(0)

        if self.cfg.daily_profit_target > 0 and self._daily_pnl >= self.cfg.daily_profit_target:
            print(f"  [BOT {_fmt_ts()}] DAILY TARGET hit: daily_pnl="
                  f"${self._daily_pnl:+.2f} >= ${self.cfg.daily_profit_target:.2f} — EXITING")
            os._exit(0)

        # FAV90 MODE: bypass the entire model gate stack. Dispatch straight to the
        # late-poll favorite executor (it picks the side from the PM book). Daily
        # kill-switch reuses the already-tracked realized _daily_pnl.
        if self.cfg.fav90_mode:
            if self.cfg.fav90_daily_stop < 0 and self._daily_pnl <= self.cfg.fav90_daily_stop:
                print(f"  [BOT {_fmt_ts()}] FAV90 kill-switch: daily_pnl="
                      f"${self._daily_pnl:+.2f} <= ${self.cfg.fav90_daily_stop:.2f} — "
                      f"trading paused for the UTC day (still logging via poll loop)")
                return
            with self._lock:
                if self._active >= self.cfg.max_concurrent_trades:
                    print(f"  [BOT {_fmt_ts()}] FAV90 skip: {self._active} positions open "
                          f"(cap={self.cfg.max_concurrent_trades})")
                    return
                self._active += 1
            threading.Thread(
                target=self._execute_trade,
                args=(window, direction, confidence, ptb, live_price, drift_pct, signals, final_up),
                daemon=True,
            ).start()
            return

        slug = window["slug"]
        end_ts = window["end_ts"]
        drift_pct = (live_price - ptb) / ptb * 100 if ptb else 0.0

        # NOTE: MM data collection is NOT triggered here. The trading loop only
        # predicts every OTHER 5-min window (the alternate is used for deep
        # analysis), so hooking MM logging to predictions would miss half the
        # windows. Instead mm_logger.start_continuous() runs on its own clock
        # (started in run_live_bot.py) and captures every consecutive window.

        if confidence < self.cfg.high_conf_threshold:
            self._record_skip(
                slug=slug, end_ts=end_ts, direction=direction, confidence=confidence,
                reason="conf_too_low",
                details=f"{confidence:.2f}% < {self.cfg.high_conf_threshold:.1f}% floor",
                ptb=ptb, live_price=live_price, drift_pct=drift_pct, final_up=final_up,
                signals=signals,
            )
            return
        if self.cfg.max_conf_threshold > 0 and confidence > self.cfg.max_conf_threshold:
            print(f"  [BOT {_fmt_ts()}] SKIP (conf too high): {confidence:.2f}% "
                  f"> {self.cfg.max_conf_threshold:.1f}% cap")
            self._record_skip(
                slug=slug, end_ts=end_ts, direction=direction, confidence=confidence,
                reason="conf_too_high",
                details=f"{confidence:.2f}% > {self.cfg.max_conf_threshold:.1f}% cap",
                ptb=ptb, live_price=live_price, drift_pct=drift_pct, final_up=final_up,
                signals=signals,
            )
            return

        # Hour-of-day blackout. OOS conditional backtest: hour band [18, 24)
        # had E[$1]=-$0.020 (n=33) — the only losing slot. Disabled when
        # blackout_hours is empty. Local time used to match trade_history.jsonl
        # timestamps (which are recorded via datetime.now()).
        if self.cfg.blackout_hours:
            local_hour = datetime.datetime.now().hour
            if _hour_in_blackout(local_hour, self.cfg.blackout_hours):
                print(f"  [BOT {_fmt_ts()}] SKIP (hour-blackout): hour={local_hour} "
                      f"in blackout {self.cfg.blackout_hours}")
                self._record_skip(
                    slug=slug, end_ts=end_ts, direction=direction, confidence=confidence,
                    reason="hour_blackout",
                    details=f"hour={local_hour} in {self.cfg.blackout_hours}",
                    ptb=ptb, live_price=live_price, drift_pct=drift_pct, final_up=final_up,
                    signals=signals,
                )
                return

        # Volatility regime gate. When recent realized vol >> rolling baseline,
        # the bot's entire signal stack degrades (all four signals become
        # momentum-following). Backtest on n=257 trades: at threshold=2.0×,
        # fires 16.7% of trades, kept-trade meanR +$0.63 vs skipped meanR
        # -$4.17. Outlier-stripped Δ = +$43; full-set Δ = +$179. On the
        # 2026-05-22→23 crash night: would have skipped 3/6 trades, saving
        # $45 (vs naive Option-A "gate just LSTM weight" which had ZERO flips
        # because LSTM weight is only 10-15% — see /predictor/CURRENT_STRATEGY.md).
        # Set BOT_VOL_GATE_THRESHOLD=0 to disable.
        # S5 mode bypasses vol-gate entirely — S5's edge depends on crash-day wins
        # (05-19, 05-20, 05-22) that vol-gate would otherwise skip. Backtest showed
        # S5 with vol-gate bypassed survives outlier-strip (+$0.71) and 10% slippage.
        # We still keep the price-history deque fresh so vol-gate stays warm if S5
        # is later disabled (deque update happens inside _vol_gate_should_skip).
        if self.cfg.s5_mode and self.cfg.vol_gate_threshold > 0 and live_price:
            self._vol_gate_should_skip(live_price)  # update deque, ignore decision
        if (not self.cfg.s5_mode) and self.cfg.vol_gate_threshold > 0 and live_price:
            should_skip, vol_now, vol_base, ratio = self._vol_gate_should_skip(live_price)
            if should_skip:
                print(f"  [BOT {_fmt_ts()}] SKIP (vol-regime-break): "
                      f"realized_vol={vol_now*100:.4f}% / baseline={vol_base*100:.4f}% "
                      f"= {ratio:.2f}× > {self.cfg.vol_gate_threshold:.2f}× threshold")
                self._record_skip(
                    slug=slug, end_ts=end_ts, direction=direction, confidence=confidence,
                    reason="vol_regime_break",
                    details=f"vol_now={vol_now*100:.4f}% baseline={vol_base*100:.4f}% "
                            f"ratio={ratio:.2f}× thresh={self.cfg.vol_gate_threshold:.2f}×",
                    ptb=ptb, live_price=live_price, drift_pct=drift_pct, final_up=final_up,
                    signals=signals,
                )
                return

        # LSTM-inverted alignment gate. The bot already inverts LSTM weight
        # in the blend (LSTM is anti-predictive on this market). When the
        # final direction *agrees* with LSTM's call, that's the bad bucket:
        # n=58 in OOS backtest, 51.7% W, E[$1]=-$0.090. Conversely
        # direction-disagrees-with-LSTM is the strongest surviving edge:
        # n=102, 74.5% W, E[$1]=+$0.245 (CI [+0.100, +0.392], no zero-cross).
        # Gate skips the lose-bucket. Set BOT_LSTM_INV_GATE=true to enable.
        if self.cfg.lstm_inv_gate_enabled and lstm_prob is not None:
            aligns_with_lstm = (
                (direction == "UP" and lstm_prob >= 0.5) or
                (direction == "DOWN" and lstm_prob < 0.5)
            )
            if aligns_with_lstm:
                print(f"  [BOT {_fmt_ts()}] SKIP (lstm-inv-contra): {direction} "
                      f"agrees with LSTM(up={lstm_prob:.3f}) — anti-predictive bucket")
                self._record_skip(
                    slug=slug, end_ts=end_ts, direction=direction, confidence=confidence,
                    reason="lstm_inv_contra",
                    details=f"{direction} with lstm_prob={lstm_prob:.3f} "
                            f"(LSTM agrees, gate fires)",
                    ptb=ptb, live_price=live_price, drift_pct=drift_pct, final_up=final_up,
                    signals=signals,
                )
                return

        # UP-direction confidence floor. Post-gate window: 21 UP trades at
        # conf<7 had per-$1 EV -$0.05; DOWN at the same conf is profitable.
        # Asymmetric — does not apply to DOWN. Set BOT_UP_HIGH_CONF=0 to disable.
        if (direction == "UP"
                and self.cfg.up_high_conf_threshold > 0
                and confidence < self.cfg.up_high_conf_threshold):
            print(f"  [BOT {_fmt_ts()}] SKIP (up-conf-too-low): UP conf={confidence:.2f}% "
                  f"< {self.cfg.up_high_conf_threshold:.1f}% UP floor")
            self._record_skip(
                slug=slug, end_ts=end_ts, direction=direction, confidence=confidence,
                reason="up_conf_too_low",
                details=f"UP conf={confidence:.2f}% < {self.cfg.up_high_conf_threshold:.1f}% UP floor",
                ptb=ptb, live_price=live_price, drift_pct=drift_pct, final_up=final_up,
                signals=signals,
            )
            return

        # Drift-noise gate: at high confidence the bot's signals collapse to
        # noise when BTC isn't moving. Fitted on lifetime data: 2/2 lifetime
        # 12%+ trades with |drift|<0.0005% lost full stake (incl. the
        # 2026-05-06 $30 bomb at drift=-0.00028%). Set noise_drift_pct=0
        # to disable.
        if (self.cfg.noise_drift_pct > 0
                and confidence >= self.cfg.noise_conf_floor
                and drift_pct is not None
                and abs(drift_pct) < self.cfg.noise_drift_pct):
            print(f"  [BOT {_fmt_ts()}] SKIP (drift-noise): {direction} "
                  f"|drift|={abs(drift_pct):.5f}% < {self.cfg.noise_drift_pct:.5f}% "
                  f"at conf={confidence:.1f}% (>= {self.cfg.noise_conf_floor:.1f}%)")
            self._record_skip(
                slug=slug, end_ts=end_ts, direction=direction, confidence=confidence,
                reason="drift_noise",
                details=f"|drift|={abs(drift_pct):.5f}% < {self.cfg.noise_drift_pct:.5f}% "
                        f"at conf={confidence:.1f}%",
                ptb=ptb, live_price=live_price, drift_pct=drift_pct, final_up=final_up,
                signals=signals,
            )
            return

        if not self.cfg.use_llm and self.cfg.contra_drift_enabled:
            noise_band = self.cfg.min_drift_pct
            contradicts = (
                (direction == "UP" and drift_pct < -noise_band) or
                (direction == "DOWN" and drift_pct > noise_band)
            )
            # Override the gate when (a) drift magnitude is already extreme
            # (gate is anti-predictive at |drift|>=0.04%, hit rate 38%) or
            # (b) model conviction is high (gate hit rate drops to 33-40%
            # at conf>=10%). Both fitted on 79 lifetime contra_drift skips.
            abs_drift = abs(drift_pct) if drift_pct is not None else 0.0
            extreme_drift = abs_drift >= 0.04
            high_conf = confidence >= 10.0
            if contradicts and not extreme_drift and not high_conf:
                print(f"  [BOT {_fmt_ts()}] SKIP (contra-drift): {direction} "
                      f"vs drift={drift_pct:+.4f}% ({confidence:.1f}%)")
                self._record_skip(
                    slug=slug, end_ts=end_ts, direction=direction, confidence=confidence,
                    reason="contra_drift",
                    details=f"{direction} vs drift={drift_pct:+.4f}% (band ±{noise_band:.4f})",
                    ptb=ptb, live_price=live_price, drift_pct=drift_pct, final_up=final_up,
                )
                return
            if contradicts and (extreme_drift or high_conf):
                print(f"  [BOT {_fmt_ts()}] contra-drift override: "
                      f"{direction} drift={drift_pct:+.4f}% conf={confidence:.1f}% "
                      f"({'extreme_drift' if extreme_drift else ''}"
                      f"{' & ' if extreme_drift and high_conf else ''}"
                      f"{'high_conf' if high_conf else ''}) — proceeding")

        print(f"  [BOT {_fmt_ts()}] decision: {direction} conf={confidence:.1f}% "
              f"drift={drift_pct:+.4f}%")

        with self._lock:
            if self._active >= self.cfg.max_concurrent_trades:
                print(f"  [BOT {_fmt_ts()}] skip: {self._active} positions already open "
                      f"(cap={self.cfg.max_concurrent_trades})")
                self._record_skip(
                    slug=slug, end_ts=end_ts, direction=direction, confidence=confidence,
                    reason="position_open",
                    details=f"{self._active} active >= cap {self.cfg.max_concurrent_trades}",
                    ptb=ptb, live_price=live_price, drift_pct=drift_pct, final_up=final_up,
                )
                return
            self._active += 1

        threading.Thread(
            target=self._execute_trade,
            args=(window, direction, confidence, ptb, live_price, drift_pct, signals, final_up),
            daemon=True,
        ).start()

    def _execute_trade(self, window, direction, confidence, ptb=None, live_price=None,
                       drift_pct=None, signals=None, final_up=None):
        slug = window["slug"]
        end_ts = window["end_ts"]
        try:
            market = self.client.resolve_market(slug)
            up_token = market["up_token"]
            down_token = market["down_token"]
            condition_id = market.get("condition_id")

            top_ask_up = self.client.get_top_ask(up_token)
            top_ask_down = self.client.get_top_ask(down_token)

            our_ask = top_ask_up if direction == "UP" else top_ask_down
            skip_kw = dict(
                slug=slug, end_ts=end_ts, direction=direction, confidence=confidence,
                ptb=ptb, live_price=live_price, drift_pct=drift_pct, final_up=final_up,
                top_ask_up=top_ask_up, top_ask_down=top_ask_down, signals=signals,
            )
            # FAV90 late-confirm branch — poll every 10s until close-60. Buy the
            # FAVORITE (higher ask) when secs_to_close <= entry_max_s AND its ask is
            # in [min_ask, max_ask] AND the PM book shows >= min_ask_depth shares
            # within 1 tick of the ask. Overrides `direction` to the favorite side;
            # logs EVERY poll (fired or not) to FAV90_LOG_PATH for the late dataset.
            if self.cfg.fav90_mode:
                poll_interval = 10.0
                window_close_ts = window.get("end_ts") or (time.time() + 240)
                deadline = window_close_ts - 60  # leave 60s for fill + settlement
                poll_count = 0
                matched = False
                while time.time() < deadline:
                    poll_count += 1
                    secs_to_close = window_close_ts - time.time()
                    try:
                        top_ask_up = self.client.get_top_ask(up_token)
                        top_ask_down = self.client.get_top_ask(down_token)
                    except Exception as e:
                        print(f"  [BOT {_fmt_ts()}] FAV90 poll {poll_count}: top_ask fetch failed ({e})")
                        time.sleep(poll_interval)
                        continue
                    if top_ask_up is not None and top_ask_down is not None:
                        side = "UP" if top_ask_up >= top_ask_down else "DOWN"
                    else:
                        side = None
                    fav_token = up_token if side == "UP" else (down_token if side == "DOWN" else None)
                    fav_asks = []
                    if fav_token is not None:
                        fb = self.client.get_full_book(fav_token)
                        if fb is not None:
                            fav_asks = fb.get("asks") or []
                    decision = fav90_decision(
                        secs_to_close=secs_to_close, top_ask_up=top_ask_up,
                        top_ask_down=top_ask_down, fav_asks=fav_asks,
                        entry_max_s=self.cfg.fav90_entry_max_s,
                        min_ask=self.cfg.fav90_min_ask, max_ask=self.cfg.fav90_max_ask,
                        min_depth=self.cfg.fav90_min_ask_depth,
                    )
                    overround = ((top_ask_up + top_ask_down - 1.0)
                                 if (top_ask_up is not None and top_ask_down is not None) else None)
                    btc_gap = ((live_price - ptb) / ptb * 100.0) if (live_price and ptb) else None
                    fa = decision["fav_ask"]
                    print(f"  [BOT {_fmt_ts()}] FAV90 poll #{poll_count}: s2c={secs_to_close:.0f}s "
                          f"fav={decision['side']} ask={fa if fa is None else f'{fa:.3f}'} "
                          f"depth={decision['depth']} "
                          f"[t={'Y' if decision['timing_ok'] else 'N'} "
                          f"p={'Y' if decision['price_ok'] else 'N'} "
                          f"d={'Y' if decision['depth_ok'] else 'N'}]")
                    try:
                        with open(FAV90_LOG_PATH, "a") as f:
                            f.write(json.dumps({
                                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                                "slug": slug,
                                "secs_to_close": round(secs_to_close, 1),
                                "fav_side": decision["side"],
                                "fav_ask": decision["fav_ask"],
                                "other_ask": (top_ask_down if decision["side"] == "UP" else top_ask_up),
                                "overround": (round(overround, 4) if overround is not None else None),
                                "ask_depth": decision["depth"],
                                "live_price": live_price,
                                "ptb": ptb,
                                "btc_gap_pct": (round(btc_gap, 4) if btc_gap is not None else None),
                                "timing_ok": decision["timing_ok"],
                                "price_ok": decision["price_ok"],
                                "depth_ok": decision["depth_ok"],
                                "fired": decision["fire"],
                            }) + "\n")
                    except Exception as _log_err:
                        print(f"  [BOT {_fmt_ts()}] fav90 log error: {_log_err}")
                    if decision["fire"]:
                        direction = decision["side"]
                        our_ask = decision["fav_ask"]
                        matched = True
                        break
                    time.sleep(poll_interval)
                if not matched:
                    print(f"  [BOT {_fmt_ts()}] SKIP (fav90-window-expired): {poll_count} polls")
                    self._record_skip(reason="fav90_window_expired",
                                      details=f"polls={poll_count}", **skip_kw)
                    return
                skip_kw["direction"] = direction
                skip_kw["top_ask_up"] = top_ask_up
                skip_kw["top_ask_down"] = top_ask_down
                print(f"  [BOT {_fmt_ts()}] FAV90 MATCH poll #{poll_count}: BUY {direction} "
                      f"ask={our_ask:.3f} (favorite, late depth-confirmed)")

            # S5 strategy gates — only active when --s5 profile is set.
            # 1) Entry in [0.40, 0.50) ∪ (0.58, 0.75]  — excludes losing middle 0.50-0.58 tier
            # 2) Side-aligned orderbook prob >= 0.85   — strong same-side book consensus
            #
            # Instead of checking once at T-4min and skipping if conditions fail, we POLL the
            # Polymarket top-of-book + Bitstamp BTC book every 10s until window close - 60s,
            # firing the moment both conditions become true (first-match wins). Direction
            # stays frozen at the T-4min decision (matches backtest); only the entry timing
            # and live ob_side are re-evaluated. Backtest n=13/23d, 85% W, +$0.923/$1,
            # perm-p=0.0004 (Bonferroni-safe).
            if self.cfg.s5_mode:
                poll_interval = 10.0
                window_close_ts = window.get("end_ts") or (time.time() + 240)
                deadline = window_close_ts - 60  # leave 60s for fill + settlement
                poll_count = 0
                matched = False
                last_ask = our_ask
                last_ob_side = None
                while time.time() < deadline:
                    poll_count += 1
                    # Re-fetch chosen-side top ask (re-using earlier value on first iter)
                    if poll_count > 1:
                        try:
                            top_ask_up = self.client.get_top_ask(up_token)
                            top_ask_down = self.client.get_top_ask(down_token)
                        except Exception as e:
                            print(f"  [BOT {_fmt_ts()}] S5 poll {poll_count}: top_ask fetch failed ({e})")
                            time.sleep(poll_interval)
                            continue
                        our_ask = top_ask_up if direction == "UP" else top_ask_down
                    last_ask = our_ask
                    if our_ask is None:
                        time.sleep(poll_interval)
                        continue

                    if self.cfg.s5_disable_lower:
                        in_s5_band = (0.58 < our_ask <= 0.75)
                    else:
                        in_s5_band = (0.40 <= our_ask < 0.50) or (0.58 < our_ask <= 0.75)

                    # === BITSTAMP ob_side (the GATE — bot trades on this) ===
                    # Signal-stack formula: ob_up_prob = clip(0.5 + 0.4*imb + 0.6*near_imb
                    #                                       + 0.3*trend, 0, 1).
                    # No trend during live polling (single snapshot), so trend term dropped.
                    book = _s5_fetch_btc_book()
                    if book is None:
                        time.sleep(poll_interval)
                        continue
                    imb = book.get("imbalance", 0.0)
                    near_imb = book.get("near_imbalance", 0.0)
                    ob_up_prob_live = max(0.0, min(1.0, 0.5 + 0.4 * imb + 0.6 * near_imb))
                    ob_side = ob_up_prob_live if direction == "UP" else (1.0 - ob_up_prob_live)
                    last_ob_side = ob_side
                    ok_ob = ob_side >= 0.85

                    # === POLYMARKET-IMPLIED side prob (SHADOW — logged, NOT gated) ===
                    # In a Polymarket binary market, top_ask is the price you pay for the
                    # YES token = market's implied prob of YES. Best symmetric estimate of
                    # the chosen-side prob:
                    #   pm_side = (top_ask_chosen + (1 - top_ask_other)) / 2
                    # i.e. average of "what I pay to bet my side" and "what the other side
                    # would lose by NOT betting against me". Tight markets: both ≈ equal.
                    pm_side_prob = None
                    if top_ask_up is not None and top_ask_down is not None:
                        if direction == "UP":
                            pm_side_prob = (top_ask_up + (1.0 - top_ask_down)) / 2.0
                        else:
                            pm_side_prob = (top_ask_down + (1.0 - top_ask_up)) / 2.0
                    pm_passes = pm_side_prob is not None and pm_side_prob >= 0.85

                    print(f"  [BOT {_fmt_ts()}] S5 poll #{poll_count}: ask={our_ask:.3f} "
                          f"[band={'Y' if in_s5_band else 'N'}]  "
                          f"bitstamp ob={ob_side:.3f} [{'Y' if ok_ob else 'N'}]  "
                          f"pm_side={pm_side_prob if pm_side_prob is None else f'{pm_side_prob:.3f}'} "
                          f"[{'Y' if pm_passes else 'N'} shadow]")

                    # Shadow log — write every poll regardless of outcome
                    try:
                        with open(S5_SHADOW_LOG_PATH, "a") as f:
                            f.write(json.dumps({
                                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                                "slug": slug,
                                "poll_n": poll_count,
                                "direction": direction,
                                "our_ask": our_ask,
                                "top_ask_up": top_ask_up,
                                "top_ask_down": top_ask_down,
                                "bitstamp_imb": round(imb, 4),
                                "bitstamp_near_imb": round(near_imb, 4),
                                "bitstamp_ob_up_prob": round(ob_up_prob_live, 4),
                                "bitstamp_ob_side": round(ob_side, 4),
                                "pm_implied_side_prob": (round(pm_side_prob, 4)
                                                          if pm_side_prob is not None else None),
                                "in_s5_band": in_s5_band,
                                "bitstamp_passes_85": ok_ob,
                                "pm_passes_85": pm_passes,
                                "matched": in_s5_band and ok_ob,
                            }) + "\n")
                    except Exception as _log_err:
                        print(f"  [BOT {_fmt_ts()}] s5 shadow log error: {_log_err}")

                    if in_s5_band and ok_ob:
                        matched = True
                        break
                    time.sleep(poll_interval)

                if not matched:
                    ask_str = f"{last_ask:.3f}" if last_ask is not None else "n/a"
                    ob_str = f"{last_ob_side:.3f}" if last_ob_side is not None else "n/a"
                    print(f"  [BOT {_fmt_ts()}] SKIP (s5-window-expired): "
                          f"{poll_count} polls, last ask={ask_str} ob_side={ob_str}")
                    self._record_skip(reason="s5_window_expired",
                                      details=f"polls={poll_count} last_ask={ask_str} "
                                              f"last_ob_side={ob_str}",
                                      **skip_kw)
                    return

                # Conditions met — commit to our 1/day budget BEFORE the actual BUY so
                # subsequent windows in the same UTC day don't fire even if this fill fails.
                self._s5_trades_today += 1
                # Refresh skip_kw with the final ask values (used by downstream gates)
                skip_kw["top_ask_up"] = top_ask_up
                skip_kw["top_ask_down"] = top_ask_down
                print(f"  [BOT {_fmt_ts()}] S5 MATCH on poll #{poll_count}: {direction} "
                      f"ask={our_ask:.3f} ob_side={last_ob_side:.3f} "
                      f"(S5 trade #{self._s5_trades_today} today UTC)")

            # Entry-band gate — primary profile filter (driven by --a / --b
            # flags in run_live_bot.py). Skip any trade whose chosen-side ask
            # falls outside [entry_min, entry_max). When entry_min=0 AND
            # entry_max>=1.0, this gate is effectively off.
            if (not self.cfg.fav90_mode) and our_ask is not None and (
                    our_ask < self.cfg.entry_min
                    or (self.cfg.entry_max < 1.0 and our_ask >= self.cfg.entry_max)):
                print(f"  [BOT {_fmt_ts()}] SKIP (entry-out-of-band): {direction} "
                      f"ask={our_ask:.3f} not in [{self.cfg.entry_min:.2f}, {self.cfg.entry_max:.2f})")
                self._record_skip(reason="entry_out_of_band",
                                  details=f"ask={our_ask:.3f} not in "
                                          f"[{self.cfg.entry_min:.2f}, {self.cfg.entry_max:.2f})",
                                  **skip_kw)
                return
            # High-entry + mid-conf gate. Fitted on post-gate window:
            # entry>=0.70 AND conf 7-12 -> per-$1 EV -$0.34 across 4 trades,
            # incl. both 2026-05-15 $70 losses. Skip when chosen-side ask is
            # >= cap AND conf < floor. Set high_entry_cap=0 to disable.
            if (not self.cfg.fav90_mode
                    and self.cfg.high_entry_cap > 0
                    and our_ask is not None
                    and our_ask >= self.cfg.high_entry_cap
                    and confidence < self.cfg.high_entry_conf_floor):
                print(f"  [BOT {_fmt_ts()}] SKIP (high-entry-low-conf): {direction} "
                      f"ask={our_ask:.3f} >= {self.cfg.high_entry_cap:.2f} cap at "
                      f"conf={confidence:.1f}% (< {self.cfg.high_entry_conf_floor:.1f}% floor)")
                self._record_skip(reason="high_entry_low_conf",
                                  details=f"ask={our_ask:.3f} >= {self.cfg.high_entry_cap:.2f} "
                                          f"with conf={confidence:.1f}% < {self.cfg.high_entry_conf_floor:.1f}%",
                                  **skip_kw)
                return
            # contra-book: only fire at low confidence. At conf>=7 the bot's
            # signal beats the book's disagreement — lifetime per-$1 EV is
            # +$0.10/skip (7-12%) and +$0.21/skip (12%+). Confidence floor
            # is configurable via BOT_CONTRA_BOOK_MAX_CONF (default 7.0).
            if (our_ask is not None and our_ask < 0.40
                    and confidence < self.cfg.contra_book_max_conf):
                print(f"  [BOT {_fmt_ts()}] SKIP (contra-book): {direction} "
                      f"top_ask={our_ask:.3f} < 0.40 at conf={confidence:.1f}% "
                      f"(< {self.cfg.contra_book_max_conf:.1f}% ceiling) — book disagrees")
                self._record_skip(reason="contra_book",
                                  details=f"{direction} top_ask={our_ask:.3f} < 0.40 at conf={confidence:.1f}%",
                                  **skip_kw)
                return
            # Mid-price gate: at high confidence the bot loses 9/13 historically
            # when buying mid-priced (entry < 0.55) sides — net -$16.97 across
            # blocked subset. Wider/configurable replacement for the old
            # hardcoded `ask<0.50 & conf>12` overconfident_contra rule.
            # Set mid_price_cap=0 to disable.
            if (self.cfg.mid_price_cap > 0
                    and our_ask is not None
                    and our_ask < self.cfg.mid_price_cap
                    and confidence >= self.cfg.mid_price_conf_floor):
                print(f"  [BOT {_fmt_ts()}] SKIP (mid-price-high-conf): {direction} "
                      f"ask={our_ask:.3f} < {self.cfg.mid_price_cap:.2f} cap at "
                      f"conf={confidence:.1f}% (>= {self.cfg.mid_price_conf_floor:.1f}%)")
                self._record_skip(reason="mid_price_high_conf",
                                  details=f"ask={our_ask:.3f} < {self.cfg.mid_price_cap:.2f} "
                                          f"with conf={confidence:.1f}% >= {self.cfg.mid_price_conf_floor:.1f}%",
                                  **skip_kw)
                return
            ptb_up_prob = (signals.get("ptb") or {}).get("up_prob") if signals else None
            strong_ptb_up = ptb_up_prob is not None and ptb_up_prob >= self.cfg.strong_ptb_up_prob
            strong_ptb_dn = ptb_up_prob is not None and (1 - ptb_up_prob) >= self.cfg.strong_ptb_up_prob
            strong_drift_up = drift_pct is not None and drift_pct >= self.cfg.strong_drift_pct
            strong_drift_dn = drift_pct is not None and drift_pct <= -self.cfg.strong_drift_pct

            # === UP-only stricter gates ===
            if direction == "UP":
                if our_ask is not None and our_ask < self.cfg.up_min_ask:
                    print(f"  [BOT {_fmt_ts()}] SKIP (up-ask-too-low): "
                          f"ask_up={our_ask:.3f} < {self.cfg.up_min_ask:.2f}")
                    self._record_skip(reason="up_ask_too_low",
                                      details=f"ask_up={our_ask:.3f} < {self.cfg.up_min_ask:.2f}",
                                      **skip_kw)
                    return
                if (self.cfg.require_drift_positive_up
                        and drift_pct is not None and drift_pct < 0):
                    print(f"  [BOT {_fmt_ts()}] SKIP (up-drift-negative): "
                          f"drift={drift_pct:+.4f}% < 0 — UP requires non-negative drift")
                    self._record_skip(reason="up_drift_negative",
                                      details=f"drift={drift_pct:+.4f}% < 0",
                                      **skip_kw)
                    return
                if (self.cfg.require_ptb_support_up
                        and ptb_up_prob is not None and ptb_up_prob < 0.50):
                    print(f"  [BOT {_fmt_ts()}] SKIP (up-no-ptb-support): "
                          f"ptb_up_prob={ptb_up_prob:.2f} < 0.50 — PTB doesn't support UP")
                    self._record_skip(reason="up_no_ptb_support",
                                      details=f"ptb_up_prob={ptb_up_prob:.2f} < 0.50",
                                      **skip_kw)
                    return
                # ask > up_max_ask cap (default 0.75) — override if PTB+drift strongly agree.
                if (our_ask is not None
                        and self.cfg.up_max_ask > 0
                        and our_ask > self.cfg.up_max_ask
                        and not (strong_ptb_up and strong_drift_up)):
                    print(f"  [BOT {_fmt_ts()}] SKIP (up-too-expensive): "
                          f"ask_up={our_ask:.3f} > {self.cfg.up_max_ask:.2f} cap (no PTB+drift override)")
                    self._record_skip(reason="up_too_expensive",
                                      details=f"ask_up={our_ask:.3f} > {self.cfg.up_max_ask:.2f}",
                                      **skip_kw)
                    return

            # === Crowd-indecision filter (UP and DOWN both) ===
            # Original observation: 22 lifetime firings, bot was wrong 73% of the
            # time; flipping the direction nets +$52 vs $0 from skipping. Updated
            # 2026-05-15 finding: the edge concentrates entirely in the conf 7-12
            # bucket (8/9 W = 88.9%, +$0.62/$1). Outside that band the flip is
            # neutral or harmful (conf <7: flip -$0.18, signal +$0.30; conf >=12:
            # n=2, both options near $0). Conf-band guard restricts the flip to
            # [crowd_flip_min_conf, crowd_flip_max_conf); outside the band the
            # original signal proceeds untouched.
            if (self.cfg.up_filter_crowd_indecision
                    and signals
                    and our_ask is not None):
                crowd_up = (signals.get("polymarket") or {}).get("up_prob")
                if crowd_up is not None and abs(crowd_up - 0.5) < 0.05 and our_ask <= 0.50:
                    in_band = (self.cfg.crowd_flip_min_conf
                               <= confidence
                               < self.cfg.crowd_flip_max_conf)
                    if not in_band:
                        print(f"  [BOT {_fmt_ts()}] crowd-indecision triggered but conf="
                              f"{confidence:.1f}% outside flip band "
                              f"[{self.cfg.crowd_flip_min_conf:.1f}, "
                              f"{self.cfg.crowd_flip_max_conf:.1f}) — proceeding with signal")
                    else:
                        orig_dir, orig_ask = direction, our_ask
                        direction = "DOWN" if orig_dir == "UP" else "UP"
                        our_ask = top_ask_up if direction == "UP" else top_ask_down
                        if our_ask is None:
                            print(f"  [BOT {_fmt_ts()}] SKIP (crowd-indecision-contra): "
                                  f"flip {orig_dir}→{direction} aborted — no opposite ask available")
                            self._record_skip(reason="crowd_indecision_contra",
                                              details=f"crowd_up={crowd_up:.3f} ask={orig_ask:.3f} (flip aborted)",
                                              **skip_kw)
                            return
                        print(f"  [BOT {_fmt_ts()}] FLIP (crowd-indecision-contra): "
                              f"{orig_dir}@{orig_ask:.3f} → {direction}@{our_ask:.3f} "
                              f"(crowd_up={crowd_up:.3f}, conf={confidence:.1f}% in flip band; "
                              f"bot 89% wrong here historically)")
                        confidence = abs((1 - final_up) - 0.5) * 200 if final_up is not None else confidence
                        skip_kw["direction"] = direction
                        skip_kw["confidence"] = confidence

            # === Expensive-fill (both directions) — relaxed from 0.75 to configurable threshold ===
            # Override: skip only if NOT (strong PTB + strong drift agreeing with direction).
            if our_ask is not None and our_ask > self.cfg.expensive_fill_threshold:
                if direction == "UP":
                    override = strong_ptb_up and strong_drift_up
                else:
                    override = strong_ptb_dn and strong_drift_dn
                if not override:
                    print(f"  [BOT {_fmt_ts()}] SKIP (expensive-fill): {direction} "
                          f"ask={our_ask:.3f} > {self.cfg.expensive_fill_threshold:.2f} (no PTB+drift override)")
                    self._record_skip(reason="expensive_fill",
                                      details=f"ask={our_ask:.3f} > {self.cfg.expensive_fill_threshold:.2f}",
                                      **skip_kw)
                    return

            if self.cfg.use_llm:
                decision = self._llm_decide(
                    window=window, direction=direction, confidence=confidence,
                    final_up=final_up, ptb=ptb, live_price=live_price,
                    drift_pct=drift_pct, signals=signals,
                    top_ask_up=top_ask_up, top_ask_down=top_ask_down,
                )
                if decision is None:
                    self._record_skip(reason="llm_skip",
                                      details="LLM reviewer voted SKIP or low llm_conf/edge",
                                      **skip_kw)
                    return
                direction = decision

            token_id = up_token if direction == "UP" else down_token

            stake = self.cfg.up_stake_usdc if direction == "UP" else self.cfg.stake_usdc

            # T+8s ask re-check: sleep, re-fetch, abort if ask moved up.
            # Disabled by default after 2026-05-15 audit (26 blocked / 22 winners).
            # Re-enable via BOT_ASK_RECHECK_ENABLED=true. Tolerance, delay,
            # timeout, and max_ref_ask are all env-configurable.
            if self.cfg.ask_recheck_enabled and our_ask is not None:
                print(f"  [BOT {_fmt_ts()}] ask-recheck: waiting {self.cfg.ask_recheck_delay_s:.0f}s "
                      f"(ref ${our_ask:.3f})")
                time.sleep(self.cfg.ask_recheck_delay_s)
                fresh_ask = None
                try:
                    with cf.ThreadPoolExecutor(max_workers=1) as _ex:
                        fresh_ask = _ex.submit(self.client.get_top_ask, token_id).result(
                            timeout=self.cfg.ask_recheck_timeout_s
                        )
                except cf.TimeoutError:
                    print(f"  [BOT {_fmt_ts()}] ask-recheck: timeout — proceeding with original")
                except Exception as e:
                    print(f"  [BOT {_fmt_ts()}] ask-recheck: error "
                          f"{type(e).__name__}: {e} — proceeding")
                if fresh_ask is not None:
                    delta = fresh_ask - our_ask
                    cheap_enough = our_ask <= self.cfg.ask_recheck_max_ref_ask
                    if delta > self.cfg.ask_recheck_tolerance and cheap_enough:
                        print(f"  [BOT {_fmt_ts()}] SKIP (ask-moved): {direction} "
                              f"ask ${our_ask:.3f} → ${fresh_ask:.3f} "
                              f"(+${delta:.3f}) > tol ${self.cfg.ask_recheck_tolerance:.2f} "
                              f"(ref<=${self.cfg.ask_recheck_max_ref_ask:.2f})")
                        self._record_skip(
                            reason="ask_moved_against",
                            details=f"ref={our_ask:.3f} fresh={fresh_ask:.3f} "
                                    f"delta=+{delta:.3f} tol={self.cfg.ask_recheck_tolerance:.2f}",
                            **skip_kw,
                        )
                        return
                    sign = "+" if delta >= 0 else ""
                    if delta > self.cfg.ask_recheck_tolerance and not cheap_enough:
                        print(f"  [BOT {_fmt_ts()}] ask-recheck: ${our_ask:.3f} → ${fresh_ask:.3f} "
                              f"({sign}${delta:.3f}) — over tol but ref>${self.cfg.ask_recheck_max_ref_ask:.2f}, proceeding")
                    else:
                        print(f"  [BOT {_fmt_ts()}] ask-recheck: ${our_ask:.3f} → ${fresh_ask:.3f} "
                              f"({sign}${delta:.3f}) — within tol, proceeding")
                    our_ask = fresh_ask  # use fresh value in downstream logging

            print(f"  [BOT {_fmt_ts()}] BUY {direction:>4} {slug} "
                  f"conf={confidence:.1f}% stake=${stake:.2f}"
                  + ("  (DRY RUN)" if self.cfg.dry_run else ""))

            # NOTE: a FOK-limit-at-band-ceiling was tested for fav90 (2026-06-04) to
            # avoid FAK kills, but it filled WORSE (0/3 vs market's 2/4) — the fast
            # late favorite gaps above the 0.95 cap before the order lands, so the
            # all-or-nothing FOK kills entirely. Reverted to the plain market FAK,
            # which crosses at the live price and fills the calmer windows.
            order_resp = self.client.buy_market(token_id, stake)
            if isinstance(order_resp, dict) and "skipped" in order_resp:
                print(f"  [BOT {_fmt_ts()}] skipped: {order_resp['skipped']}")
                self._record_skip(reason="book_vanished",
                                  details=str(order_resp.get('skipped')),
                                  **skip_kw)
                return
            print(f"  [BOT {_fmt_ts()}] raw resp: {order_resp}")
            fill = _parse_fill(order_resp, stake)
            if fill:
                shares, fill_px, actual_filled_usdc = fill
                print(f"  [BOT {_fmt_ts()}] filled: {shares:.2f} shares @ ~${fill_px:.3f} "
                      f"(${actual_filled_usdc:.3f} of ${stake:.2f} stake used)")
            else:
                shares, fill_px, actual_filled_usdc = None, None, None
                print(f"  [BOT {_fmt_ts()}] order resp: {order_resp}")

            cashed = None
            if shares and fill_px and self.cfg.enable_cash_out:
                cashed = self._monitor_cash_out(
                    token_id=token_id, shares=shares, fill_px=fill_px,
                    stake=stake, end_ts=end_ts,
                )

            extras = dict(
                ptb=ptb, live_price=live_price, drift_pct=drift_pct,
                final_up=final_up, signals=signals,
                top_ask_up=top_ask_up, top_ask_down=top_ask_down,
                shares=shares, fill_px=fill_px, stake=stake,
                actual_filled_usdc=actual_filled_usdc,
            )
            if cashed is not None:
                self._record(slug, direction, confidence, cashed["won"], cashed["pnl"], **extras)
                badge = "CASHOUT"
                print(f"  [BOT {_fmt_ts()}] {slug}: {badge} pnl=${cashed['pnl']:+.2f} "
                      f"(sold {cashed['shares_sold']:.2f} @ ~${cashed['sell_px']:.3f})")
                self._print_summary()
                return

            # FAV90 STOP-LOSS: synchronous monitor — poll the held position's bid
            # (also logs trajectory). If bid <= fav90_stop_bid, SELL with
            # retry-until-flat and book the realized cut PnL (skips resolution).
            # Returns None if it held to close, then we resolve normally below.
            if self.cfg.fav90_mode and shares:
                cut = self._fav90_monitor(
                    token_id=token_id, shares=shares, fill_px=fill_px,
                    direction=direction, slug=slug, end_ts=end_ts, stake=stake,
                )
                if cut is not None:
                    self._record(slug, direction, confidence, cut["won"], cut["pnl"], **extras)
                    print(f"  [BOT {_fmt_ts()}] {slug}: STOP-SOLD pnl=${cut['pnl']:+.2f} "
                          f"@ bid {cut['stop_bid']:.3f} (recovered ${cut['sold_usdc']:.2f}"
                          f"{' INCOMPLETE' if cut.get('incomplete') else ''})")
                    self._print_summary()
                    return

            resolution = self.client.wait_for_resolution(slug, end_ts + 600)
            if resolution is None:
                print(f"  [BOT {_fmt_ts()}] {slug}: resolution timeout")
                self._record(slug, direction, confidence, None, 0.0, **extras)
                return

            won = (direction == "UP" and resolution["up_won"]) or \
                  (direction == "DOWN" and not resolution["up_won"])

            cost_basis = actual_filled_usdc if actual_filled_usdc is not None else stake
            if won and shares is not None:
                pnl = shares - cost_basis
            elif won:
                pnl = cost_basis
            else:
                pnl = -cost_basis

            self._record(slug, direction, confidence, won, pnl, **extras)
            badge = "WIN " if won else "LOSS"
            print(f"  [BOT {_fmt_ts()}] {slug}: {badge} pnl=${pnl:+.2f} "
                  f"(up_px={resolution['up_price']:.2f} down_px={resolution['down_price']:.2f})")

            if won and condition_id:
                tx = self.client.redeem_position(condition_id)
                if tx:
                    print(f"  [BOT {_fmt_ts()}] redeemed -> 0x{tx.lstrip('0x')}")
                else:
                    print(f"  [BOT {_fmt_ts()}] redeem failed or no shares")

            self._print_summary()

        except PolymarketError as e:
            print(f"  [BOT {_fmt_ts()}] polymarket error: {e}")
            self._record(slug, direction, confidence, None, 0.0,
                         ptb=ptb, live_price=live_price, drift_pct=drift_pct,
                         final_up=final_up, signals=signals)
        except Exception as e:
            print(f"  [BOT {_fmt_ts()}] trade error: {type(e).__name__}: {e}")
            self._record(slug, direction, confidence, None, 0.0,
                         ptb=ptb, live_price=live_price, drift_pct=drift_pct,
                         final_up=final_up, signals=signals)
        finally:
            with self._lock:
                self._active = max(0, self._active - 1)

    def _execute_dbmodel_trade(self, window, direction, p_up, drift_pct=None,
                               live_price=None, ptb=None, limit_price=None):
        """Dollar-bar PTB model execution: BUY `direction` at market, $1, hold to
        resolution. Deliberately bypasses the directional gate stack used by
        _execute_trade — this mode bets the model's side every window with no
        price/confidence filter. Isolated so it can never disturb the other
        profiles. Logs one record per fire to DBMODEL_LOG_PATH."""
        slug = window["slug"]
        end_ts = window["end_ts"]
        # confidence (0-100) is for the shared trade log/summary only, not a gate.
        confidence = abs(p_up - 0.5) * 200.0
        try:
            market = self.client.resolve_market(slug)
            up_token = market["up_token"]
            down_token = market["down_token"]
            condition_id = market.get("condition_id")
            token_id = up_token if direction == "UP" else down_token
            stake = self.cfg.stake_usdc

            # Snapshot the side's ask for the record (NOT a gate — we take at market).
            try:
                top_ask_up = self.client.get_top_ask(up_token)
                top_ask_down = self.client.get_top_ask(down_token)
            except Exception:
                top_ask_up = top_ask_down = None
            our_ask = top_ask_up if direction == "UP" else top_ask_down

            print(f"  [DBMODEL {_fmt_ts()}] BUY {direction:>4} {slug} "
                  f"P(up)={p_up:.3f} ask={our_ask if our_ask is None else f'{our_ask:.3f}'} "
                  f"stake=${stake:.2f}" + ("  (DRY RUN)" if self.cfg.dry_run else ""))

            shares = fill_px = actual_filled_usdc = None
            if not self.cfg.dry_run:
                if limit_price is not None:
                    size = round(stake / limit_price, 2)
                    order_resp = self.client.buy_limit_fok(token_id, limit_price, size)
                else:
                    order_resp = self.client.buy_market(token_id, stake)
                if isinstance(order_resp, dict) and "skipped" in order_resp:
                    print(f"  [DBMODEL {_fmt_ts()}] skipped: {order_resp['skipped']}")
                    self._record_skip(reason="book_vanished",
                                      details=str(order_resp.get("skipped")),
                                      slug=slug, end_ts=end_ts, direction=direction,
                                      confidence=confidence, ptb=ptb, live_price=live_price,
                                      drift_pct=drift_pct, final_up=(p_up >= 0.5),
                                      top_ask_up=top_ask_up, top_ask_down=top_ask_down,
                                      signals=None)
                    self._dbmodel_log(slug, p_up, direction, our_ask,
                                      None, None, stake, None, None)
                    return
                fill = _parse_fill(order_resp, stake)
                if fill:
                    shares, fill_px, actual_filled_usdc = fill
                    print(f"  [DBMODEL {_fmt_ts()}] filled: {shares:.2f} shares "
                          f"@ ~${fill_px:.3f} (${actual_filled_usdc:.3f} used)")

            extras = dict(
                ptb=ptb, live_price=live_price, drift_pct=drift_pct,
                final_up=(p_up >= 0.5), signals=None,
                top_ask_up=top_ask_up, top_ask_down=top_ask_down,
                shares=shares, fill_px=fill_px, stake=stake,
                actual_filled_usdc=actual_filled_usdc,
            )

            if self.cfg.dry_run:
                # Paper trade: no order placed, but settle the hypothetical fill so
                # the record carries real win/loss + PnL. Hypothetical fill = the
                # limit price (if set) or the snapshot ask on the chosen side.
                hypo_ask = limit_price if limit_price is not None else our_ask
                hypo_shares = (stake / hypo_ask) if hypo_ask else None
                # Sample the held side's bid/ask through the window BEFORE settling,
                # so the record carries the real intra-window exit-price path
                # (bid = what we could sell for) for stop-loss/take-profit backtests.
                path = self._sample_book_path(token_id, end_ts)
                resolution = self.client.wait_for_resolution(slug, end_ts + 600)
                bw = None
                ws = window.get("ws")
                if ws is not None:
                    # slug = "{asset}-updown-{tf}-{ws}" -> Binance pair + kline interval
                    parts = slug.split("-")
                    b_sym = f"{parts[0].upper()}USDT" if parts else "BTCUSDT"
                    b_int = parts[2] if len(parts) > 2 else "5m"
                    bw = self.client.binance_window_prices(ws, symbol=b_sym, interval=b_int)
                if resolution is None:
                    print(f"  [DBMODEL {_fmt_ts()}] {slug}: paper resolution timeout")
                    self._record(slug, direction, confidence, None, 0.0, **extras)
                    self._dbmodel_paper_log(window, p_up, direction, confidence,
                                            drift_pct, hypo_ask, top_ask_up, top_ask_down,
                                            hypo_shares, stake, None, None, bw, path=path)
                    return
                won = (direction == "UP") == bool(resolution["up_won"])
                if won and hypo_shares is not None:
                    pnl = hypo_shares - stake
                elif won:
                    pnl = None          # couldn't price the fill (no ask)
                else:
                    pnl = -stake
                self._record(slug, direction, confidence, won, pnl or 0.0, **extras)
                self._dbmodel_paper_log(window, p_up, direction, confidence,
                                        drift_pct, hypo_ask, top_ask_up, top_ask_down,
                                        hypo_shares, stake, won, pnl, bw, resolution,
                                        path=path)
                badge = "WIN " if won else "LOSS"
                mv = f"{bw['move_pct']:+.3f}%" if bw else "n/a"
                print(f"  [DBMODEL {_fmt_ts()}] {slug}: PAPER {badge} "
                      f"pnl=${(pnl if pnl is not None else 0.0):+.3f} "
                      f"(up_won={resolution['up_won']} binance_move={mv})")
                self._print_summary()
                return

            resolution = self.client.wait_for_resolution(slug, end_ts + 600)
            if resolution is None:
                print(f"  [DBMODEL {_fmt_ts()}] {slug}: resolution timeout")
                self._record(slug, direction, confidence, None, 0.0, **extras)
                self._dbmodel_log(slug, p_up, direction, our_ask,
                                  shares, fill_px, stake, None, None)
                return

            won = (direction == "UP" and resolution["up_won"]) or \
                  (direction == "DOWN" and not resolution["up_won"])
            cost_basis = actual_filled_usdc if actual_filled_usdc is not None else stake
            if won and shares is not None:
                pnl = shares - cost_basis
            elif won:
                pnl = cost_basis
            else:
                pnl = -cost_basis

            self._record(slug, direction, confidence, won, pnl, **extras)
            self._dbmodel_log(slug, p_up, direction, our_ask,
                              shares, fill_px, stake, won, pnl, actual_filled_usdc)
            badge = "WIN " if won else "LOSS"
            print(f"  [DBMODEL {_fmt_ts()}] {slug}: {badge} pnl=${pnl:+.2f} "
                  f"(up_px={resolution['up_price']:.2f} down_px={resolution['down_price']:.2f})")

            if won and condition_id:
                if self.cfg.dbmodel_delegate_redeem:
                    # Shared-wallet cohort: the centralized redeemer daemon sweeps
                    # this winner. Bots never send on-chain txs, so N processes on
                    # one wallet can't collide on the in-flight-tx limit.
                    print(f"  [DBMODEL {_fmt_ts()}] win held for daemon redeem "
                          f"(delegate_redeem)")
                else:
                    tx = self.client.redeem_position(condition_id)
                    if tx:
                        print(f"  [DBMODEL {_fmt_ts()}] redeemed -> 0x{tx.lstrip('0x')}")
                    else:
                        print(f"  [DBMODEL {_fmt_ts()}] redeem failed or no shares")
            self._print_summary()

        except PolymarketError as e:
            print(f"  [DBMODEL {_fmt_ts()}] polymarket error: {e}")
            self._record(slug, direction, confidence, None, 0.0,
                         drift_pct=drift_pct, live_price=live_price, ptb=ptb,
                         final_up=(p_up >= 0.5))
        except Exception as e:
            print(f"  [DBMODEL {_fmt_ts()}] trade error: {type(e).__name__}: {e}")
            self._record(slug, direction, confidence, None, 0.0,
                         drift_pct=drift_pct, live_price=live_price, ptb=ptb,
                         final_up=(p_up >= 0.5))
        finally:
            with self._lock:
                self._active = max(0, self._active - 1)

    def _dbmodel_log(self, slug, p_up, direction, ask, shares, fill_px, stake, won, pnl,
                     actual_filled_usdc=None):
        try:
            with open(DBMODEL_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    "slug": slug,
                    "p_up": round(p_up, 4),
                    "direction": direction,
                    "ask": round(ask, 4) if ask is not None else None,
                    "shares": shares,
                    "fill_px": round(fill_px, 4) if fill_px is not None else None,
                    "stake": stake,
                    # real USDC spent per the CLOB response (makingAmount). Persisted
                    # so booked cost can be reconciled against the on-chain wallet
                    # debit — that gap is what the real-balance stop-loss exists for.
                    "actual_filled_usdc": actual_filled_usdc,
                    "won": won,
                    "pnl": round(pnl, 4) if pnl is not None else None,
                }) + "\n")
        except Exception as _e:
            print(f"  [DBMODEL {_fmt_ts()}] log error: {_e}")

    def _sample_book_path(self, token_id, end_ts, cadence_s=15):
        """Dry-run only: poll the HELD side's bid/ask every ~cadence_s seconds
        until window close. `bid` is what we could actually sell our shares for,
        so this captures the real intra-window price path — exactly what we need
        to backtest stop-loss / take-profit exits at prices we could really get,
        instead of guessing an exit value. Returns a list of {s2c, bid, ask}
        samples (best-effort; book hiccups are skipped). Runs in the per-window
        thread, which already blocks until resolution, so it adds no concurrency."""
        path = []
        while True:
            s2c = end_ts - time.time()
            if s2c <= 2:
                break
            try:
                bid = self.client.get_top_bid(token_id)
                ask = self.client.get_top_ask(token_id)
                path.append({
                    "s2c": round(s2c, 1),
                    "bid": round(bid, 4) if bid is not None else None,
                    "ask": round(ask, 4) if ask is not None else None,
                })
            except Exception:
                pass
            time.sleep(min(cadence_s, max(s2c - 2, 0.1)))
        return path

    def _dbmodel_paper_log(self, window, p_up, direction, confidence, drift_pct,
                           our_ask, top_ask_up, top_ask_down, hypo_shares, stake,
                           won, pnl, bw, resolution=None, path=None):
        """Rich per-window paper-trade record (dry-run only). One JSONL line that
        carries the full feature vector + raw/calibrated proba + both asks + the
        hypothetical fill + the outcome settled by gamma and cross-checked vs
        Binance. See DBMODEL_PAPER_LOG_PATH."""
        try:
            ws = window.get("ws")
            features = window.get("features") or {}
            raw = window.get("raw_proba")
            up_won = bool(resolution["up_won"]) if resolution else None
            up_price = resolution.get("up_price") if resolution else None
            down_price = resolution.get("down_price") if resolution else None
            b_open = bw.get("open") if bw else None
            b_close = bw.get("close") if bw else None
            b_move = bw.get("move_pct") if bw else None
            agree = None
            if up_won is not None and b_move is not None:
                agree = (up_won == (b_move > 0))
            rec = {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "slug": window.get("slug"),
                "ws": ws,
                "session_hour": (datetime.datetime.utcfromtimestamp(ws).hour
                                 if ws is not None else None),
                "p_up": round(p_up, 4),
                "raw_proba": round(raw, 4) if raw is not None else None,
                "direction": direction,
                "confidence": round(confidence, 2),
                "drift_pct": round(drift_pct, 4) if drift_pct is not None else None,
                "strike": window.get("strike"),
                "monitor_start_s": window.get("monitor_start_s"),
                "window_s": window.get("window_s"),
                "features": {k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in features.items()},
                "top_ask_up": round(top_ask_up, 4) if top_ask_up is not None else None,
                "top_ask_down": round(top_ask_down, 4) if top_ask_down is not None else None,
                "our_ask": round(our_ask, 4) if our_ask is not None else None,
                "hypo_shares": round(hypo_shares, 4) if hypo_shares is not None else None,
                "gamma_up_won": up_won,
                "up_price": up_price,
                "down_price": down_price,
                "binance_open": b_open,
                "binance_close": b_close,
                "binance_move_pct": round(b_move, 4) if b_move is not None else None,
                "agree": agree,
                "path": path,
                "won": won,
                "pnl": round(pnl, 4) if pnl is not None else None,
            }
            with open(DBMODEL_PAPER_LOG_PATH, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as _e:
            print(f"  [DBMODEL {_fmt_ts()}] paper log error: {_e}")

    def _fav90_traj_write(self, slug, direction, fill_px, s2c, bid):
        """Append one post-entry trajectory record (instrumentation, no orders)."""
        try:
            with open(FAV90_TRAJ_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    "slug": slug,
                    "direction": direction,
                    "fill_px": round(fill_px, 4) if fill_px is not None else None,
                    "secs_to_close": round(s2c, 1),
                    "bid": bid,
                }) + "\n")
        except Exception as _e:
            print(f"  [BOT {_fmt_ts()}] fav90 traj log error: {_e}")

    def _fav90_monitor(self, *, token_id, shares, fill_px, direction, slug, end_ts, stake):
        """Synchronous stop-loss monitor for a held fav90 position. Polls the top
        bid every 6s (logging each poll). If bid <= fav90_stop_bid, SELLS the
        position with retry-until-flat and returns the realized cut result
        {won,pnl,stop_bid,sold_usdc,incomplete}. Returns None if it held to ~close
        (caller then resolves normally). Runs in the existing per-window thread."""
        STOP = self.cfg.fav90_stop_bid
        POLL = 6.0
        while True:
            s2c = end_ts - time.time()
            if s2c <= 6:
                return None   # too close to sell + settle — hold to resolution
            bid = None
            try:
                bid = self.client.get_top_bid(token_id)
            except Exception:
                pass
            self._fav90_traj_write(slug, direction, fill_px, s2c, bid)
            if bid is not None and bid <= STOP:
                print(f"  [BOT {_fmt_ts()}] FAV90 STOP hit: bid=${bid:.3f} <= ${STOP:.2f} "
                      f"(fill ${fill_px:.3f}) — selling {shares:.2f} sh, retry till flat")
                sold_usdc, incomplete = self._fav90_sell_all(token_id, shares, end_ts)
                pnl = sold_usdc - stake
                return {"won": pnl > 0, "pnl": pnl, "stop_bid": bid,
                        "sold_usdc": sold_usdc, "incomplete": incomplete}
            time.sleep(POLL)

    def _fav90_sell_all(self, token_id, shares, end_ts):
        """Sell `shares` via repeated FAK market sells until the position is flat
        (or the window is ~closed). Walks down with the falling bid; refreshes the
        true remaining balance from chain each loop. Returns (total_usdc, incomplete)."""
        total = 0.0
        remaining = shares
        attempts = 0
        while remaining > 0.05 and (end_ts - time.time()) > 3 and attempts < 25:
            attempts += 1
            resp = None
            try:
                resp = self.client.sell_market(token_id, remaining)
            except Exception as e:
                print(f"  [BOT {_fmt_ts()}] FAV90 stop-sell {attempts}: error {e} — retry")
            if isinstance(resp, dict) and "skipped" in resp:
                print(f"  [BOT {_fmt_ts()}] FAV90 stop-sell {attempts}: {resp['skipped']} — retry")
            elif resp is not None:
                sf = _parse_sell_fill(resp, remaining)
                if sf:
                    usdc_recv, shares_sold, sell_px = sf
                    total += usdc_recv
                    print(f"  [BOT {_fmt_ts()}] FAV90 stop-sell {attempts}: "
                          f"sold {shares_sold:.2f}@${sell_px:.3f} (got ${usdc_recv:.2f})")
            time.sleep(1.0)
            # authoritative remaining from chain
            bal = self.client.get_conditional_balance(token_id)
            if bal is not None:
                remaining = bal
        incomplete = remaining > 0.05
        if incomplete:
            print(f"  [BOT {_fmt_ts()}] FAV90 stop-sell INCOMPLETE: {remaining:.2f} sh unsold "
                  f"after {attempts} tries — leftover resolves at close")
        return total, incomplete

    def _monitor_cash_out(self, *, token_id, shares, fill_px, stake, end_ts):
        """Wait cash_out_delay_sec, then peek at top bid. If bid has dropped
        below threshold, dump the position and return the realized PnL.

        Threshold = max(floor, fill_px - drop). If we bought near the floor
        already (fill_px <= floor+drop), effectively no cash-out trigger.
        Returns None if we held the position through to resolution.
        """
        target = time.time() + self.cfg.cash_out_delay_sec
        if end_ts - target < 30:
            return None

        time.sleep(max(0.0, target - time.time()))

        bid = self.client.get_top_bid(token_id)
        threshold = max(self.cfg.cash_out_floor, fill_px - self.cfg.cash_out_drop)

        if bid is None:
            print(f"  [BOT {_fmt_ts()}] cash-out check: bid unavailable — holding")
            return None

        if bid > threshold:
            print(f"  [BOT {_fmt_ts()}] cash-out check: bid=${bid:.3f} > "
                  f"threshold=${threshold:.3f} — holding")
            return None

        print(f"  [BOT {_fmt_ts()}] CASH-OUT trigger: bid=${bid:.3f} ≤ "
              f"threshold=${threshold:.3f} (fill=${fill_px:.3f})")
        sell_resp = self.client.sell_market(token_id, shares)
        if isinstance(sell_resp, dict) and "skipped" in sell_resp:
            print(f"  [BOT {_fmt_ts()}] cash-out SELL skipped: {sell_resp['skipped']} — holding")
            return None
        print(f"  [BOT {_fmt_ts()}] cash-out raw: {sell_resp}")
        sell_fill = _parse_sell_fill(sell_resp, shares)
        if not sell_fill:
            print(f"  [BOT {_fmt_ts()}] cash-out fill parse failed — holding")
            return None

        usdc_received, shares_sold, sell_px = sell_fill
        pnl = usdc_received - stake
        won = pnl > 0
        return {
            "won": won, "pnl": pnl,
            "shares_sold": shares_sold, "sell_px": sell_px,
            "usdc_received": usdc_received,
        }

    def _llm_decide(self, *, window, direction, confidence, final_up, ptb, live_price,
                    drift_pct, signals, top_ask_up, top_ask_down) -> str | None:
        recent = [
            {"direction": t["direction"], "conf": round(t["confidence"], 1),
             "won": t["won"], "pnl": round(t["pnl"], 2)}
            for t in self._trades[-5:]
        ]
        features = {
            "window": window.get("window") or window.get("slug"),
            "predicted_direction": direction,
            "predicted_confidence_pct": round(confidence, 2),
            "model_prob_up": round(final_up, 4) if final_up is not None else None,
            "drift_pct": round(drift_pct, 4) if drift_pct is not None else None,
            "ptb": ptb,
            "live_price": live_price,
            "signals": signals or {},
            "top_ask_up": top_ask_up,
            "top_ask_down": top_ask_down,
            "recent_trades": recent,
        }
        try:
            review = llm_reviewer.review_trade(
                features,
                model=self.cfg.llm_model,
                api_key=self.cfg.anthropic_api_key,
            )
        except Exception as e:
            print(f"  [LLM {_fmt_ts()}] error: {type(e).__name__}: {e} — falling back to SKIP")
            return None

        action = review["action"]
        llm_conf = review["confidence"]
        reason = review["reason"]
        print(f"  [LLM {_fmt_ts()}] action={action} conf={llm_conf} reason={reason!r}")

        if action == "SKIP":
            return None
        if llm_conf < self.cfg.llm_min_conf:
            print(f"  [LLM {_fmt_ts()}] SKIP (low llm_conf): {llm_conf} < {self.cfg.llm_min_conf:.0f}")
            return None

        final = "UP" if action == "BUY_UP" else "DOWN"

        if final_up is not None:
            if final == "UP" and top_ask_up is not None:
                edge = final_up - top_ask_up
            elif final == "DOWN" and top_ask_down is not None:
                edge = (1.0 - final_up) - top_ask_down
            else:
                edge = None
            if edge is not None and edge < self.cfg.llm_min_edge:
                print(f"  [LLM {_fmt_ts()}] OVERRIDE SKIP: {final} edge={edge:+.3f} "
                      f"< min={self.cfg.llm_min_edge:+.3f}")
                return None

        if final != direction:
            print(f"  [LLM {_fmt_ts()}] FLIP: predictor={direction} -> llm={final}")
        return final

    def _record_skip(self, *, slug, end_ts, direction, confidence, reason, details,
                     ptb=None, live_price=None, drift_pct=None, final_up=None,
                     top_ask_up=None, top_ask_down=None, signals=None):
        """Log a skipped prediction with its eventual real-world outcome.

        Spawns a daemon thread that waits for the market to settle, computes
        what the bot *would* have won/lost, and appends to skip_history.jsonl.
        Skips that share a window with an active trade are logged the same way
        — analysis later will tell us whether each skip rule earned its keep.
        """
        threading.Thread(
            target=self._resolve_skip_and_log,
            kwargs=dict(
                slug=slug, end_ts=end_ts, direction=direction, confidence=confidence,
                reason=reason, details=details,
                ptb=ptb, live_price=live_price, drift_pct=drift_pct, final_up=final_up,
                top_ask_up=top_ask_up, top_ask_down=top_ask_down, signals=signals,
            ),
            daemon=True,
        ).start()

    def _resolve_skip_and_log(self, *, slug, end_ts, direction, confidence, reason,
                              details, ptb, live_price, drift_pct, final_up,
                              top_ask_up, top_ask_down, signals=None):
        try:
            # Fetch book snapshot if caller didn't have one (early-skip cases).
            if top_ask_up is None or top_ask_down is None:
                try:
                    market = self.client.resolve_market(slug)
                    if top_ask_up is None:
                        top_ask_up = self.client.get_top_ask(market["up_token"])
                    if top_ask_down is None:
                        top_ask_down = self.client.get_top_ask(market["down_token"])
                except Exception:
                    pass

            deadline = (end_ts + 600) if end_ts else (int(time.time()) + 1200)
            resolution = self.client.wait_for_resolution(slug, deadline)
            if resolution is None:
                return  # don't write incomplete entries — no outcome means no signal

            up_won = resolution["up_won"]
            would_have_won = (
                (direction == "UP" and up_won) or
                (direction == "DOWN" and not up_won)
            )
            our_ask = top_ask_up if direction == "UP" else top_ask_down
            stake = self.cfg.up_stake_usdc if direction == "UP" else self.cfg.stake_usdc
            if our_ask and 0 < our_ask < 1:
                shares = stake / our_ask
                would_have_pnl = (shares - stake) if would_have_won else -stake
            else:
                would_have_pnl = None

            # Shadow logging: for lstm_inv_contra skips, also record what
            # FLIPPING direction would have netted (bet opposite side at its
            # top_ask). Binary market => flip wins iff bot's dir lost.
            # Analysis: run shadow_flip_analysis.py periodically.
            flip_direction = None
            flip_entry_price = None
            flip_would_have_won = None
            flip_would_have_pnl = None
            if reason == "lstm_inv_contra":
                flip_direction = "DOWN" if direction == "UP" else "UP"
                flip_entry_price = top_ask_down if direction == "UP" else top_ask_up
                flip_would_have_won = not would_have_won
                if flip_entry_price and 0 < flip_entry_price < 1:
                    flip_shares = stake / flip_entry_price
                    flip_would_have_pnl = (flip_shares - stake) if flip_would_have_won else -stake

            sig = _flatten_signals(signals) or {}
            market_mid = None
            if top_ask_up is not None and top_ask_down is not None:
                market_mid = round((top_ask_up + (1.0 - top_ask_down)) / 2.0, 4)
            entry = {
                "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                "slug": slug,
                "direction": direction,
                "confidence": confidence,
                "skip_reason": reason,
                "skip_details": details,
                # counterfactual outcome (would_have_*)
                "would_have_won": would_have_won,
                "would_have_pnl": would_have_pnl,
                "up_price": resolution["up_price"],
                "down_price": resolution["down_price"],
                # sizing (would-be)
                "stake_usdc": stake,
                "actual_filled_usdc": None,
                "shares": None,
                "entry_price": (top_ask_up if direction == "UP" else top_ask_down),
                # book snapshot
                "top_ask_up": top_ask_up,
                "top_ask_down": top_ask_down,
                "market_mid": market_mid,
                # price / drift
                "ptb": ptb,
                "live_price": live_price,
                "ptb_distance_pct": drift_pct,
                "btc_drift_pct": drift_pct,
                # signal components
                "lstm_prob": sig.get("lstm_up_prob"),
                "orderbook_prob": sig.get("orderbook_up_prob"),
                "ptb_prob": sig.get("ptb_up_prob"),
                "crowd_prob": sig.get("polymarket_up_prob"),
                "final_blended_prob": final_up,
                "signals": sig or None,
                # Shadow flip fields — only populated for lstm_inv_contra skips
                "flip_direction": flip_direction,
                "flip_entry_price": flip_entry_price,
                "flip_would_have_won": flip_would_have_won,
                "flip_would_have_pnl": flip_would_have_pnl,
            }
            with open(SKIP_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"  [BOT {_fmt_ts()}] skip log error: {type(e).__name__}: {e}")

    def _record(self, slug, direction, confidence, won, pnl, *,
                ptb=None, live_price=None, drift_pct=None, final_up=None,
                signals=None, top_ask_up=None, top_ask_down=None,
                shares=None, fill_px=None, stake=None, actual_filled_usdc=None):
        sig = _flatten_signals(signals) or {}
        market_mid = None
        if top_ask_up is not None and top_ask_down is not None:
            market_mid = round((top_ask_up + (1.0 - top_ask_down)) / 2.0, 4)
        entry = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "slug": slug,
            "direction": direction,
            "confidence": confidence,
            "won": won,
            "pnl": pnl,
            # sizing / fill
            "stake_usdc": stake,
            "actual_filled_usdc": actual_filled_usdc,
            "shares": shares,
            "entry_price": fill_px,
            # book snapshot
            "top_ask_up": top_ask_up,
            "top_ask_down": top_ask_down,
            "market_mid": market_mid,
            # price / drift
            "ptb": ptb,
            "live_price": live_price,
            "ptb_distance_pct": drift_pct,
            "btc_drift_pct": drift_pct,
            # signal components (top-level)
            "lstm_prob": sig.get("lstm_up_prob"),
            "orderbook_prob": sig.get("orderbook_up_prob"),
            "ptb_prob": sig.get("ptb_up_prob"),
            "crowd_prob": sig.get("polymarket_up_prob"),
            "final_blended_prob": final_up,
            # raw signals dict (with weights) preserved for analysis
            "signals": sig or None,
        }
        self._trades.append(entry)
        if won is True:
            self._wins += 1
            self._pnl += pnl
            self._session_pnl += pnl
            self._daily_pnl += pnl
        elif won is False:
            self._losses += 1
            self._pnl += pnl
            self._session_pnl += pnl
            self._daily_pnl += pnl
        try:
            with open(TRADE_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            print(f"  [BOT {_fmt_ts()}] trade log write failed: {e}")

    def _print_summary(self):
        total = self._wins + self._losses
        acc = (self._wins / total * 100) if total else 0.0
        print(f"  [BOT] RUNNING: {self._wins}W/{self._losses}L "
              f"({acc:.1f}%) pnl=${self._pnl:+.2f}")
