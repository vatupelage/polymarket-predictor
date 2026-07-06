"""
LLM-as-filter backtest. For each historical Profile A trade T:
- Build a pre-trade snapshot (no outcome fields)
- Build walk-forward priors from trades [0..T-1] only
- Ask Claude Opus 4.7: TAKE or SKIP + confidence + reasoning
- Cache to llm_filter_cache.jsonl (resumable)

Markov-clean: priors are aggregate stats from prior trades, no path-dependence.
Look-ahead-safe: outcome of trade T is NEVER passed in.
"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from anthropic import Anthropic

SCRIPT_DIR = Path(__file__).parent
CACHE = SCRIPT_DIR / "llm_filter_cache.jsonl"
HISTORY = SCRIPT_DIR / "trade_history.jsonl"
MODEL = "claude-opus-4-7"

SYSTEM_PROMPT = """You are a quant filter for a 5-minute BTC up/down binary prediction bot on Polymarket.

PROBLEM
Every 5 minutes the bot examines one binary market: "Will BTC end above $X at HH:MM?"
The bot fuses several signals to pick a direction (UP/DOWN) and a confidence. You decide
TAKE (execute the trade) or SKIP (don't execute).

PAYOFF MATH (memorize this)
- entry_price is the cost per share (range 0.0-1.0). A winning share pays $1.
- WIN profit per dollar staked: (1 - entry_price) / entry_price
- LOSS: -1.0 per dollar staked (you lose 100% of what you paid)
- Breakeven win rate equals entry_price. At entry 0.55 you need 55%+ W%, at entry 0.70 you need 70%+.
- High entry prices already imply the market thinks the direction is likely. The edge has to come
  from a probability mismatch, not from "the signal says UP."

SIGNALS IN THE SNAPSHOT
- direction: the bot's blended pick (UP/DOWN)
- confidence_pct: bot's blended conviction (0-100)
- entry_price: cost of share for the chosen direction
- btc_drift_pct: BTC price drift since the 5-min window opened
- ptb_distance_pct: % distance from current price to the strike (price-to-beat)
- signals.lstm_up_prob: trained LSTM's raw P(UP) on recent BTC data
- signals.ptb_up_prob: model based on PTB distance
- signals.orderbook_up_prob: BTC perp orderbook imbalance signal
- signals.crowd_up_prob: Polymarket consensus
- top_ask_up / top_ask_down: best asks for each side of the binary
- hour_local: 0-23, local time

PRIORS (walk-forward — only includes trades before this one)
- overall_win_rate: cumulative W% so far
- by_direction: W% conditional on the chosen direction
- by_entry_band: W% in 5pp entry-price buckets [0.50,0.55), [0.55,0.60), etc.
- by_hour_bucket_local: W% in 6h time-of-day buckets
- recent_20: W% and mean per-dollar PnL over the most recent 20 trades
You should LEAN on these. If the sub-population this trade belongs to has historical
W% below its breakeven (= entry_price), the rational answer is SKIP — even if confidence is high.

OUTPUT FORMAT
Return ONLY a single JSON object — no markdown fences, no preamble, no trailing text:
{"decision": "TAKE" or "SKIP", "confidence": <float 0.0-1.0>, "reasoning": "<one or two sentences>"}

`confidence` is how sure YOU are of your TAKE/SKIP call. Be honest — if you're 50/50, say 0.5.
"""

def load_trades():
    out = []
    with open(HISTORY) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except Exception:
                continue
            if t.get("stake_usdc") in (None, 0, 0.0):
                continue
            if t.get("pnl") is None:
                continue
            ep = t.get("entry_price")
            if ep is None or not (0.50 <= ep < 0.75):
                continue
            if "lstm_prob" not in t:
                continue
            try:
                t["_dt"] = datetime.fromisoformat(t["ts"])
            except Exception:
                continue
            out.append(t)
    out.sort(key=lambda x: x["_dt"])
    return out

def load_cache():
    seen = {}
    if not CACHE.exists():
        return seen
    with open(CACHE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                seen[rec["ts"]] = rec
            except Exception:
                continue
    return seen

def winrate(group):
    n = len(group)
    if n == 0:
        return None
    w = sum(1 for t in group if t["won"])
    return {"n": n, "w_pct": round(100 * w / n, 1)}

def build_priors(prior_trades):
    if not prior_trades:
        return {"trades_so_far": 0, "note": "no prior trades — rely on feature signals only"}
    overall = winrate(prior_trades)
    by_dir = {
        "UP":   winrate([t for t in prior_trades if t["direction"] == "UP"]),
        "DOWN": winrate([t for t in prior_trades if t["direction"] == "DOWN"]),
    }
    by_entry = {}
    for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75)]:
        by_entry[f"[{lo:.2f},{hi:.2f})"] = winrate([t for t in prior_trades if lo <= t["entry_price"] < hi])
    by_hour = {}
    for lo, hi in [(0, 6), (6, 12), (12, 18), (18, 24)]:
        by_hour[f"[{lo:02d},{hi:02d})"] = winrate(
            [t for t in prior_trades if lo <= datetime.fromisoformat(t["ts"]).hour < hi]
        )
    recent20 = prior_trades[-20:]
    recent = winrate(recent20)
    if recent and recent20:
        pnls = [t["pnl"] / t["stake_usdc"] for t in recent20]
        recent["mean_per_dollar"] = round(sum(pnls) / len(pnls), 3)
    return {
        "trades_so_far": len(prior_trades),
        "overall_win_rate": overall,
        "by_direction": by_dir,
        "by_entry_band": by_entry,
        "by_hour_bucket_local": by_hour,
        "recent_20": recent,
    }

def build_snapshot(t):
    return {
        "direction": t["direction"],
        "confidence_pct": round(t["confidence"], 2),
        "entry_price": round(t["entry_price"], 4),
        "btc_drift_pct": round(t.get("btc_drift_pct") or 0, 5),
        "ptb_distance_pct": round(t.get("ptb_distance_pct") or 0, 5),
        "signals": {
            "lstm_up_prob": round(t.get("lstm_prob") or 0, 4),
            "ptb_up_prob": round(t.get("ptb_prob") or 0, 4),
            "orderbook_up_prob": round(t.get("orderbook_prob") or 0, 4),
            "crowd_up_prob": round(t.get("crowd_prob") or 0, 4),
        },
        "market": {
            "top_ask_up": round(t.get("top_ask_up") or 0, 4),
            "top_ask_down": round(t.get("top_ask_down") or 0, 4),
            "market_mid": round(t.get("market_mid") or 0, 4),
        },
        "hour_local": datetime.fromisoformat(t["ts"]).hour,
    }

def parse_response(text):
    s = text.strip()
    if s.startswith("```"):
        # strip markdown fence if model wrapped it anyway
        s = s.split("```", 2)
        s = s[1] if len(s) > 1 else text
        if s.startswith("json"):
            s = s[4:]
        s = s.strip().rstrip("`").strip()
    return json.loads(s)

def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    trades = load_trades()
    seen = load_cache()
    if limit:
        print(f"LIMIT mode: processing only first {limit} uncached trades")
    print(f"Total trades: {len(trades)}  cached: {len(seen)}  remaining: {len(trades) - len(seen)}")
    if trades:
        print(f"Date range: {trades[0]['_dt'].date()} -> {trades[-1]['_dt'].date()}")
    print(f"Model: {MODEL}")
    print()

    client = Anthropic()
    out_file = open(CACHE, "a")

    total_in_toks = 0
    total_out_toks = 0
    total_cache_read = 0
    total_cache_write = 0
    n_calls = 0
    start = time.time()

    for i, t in enumerate(trades):
        if t["ts"] in seen:
            continue
        if limit and n_calls >= limit:
            break

        priors = build_priors(trades[:i])
        snapshot = build_snapshot(t)
        user_msg = json.dumps({"trade": snapshot, "priors": priors}, indent=2)

        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=512,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_msg}],
            )
        except Exception as e:
            print(f"  trade {i+1}/{len(trades)} {t['ts']}: API error: {e}", flush=True)
            time.sleep(2)
            continue

        text = resp.content[0].text
        try:
            parsed = parse_response(text)
            decision = str(parsed.get("decision", "")).upper().strip()
            if decision not in ("TAKE", "SKIP"):
                decision = "PARSE_ERROR"
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = str(parsed.get("reasoning", ""))
        except Exception:
            decision = "PARSE_ERROR"
            confidence = 0.0
            reasoning = text[:400]

        usage = resp.usage
        in_toks = getattr(usage, "input_tokens", 0)
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        out_toks = getattr(usage, "output_tokens", 0)
        total_in_toks += in_toks
        total_out_toks += out_toks
        total_cache_read += cache_read
        total_cache_write += cache_write
        n_calls += 1

        rec = {
            "ts": t["ts"],
            "slug": t["slug"],
            "i": i,
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "actual_won": t["won"],
            "actual_pnl": t["pnl"],
            "stake_usdc": t["stake_usdc"],
            "direction": t["direction"],
            "entry_price": t["entry_price"],
            "in_toks": in_toks,
            "cache_read": cache_read,
            "cache_write": cache_write,
            "out_toks": out_toks,
        }
        out_file.write(json.dumps(rec) + "\n")
        out_file.flush()

        actual = "WIN " if t["won"] else "LOSS"
        # cost so far (Opus 4.7: $15/Mtok input, $75/Mtok output, $1.50/Mtok cache read, $18.75/Mtok cache write)
        cost = (
            (total_in_toks * 15 + total_cache_write * 18.75 + total_cache_read * 1.50 + total_out_toks * 75)
            / 1_000_000.0
        )
        elapsed = time.time() - start
        rate = n_calls / elapsed if elapsed > 0 else 0
        eta_remaining = (len(trades) - len(seen) - n_calls) / rate if rate > 0 else 0
        print(
            f"  {i+1:>3}/{len(trades)} {t['ts']} {t['direction']:<4} ep={t['entry_price']:.3f} "
            f"conf={t['confidence']:>5.1f}% -> {decision:<5} c={confidence:.2f} | actual={actual} | "
            f"toks(in/cr/cw/out)={in_toks}/{cache_read}/{cache_write}/{out_toks} "
            f"$={cost:.2f} eta={eta_remaining:.0f}s",
            flush=True,
        )

    out_file.close()
    print()
    print(f"Done. calls={n_calls} elapsed={time.time()-start:.0f}s")
    print(f"Tokens — input(uncached)={total_in_toks} cache_read={total_cache_read} cache_write={total_cache_write} output={total_out_toks}")
    final_cost = (
        (total_in_toks * 15 + total_cache_write * 18.75 + total_cache_read * 1.50 + total_out_toks * 75)
        / 1_000_000.0
    )
    print(f"Estimated cost: ${final_cost:.2f}")

if __name__ == "__main__":
    main()
