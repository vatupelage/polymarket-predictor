"""Deep-dive on the 107 skipped signals — what separates the 64 wins from 43 losses?

Extracts the per-window feature row (PTB/OB/LSTM/PM votes, final P(UP), drift)
from the log, joins with Polymarket outcomes, and scores candidate rules so we
can tell the LLM (or a code gate) which windows to actually trade.
"""

import json
import re
import time
from pathlib import Path

import requests

LOG = Path("/home/vidura/btcpredictor/predictor/live_bot.log")
GAMMA = "https://gamma-api.polymarket.com/events?slug={}"

# row format: "HH:MM-HH:MM | $   74,692 | $   74,692 | +0.0000% |   UP | 11.4% |   50% |   51% |   56% |   80% |   56% <<<"
row_re = re.compile(
    r"[\d:]+-[\d:]+\s*\|\s*\$\s*[\d,]+\s*\|\s*\$\s*[\d,]+\s*\|\s*([+-]?[\d.]+)%\s*\|\s*(\w+)\s*\|\s*([\d.]+)%\s*\|"
    r"\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*\|\s*([\d.]+)%\s*<<<"
)
win_re = re.compile(r">>> Window .* Slug:\s+(\S+)")
llm_re = re.compile(r"\[LLM [\d:]+\] action=(\S+) conf=(\d+)")
floor_re = re.compile(r"\[LLM [\d:]+\] SKIP \(low llm_conf\): (\d+) <")


def parse_log() -> list[dict]:
    lines = LOG.read_text().splitlines()
    events: list[dict] = []

    last_row: dict | None = None
    last_slug: str | None = None
    last_llm_action: str | None = None

    for line in lines:
        m = row_re.search(line)
        if m:
            last_row = {
                "drift": float(m.group(1)),
                "dir": m.group(2),
                "conf": float(m.group(3)),
                "ptb": float(m.group(4)),
                "ob": float(m.group(5)),
                "lstm": float(m.group(6)),
                "pm": float(m.group(7)),
                "final_up": float(m.group(8)),
            }
            continue

        m = win_re.search(line)
        if m:
            last_slug = m.group(1)
            last_llm_action = None
            continue

        m = llm_re.search(line)
        if m and last_slug and last_row:
            action = m.group(1)
            llm_conf = int(m.group(2))
            if action == "SKIP":
                events.append({"slug": last_slug, **last_row, "llm_action": "SKIP", "llm_conf": llm_conf})
                last_row = None
            else:
                last_llm_action = action
            continue

        m = floor_re.search(line)
        if m and last_slug and last_row:
            llm_conf = int(m.group(1))
            events.append({
                "slug": last_slug, **last_row,
                "llm_action": last_llm_action or "?", "llm_conf": llm_conf,
            })
            last_row = None

    return events


def get_outcome(slug: str) -> bool | None:
    try:
        r = requests.get(GAMMA.format(slug), timeout=6)
        ev = r.json()
        if not ev:
            return None
        mkt = ev[0]["markets"][0]
        if not mkt.get("closed", False):
            return None
        prices = json.loads(mkt["outcomePrices"])
        return float(prices[0]) >= 0.99  # True = UP won
    except Exception:
        return None


def won_trade(ev: dict, up_won: bool) -> bool:
    return (ev["dir"] == "UP" and up_won) or (ev["dir"] == "DOWN" and not up_won)


def pct(wins: int, total: int) -> str:
    return f"{wins}/{total} ({wins/total*100:.1f}%)" if total else f"0/0 (—)"


def score_rule(rows: list[dict], name: str, pred) -> None:
    sub = [r for r in rows if pred(r)]
    wins = sum(1 for r in sub if r["won"])
    total = len(sub)
    if total == 0:
        print(f"  {name:<55} n=0")
        return
    wr = wins / total * 100
    pnl = wins * 5 - (total - wins) * 5
    print(f"  {name:<55} n={total:>3}  wins={wins:>3}  wr={wr:5.1f}%  pnl=${pnl:+d}")


def main():
    skips = parse_log()
    print(f"Parsed {len(skips)} skipped events with full feature rows\n")

    enriched: list[dict] = []
    for s in skips:
        up_won = get_outcome(s["slug"])
        if up_won is None:
            continue
        s["up_won"] = up_won
        s["won"] = won_trade(s, up_won)
        enriched.append(s)
        time.sleep(0.08)

    wins = [e for e in enriched if e["won"]]
    losses = [e for e in enriched if not e["won"]]
    print(f"Resolved: {len(enriched)}  W: {len(wins)}  L: {len(losses)}  "
          f"wr={len(wins)/len(enriched)*100:.1f}%\n")

    # ---- distribution comparison ----
    def stats(key, rows):
        vals = [r[key] for r in rows]
        return f"mean={sum(vals)/len(vals):5.1f} min={min(vals):5.1f} max={max(vals):5.1f}"

    keys = ["conf", "drift", "final_up", "ptb", "ob", "lstm", "pm", "llm_conf"]
    print("Feature distribution — WINS vs LOSSES:")
    print(f"{'feature':<10} {'winners':<50} losers")
    print("-" * 110)
    for k in keys:
        print(f"  {k:<8} {stats(k, wins):<48} {stats(k, losses)}")
    print()

    # ---- alignment counts ----
    def aligned_count(r):
        """How many of PTB/OB/LSTM/PM agree with predictor's dir."""
        votes = [r["ptb"], r["ob"], r["lstm"], r["pm"]]
        if r["dir"] == "UP":
            return sum(1 for v in votes if v > 50)
        else:
            return sum(1 for v in votes if v < 50)

    for r in enriched:
        r["aligned"] = aligned_count(r)

    print("By # aligned sub-signals (of PTB/OB/LSTM/PM):")
    for n in range(5):
        sub = [r for r in enriched if r["aligned"] == n]
        if sub:
            w = sum(1 for r in sub if r["won"])
            print(f"  {n}/4 aligned: n={len(sub):>3}  wr={pct(w, len(sub))}  pnl=${w*5 - (len(sub)-w)*5:+d}")
    print()

    # ---- candidate rules ----
    print("Candidate filter rules (fire on skip only if rule passes):\n")
    score_rule(enriched, "ALL (baseline, fire every skip)", lambda r: True)
    score_rule(enriched, "conf >= 15%", lambda r: r["conf"] >= 15)
    score_rule(enriched, "conf >= 20%", lambda r: r["conf"] >= 20)
    score_rule(enriched, "conf >= 25%", lambda r: r["conf"] >= 25)
    score_rule(enriched, "final_up extreme (>=55 or <=45)", lambda r: abs(r["final_up"] - 50) >= 5)
    score_rule(enriched, "final_up extreme (>=58 or <=42)", lambda r: abs(r["final_up"] - 50) >= 8)
    score_rule(enriched, ">=3/4 sub-signals align", lambda r: r["aligned"] >= 3)
    score_rule(enriched, "4/4 sub-signals align", lambda r: r["aligned"] >= 4)
    score_rule(enriched, "LSTM aligns with predictor",
               lambda r: (r["dir"] == "UP" and r["lstm"] > 50) or (r["dir"] == "DOWN" and r["lstm"] < 50))
    score_rule(enriched, "LSTM strong-aligned (>=60 or <=40)",
               lambda r: (r["dir"] == "UP" and r["lstm"] >= 60) or (r["dir"] == "DOWN" and r["lstm"] <= 40))
    score_rule(enriched, "OB aligns with predictor",
               lambda r: (r["dir"] == "UP" and r["ob"] > 50) or (r["dir"] == "DOWN" and r["ob"] < 50))
    score_rule(enriched, "PM aligns with predictor",
               lambda r: (r["dir"] == "UP" and r["pm"] > 50) or (r["dir"] == "DOWN" and r["pm"] < 50))
    score_rule(enriched, "conf>=15 AND >=3/4 align",
               lambda r: r["conf"] >= 15 and r["aligned"] >= 3)
    score_rule(enriched, "conf>=15 AND LSTM aligns",
               lambda r: r["conf"] >= 15 and
                         ((r["dir"] == "UP" and r["lstm"] > 50) or (r["dir"] == "DOWN" and r["lstm"] < 50)))
    score_rule(enriched, "conf>=10 AND >=3/4 align AND LSTM aligns",
               lambda r: r["conf"] >= 10 and r["aligned"] >= 3 and
                         ((r["dir"] == "UP" and r["lstm"] > 50) or (r["dir"] == "DOWN" and r["lstm"] < 50)))
    score_rule(enriched, "final_up>=55 for UP, final_up<=45 for DOWN",
               lambda r: (r["dir"] == "UP" and r["final_up"] >= 55) or (r["dir"] == "DOWN" and r["final_up"] <= 45))
    score_rule(enriched, "final_up>=58 for UP, final_up<=42 for DOWN",
               lambda r: (r["dir"] == "UP" and r["final_up"] >= 58) or (r["dir"] == "DOWN" and r["final_up"] <= 42))
    score_rule(enriched, "UP direction only",  lambda r: r["dir"] == "UP")
    score_rule(enriched, "DOWN direction only", lambda r: r["dir"] == "DOWN")


if __name__ == "__main__":
    main()
