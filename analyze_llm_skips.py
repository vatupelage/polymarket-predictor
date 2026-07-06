"""Counterfactual report on LLM-era skipped signals.

For each `[LLM ...] action=SKIP` in the bot log, finds the preceding
`>>> Window ... Slug: ...` and `[BOT ...] decision: DIR` lines, queries
Polymarket Gamma for the market's resolution, and computes what we
would have made/lost taking the predictor's direction at $5 stake.

For SKIPs that came after a Claude BUY_UP/BUY_DOWN, the LLM's intended
direction is used (so we can see what the conviction-floor cost us).
"""

import json
import re
import time
from pathlib import Path

import requests

LOG = Path("/home/vidura/btcpredictor/predictor/live_bot.log")
GAMMA = "https://gamma-api.polymarket.com/events?slug={}"
STAKE = 5.0


def parse_log() -> list[dict]:
    text = LOG.read_text()
    lines = text.splitlines()

    events: list[dict] = []
    last_slug: str | None = None
    last_decision: dict | None = None
    last_llm_action: str | None = None

    win_re = re.compile(r">>> Window .* Slug:\s+(\S+)")
    decision_re = re.compile(r"\[BOT [\d:]+\] decision: (\S+) conf=([\d.]+)% drift=([+-][\d.]+)%")
    llm_re = re.compile(
        r"\[LLM [\d:]+\] action=(\S+) conf=(\d+) reason='(.*?)'?$"
    )
    floor_re = re.compile(r"\[LLM [\d:]+\] SKIP \(low llm_conf\): (\d+) <")

    for line in lines:
        m = win_re.search(line)
        if m:
            last_slug = m.group(1)
            last_decision = None
            last_llm_action = None
            continue

        m = decision_re.search(line)
        if m and last_slug:
            last_decision = {
                "direction": m.group(1),
                "confidence": float(m.group(2)),
                "drift": float(m.group(3)),
            }
            continue

        m = llm_re.search(line)
        if m and last_slug and last_decision:
            action = m.group(1)
            llm_conf = int(m.group(2))
            reason = m.group(3)[:100]
            if action == "SKIP":
                events.append({
                    "slug": last_slug,
                    "predictor_dir": last_decision["direction"],
                    "predictor_conf": last_decision["confidence"],
                    "llm_action": "SKIP",
                    "llm_conf": llm_conf,
                    "reason": reason,
                })
            else:
                last_llm_action = action
            continue

        m = floor_re.search(line)
        if m and last_slug and last_decision:
            llm_conf = int(m.group(1))
            events.append({
                "slug": last_slug,
                "predictor_dir": last_decision["direction"],
                "predictor_conf": last_decision["confidence"],
                "llm_action": last_llm_action or "?",
                "llm_conf": llm_conf,
                "reason": f"floor: {llm_conf}<60",
            })

    return events


def get_outcome(slug: str) -> dict | None:
    try:
        r = requests.get(GAMMA.format(slug), timeout=6)
        events = r.json()
        if not events:
            return None
        mkt = events[0]["markets"][0]
        if not mkt.get("closed", False):
            return None
        prices = json.loads(mkt["outcomePrices"])
        return {
            "up_won": float(prices[0]) >= 0.99,
            "up_price": float(prices[0]),
        }
    except Exception:
        return None


def main():
    skips = parse_log()
    print(f"Total skipped events parsed: {len(skips)}\n")

    # Counterfactual: take the predictor's direction at $5 stake.
    # Without the actual fill price logged at decision time, we approximate:
    #   if predictor is RIGHT and we had filled at the average top_ask of
    #   ~$0.40 (rough), shares = $5 / 0.40 = 12.5, profit = +$7.50
    #   if WRONG, lose -$5.00
    # This is rough — overstates wins, accurate on losses.

    rows: list[dict] = []
    wins, losses, unresolved = 0, 0, 0
    total_pnl_rough = 0.0

    for s in skips:
        out = get_outcome(s["slug"])
        if out is None:
            unresolved += 1
            verdict = "?"
            pnl = 0.0
        else:
            won = (s["predictor_dir"] == "UP" and out["up_won"]) or \
                  (s["predictor_dir"] == "DOWN" and not out["up_won"])
            verdict = "WIN" if won else "LOSS"
            if won:
                wins += 1
                pnl = +5.0  # rough: assumes ~0.50 fill
            else:
                losses += 1
                pnl = -5.0
            total_pnl_rough += pnl

        rows.append({**s, "verdict": verdict, "pnl_rough": pnl,
                     "up_price": out["up_price"] if out else None})
        time.sleep(0.12)

    print(f"Resolved: {wins + losses}  Unresolved: {unresolved}")
    print(f"Would-have-WON:  {wins}")
    print(f"Would-have-LOST: {losses}")
    if (wins + losses) > 0:
        print(f"Would-have win-rate: {wins/(wins+losses)*100:.1f}%")
    print(f"Rough counterfactual PnL @ ${STAKE} flat (assumes 50/50 fills): "
          f"${total_pnl_rough:+.2f}")
    print()
    print(f"{'Slug':<32} {'Pred':<5} {'Conf':>6} {'LLM':<8} {'lconf':>5}  {'Verdict':<7} {'PnL':>7}  Reason")
    print("-" * 130)
    for r in rows:
        print(f"  {r['slug']:<30} {r['predictor_dir']:<5} {r['predictor_conf']:>5.1f}% "
              f"{r['llm_action']:<8} {r['llm_conf']:>4}   {r['verdict']:<7} "
              f"${r['pnl_rough']:+5.2f}  {r['reason']}")


if __name__ == "__main__":
    main()
