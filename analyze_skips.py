"""Analyze all SKIP events in live_bot.log - compute counterfactual PnL
had we invested $10 per skipped trade at the top-ask at bot decision time.

Limitation: we don't log the fill price at skip time, so we use the final
outcome price (0 or 1) which overstates PnL - but market is binary so
this at least tells us which direction won. Sign of PnL is correct; the
magnitude assumes a fill at ~0.5 for losses (won=False -> -$10) and
outcome-share count shares at ~0.5 for wins (won=True -> rough +$10).
"""

import json
import re
import time
from pathlib import Path
import requests

LOG = Path("/home/vidura/btcpredictor/predictor/live_bot.log")
GAMMA = "https://gamma-api.polymarket.com/events?slug={}"
STAKE = 10.0

text = LOG.read_text()
lines = text.splitlines()

# Walk the log; for each SKIP line, find the preceding ">>> Window ... Slug:"
skips = []
last_slug = None
last_ptb = None

for line in lines:
    m = re.search(r">>> Window .* Slug: (\S+)", line)
    if m:
        last_slug = m.group(1)
        continue
    m = re.search(r"SKIP \((drift|price)\):.*\(([A-Z]+) @ ([\d.]+)%\)", line)
    if m and last_slug:
        gate, direction, conf = m.group(1), m.group(2), float(m.group(3))
        skips.append({
            "slug": last_slug,
            "gate": gate,
            "direction": direction,
            "confidence": conf,
        })

print(f"Total SKIP events: {len(skips)}\n")

# Query resolution for each unique slug
slugs = list({s["slug"] for s in skips})
results = {}
for slug in slugs:
    try:
        r = requests.get(GAMMA.format(slug), timeout=6)
        events = r.json()
        if not events:
            continue
        mkt = events[0]["markets"][0]
        if not mkt.get("closed", False):
            results[slug] = None
            continue
        prices = json.loads(mkt["outcomePrices"])
        up_won = float(prices[0]) >= 0.99
        results[slug] = {"up_won": up_won, "up_price": float(prices[0]),
                         "down_price": float(prices[1])}
    except Exception as e:
        results[slug] = None
    time.sleep(0.2)

# Compute counterfactual PnL (assuming $10 filled at ~$0.50 avg -> 20 shares)
# Approximation: winner paid $1/share, loser paid $0 -> PnL = (20 × 1 - 10) if win
# This is rough; without real fill prices we assume balanced ~0.5.
wins, losses, unresolved = 0, 0, 0
pnl_est = 0.0
for s in skips:
    res = results.get(s["slug"])
    if not res:
        unresolved += 1
        continue
    won = (s["direction"] == "UP" and res["up_won"]) or \
          (s["direction"] == "DOWN" and not res["up_won"])
    if won:
        wins += 1
        pnl_est += 10.0   # rough: doubled
    else:
        losses += 1
        pnl_est -= 10.0

print(f"Resolved: {wins + losses}  Unresolved: {unresolved}")
print(f"Would-have-won:  {wins}")
print(f"Would-have-lost: {losses}")
print(f"Rough PnL (assuming ~0.5 fills): ${pnl_est:+.2f}")
print()

# Detailed breakdown
for s in skips:
    res = results.get(s["slug"])
    if not res:
        verdict = "unresolved"
    else:
        won = (s["direction"] == "UP" and res["up_won"]) or \
              (s["direction"] == "DOWN" and not res["up_won"])
        verdict = "WIN " if won else "LOSS"
    print(f"  {s['gate']:5s} {s['direction']:4s} @ {s['confidence']:5.1f}%  {s['slug']}  {verdict}")
