"""Detail every skipped-trade that would-have-LOST: pull signal breakdown,
drift, and outcome."""

import json
import re
import time
from pathlib import Path
import requests

LOG = Path("/home/vidura/btcpredictor/predictor/live_bot.log")
GAMMA = "https://gamma-api.polymarket.com/events?slug={}"

text = LOG.read_text()
lines = text.splitlines()

# For each window, extract slug, PTB, live_price, drift, direction, conf, signals, skip-line
# Signals line example: "13:50-13:55 | $ 76,879 | $ 76,750 | -0.1678% | DOWN | 5.2% | 47% | 55% | 46% | 84% | 47% <<<"
# Columns: Window | PTB | Live | Drift | Dir | Final conf | PTB_sig | OB_sig | ??? | LSTM | PM

sig_row_re = re.compile(
    r"(\d{2}:\d{2}-\d{2}:\d{2})\s*\|\s*\$\s*([\d,]+)\s*\|\s*\$\s*([\d,]+)\s*\|\s*"
    r"([+-][\d.]+)%\s*\|\s*([A-Z]+)\s*\|\s*([\d.]+)%\s*\|\s*(\d+)%\s*\|\s*(\d+)%\s*\|\s*"
    r"(\d+)%\s*\|\s*(\d+)%\s*\|\s*(\d+)%"
)

windows = []  # each: dict with signals + meta
last_window_meta = {}

for line in lines:
    m = re.search(r">>> Window (\d{2}:\d{2}-\d{2}:\d{2}) \| Price To Beat: \$([\d,]+) \| Slug: (\S+)", line)
    if m:
        last_window_meta = {
            "window": m.group(1),
            "ptb": int(m.group(2).replace(",", "")),
            "slug": m.group(3),
        }
        continue
    m = sig_row_re.search(line)
    if m and last_window_meta:
        w = dict(last_window_meta)
        w["live"] = int(m.group(3).replace(",", ""))
        w["drift_pct"] = float(m.group(4))
        w["direction"] = m.group(5)
        w["final_conf"] = float(m.group(6))
        w["ptb_sig"] = int(m.group(7))
        w["ob_sig"] = int(m.group(8))
        w["col3"] = int(m.group(9))
        w["lstm_sig"] = int(m.group(10))
        w["pm_sig"] = int(m.group(11))
        windows.append(w)

# Also capture SKIP lines with confidence to match later
skips_by_slug = {}
last_slug = None
for line in lines:
    m = re.search(r">>> Window .* Slug: (\S+)", line)
    if m:
        last_slug = m.group(1)
        continue
    m = re.search(r"SKIP \((drift|price)\):.*\(([A-Z]+) @ ([\d.]+)%\)", line)
    if m and last_slug:
        skips_by_slug[last_slug] = {
            "gate": m.group(1),
            "direction": m.group(2),
            "confidence": float(m.group(3)),
        }

# Query outcomes
def get_outcome(slug):
    try:
        r = requests.get(GAMMA.format(slug), timeout=6)
        events = r.json()
        mkt = events[0]["markets"][0]
        if not mkt.get("closed", False):
            return None
        prices = json.loads(mkt["outcomePrices"])
        return {
            "up_won": float(prices[0]) >= 0.99,
            "up_price": float(prices[0]),
            "down_price": float(prices[1]),
        }
    except Exception:
        return None

print(f"{'Slug':<30} {'Dir':<4} {'Conf':>6} {'Drift':>8}  {'PTB':>4} {'OB':>4} {'LSTM':>4} {'PM':>4}  {'Outcome':<8}")
print("-" * 90)

losses = []
for w in windows:
    slug = w["slug"]
    if slug not in skips_by_slug:
        continue
    skip = skips_by_slug[slug]
    out = get_outcome(slug)
    if not out:
        continue
    would_have_won = (skip["direction"] == "UP" and out["up_won"]) or \
                     (skip["direction"] == "DOWN" and not out["up_won"])
    if would_have_won:
        continue

    losses.append({
        "slug": slug,
        "direction": skip["direction"],
        "conf": skip["confidence"],
        "drift": w["drift_pct"],
        "ptb_sig": w["ptb_sig"],
        "ob_sig": w["ob_sig"],
        "lstm_sig": w["lstm_sig"],
        "pm_sig": w["pm_sig"],
        "up_price": out["up_price"],
    })
    winner = "UP won" if out["up_won"] else "DOWN won"
    print(f"  {slug:<28} {skip['direction']:<4} {skip['confidence']:>5.1f}%  {w['drift_pct']:>+7.4f}%  "
          f"{w['ptb_sig']:>3}% {w['ob_sig']:>3}% {w['lstm_sig']:>3}% {w['pm_sig']:>3}%  {winner}")
    time.sleep(0.15)

print(f"\nTotal losing skips: {len(losses)}  (would have cost: -${len(losses) * 10:.2f})")
