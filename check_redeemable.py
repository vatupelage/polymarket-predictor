"""
Scan-only: check what Polymarket positions exist for the funder wallet.
Identifies orphan winners that need redemption. NO on-chain writes — safe to run.

Run with:  python3 check_redeemable.py
"""
import os
import sys
import json
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from live_trader.config import load_config

cfg = load_config(dotenv_path=os.path.join(HERE, ".env"))
addr = cfg.funder_address

print(f"Wallet: {addr}")
print()

# 1) All open / unsettled positions
print("=" * 90)
print("ALL OPEN POSITIONS  (data-api.polymarket.com/positions)")
print("=" * 90)
url_all = f"https://data-api.polymarket.com/positions?user={addr}&sizeThreshold=0.1"
try:
    r = requests.get(url_all, timeout=15)
    r.raise_for_status()
    positions = r.json()
except Exception as e:
    print(f"  API error: {type(e).__name__}: {e}")
    sys.exit(1)

if not isinstance(positions, list):
    print(f"  unexpected response: {positions}")
    sys.exit(1)

print(f"  Total positions returned: {len(positions)}")
print()

if positions:
    print(f"  {'slug':<35} {'side':<5} {'size':>8} {'curPx':>7} {'value':>9} {'redeem?':<8}")
    print("  " + "-" * 80)
    for p in positions:
        slug = (p.get("eventSlug") or p.get("slug") or "?")[:33]
        side = (p.get("outcome") or "")[:4]
        size = float(p.get("size", 0))
        cur  = float(p.get("curPrice", 0))
        redeemable = p.get("redeemable", False)
        value = size * cur
        flag = "YES" if redeemable else "no"
        print(f"  {slug:<35} {side:<5} {size:>8.2f} {cur:>7.3f} ${value:>7.2f}  {flag:<8}")
    print()

# 2) Focus on redeemable winners
print("=" * 90)
print("REDEEMABLE POSITIONS  (winners ready to claim)")
print("=" * 90)
url_red = f"https://data-api.polymarket.com/positions?user={addr}&redeemable=true&sizeThreshold=0.1"
r = requests.get(url_red, timeout=15)
r.raise_for_status()
red_positions = r.json()

if not isinstance(red_positions, list):
    print(f"  unexpected response: {red_positions}")
    sys.exit(1)

winners = [p for p in red_positions if float(p.get("curPrice", 0)) >= 0.99 and float(p.get("size", 0)) >= 0.1]
print(f"  Redeemable returned: {len(red_positions)}")
print(f"  Filtered winners (curPx>=0.99, size>=0.1): {len(winners)}")
print()

if winners:
    total_value = 0
    print(f"  {'slug':<35} {'side':<5} {'size':>8} {'curPx':>7} {'value':>9} {'cond_id':<12}")
    print("  " + "-" * 90)
    for p in winners:
        slug = (p.get("eventSlug") or "?")[:33]
        side = (p.get("outcome") or "")[:4]
        size = float(p.get("size", 0))
        cur  = float(p.get("curPrice", 0))
        value = size * cur
        cond_id = (p.get("conditionId") or "?")[:12]
        total_value += value
        print(f"  {slug:<35} {side:<5} {size:>8.2f} {cur:>7.3f} ${value:>7.2f}  {cond_id}")
    print()
    print(f"  >>> Total redeemable value: ${total_value:.2f} <<<")
else:
    print("  No redeemable winners found.")

print()
print("If there are redeemable winners, run:  python3 do_redeem.py")
