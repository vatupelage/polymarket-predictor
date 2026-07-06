#!/usr/bin/env python3
"""One-shot: wait for the open 5-min position to resolve, then redeem.

Open position at shutdown:
  slug:    btc-updown-5m-1776916800   (window 09:30-09:35 UTC, 2026-04-23)
  side:    UP, 13.16 shares
"""
import os, sys, time, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from live_trader.config import load_config
from live_trader.polymarket import PolymarketBotClient

SLUG = "btc-updown-5m-1776916800"
WIN_END = 1776917100  # 09:35:00 UTC
DEADLINE = WIN_END + 600  # give the oracle 10 min to report

cfg = load_config(dotenv_path=os.path.join(HERE, ".env"))
client = PolymarketBotClient(cfg)

print(f"[{time.strftime('%H:%M:%S')}] resolving {SLUG} ...")
mkt = client.resolve_market(SLUG)
cond_id = mkt["condition_id"]
print(f"  condition_id: {cond_id}")

print(f"[{time.strftime('%H:%M:%S')}] waiting for resolution (deadline {time.strftime('%H:%M:%S', time.gmtime(DEADLINE))})...")
res = client.wait_for_resolution(SLUG, DEADLINE, poll_sec=10)
if not res:
    print("  no resolution within deadline; bailing")
    sys.exit(1)

print(f"  result: up_won={res['up_won']} up_px={res['up_price']} down_px={res['down_price']}")

# We bought UP — win iff up_won
won = bool(res["up_won"])
print(f"  our UP position: {'WIN' if won else 'LOSS'}")

if not won:
    print("  loss — nothing to redeem (DOWN shares we don't hold)")
    sys.exit(0)

print(f"[{time.strftime('%H:%M:%S')}] redeeming condition_id={cond_id} ...")
tx = client.redeem_position(cond_id)
if tx:
    print(f"  redeemed -> {tx}")
else:
    print("  redemption returned None (oracle not ready or other error)")
    sys.exit(2)
