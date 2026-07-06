"""
Check the on-chain status of today's two pending trades:
- 2026-05-22T03:15:20  DOWN  ep=0.68  14.06 shares  slug=btc-updown-5m-1779399000
- 2026-05-22T04:55:13  DOWN  ep=0.70  34.05 shares  slug=btc-updown-5m-1779405000

Read-only. Resolves each market and queries on-chain CTF balance for the side
we took. Reports WIN/LOSS and whether we still own shares that need redemption.
"""
import os, sys, json, requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from live_trader.config import load_config
from live_trader.polymarket import PolymarketBotClient

cfg = load_config(dotenv_path=os.path.join(HERE, ".env"))
client = PolymarketBotClient(cfg)
addr = cfg.funder_address

PENDING = [
    {"ts": "2026-05-22T03:15:20", "slug": "btc-updown-5m-1779399000", "side": "DOWN", "shares": 14.06, "spent": 9.56},
    {"ts": "2026-05-22T04:55:13", "slug": "btc-updown-5m-1779405000", "side": "DOWN", "shares": 34.05, "spent": 23.84},
]

print(f"Wallet: {addr}")
print()

for p in PENDING:
    print(f"=== {p['ts']}  {p['slug']}  {p['side']} {p['shares']} sh (${p['spent']:.2f} paid) ===")
    try:
        mkt = client.resolve_market(p["slug"])
    except Exception as e:
        print(f"  resolve_market failed: {type(e).__name__}: {e}")
        continue

    cond_id = mkt["condition_id"]
    up_tok = mkt.get("up_token_id") or mkt.get("up_token")
    down_tok = mkt.get("down_token_id") or mkt.get("down_token")
    our_tok = down_tok if p["side"] == "DOWN" else up_tok

    print(f"  condition_id: {cond_id}")
    print(f"  our_token (CTF positionId for {p['side']}): {our_tok}")

    # Check our on-chain balance
    try:
        bal = client.get_conditional_balance(str(our_tok)) or 0.0
    except Exception as e:
        print(f"  balance check failed: {type(e).__name__}: {e}")
        continue
    print(f"  on-chain balance: {bal:.4f} shares")

    # Try to determine resolution
    res = None
    try:
        # Short timeout — just check current state, don't wait
        import time
        res = client.wait_for_resolution(p["slug"], int(time.time()) + 1, poll_sec=1)
    except Exception as e:
        print(f"  resolution check error: {type(e).__name__}: {e}")

    if res:
        up_won = bool(res.get("up_won"))
        won = up_won if p["side"] == "UP" else (not up_won)
        print(f"  resolution: up_won={up_won}  up_px={res.get('up_price')}  down_px={res.get('down_price')}")
        print(f"  our {p['side']} side: {'WIN ' if won else 'LOSS'}")
        if won and bal >= 0.01:
            value = bal * 1.0
            print(f"  ** REDEEMABLE: {bal:.4f} shares worth ~${value:.2f} **")
        elif won and bal < 0.01:
            print(f"  WIN but on-chain balance is zero — already redeemed somewhere")
        else:
            print(f"  LOSS — nothing to redeem")
    else:
        print(f"  resolution: not yet resolved (or oracle hasn't reported)")
        if bal >= 0.01:
            print(f"  But you hold {bal:.4f} shares on-chain. Wait for market to settle.")

    print()
