"""Independent redeem sweep: finds every resolved WINNING position (curPrice>0.5,
size>0) for the funder and redeems it to USDC. Safe to run repeatedly (already-
redeemed positions just drop off). Belt-and-suspenders so wins always get cashed
even if the bot's own redeem path hiccups.
"""
import requests, os, time, datetime
from live_trader.config import load_config
from live_trader.polymarket import PolymarketBotClient
from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType

cfg = load_config(dotenv_path=os.path.join(os.getcwd(), ".env"))
c = PolymarketBotClient(cfg)
funder = cfg.funder_address

def usdc():
    r = c._client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=cfg.signature_type))
    raw = r.get("balance") if isinstance(r, dict) else getattr(r, "balance", None)
    v = float(raw); return v / 1e6 if v >= 1e5 else v

def winners():
    out = []; off = 0
    while off < 1000:
        try:
            r = requests.get("https://data-api.polymarket.com/positions",
                             params={"user": funder, "sizeThreshold": "0.1", "limit": "200", "offset": str(off)}, timeout=15)
            b = r.json()
        except Exception:
            break
        if not isinstance(b, list) or not b:
            break
        for p in b:
            if float(p.get("curPrice", 0)) > 0.5 and float(p.get("size", 0)) > 0.01:
                out.append((p.get("conditionId") or p.get("condition_id"), p.get("title", "")[:34]))
        if len(b) < 200:
            break
        off += 200
    return out

ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
w = winners()
if not w:
    print(f"[{ts}] no redeemable winners (balance ${usdc():.2f})")
else:
    b0 = usdc()
    print(f"[{ts}] {len(w)} winner(s) to redeem (balance ${b0:.2f})")
    for cid, title in w:
        try:
            res = c.redeem_position(cid, force=True)
            print(f"  REDEEM {title:34s} -> {str(res)[:70]}")
        except Exception as e:
            print(f"  {title:34s} ERR: {str(e)[:90]}")
        time.sleep(4)
    time.sleep(8)
    print(f"[{ts}] balance ${b0:.2f} -> ${usdc():.2f}")
