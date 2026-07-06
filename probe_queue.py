"""LIVE queue-position probe (~$1 risk). Posts a tiny resting GTC bid at the
touch, times the post (our speed-to-rest), measures size_ahead (queue depth in
front of us), then cancels. Repeats. Tells us empirically: how fast can we get a
resting order live, and do we land at the FRONT (size_ahead~0) or behind the
incumbents' resting size?
"""
import time, os, math
from live_trader.config import load_config
from live_trader.polymarket import PolymarketBotClient
from py_clob_client_v2.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions, BalanceAllowanceParams, AssetType

cfg = load_config(dotenv_path=os.path.join(os.getcwd(), ".env"))
cl = PolymarketBotClient(cfg)

def usdc():
    r = cl._client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL, signature_type=cfg.signature_type))
    raw = r.get("balance") if isinstance(r, dict) else getattr(r, "balance", None)
    v = float(raw); return v / 1e6 if v >= 1e5 else v

now = int(time.time()); slug = f"btc-updown-5m-{(now//300)*300}"
mkt = cl.resolve_market(slug); tok = mkt["up_token"]
cl.warm_token(tok); cl.warm_signer(tok)
ts_, nr = cl._tok_meta.get(tok, (None, None))
opts = PartialCreateOrderOptions(tick_size=ts_, neg_risk=nr)
bal0 = usdc()
print(f"slug={slug}  USDC start ${bal0:.2f}  (probing UP token)")
print(f"{'#':>2} | {'best_bid':>8} {'bid_size':>8} | {'post_ms':>7} | {'size_ahead':>10} | {'we_first?':>9}")

post_lats = []; aheads = []; first_ct = 0; n = 0
for i in range(8):
    book = cl.get_full_book(tok)
    if not book or not book["bids"] or not book["asks"]:
        time.sleep(2); continue
    bb = book["bids"][0][0]; bbsz = book["bids"][0][1]; ba = book["asks"][0][0]
    size = max(5.0, math.ceil(1.05 / max(bb, 0.02)))   # >= $1 notional, min 5 sh
    s = cl._client.create_order(OrderArgs(token_id=tok, price=bb, size=size, side="BUY"), opts)
    t = time.perf_counter()
    oid = None
    try:
        resp = cl._client.post_order(s, OrderType.GTC)
        post_ms = (time.perf_counter() - t) * 1000
        oid = (resp.get("orderID") or resp.get("orderId")) if isinstance(resp, dict) else None
        status = resp.get("status") if isinstance(resp, dict) else resp
    except Exception as e:
        print(f"{i:>2} | post err: {str(e)[:50]}"); time.sleep(2); continue
    # re-read book: size now at our price minus our size = queue ahead of us
    b2 = cl.get_full_book(tok)
    cur = next((sz for (p, sz) in b2["bids"] if abs(p - bb) < 1e-9), 0.0) if b2 and b2["bids"] else 0.0
    ahead = max(0.0, cur - size)
    first = ahead <= size  # roughly alone / at front
    post_lats.append(post_ms); aheads.append(ahead); n += 1
    if first: first_ct += 1
    print(f"{i:>2} | {bb:>8.3f} {bbsz:>8.0f} | {post_ms:>6.0f} | {ahead:>10.0f} | {'YES' if first else 'no':>9}  status={status}")
    try:
        cl._client.cancel_all()
    except Exception:
        pass
    time.sleep(3)

# flatten any accidental inventory
try:
    cl._client.cancel_all()
except Exception:
    pass
bal1 = usdc()
post_lats.sort(); aheads.sort()
if n:
    print()
    print(f"median post latency: {post_lats[n//2]:.0f} ms   (our speed to get a resting order live)")
    print(f"median size_ahead:   {aheads[n//2]:.0f} sh    (shares queued in FRONT of us at the touch)")
    print(f"landed at front (size_ahead<=our size): {first_ct}/{n}")
print(f"USDC end ${bal1:.2f}  (delta ${bal1-bal0:+.2f})")
print()
print("read: low post latency = we ACT fast. but if size_ahead is consistently large,")
print("incumbents already hold the front -> we sit behind -> toxic overflow fills -> the")
print("queue-aware backtest's negative result is REAL, not pessimistic.")
