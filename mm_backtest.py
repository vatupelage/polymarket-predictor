"""Interim maker-strategy backtest on collected MM data.
Models a BEST-CASE maker: joins the best bid/ask (tightest quote), front of queue,
holds inventory to window close. Measures markout (adverse-selection-inclusive)
net edge per share, plus a rebate UPPER BOUND from the taker-fee pool our fills
would have generated. Decisive number: net edge/share > 0 => making is viable.
"""
import json, datetime, collections

def epoch(iso):
    try:
        return datetime.datetime.fromisoformat(iso).timestamp()
    except Exception:
        return None

book = collections.defaultdict(list)   # token_id -> [(ts, best_bid, best_ask, mid, secs_to_close)]
tok_slug = {}
for ln in open("mm_book.jsonl"):
    try:
        r = json.loads(ln)
    except Exception:
        continue
    if "token_id" not in r:
        continue
    t = epoch(r["ts"])
    if t is None:
        continue
    book[r["token_id"]].append((t, r.get("best_bid"), r.get("best_ask"), r.get("mid"), r.get("secs_to_close")))
    tok_slug[r["token_id"]] = r["slug"]
for k in book:
    book[k].sort()

# resolution proxy = final mid per token among snapshots seen near close (<=15s)
res = {}
closed = set()
for tid, seq in book.items():
    near = [m for (t, bb, ba, m, s) in seq if s is not None and s <= 15 and m is not None]
    if near:
        res[tid] = near[-1]
        closed.add(tok_slug[tid])

def prevailing(tid, ts):
    best = None
    for (t, bb, ba, m, s) in book.get(tid, []):
        if t <= ts + 2.0:
            best = (bb, ba)
        else:
            break
    return best

EPS = 1e-9
buys = []   # (markout_per_share, size)
sells = []
fee_pool = 0.0
for ln in open("mm_tape.jsonl"):
    try:
        r = json.loads(ln)
    except Exception:
        continue
    tr = r.get("trade", {})
    tid = tr.get("asset")
    if tid not in res:
        continue
    ts = tr.get("ts")
    if ts is None:
        continue
    side = str(tr.get("side", "")).upper()
    px = float(tr.get("price", 0))
    sz = float(tr.get("size", 0))
    pv = prevailing(tid, ts)
    if not pv or pv[0] is None or pv[1] is None:
        continue
    bb, ba = pv
    R = res[tid]
    fee_frac = 0.07 * px * (1 - px)   # ~3.5% of notional at p=0.5
    if side == "SELL" and px <= bb + EPS:      # we buy at our joined bid -> long
        buys.append((R - bb, sz)); fee_pool += sz * px * fee_frac
    elif side == "BUY" and px >= ba - EPS:     # we sell at our joined ask -> short
        sells.append((ba - R, sz)); fee_pool += sz * px * fee_frac

def agg(rows):
    v = sum(s for _, s in rows)
    p = sum(m * s for m, s in rows)
    return v, (p / v if v else 0.0)

bv, bps = agg(buys)
sv, sps = agg(sells)
tv = bv + sv
gross = ((sum(m * s for m, s in buys) + sum(m * s for m, s in sells)) / tv) if tv else 0.0
reb = (fee_pool / tv) if tv else 0.0

print("=== INTERIM MAKER BACKTEST (best-case: join touch, front of queue, hold to close) ===")
print(f"windows observed-to-close: {len(closed)}   tokens w/ resolution proxy: {len(res)}")
print(f"BID fills (we buy):  {len(buys):4d} prints  {bv:8.0f} sh  markout/sh {bps*100:+.2f}%")
print(f"ASK fills (we sell): {len(sells):4d} prints  {sv:8.0f} sh  markout/sh {sps*100:+.2f}%")
print(f"ALL fills:           {tv:8.0f} sh  GROSS markout/sh (adverse-sel incl, NO rebate) = {gross*100:+.3f}%")
print()
reb_crypto = 0.20 * reb   # Maker Rebates Program: 20% of crypto taker fees -> makers,
                          # and our share ~= 20% x fees OUR fills generated (per-market, proportional)
print(f"taker-fee pool our fills generated: ${fee_pool:.2f} over {tv:.0f} sh")
print(f"  rebate if 100% returned (only-maker, WRONG): {reb*100:+.2f}%/sh")
print(f"  REAL rebate = 20% of crypto taker fees:      {reb_crypto*100:+.3f}%/sh")
print()
print(f"NET edge/sh (markout {gross*100:+.3f}% + real 20% rebate {reb_crypto*100:+.3f}%) = {(gross+reb_crypto)*100:+.3f}%   [>0 = viable]")
print("  NOTE: still BEST-CASE fills (join touch, front of queue, fill 100% of touch flow).")
print("  Realistic queue position fills less good-flow / more toxic-flow -> markout worse than this.")
print("prior adverse selection on this book: -5.5%/sh")
