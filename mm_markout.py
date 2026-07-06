"""Maker mark-out backtest done RIGHT: measure short-horizon mark-out (spread
captured minus adverse selection BEFORE we flatten), not hold-to-resolution.
A real MM flattens fast; resolution-markout = directional bet (wrong metric).

For each touch fill at price f on token at time t:
  buy  (taker SELL hit our bid): markout(H) = mid(t+H) - f   [we're long at f, flatten at mid later]
  sell (taker BUY lifted ask):   markout(H) = f - mid(t+H)
Positive = we captured more than we gave back. Add 20% crypto rebate.
"""
import json, datetime, collections, bisect

def epoch(iso):
    try: return datetime.datetime.fromisoformat(iso).timestamp()
    except Exception: return None

mids = collections.defaultdict(list)   # token_id -> [(ts, mid)]
touch = collections.defaultdict(list)  # token_id -> [(ts, best_bid, best_ask)]
for ln in open("mm_book.jsonl"):
    try: r = json.loads(ln)
    except Exception: continue
    if "token_id" not in r: continue
    t = epoch(r["ts"])
    if t is None or r.get("mid") is None: continue
    mids[r["token_id"]].append((t, r["mid"]))
    touch[r["token_id"]].append((t, r.get("best_bid"), r.get("best_ask")))
for k in mids: mids[k].sort()
for k in touch: touch[k].sort()

def mid_at(tid, target):
    seq = mids.get(tid)
    if not seq: return None
    ts = [x[0] for x in seq]
    i = bisect.bisect_left(ts, target)
    if i >= len(seq): i = len(seq) - 1
    return seq[i][1]

def touch_at(tid, target):
    seq = touch.get(tid)
    if not seq: return None
    ts = [x[0] for x in seq]
    i = bisect.bisect_right(ts, target + 2.0) - 1
    if i < 0: return None
    return seq[i][1], seq[i][2]

H = [5, 15, 30, 60]
fills = []   # (token_id, ts, side, fill_price)  side: +1 buy / -1 sell
fee_pool = 0.0
EPS = 1e-9
for ln in open("mm_tape.jsonl"):
    try: r = json.loads(ln)
    except Exception: continue
    tr = r.get("trade", {}); tid = tr.get("asset"); ts = tr.get("ts")
    if tid not in mids or ts is None: continue
    side = str(tr.get("side", "")).upper(); px = float(tr.get("price", 0)); sz = float(tr.get("size", 0))
    tk = touch_at(tid, ts)
    if not tk or tk[0] is None or tk[1] is None: continue
    bb, ba = tk
    fee_frac = 0.07 * px * (1 - px)
    if side == "SELL" and px <= bb + EPS:
        fills.append((tid, ts, +1, bb, sz)); fee_pool += sz * px * fee_frac
    elif side == "BUY" and px >= ba - EPS:
        fills.append((tid, ts, -1, ba, sz)); fee_pool += sz * px * fee_frac

tv = sum(f[4] for f in fills)
reb = 0.20 * (fee_pool / tv) if tv else 0.0
print("=== MAKER MARK-OUT (flatten-fast model; short-horizon adverse selection) ===")
print(f"touch fills: {len(fills)}  volume: {tv:.0f} sh   real rebate (20% crypto): {reb*100:+.3f}%/sh")
print()
print(f"{'horizon':>8} | {'markout/sh':>11} | {'+rebate':>9} | net verdict")
for h in H:
    tot = 0.0; vol = 0.0
    for (tid, ts, sgn, f, sz) in fills:
        m = mid_at(tid, ts + h)
        if m is None: continue
        mk = sgn * (m - f)          # buy: mid-f ; sell: f-mid
        tot += mk * sz; vol += sz
    mo = (tot / vol) if vol else 0.0
    net = mo + reb
    print(f"{h:>6}s  | {mo*100:>+9.3f}% | {net*100:>+7.3f}% | {'VIABLE' if net>0 else 'loses'}")
print()
print("read: markout at small H = spread captured; if it stays >0 (or >-rebate) a fast-flatten")
print("maker earns. if negative even at 5s, informed flow picks us off before we can flatten.")
