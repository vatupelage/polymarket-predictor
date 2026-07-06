"""HONEST MM edge test: queue-aware fills + short-horizon markout-to-mid.
No inventory accumulation (that = directional bet). Each filled share is valued
at the mid H seconds later = the realistic value of flattening via the other
side, net of adverse selection over the hold. Queue-aware: we sit behind the
resting touch size, so a print only fills our overflow (toxic selection).
"""
import json, datetime, collections, bisect

S = 100.0; EPS = 1e-9
def fee_frac(p): return 0.07 * p * (1 - p)
def epoch(iso):
    try: return datetime.datetime.fromisoformat(iso).timestamp()
    except Exception: return None

ev = collections.defaultdict(list)
mids = collections.defaultdict(list)
for ln in open("mm_book.jsonl"):
    try: r = json.loads(ln)
    except Exception: continue
    if "token_id" not in r: continue
    t = epoch(r["ts"]);
    if t is None: continue
    bids = r.get("bids") or []; asks = r.get("asks") or []
    bb = bids[0][0] if bids else None; bbsz = bids[0][1] if bids else 0.0
    ba = asks[0][0] if asks else None; basz = asks[0][1] if asks else 0.0
    ev[r["token_id"]].append((t, "book", (bb, bbsz, ba, basz)))
    if r.get("mid") is not None: mids[r["token_id"]].append((t, r["mid"]))
for ln in open("mm_tape.jsonl"):
    try: r = json.loads(ln)
    except Exception: continue
    tr = r.get("trade", {}); tid = tr.get("asset"); t = tr.get("ts")
    if tid is None or t is None: continue
    ev[tid].append((float(t), "trade", (str(tr.get("side","")).upper(), float(tr.get("price",0)), float(tr.get("size",0)))))
for k in ev: ev[k].sort(key=lambda x: x[0])
for k in mids: mids[k].sort()

def mid_at(tid, target):
    seq = mids.get(tid)
    if not seq: return None
    ts = [x[0] for x in seq]; i = bisect.bisect_left(ts, target)
    if i >= len(seq): i = len(seq)-1
    return seq[i][1]

H = [5, 15, 30, 60]
fills = []   # (tid, ts, sgn, price, size)
fees = 0.0; buyv = sellv = 0.0
for tid, stream in ev.items():
    bb=bbsz=ba=basz=None; qbid=qask=0.0; fbid=fsell=0.0
    for (t, kind, p) in stream:
        if kind == "book":
            nbb,nbbsz,nba,nbasz = p
            if nbb != bb: qbid = nbbsz or 0.0; fbid = 0.0
            if nba != ba: qask = nbasz or 0.0; fsell = 0.0
            bb,bbsz,ba,basz = nbb,nbbsz,nba,nbasz
        else:
            side, px, sz = p
            if side=="SELL" and bb is not None and px <= bb+EPS:
                if sz <= qbid: qbid -= sz
                else:
                    over = sz-qbid; qbid=0.0; fill=min(over, S-fbid)
                    if fill>0: fbid+=fill; fills.append((tid,t,+1,bb,fill)); fees+=fill*px*fee_frac(px); buyv+=fill
            elif side=="BUY" and ba is not None and px >= ba-EPS:
                if sz <= qask: qask -= sz
                else:
                    over = sz-qask; qask=0.0; fill=min(over, S-fsell)
                    if fill>0: fsell+=fill; fills.append((tid,t,-1,ba,fill)); fees+=fill*px*fee_frac(px); sellv+=fill

fv = buyv+sellv; reb = 0.20*(fees/fv) if fv else 0.0
print("=== QUEUE-AWARE FILLS + SHORT-HORIZON MARKOUT (honest MM edge) ===")
print(f"queue-aware filled volume: {fv:.0f} sh (buys {buyv:.0f} / sells {sellv:.0f})  real rebate {reb*100:+.3f}%/sh")
print(f"{'H':>5} | {'markout/sh':>11} | {'+rebate':>9} | verdict")
for h in H:
    tot=vol=0.0
    for (tid,t,sgn,f,sz) in fills:
        m = mid_at(tid, t+h)
        if m is None: continue
        tot += sgn*(m-f)*sz; vol += sz
    mo = tot/vol if vol else 0.0; net = mo+reb
    print(f"{h:>4}s | {mo*100:>+9.4f}% | {net*100:>+7.4f}% | {'VIABLE' if net>0 else 'loses'}")
print()
print("vs front-of-queue earlier: +0.34..+0.47% markout. Queue-aware = toxic selection;")
print("if markout stays >0 here, the edge survives realistic queue position.")
