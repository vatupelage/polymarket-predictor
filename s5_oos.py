"""S5 out-of-sample / robustness validation. Tests the two overfit signatures
the 1596-window analysis flagged:
  (1) per-day concentration  ("collapses to a few lucky days")
  (2) threshold/band sensitivity ("sign-flips across selection rules")
plus a temporal first-half/second-half split. Uses the shadow log + raw
bitstamp_ob_side; outcome inferred from final-poll prices; net of dynamic fee.
"""
import json, collections, statistics

def fee_of_stake(p): return 0.07 * p * (1 - p)

by_slug = collections.defaultdict(list)
for ln in open("s5_shadow.jsonl"):
    try: r = json.loads(ln)
    except Exception: continue
    by_slug[r["slug"]].append(r)

outcome = {}; slug_day = {}
for slug, polls in by_slug.items():
    polls.sort(key=lambda x: x.get("poll_n", 0))
    last = polls[-1]; au = last.get("top_ask_up"); ad = last.get("top_ask_down")
    slug_day[slug] = polls[0]["ts"][:10]
    if au is None or ad is None: continue
    if au >= 0.65 and ad <= 0.35: outcome[slug] = "UP"
    elif ad >= 0.65 and au <= 0.35: outcome[slug] = "DOWN"

def entries(thr, band):
    out = []
    for slug, polls in by_slug.items():
        if slug not in outcome: continue
        for p in polls:
            ask = p.get("our_ask"); ob = p.get("bitstamp_ob_side")
            if ask is None or ob is None: continue
            if band[0] < ask <= band[1] and ob >= thr:
                w = (p["direction"] == outcome[slug])
                pnl = (1-ask) if w else (-ask)
                out.append((slug_day[slug], w, pnl - fee_of_stake(ask)))
                break
    return out

def summ(rows):
    if not rows: return "no entries"
    n=len(rows); win=sum(int(w) for _,w,_ in rows); net=sum(p for _,_,p in rows)
    return f"n={n:3d}  win={win/n*100:4.1f}%  net={net/n*100:+6.2f}%/trade  total={net*100:+6.1f}% of $1/trade"

print("============ (1) PER-DAY DECOMPOSITION  (bitstamp>=0.85, band 0.58-0.75) ============")
rows = entries(0.85, (0.58, 0.75))
byday = collections.defaultdict(list)
for d, w, p in rows: byday[d].append((w, p))
for d in sorted(byday):
    rs = byday[d]; n=len(rs); win=sum(int(w) for w,_ in rs); net=sum(p for _,p in rs)
    print(f"  {d}: n={n}  win={win}/{n}  net={net*100:+.1f}%")
days_positive = sum(1 for d in byday if sum(p for _,p in byday[d])>0)
print(f"  -> {len(byday)} trading days, {days_positive} net-positive. "
      f"top day = {max(sum(p for _,p in byday[d]) for d in byday)*100:+.1f}%, "
      f"total = {sum(p for _,_,p in rows)*100:+.1f}%")
print(f"  ALL: {summ(rows)}")

print("============ (2) THRESHOLD / BAND SENSITIVITY (does the edge hold or flip?) ============")
for thr in (0.80, 0.85, 0.90):
    for band in ((0.58,0.75),(0.50,0.75),(0.50,0.80),(0.40,0.75)):
        print(f"  thr={thr} band{band}: {summ(entries(thr, band))}")

print("============ (3) TEMPORAL SPLIT (first half vs second half by date) ============")
rows = entries(0.85, (0.50, 0.75))
days = sorted(set(d for d,_,_ in rows))
if len(days) >= 2:
    mid = days[len(days)//2]
    first = [(w,p) for d,w,p in rows if d < mid]
    second = [(w,p) for d,w,p in rows if d >= mid]
    print(f"  split at {mid}")
    print(f"  FIRST  half: {summ([('',w,p) for w,p in first])}")
    print(f"  SECOND half: {summ([('',w,p) for w,p in second])}")
