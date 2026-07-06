"""Evaluate S5 SIGNAL edge from shadow log (execution-free): for each window,
does the S5 trigger (entry ask in band + signal>=0.85) pick the winner? Compare
the original Bitstamp signal vs the Polymarket-orderbook signal, gross and net
of the price-dependent dynamic fee. Outcome inferred from the final poll's asks
(winner's ask -> ~1, loser -> ~0).
"""
import json, collections

def fee_of_stake(p):  # ~3.5% of notional at p=0.5 (crypto feeRate=0.07)
    return 0.07 * p * (1 - p)

by_slug = collections.defaultdict(list)
for ln in open("s5_shadow.jsonl"):
    try: r = json.loads(ln)
    except Exception: continue
    by_slug[r["slug"]].append(r)

# outcome per slug from last poll: winner = side with high ask near close
outcome = {}
for slug, polls in by_slug.items():
    polls.sort(key=lambda x: x.get("poll_n", 0))
    last = polls[-1]
    au = last.get("top_ask_up"); ad = last.get("top_ask_down")
    if au is None or ad is None: continue
    if au >= 0.65 and ad <= 0.35: outcome[slug] = "UP"
    elif ad >= 0.65 and au <= 0.35: outcome[slug] = "DOWN"
    # else ambiguous/unresolved -> skip

def evaluate(signal_key, band):
    n=win=0; gross=0.0; net=0.0
    for slug, polls in by_slug.items():
        if slug not in outcome: continue
        # first poll where the S5 trigger fires for this signal
        entry = None
        for p in polls:
            ask = p.get("our_ask")
            if ask is None: continue
            in_band = (band[0] < ask <= band[1])
            if in_band and p.get(signal_key):
                entry = p; break
        if entry is None: continue
        ask = entry["our_ask"]; d = entry["direction"]
        w = (d == outcome[slug])
        pnl = (1 - ask) if w else (-ask)
        f = fee_of_stake(ask)
        n += 1; win += int(w); gross += pnl; net += (pnl - f)
    if not n:
        print(f"  {signal_key:18s} band{band}: no entries"); return
    print(f"  {signal_key:18s} band{band}: n={n:3d}  win={win/n*100:4.1f}%  "
          f"gross={gross/n*100:+6.2f}%/trade  net(after fee)={net/n*100:+6.2f}%/trade")

print(f"windows with clear outcome: {len(outcome)} / {len(by_slug)} total")
print("=== S5 SIGNAL EDGE (shadow, execution-free) — UPPER band (0.58, 0.75] ===")
evaluate("bitstamp_passes_85", (0.58, 0.75))
evaluate("pm_passes_85",       (0.58, 0.75))
print("=== same, FULL S5 band (0.40, 0.75] (incl. disabled lower band) ===")
evaluate("bitstamp_passes_85", (0.40, 0.75))
evaluate("pm_passes_85",       (0.40, 0.75))
print()
print("read: win% > entry-price-implied AND net > 0 => real signal edge (live-flat was")
print("execution, low latency could capture it). net <= 0 => no edge, latency won't help.")
