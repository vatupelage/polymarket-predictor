"""Empirically test: does the POLYMARKET orderbook imbalance work as an S5
directional signal under low latency? Reproduces the efficiency test on fresh
co-located data: measure (a) does PM-book imbalance PREDICT the winner, vs
(b) does TRADING it (cross the ask) make money gross, and (c) net of the ~3%
dynamic taker fee. High predictive power + negative PnL = endogenous signal,
already in the price you pay; low latency cannot fix that.
"""
import json, collections

def fee_frac(p): return 0.07 * p * (1 - p)   # ~3.5% of notional at 0.5

# per (slug, side) -> list of (secs_to_close, best_bid, best_ask, mid, bid_depth2c, ask_depth2c, microprice)
W = collections.defaultdict(dict)   # slug -> {"UP":[...], "DOWN":[...]}
for ln in open("mm_book.jsonl"):
    try: r = json.loads(ln)
    except Exception: continue
    s = r.get("slug"); tok = r.get("token")
    if not s or tok not in ("UP", "DOWN"): continue
    W.setdefault(s, {}).setdefault(tok, []).append((
        r.get("secs_to_close"), r.get("best_bid"), r.get("best_ask"),
        r.get("mid"), r.get("bid_depth_2c"), r.get("ask_depth_2c"), r.get("microprice")))

DEC = 120   # decide ~2 min before close (S5-style)
def at(seq, secs):
    cand = [x for x in seq if x[0] is not None]
    if not cand: return None
    return min(cand, key=lambda x: abs(x[0] - secs))
def final_mid(seq):
    near = [x for x in seq if x[0] is not None and x[0] <= 20 and x[3] is not None]
    return near[-1][3] if near else None

n=0; correct=0; gross=0.0; net=0.0; signal_sum=0.0
flat_acc=[]
for slug, sides in W.items():
    if "UP" not in sides or "DOWN" not in sides: continue
    up = sides["UP"]; dn = sides["DOWN"]
    fu = final_mid(up)
    if fu is None: continue
    up_won = 1 if fu > 0.5 else 0
    du = at(up, DEC)
    if du is None or du[4] is None or du[5] is None or (du[4]+du[5]) == 0: continue
    # PM-book imbalance on UP token: bid-heavy => up favored
    imb = (du[4] - du[5]) / (du[4] + du[5])
    pred_up = 1 if imb > 0 else 0
    if pred_up == up_won: correct += 1
    # trade: enter predicted side at ITS ask, settle to outcome
    if pred_up:
        ask = at(up, DEC)[2]
        if ask is None: continue
        pnl = (1 - ask) if up_won else (-ask)
        f = fee_frac(ask)
    else:
        dask = at(dn, DEC)[2]
        if dask is None: continue
        dn_won = 1 - up_won
        pnl = (1 - dask) if dn_won else (-dask)
        f = fee_frac(dask)
    gross += pnl; net += pnl - f; n += 1
    signal_sum += abs(imb)

if n:
    acc = correct / n
    print("=== S5 with POLYMARKET-orderbook signal (fresh low-latency data) ===")
    print(f"windows tested: {n}   decision @ ~{DEC}s to close")
    print(f"predictive accuracy (book imbalance -> winner): {acc*100:.1f}%   (50% = no edge)")
    print(f"GROSS PnL/trade (cross the ask, no fee):  ${gross/n:+.4f}  per $1 staked = {gross/n*100:+.2f}%")
    print(f"NET PnL/trade (after ~3% dynamic fee):    ${net/n:+.4f}  per $1 staked = {net/n*100:+.2f}%")
    print()
    print("interpretation: if accuracy > 50% but PnL < 0 -> the signal is REAL but")
    print("already priced into the ask you cross; + the fee. Low latency can't fix")
    print("buying the favorite at an inflated, fee'd price. (small sample; mechanism demo.)")
else:
    print("no usable windows (need both UP/DOWN books + outcome + decision snapshot)")
