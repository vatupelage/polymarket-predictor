"""Two-sided maker simulation with REALISM: queue-aware fills + flatten cost.

Per token, per window: quote size S at the touch (best_bid/best_ask), sitting
BEHIND the resting size (queue_ahead). A taker print only fills us with its
overflow after clearing the queue (-> we catch the sweeping/toxic prints, miss
the benign flow the front of queue absorbs). Track inventory + cash; at window
close flatten residual by CROSSING (give up spread + pay ~3% taker fee). Add the
real 20% crypto maker rebate on fees our fills generated.

PnL/share net of all of this = the honest market-making edge.
"""
import json, datetime, collections

S = 100.0          # our quote size per side (shares)
EPS = 1e-9
def fee_frac(p):   # dynamic taker fee ~3.5% of notional at p=0.5
    return 0.07 * p * (1 - p)
def epoch(iso):
    try: return datetime.datetime.fromisoformat(iso).timestamp()
    except Exception: return None

# build per-token chronological event stream
ev = collections.defaultdict(list)   # token_id -> [(ts, kind, payload)]
slug_of = {}
for ln in open("mm_book.jsonl"):
    try: r = json.loads(ln)
    except Exception: continue
    if "token_id" not in r: continue
    t = epoch(r["ts"])
    if t is None: continue
    bids = r.get("bids") or []; asks = r.get("asks") or []
    bb = bids[0][0] if bids else None; bbsz = bids[0][1] if bids else 0.0
    ba = asks[0][0] if asks else None; basz = asks[0][1] if asks else 0.0
    ev[r["token_id"]].append((t, "book", (bb, bbsz, ba, basz)))
    slug_of[r["token_id"]] = r["slug"]
for ln in open("mm_tape.jsonl"):
    try: r = json.loads(ln)
    except Exception: continue
    tr = r.get("trade", {}); tid = tr.get("asset"); t = tr.get("ts")
    if tid is None or t is None: continue
    ev[tid].append((float(t), "trade", (str(tr.get("side","")).upper(), float(tr.get("price",0)), float(tr.get("size",0)))))
for k in ev: ev[k].sort(key=lambda x: x[0])

tot_pnl = 0.0; tot_fillvol = 0.0; tot_fees = 0.0
tot_buy = 0.0; tot_sell = 0.0; resid_abs = 0.0; n_tok = 0
flatten_cost_tot = 0.0
for tid, stream in ev.items():
    bb=bbsz=ba=basz=None
    qbid=qask=0.0; fbid=fsell=0.0   # queue ahead + our filled-at-level (bid/ask)
    inv=0.0; cash=0.0; fees=0.0; buyv=0.0; sellv=0.0
    last_bb=last_ba=None
    for (t, kind, p) in stream:
        if kind == "book":
            nbb, nbbsz, nba, nbasz = p
            # touch moved -> we cancel/repost: reset queue ahead to current resting size, reset our level fills
            if nbb != bb: qbid = nbbsz or 0.0; fbid = 0.0
            if nba != ba: qask = nbasz or 0.0; fsell = 0.0
            bb,bbsz,ba,basz = nbb,nbbsz,nba,nbasz
            if bb is not None: last_bb=bb
            if ba is not None: last_ba=ba
        else:
            side, px, sz = p
            if side == "SELL" and bb is not None and px <= bb + EPS:
                # consumes bid queue ahead, then us
                if sz <= qbid: qbid -= sz
                else:
                    over = sz - qbid; qbid = 0.0
                    fill = min(over, S - fbid)
                    if fill > 0:
                        fbid += fill; inv += fill; cash -= fill*bb
                        fees += fill*px*fee_frac(px); buyv += fill
            elif side == "BUY" and ba is not None and px >= ba - EPS:
                if sz <= qask: qask -= sz
                else:
                    over = sz - qask; qask = 0.0
                    fill = min(over, S - fsell)
                    if fill > 0:
                        fsell += fill; inv -= fill; cash += fill*ba
                        fees += fill*px*fee_frac(px); sellv += fill
    # flatten residual by CROSSING (spread + taker fee)
    if abs(inv) > 1e-6 and last_bb is not None and last_ba is not None:
        if inv > 0:   # long -> sell at bid
            px=last_bb; cash += inv*px - inv*px*fee_frac(px); flatten_cost_tot += inv*((last_ba+last_bb)/2-px)+inv*px*fee_frac(px)
        else:         # short -> buy at ask
            px=last_ba; cash += inv*px - abs(inv)*px*fee_frac(px); flatten_cost_tot += abs(inv)*(px-(last_ba+last_bb)/2)+abs(inv)*px*fee_frac(px)
        resid_abs += abs(inv); inv=0.0
    tot_pnl += cash; tot_fees += fees; buy=buyv; sell=sellv
    tot_buy += buyv; tot_sell += sellv; tot_fillvol += buyv+sellv; n_tok += 1

rebate = 0.20 * tot_fees
net = tot_pnl + rebate
fv = tot_fillvol
print("=== TWO-SIDED MM SIM (queue-aware fills + flatten-by-crossing cost) ===")
print(f"quote size S={S:.0f}/side  tokens simulated={n_tok}")
print(f"filled volume: {fv:.0f} sh  (buys {tot_buy:.0f} / sells {tot_sell:.0f})  residual flattened {resid_abs:.0f} sh")
print(f"  [front-of-queue model earlier filled 135k sh -> queue-aware fills far less]")
print(f"trading PnL (incl flatten cost): ${tot_pnl:+.2f}   flatten cost paid: ${flatten_cost_tot:.2f}")
print(f"maker rebate (20% crypto): ${rebate:+.2f}")
print(f"NET PnL: ${net:+.2f}  over {fv:.0f} sh = {(net/fv*100 if fv else 0):+.4f}%/sh   [>0 = viable]")
print(f"   gross markout/sh (pre-rebate): {(tot_pnl/fv*100 if fv else 0):+.4f}%   rebate/sh: {(rebate/fv*100 if fv else 0):+.4f}%")
