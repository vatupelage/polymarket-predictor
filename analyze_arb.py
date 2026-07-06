#!/usr/bin/env python3
"""Analyze arb_history.jsonl from the arb executor (dry-run or live).

The decisive question this answers: how big is the gap between *detected* edge
and *fillable* edge? The May-22 -$25 loss happened because the bot acted on
detected edge while the actual fillable depth was tiny. This report quantifies
that gap and projects a realistic $/day BEFORE you flip to live.

Usage:
    python3 predictor/analyze_arb.py
    python3 predictor/analyze_arb.py /path/to/arb_history.jsonl
"""

import datetime
import json
import os
import statistics
import sys


def _load(path):
    if not os.path.exists(path):
        print(f"No arb history at {path}\n"
              f"Run the dry-run collector first:  python3 -u predictor/run_live_bot.py --arb")
        sys.exit(0)
    rows = []
    for ln in open(path):
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return rows


def _span_days(rows):
    ts = [r["ts"] for r in rows if "ts" in r]
    if not ts:
        return 0.0, None, None
    t0 = datetime.datetime.fromisoformat(min(ts))
    t1 = datetime.datetime.fromisoformat(max(ts))
    days = max((t1 - t0).total_seconds() / 86400.0, 1e-9)
    return days, min(ts)[:16], max(ts)[:16]


def _dist(vals, label, pct=True):
    if not vals:
        print(f"  {label}: (none)")
        return
    vals = sorted(vals)
    def q(p):
        return vals[min(len(vals) - 1, int(p * len(vals)))]
    fmt = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:.1f}")
    print(f"  {label}: median={fmt(statistics.median(vals))}  "
          f"p75={fmt(q(.75))}  p90={fmt(q(.90))}  max={fmt(max(vals))}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "arb_history.jsonl")
    rows = _load(path)
    errs = [r for r in rows if "loop_error" in r or "scan_error" in r]
    rtts = [r for r in rows if "rtt_ms_median" in r]      # startup network probes
    rows = [r for r in rows if "edge" in r]
    days, t0, t1 = _span_days(rows)

    print("=" * 64)
    print(f"ARB HISTORY ANALYSIS — {len(rows)} records over {days:.2f} days")
    print(f"  span: {t0}  ->  {t1}")
    if errs:
        print(f"  ⚠ {len(errs)} loop/scan errors logged (check arb_history.jsonl)")
    print("=" * 64)

    dry = [r for r in rows if r.get("mode") == "DRY"]
    live = [r for r in rows if r.get("mode") == "LIVE"]

    # ---------- DRY-RUN MEASUREMENT ----------
    if dry:
        acted = [r for r in dry if r.get("acted")]               # fillable opportunities
        thin = [r for r in dry if not r.get("acted") and r.get("reason") == "depth<min"]
        print(f"\n--- DRY-RUN MEASUREMENT ({len(dry)} detections) ---")
        print(f"  fillable opportunities (depth ok): {len(acted)}")
        print(f"  detected but TOO THIN to fill:     {len(thin)}   "
              f"<- the May-22 trap: edge present, depth absent")
        if dry:
            frac = 100 * len(thin) / len(dry)
            print(f"  => {frac:.0f}% of detected edges are NOT realistically capturable")

        print("\n  edge distribution:")
        _dist([r["edge"] for r in acted], "  fillable edges")
        _dist([r["edge"] for r in thin], "  too-thin edges ")
        _dist([r.get("fillable_size", 0) for r in acted], "  fillable size (shares)", pct=False)

        prof = [r.get("expected_profit", 0) for r in acted]
        if prof:
            total = sum(prof)
            print(f"\n  expected profit (fillable only, at detected prices):")
            print(f"    total over {days:.2f}d: ${total:.2f}   "
                  f"avg/opportunity: ${statistics.mean(prof):.3f}")
            print(f"    ** GROSS projection: ${total/days:.0f}/day **")
            print(f"       (upper bound — assumes you WIN every fill vs competing bots,")
            print(f"        both FOK legs always fill, and no fees. Real will be lower.)")
            # haircut illustration
            for win_rate in (0.5, 0.25):
                print(f"       at {win_rate:.0%} fill-race win rate: ~${total/days*win_rate:.0f}/day")

    # ---------- LIVE RESULTS ----------
    if live:
        locked = [r for r in live if r.get("result") == "ARB_LOCKED"]
        unwound = [r for r in live if r.get("result") in
                   ("ONE_LEG_UNWOUND", "LEG2_KILLED_UNWOUND")]  # both naming eras
        nofill = [r for r in live if r.get("result") in ("both_killed", "leg1_killed")]
        lat = [r["latency_ms"] for r in live if isinstance(r.get("latency_ms"), (int, float))]
        print(f"\n--- LIVE RESULTS ({len(live)} actions) ---")
        print(f"  ARB_LOCKED (both legs filled): {len(locked)}")
        if locked:
            lp = sum(r.get("locked_profit", 0) for r in locked)
            print(f"    realized locked profit: ${lp:+.2f}")
        print(f"  no fill (no position, no loss): {len(nofill)}")
        print(f"  ONE LEG -> UNWOUND (danger events): {len(unwound)}")
        if unwound:
            print(f"    ⚠ leg-miss — flattened automatically, small loss each. Check log.")
            print(f"    leg-miss rate: {100*len(unwound)/max(len(locked)+len(unwound),1):.0f}% "
                  f"of attempted pairs")
        if lat:
            lat_s = sorted(lat)
            def q(p): return lat_s[min(len(lat_s) - 1, int(p * len(lat_s)))]
            print(f"  latency (detect->fill, n={len(lat_s)}): median={q(.5)}ms  "
                  f"p90={q(.9)}ms  max={max(lat_s)}ms")
        else:
            print(f"  latency (detect->fill): NOT RECORDED — no numeric samples "
                  f"(REST path now timestamps detect_ts; restart to start measuring)")
        # net realized if locked_profit present (unwind losses not auto-summed —
        # they need on-chain reconciliation; flagged above)
        if locked:
            print(f"\n  NOTE: unwind losses are not auto-summed (need on-chain check).")
            print(f"        Net live edge = locked profit - unwind losses - fees.")

    # ---------- LATENCY & NETWORK ----------
    print("\n--- LATENCY & NETWORK (the leg-race verdict) ---")
    if rtts:
        meds = sorted(r["rtt_ms_median"] for r in rtts)
        last = rtts[-1]
        print(f"  network RTT to CLOB (GET /time), {len(rtts)} probe(s):")
        print(f"    latest: median={last['rtt_ms_median']}ms "
              f"(min={last.get('rtt_ms_min','?')} max={last.get('rtt_ms_max','?')})")
        print(f"    across runs: best={min(meds)}ms  worst={max(meds)}ms")
        r = last["rtt_ms_median"]
        if r > 100:
            print(f"  => {r}ms is HIGH. Each leg crosses the wire in ~{r}ms; the cheap")
            print(f"     quote is usually gone before your order lands. Co-locate to")
            print(f"     eu-west-1 (expect single-digit ms) before expecting to lock arbs.")
        elif r > 30:
            print(f"  => {r}ms is borderline. Some races winnable, many lost.")
        else:
            print(f"  => {r}ms is co-located-grade. If you still see 0 locked here,")
            print(f"     the edge is gone to faster bots — not a latency problem.")
    else:
        print("  No RTT probes logged yet. Restart the --arb bot to record one at")
        print("  startup (compare local vs eu-west-1 to justify co-location).")

    if not dry and not live:
        print("\n  No edge records yet — let the dry-run collector run longer.")

    print("\n" + "=" * 64)
    print("READ THIS: if 'too thin to fill' is most detections, the realistic")
    print("$/day is near the LOW end and live trading is marginal. If fillable")
    print("opportunities are frequent with healthy size, it's worth going live")
    print("(set BOT_ARB_DRY_RUN=false) with the existing risk caps.")
    print("=" * 64)


if __name__ == "__main__":
    main()
