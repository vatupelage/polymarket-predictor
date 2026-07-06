#!/usr/bin/env python3
"""Analyze the dbmodel paper-trading log (dbmodel_paper.jsonl).

The paper run logs one rich record per 5-min window: the full feature vector,
raw+calibrated proba, both asks, a hypothetical fill at the snapshot ask, and the
realized outcome settled by Polymarket gamma (ground truth) + a Binance close
cross-check. This script summarizes it and replays a few candidate gates so we can
see whether ANY variant of the strategy has an edge — without re-collecting data.

  python3 analyze_dbmodel_paper.py [path]   # default: ./dbmodel_paper.jsonl
"""
import json
import os
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows, label):
    """Print n / hit-rate / total & mean PnL for a subset (settled rows only)."""
    settled = [r for r in rows if r.get("won") is not None and r.get("pnl") is not None]
    n = len(settled)
    if n == 0:
        print(f"  {label:<34} n=0")
        return
    wins = sum(1 for r in settled if r["won"])
    pnl = sum(r["pnl"] for r in settled)
    print(f"  {label:<34} n={n:>4}  hit={wins/n:6.1%}  "
          f"pnl=${pnl:+8.3f}  mean=${pnl/n:+.4f}")


def bucket(rows, keyfn, label, order=None):
    print(f"\n{label}")
    groups = defaultdict(list)
    for r in rows:
        k = keyfn(r)
        if k is not None:
            groups[k].append(r)
    keys = order if order else sorted(groups)
    for k in keys:
        if k in groups:
            summarize(groups[k], f"  {k}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "dbmodel_paper.jsonl"
    if not os.path.exists(path):
        print(f"no log at {path}"); sys.exit(1)
    rows = load(path)
    settled = [r for r in rows if r.get("won") is not None]
    print(f"loaded {len(rows)} records ({len(settled)} settled) from {path}\n")

    # data-quality: gamma vs binance agreement
    checked = [r for r in rows if r.get("agree") is not None]
    if checked:
        agree = sum(1 for r in checked if r["agree"])
        print(f"gamma vs binance agreement: {agree}/{len(checked)} = {agree/len(checked):.1%}")

    print("\n=== OVERALL (bet model side at market, hold to resolution) ===")
    summarize(rows, "all")

    # buy-low thesis: only the chosen side priced cheap
    bucket(rows, lambda r: ("ask<0.50" if (r.get("our_ask") is not None and r["our_ask"] < 0.50)
                            else ("ask>=0.50" if r.get("our_ask") is not None else None)),
           "=== by ask level on chosen side (buy-low thesis) ===")

    # confidence buckets
    def cbucket(r):
        c = r.get("confidence")
        if c is None:
            return None
        for lo in (0, 20, 40, 60, 80):
            if c < lo + 20:
                return f"conf[{lo:>2}-{lo+20})"
        return "conf[80-100]"
    bucket(rows, cbucket, "=== by confidence |p_up-0.5|*200 ===",
           order=["conf[ 0-20)", "conf[20-40)", "conf[40-60)", "conf[60-80)", "conf[80-100]"])

    # drift sign: did the model bet WITH the early-window drift (continuation)?
    def dbucket(r):
        d, dir_ = r.get("drift_pct"), r.get("direction")
        if d is None or dir_ is None:
            return None
        with_drift = (dir_ == "UP" and d > 0) or (dir_ == "DOWN" and d < 0)
        return "continuation (bet w/ drift)" if with_drift else "reversion (bet vs drift)"
    bucket(rows, dbucket, "=== by drift alignment ===")

    # session hour (UTC)
    bucket(rows, lambda r: f"h{r['session_hour']:02d}" if r.get("session_hour") is not None else None,
           "=== by UTC hour ===")

    # candidate gate replay: combine buy-low + a confidence floor
    print("\n=== candidate gate replay (subset PnL) ===")
    def gate(r, max_ask, min_conf):
        a, c = r.get("our_ask"), r.get("confidence")
        return a is not None and c is not None and a <= max_ask and c >= min_conf
    for max_ask in (0.50, 0.45, 0.40):
        for min_conf in (0, 20, 40):
            sub = [r for r in rows if gate(r, max_ask, min_conf)]
            summarize(sub, f"ask<={max_ask:.2f} & conf>={min_conf}")


if __name__ == "__main__":
    main()
