"""Multi-bot bankroll allocation simulator.

Given $200 shared capital and three dbmodel bots (btc_5m, bnb_15m, eth_15m),
solve for a fixed dollar stake per bot that maximizes 30-day median growth
subject to P(bankroll ever < 50% of start) < 5%, under a stressed (decayed) edge.

Spec: docs/superpowers/specs/2026-06-18-multibot-bankroll-allocation-design.md

Return model (per $1 staked, per trade):
    gross = (1/ask - 1) if won else -1
    fee   = 0.07 * (1 - ask)     # Polymarket crypto taker fee, entry only
    net   = gross - fee

Math kernels are pure and unit-tested (tests/test_sim_bankroll_allocation.py).
"""
import json
import math

import numpy as np

CRYPTO_FEE_RATE = 0.07     # Polymarket crypto-category taker feeRate (docs/trading/fees)


# --------------------------------------------------------------------------
# Pure return-model kernels
# --------------------------------------------------------------------------
def taker_fee_frac(ask: float) -> float:
    """Taker fee as a fraction of stake for a fixed-$ buy at `ask`.

    Polymarket fee = shares * feeRate * p * (1-p); for stake/ask shares this
    collapses to feeRate * (1 - ask)."""
    return CRYPTO_FEE_RATE * (1.0 - ask)


def net_return(won: bool, ask: float) -> float:
    """Per-$1 net return of one trade: gross payoff minus the entry fee."""
    gross = (1.0 / ask - 1.0) if won else -1.0
    return gross - taker_fee_frac(ask)


def implied_cost_frac(won: bool, ask: float, actual_ret: float) -> float:
    """Cost (fee+slippage) as a fraction of stake, implied by a realized trade:
    theoretical gross-at-ask minus the actual per-$1 return."""
    gross = (1.0 / ask - 1.0) if won else -1.0
    return gross - actual_ret


def load_returns(path: str) -> list[float]:
    """Read a paper/trade jsonl into a list of per-$1 net returns.

    Skips unresolved rows (won is None) and rows without a chosen-side ask."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            won = r.get("won")
            ask = r.get("our_ask")
            if won is None or not ask:
                continue
            out.append(net_return(bool(won), float(ask)))
    return out


def kelly_fraction(returns: list[float], grid: int = 2000) -> float:
    """Growth-optimal bankroll fraction f* maximizing E[log(1 + f*r)].

    Constrained to f in [0, f_max) where f_max keeps 1 + f*min(r) > 0 so the
    log stays defined. Returns 0 when there is no positive-growth fraction."""
    r = np.asarray(returns, dtype=float)
    worst = r.min()
    if worst >= 0:          # never loses -> unbounded; cap at full bankroll
        return 1.0
    f_max = -1.0 / worst    # 1 + f*worst > 0  =>  f < -1/worst
    fs = np.linspace(0.0, f_max * 0.999, grid)
    # E[log(1 + f*r)] for each candidate f (vectorized over trades)
    growth = np.array([np.mean(np.log1p(f * r)) for f in fs])
    best = fs[int(np.argmax(growth))]
    return float(max(0.0, best))


def apply_decay(returns: list[float], factor: float = 0.5) -> list[float]:
    """Stress the edge: halve (factor=0.5) the residual mean via a parallel
    downward shift, preserving variance/shape."""
    arr = np.asarray(returns, dtype=float)
    mean = arr.mean()
    shift = mean * (1.0 - factor)
    return (arr - shift).tolist()


# --------------------------------------------------------------------------
# Monte-Carlo engine
# --------------------------------------------------------------------------
def simulate(stakes: dict, returns_by_bot: dict, rates: dict,
             bankroll: float, n_trades: int, n_paths: int, seed: int,
             dd_frac: float = 0.5) -> dict:
    """Simulate `n_paths` bankroll paths of `n_trades` i.i.d. interleaved trades.

    Each trade is assigned to a bot proportional to `rates`, draws a net return
    (bootstrap) from that bot's empirical distribution, and moves the bankroll by
    stake_bot * return. A trade is skipped if bankroll < its stake.

    Returns: median_final, mean_final, p_dd50 (P bankroll ever < dd_frac*start),
    p_profit, dd_p50/p90/p99 (max drawdown-from-start fraction percentiles)."""
    rng = np.random.default_rng(seed)
    bots = list(returns_by_bot.keys())
    weights = np.array([rates[b] for b in bots], dtype=float)
    weights = weights / weights.sum()
    stake_vec = np.array([stakes[b] for b in bots], dtype=float)
    pools = [np.asarray(returns_by_bot[b], dtype=float) for b in bots]

    bank = np.full(n_paths, float(bankroll))
    run_min = bank.copy()
    ruin_level = bankroll * dd_frac

    for _ in range(n_trades):
        who = rng.choice(len(bots), size=n_paths, p=weights)
        ret = np.empty(n_paths)
        for bi, pool in enumerate(pools):
            m = who == bi
            cnt = int(m.sum())
            if cnt:
                ret[m] = pool[rng.integers(0, len(pool), size=cnt)]
        stk = stake_vec[who]
        can_play = bank >= stk
        bank = bank + np.where(can_play, stk * ret, 0.0)
        run_min = np.minimum(run_min, bank)

    dd = np.clip((bankroll - run_min) / bankroll, 0.0, None)
    return {
        "median_final": float(np.median(bank)),
        "mean_final": float(np.mean(bank)),
        "p_dd50": float(np.mean(run_min < ruin_level)),
        "p_profit": float(np.mean(bank > bankroll)),
        "dd_p50": float(np.percentile(dd, 50)),
        "dd_p90": float(np.percentile(dd, 90)),
        "dd_p99": float(np.percentile(dd, 99)),
    }


def optimize(returns_by_bot: dict, rates: dict, bankroll: float,
             n_trades: int, n_paths: int, target_p_dd: float,
             dd_frac: float, seed: int, weights: dict | None = None) -> dict:
    """Scale a weighted stake vector to the ruin boundary.

    Relative stakes default to each bot's Kelly fraction (on its own returns);
    pass `weights` to compare other allocation schemes at equal risk. A single
    multiplier `scale` is binary-searched so P(drawdown>dd_frac) == target_p_dd.
    Returns the resulting per-bot fixed-$ stakes, the scale, and the metrics."""
    bots = list(returns_by_bot.keys())
    kelly = {b: kelly_fraction(returns_by_bot[b]) for b in bots}
    w = kelly if weights is None else weights
    base = np.array([w[b] for b in bots])
    if base.sum() <= 0:
        return {"scale": 0.0, "stakes": {b: 0.0 for b in bots},
                "kelly": kelly, "weights": dict(w),
                "metrics": simulate({b: 0.0 for b in bots}, returns_by_bot,
                                    rates, bankroll, n_trades, n_paths, seed,
                                    dd_frac)}

    def stakes_for(scale):
        # scale * weight * bankroll = fixed $ stake per bot
        return {b: scale * w[b] * bankroll for b in bots}

    def p_dd(scale):
        return simulate(stakes_for(scale), returns_by_bot, rates, bankroll,
                        n_trades, n_paths, seed, dd_frac)["p_dd50"]

    lo, hi = 0.0, 1.0
    # expand hi until it exceeds the target (or cap it — full Kelly is the ceiling)
    while p_dd(hi) < target_p_dd and hi < 4.0:
        hi *= 2.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if p_dd(mid) < target_p_dd:
            lo = mid
        else:
            hi = mid
    scale = lo   # largest scale still under the constraint
    stakes = stakes_for(scale)
    metrics = simulate(stakes, returns_by_bot, rates, bankroll,
                       n_trades, n_paths, seed, dd_frac)
    return {"scale": scale, "stakes": stakes, "kelly": kelly,
            "weights": dict(w), "metrics": metrics}


# --------------------------------------------------------------------------
# Analysis driver (real data)
# --------------------------------------------------------------------------
def _chosen_ask(rec):
    """Chosen-side ask for a trade record: our_ask if present, else the
    direction-matched top ask (live trade_history rows)."""
    if rec.get("our_ask"):
        return float(rec["our_ask"])
    d = (rec.get("direction") or "").upper()
    a = rec.get("top_ask_up") if d == "UP" else rec.get("top_ask_down")
    return float(a) if a else None


def load_records(path):
    """Resolved trade records (won is not None) with a usable chosen ask + ts."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("won") is None:
                continue
            ask = _chosen_ask(r)
            if not ask:
                continue
            r["_ask"] = ask
            out.append(r)
    return out


def file_stats(path):
    """n, days span, trades/day, win rate, avg ask, mean net edge/$1."""
    from datetime import datetime
    recs = load_records(path)
    n = len(recs)
    asks = [r["_ask"] for r in recs]
    wins = sum(1 for r in recs if r["won"])
    nets = [net_return(bool(r["won"]), r["_ask"]) for r in recs]
    ts = sorted(datetime.fromisoformat(r["ts"]) for r in recs if r.get("ts"))
    days = max((ts[-1] - ts[0]).total_seconds() / 86400.0, 1e-9) if len(ts) > 1 else 1.0
    return {"n": n, "days": days, "per_day": n / days,
            "win_rate": wins / n if n else 0.0,
            "avg_ask": sum(asks) / n if n else 0.0,
            "mean_net": sum(nets) / n if n else 0.0}


def validate_fee_model(live_path, stake_filter=5.0):
    """Compare modeled fee 0.07*(1-ask) to the empirical cost gap on real fills.

    For each live trade: implied cost = theoretical gross-at-ask minus realized
    pnl/stake. Returns median implied cost vs median modeled fee."""
    import statistics
    recs = [r for r in load_records(live_path)
            if r.get("stake_usdc") == stake_filter and r.get("pnl") is not None]
    implied, modeled = [], []
    for r in recs:
        actual = float(r["pnl"]) / float(r["stake_usdc"])
        implied.append(implied_cost_frac(bool(r["won"]), r["_ask"], actual))
        modeled.append(taker_fee_frac(r["_ask"]))
    if not implied:
        return None
    return {"n": len(implied),
            "median_implied_cost": statistics.median(implied),
            "median_modeled_fee": statistics.median(modeled),
            "mean_implied_cost": statistics.mean(implied),
            "mean_modeled_fee": statistics.mean(modeled)}


def _fmt_metrics(m):
    return (f"median final ${m['median_final']:6.0f} | "
            f"P(profit) {m['p_profit']*100:4.1f}% | "
            f"P(dd>50%) {m['p_dd50']*100:4.1f}% | "
            f"dd P50/P90/P99 {m['dd_p50']*100:3.0f}/{m['dd_p90']*100:3.0f}/"
            f"{m['dd_p99']*100:3.0f}%")


def main():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(here, "sim_data")
    sources = {
        "btc_5m":  os.path.join(d, "btc_5m_dbmodel_paper.jsonl"),
        "bnb_15m": os.path.join(d, "paper_bnb_15m.jsonl"),
        "eth_15m": os.path.join(d, "paper_eth_15m.jsonl"),
    }
    BANKROLL, HORIZON_DAYS = 200.0, 30
    TARGET_PDD, DD_FRAC, SEED = 0.05, 0.5, 2026
    N_PATHS = 12000

    print("=" * 78)
    print("MULTI-BOT BANKROLL ALLOCATION  —  $200 shared, 30d horizon, no stop-loss")
    print("=" * 78)

    # --- fee-model validation (btc live fills) ---
    fv = validate_fee_model(os.path.join(d, "btc_5m_trade_history.jsonl"))
    print("\n[1] FEE-MODEL VALIDATION  (btc_5m live $5 fills)")
    if fv:
        print(f"    n={fv['n']}  empirical cost/stake median={fv['median_implied_cost']*100:+.2f}% "
              f"mean={fv['mean_implied_cost']*100:+.2f}%")
        print(f"              modeled 0.07*(1-ask) median={fv['median_modeled_fee']*100:+.2f}% "
              f"mean={fv['mean_modeled_fee']*100:+.2f}%")
        if abs(fv['mean_implied_cost'] - fv['mean_modeled_fee']) < 0.015:
            verdict = "MATCH — fee model confirmed by live fills"
        elif abs(fv['mean_implied_cost']) < 0.01:
            verdict = ("LOG IS FEE-BLIND — pnl booked at quoted ask, fee not "
                       "observable here; corroborated by real-money ~3% "
                       "(dbmodel_live_execution_cost). Keeping modeled fee.")
        else:
            verdict = "DIVERGES — prefer empirical cost over model"
        print(f"    -> {verdict}")

    # --- per-bot stats + return distributions ---
    print("\n[2] PER-BOT (paper, net of modeled fee)")
    rets, rates, stats = {}, {}, {}
    for b, p in sources.items():
        s = file_stats(p)
        stats[b] = s
        rets[b] = load_returns(p)
        rates[b] = s["per_day"]
        print(f"    {b:8s} n={s['n']:4d} {s['per_day']:5.0f}/day  win={s['win_rate']*100:4.1f}%  "
              f"ask={s['avg_ask']:.3f}  net edge/$1={s['mean_net']:+.4f}")
    total_day = sum(rates.values())
    n_trades = int(total_day * HORIZON_DAYS)
    print(f"    combined ~{total_day:.0f} trades/day  ->  {n_trades} trades over {HORIZON_DAYS}d")

    # --- stressed returns (halve residual edge) ---
    rets_stress = {b: apply_decay(rets[b], 0.5) for b in sources}

    print("\n[3] KELLY FRACTIONS  (full-Kelly, per bot)")
    for b in sources:
        print(f"    {b:8s} base f*={kelly_fraction(rets[b]):.3f}   "
              f"stressed f*={kelly_fraction(rets_stress[b]):.3f}")

    # --- recommendation: Kelly-weighted, scaled to 5% ruin under STRESS ---
    print("\n[4] RECOMMENDED SIZING  (Kelly-weighted, scaled to P(dd>50%)=5% under STRESSED edge)")
    rec = optimize(rets_stress, rates, BANKROLL, n_trades, N_PATHS,
                   TARGET_PDD, DD_FRAC, SEED)
    for b in sources:
        print(f"    {b:8s} stake = ${rec['stakes'][b]:5.2f}")
    print(f"    (scale on Kelly vector = {rec['scale']:.3f})")
    print(f"    STRESSED:  {_fmt_metrics(rec['metrics'])}")
    base_at_rec = simulate(rec["stakes"], rets, rates, BANKROLL, n_trades,
                           N_PATHS, SEED, DD_FRAC)
    print(f"    BASE edge: {_fmt_metrics(base_at_rec)}")

    # --- scheme comparison at equal risk (5% ruin, stressed) ---
    print("\n[5] ALLOCATION SCHEMES  (each scaled to 5% ruin under stress)")
    edge_w = {b: max(stats[b]["mean_net"], 0.0) for b in sources}
    eq_w = {b: 1.0 for b in sources}
    schemes = {"Kelly (rec)": None, "equal-$": eq_w, "edge-prop": edge_w}
    for name, w in schemes.items():
        r = optimize(rets_stress, rates, BANKROLL, n_trades, N_PATHS,
                     TARGET_PDD, DD_FRAC, SEED, weights=w)
        st = "  ".join(f"{b.split('_')[0]}=${r['stakes'][b]:.2f}" for b in sources)
        print(f"    {name:12s} [{st}]")
        print(f"                 {_fmt_metrics(r['metrics'])}")


if __name__ == "__main__":
    main()
