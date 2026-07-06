"""Tests for the multi-bot bankroll allocation simulator (math kernels).

TDD-first: these define the contract for sim_bankroll_allocation.py.
Spec: docs/superpowers/specs/2026-06-18-multibot-bankroll-allocation-design.md
"""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sim_bankroll_allocation as sim


# ---- fee model: fee/stake = 0.07 * (1 - ask) ------------------------------
def test_taker_fee_frac_peaks_low_ask():
    assert sim.taker_fee_frac(0.50) == pytest.approx(0.035)
    assert sim.taker_fee_frac(0.60) == pytest.approx(0.028)
    assert sim.taker_fee_frac(0.90) == pytest.approx(0.007)


# ---- per-$1 net return = gross - fee --------------------------------------
def test_net_return_win_subtracts_fee():
    # win at 0.50: gross (1/0.5 - 1)=1.0, fee 0.035 -> 0.965
    assert sim.net_return(True, 0.50) == pytest.approx(0.965)


def test_net_return_loss_includes_fee():
    # loss: gross -1, fee 0.035 -> -1.035
    assert sim.net_return(False, 0.50) == pytest.approx(-1.035)


# ---- load_returns reads a paper jsonl into net per-$1 returns -------------
def test_load_returns_parses_won_and_ask(tmp_path):
    p = tmp_path / "paper.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"won": True, "our_ask": 0.50}) + "\n")
        f.write(json.dumps({"won": False, "our_ask": 0.50}) + "\n")
        f.write(json.dumps({"won": None, "our_ask": 0.50}) + "\n")   # unresolved -> skip
        f.write(json.dumps({"won": True}) + "\n")                     # no ask -> skip
    rets = sim.load_returns(str(p))
    assert len(rets) == 2
    assert rets[0] == pytest.approx(0.965)
    assert rets[1] == pytest.approx(-1.035)


# ---- implied cost: gap between theoretical gross and realized return -------
def test_implied_cost_frac_recovers_fee():
    # a win at 0.50 whose realized per-$1 return was 0.965 implies a 0.035 cost
    assert sim.implied_cost_frac(True, 0.50, 0.965) == pytest.approx(0.035)


def test_implied_cost_frac_loss():
    # a loss that returned exactly -1.0 (no extra cost beyond stake) -> 0 gap
    assert sim.implied_cost_frac(False, 0.50, -1.0) == pytest.approx(0.0)


# ---- Kelly fraction --------------------------------------------------------
def test_kelly_even_money_sixty_forty():
    # even-money 60/40 -> Kelly = p - q = 0.20
    rets = [1.0] * 60 + [-1.0] * 40
    assert sim.kelly_fraction(rets) == pytest.approx(0.20, abs=0.01)


def test_kelly_zero_when_no_edge():
    rets = [1.0] * 50 + [-1.0] * 50
    assert sim.kelly_fraction(rets) == pytest.approx(0.0, abs=0.01)


# ---- decay stress: halve the residual mean edge ---------------------------
def test_apply_decay_halves_mean():
    rets = [0.9, -1.0, 0.9, -1.0, 0.5]   # some mean m
    m = sum(rets) / len(rets)
    decayed = sim.apply_decay(rets, factor=0.5)
    assert sum(decayed) / len(decayed) == pytest.approx(m / 2)
    # variance preserved (parallel shift)
    def var(xs):
        mu = sum(xs) / len(xs)
        return sum((x - mu) ** 2 for x in xs) / len(xs)
    assert var(decayed) == pytest.approx(var(rets))


# ---- simulate --------------------------------------------------------------
def test_simulate_zero_stake_keeps_bankroll_flat():
    out = sim.simulate(
        stakes={"a": 0.0}, returns_by_bot={"a": [0.5, -1.0]},
        rates={"a": 1.0}, bankroll=200.0, n_trades=100, n_paths=500, seed=1)
    assert out["median_final"] == pytest.approx(200.0)
    assert out["p_dd50"] == pytest.approx(0.0)
    assert out["p_profit"] == pytest.approx(0.0)


def test_simulate_is_deterministic_with_seed():
    args = dict(stakes={"a": 1.0}, returns_by_bot={"a": [0.9, -1.0]},
                rates={"a": 1.0}, bankroll=200.0, n_trades=200, n_paths=500)
    a = sim.simulate(seed=42, **args)
    b = sim.simulate(seed=42, **args)
    assert a["median_final"] == b["median_final"]
    assert a["p_dd50"] == b["p_dd50"]


def test_simulate_positive_edge_grows_bankroll():
    # strongly +EV stream should grow median bankroll above start
    out = sim.simulate(
        stakes={"a": 2.0}, returns_by_bot={"a": [0.9] * 70 + [-1.0] * 30},
        rates={"a": 1.0}, bankroll=200.0, n_trades=300, n_paths=1000, seed=7)
    assert out["median_final"] > 200.0
    assert out["p_profit"] > 0.5


def test_simulate_ruin_metric_triggers_on_big_stake():
    # huge stake on a coin-flip-ish stream must drive some paths below $100
    out = sim.simulate(
        stakes={"a": 40.0}, returns_by_bot={"a": [0.9, -1.0]},
        rates={"a": 1.0}, bankroll=200.0, n_trades=300, n_paths=1000, seed=3)
    assert out["p_dd50"] > 0.0


# ---- optimize: scale to the ruin boundary ---------------------------------
def test_optimize_respects_ruin_constraint():
    returns_by_bot = {
        "hi": [0.9] * 65 + [-1.0] * 35,    # decent edge
        "lo": [0.9] * 55 + [-1.0] * 45,    # thin edge
    }
    rates = {"hi": 1.0, "lo": 1.0}
    rec = sim.optimize(returns_by_bot, rates, bankroll=200.0,
                       n_trades=300, n_paths=800, target_p_dd=0.05,
                       dd_frac=0.5, seed=11)
    # returned config's ruin prob is at or under target (within MC tolerance)
    assert rec["metrics"]["p_dd50"] <= 0.05 + 0.02
    # higher-edge bot gets the larger stake
    assert rec["stakes"]["hi"] >= rec["stakes"]["lo"]
    assert all(s >= 0 for s in rec["stakes"].values())
