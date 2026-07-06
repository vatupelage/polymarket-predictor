import numpy as np
from edgelab import capsim_run

def _data():
    bt = np.arange(0.0, 10.0, 0.1)
    bmid = 100.0 + (bt >= 5.0) * 1.0                      # +1% step up at t=5
    up = {"t": bt.copy(), "ask": np.full_like(bt, 0.50),
          "ask_sz": np.full_like(bt, 100.0),
          "mid": 0.50 + (bt >= 5.3) * 0.05}              # clob reprices up at 5.3 (laggy)
    return {"bt": bt, "bmid": bmid, "assets": {"UP": up}, "ot": np.array([]),
            "oval": np.array([])}

def test_asset_polarity_detects_up():
    d = _data()
    assert capsim_run.asset_polarity(d) == {"UP": +1}

def test_assemble_fills_buys_predicted_side_before_reprice():
    d = _data()
    pol = {"UP": +1}
    trg = [(5.05, +1)]                                    # detected the step up
    outcomes = {"UP": 1}                                  # market resolved up (won)
    fills = capsim_run.assemble_fills(d, pol, trg, R=0.1, stake=5.0, H=1.0,
                                      outcomes=outcomes)
    assert len(fills) == 1
    f = fills[0]
    assert f["asset"] == "UP" and abs(f["ask"] - 0.50) < 1e-9
    # bought at 0.50 before reprice; mark at arrival+H=6.15 -> mid 0.55
    assert abs(f["mark_net"] - (0.55 - 0.50 - 0.0175)) < 1e-9
    # resolution: won -> 1 - 0.50 - 0.0175
    assert abs(f["res_net"] - (1.0 - 0.50 - 0.0175)) < 1e-9

def test_assemble_fills_skips_wrong_polarity_and_no_book():
    d = _data()
    fills = capsim_run.assemble_fills(d, {"UP": +1}, [(5.05, -1)], R=0.1, stake=5.0,
                                      H=1.0, outcomes={})
    assert fills == []                                    # trigger dir -1 != polarity +1

def test_summarize_and_days_to_N():
    fills = [{"ask": 0.5, "mark_net": 0.03, "res_net": 0.48},
             {"ask": 0.5, "mark_net": -0.01, "res_net": -0.52},
             {"ask": 0.5, "mark_net": 0.02, "res_net": None}]
    s = capsim_run.summarize(fills)
    assert s["n"] == 3 and s["n_resolved"] == 2
    assert abs(s["res_median"] - (-0.02)) < 1e-9          # median of [0.48,-0.52]
    assert abs(s["res_frac_pos"] - 0.5) < 1e-9
    assert capsim_run.days_to_N(0, 3600) == float("inf")
    assert abs(capsim_run.days_to_N(30, 86400) - 1.0) < 1e-6   # 30/day -> 1 day

def test_run_sweep_has_momentum_and_random_cells():
    bt = np.arange(0.0, 60.0, 0.1)
    bmid = 100.0 + np.floor(bt / 10.0)                    # periodic up-steps
    up = {"t": bt.copy(), "ask": np.full_like(bt, 0.5),
          "ask_sz": np.full_like(bt, 100.0), "mid": 0.5 + 0.01*np.floor(bt/10.0)}
    data = {"bt": bt, "bmid": bmid, "assets": {"UP": up}, "ot": np.array([]),
            "oval": np.array([])}
    pol = {"UP": +1}
    rep = capsim_run.run_sweep(data, pol, outcomes={"UP": 1}, thetas=[50.0], Rs=[0.1],
                               Hs=[1.0], stake=5.0, lookback_s=0.5, cooldown_s=0.5, seed=1)
    c = rep["cells"][0]
    assert c["theta"] == 50.0 and c["R"] == 0.1
    assert "momentum" in c and "random" in c and "fee_bar" in c and "days_to_N" in c
    assert c["momentum"]["n"] >= 1

def test_print_report_leads_with_resolution(capsys):
    report = {"span_s": 3600.0, "cells": [{
        "theta": 50.0, "R": 0.1, "H": 1.0, "fee_bar": 0.0525, "days_to_N": 2.5,
        "momentum": {"n": 12, "n_resolved": 4, "res_median": -0.03,
                     "res_frac_pos": 0.5, "mark_median": 0.02, "median_ask": 0.5},
        "random": {"n": 12, "n_resolved": 4, "res_median": -0.04,
                   "res_frac_pos": 0.5, "mark_median": 0.018, "median_ask": 0.5}}]}
    capsim_run.print_report(report)
    out = capsys.readouterr().out
    assert "n_resolved=4" in out          # leads with the honest sample size
    assert "res_median" in out and "fee_bar" in out and "days_to_N" in out

def test_small_bps_move_fires_trigger():
    from edgelab import capsim
    # a clean +2.5bps step over 0.5s must produce >=1 trigger at theta=2bps
    t = np.arange(0.0, 5.0, 0.1)
    mid = 100.0 * (1.0 + (t >= 2.0) * 0.00025)   # +2.5 bps step at t=2.0
    trg = capsim.momentum_triggers(t, mid, theta_bps=2.0, lookback_s=0.5, cooldown_s=0.5)
    assert len(trg) >= 1 and trg[0][1] == +1
