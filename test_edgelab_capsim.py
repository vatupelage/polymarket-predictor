import numpy as np
from edgelab import capsim

def test_bps_returns_basic():
    t = np.array([0.0, 0.4, 0.8, 1.2])
    mid = np.array([100.0, 100.0, 101.0, 101.0])  # +1% = +100 bps by t=0.8
    r = capsim.bps_returns(t, mid, lookback_s=0.5)
    assert np.isnan(r[0])                       # nothing 0.5s before t=0
    assert abs(r[2] - 100.0) < 1e-6             # 101/100-1 over [0.3,0.8] -> 100bps
    assert abs(r[3] - 100.0) < 1e-6

def test_momentum_triggers_threshold_and_cooldown():
    t = np.array([0.0, 0.4, 0.8, 1.2, 5.0])
    mid = np.array([100.0, 100.0, 101.0, 102.0, 100.0])
    trg = capsim.momentum_triggers(t, mid, theta_bps=50.0, lookback_s=0.5, cooldown_s=0.5)
    # first cross at t=0.8 (+100bps); t=1.2 within cooldown -> suppressed; t=5.0 is a drop
    assert trg[0][0] == 0.8 and trg[0][1] == +1
    assert all(abs(a[0] - 0.8) > 0.49 for a in trg[1:])   # no re-fire inside cooldown
    assert any(d == -1 for _, d in trg)                   # the drop fires a -1

def test_random_triggers_deterministic_and_bounded():
    a = capsim.random_triggers(10.0, 20.0, n=5, seed=7)
    b = capsim.random_triggers(10.0, 20.0, n=5, seed=7)
    assert a == b and len(a) == 5
    assert all(10.0 <= ti < 20.0 and di in (-1, 1) for ti, di in a)
    assert a == sorted(a)

def test_polarity_up_and_down_token():
    ta = np.arange(0, 10, 0.1); spot = 100 + np.sin(ta)        # oscillating spot
    up = 0.5 + 0.01 * np.sin(ta)                                # moves WITH spot
    dn = 0.5 - 0.01 * np.sin(ta)                                # moves AGAINST spot
    assert capsim.polarity_from_levels(ta, spot, ta, up) == +1
    assert capsim.polarity_from_levels(ta, spot, ta, dn) == -1

def test_polarity_degenerate_returns_zero():
    ta = np.arange(0, 5, 0.1); spot = 100 + np.sin(ta)
    flat = np.full_like(ta, 0.5)                                # no variance
    assert capsim.polarity_from_levels(ta, spot, ta, flat) == 0
    assert capsim.polarity_from_levels(ta, spot, np.array([100.0]), np.array([0.5])) == 0

def test_fee_peaks_at_half():
    assert abs(capsim.fee_per_share(0.5) - 0.0175) < 1e-9     # 1.75% at the money
    assert capsim.fee_per_share(0.1) < capsim.fee_per_share(0.5)
    assert abs(capsim.fee_per_share(0.1) - capsim.fee_per_share(0.9)) < 1e-9  # symmetric

def test_hittable_ask_depth_caps():
    ta = np.array([0.0, 1.0, 2.0])
    ask = np.array([0.50, 0.51, 0.52])
    sz  = np.array([100.0, 3.0, 100.0])
    # arrival 1.5 -> last<=1.5 is index1: ask 0.51, size 3 shares; stake $5 wants 9.8 shares
    px, sh = capsim.hittable_ask(ta, ask, sz, arrival=1.5, stake=5.0)
    assert abs(px - 0.51) < 1e-9 and abs(sh - 3.0) < 1e-9     # capped by size
    # deep size -> capped by stake
    px2, sh2 = capsim.hittable_ask(ta, ask, np.array([100.,100.,100.]), 1.5, 5.0)
    assert abs(sh2 - 5.0/0.51) < 1e-6

def test_hittable_ask_no_book_returns_nan():
    ta = np.array([2.0]); ask = np.array([0.5]); sz = np.array([10.0])
    px, sh = capsim.hittable_ask(ta, ask, sz, arrival=1.0, stake=5.0)  # arrival before any book
    assert np.isnan(px) and sh == 0.0

def test_mark_value_uses_arrival_plus_H():
    tm = np.array([0.0, 1.0, 2.0]); mid = np.array([0.50, 0.55, 0.60])
    assert abs(capsim.mark_value(tm, mid, arrival=1.0, H=1.0) - 0.60) < 1e-9

def test_scalar_ffill_nan_value_returns_nan():
    assert np.isnan(capsim.scalar_ffill(np.array([0.0]), np.array([np.nan]), 0.0))
    assert np.isnan(capsim.scalar_ffill(np.array([1.0]), np.array([0.5]), 0.0))  # q before first

def test_hittable_ask_nan_ask_and_bad_size_return_nan():
    ta = np.array([0.0, 1.0])
    # explicit NaN ask at a reachable time
    px, sh = capsim.hittable_ask(ta, np.array([np.nan, np.nan]), np.array([10.0, 10.0]),
                                 arrival=1.5, stake=5.0)
    assert np.isnan(px) and sh == 0.0
    # zero size
    px, sh = capsim.hittable_ask(ta, np.array([0.5, 0.5]), np.array([0.0, 0.0]),
                                 arrival=1.5, stake=5.0)
    assert np.isnan(px) and sh == 0.0
    # NaN size
    px, sh = capsim.hittable_ask(ta, np.array([0.5, 0.5]), np.array([np.nan, np.nan]),
                                 arrival=1.5, stake=5.0)
    assert np.isnan(px) and sh == 0.0

def test_net_edge_entry_only_fee():
    # win at $1 bought at 0.5: edge = 1 - 0.5 - 0.0175 = 0.4825
    assert abs(capsim.net_edge_per_share(1.0, 0.5) - 0.4825) < 1e-9
    # loss at $0 bought at 0.5: edge = -0.5 - 0.0175
    assert abs(capsim.net_edge_per_share(0.0, 0.5) - (-0.5175)) < 1e-9
    # mark valuation uses the SAME entry-only fee (no second fee on the notional sale)
    assert abs(capsim.net_edge_per_share(0.55, 0.50) - (0.05 - 0.0175)) < 1e-9
