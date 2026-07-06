"""TDD for edgelab.ofi — Cont-Kukanov-Stoikov level-1 Order-Flow Imbalance.

Sign convention (canonical CKS): positive OFI = net buying pressure (price
should rise). Values below are hand-computed from the definition:

  bid:  P_b' > P_b -> e_b = Q_b'        ;  P_b' = P_b -> e_b = Q_b' - Q_b
        P_b' < P_b -> e_b = -Q_b
  ask:  P_a' > P_a -> e_a = -Q_a        ;  P_a' = P_a -> e_a = Q_a' - Q_a
        P_a' < P_a -> e_a = Q_a'
  OFI = e_b - e_a

A state is (bid_price, bid_size, ask_price, ask_size).
"""

from edgelab.ofi import ofi_increment, OFIAccumulator


def test_no_prior_state_is_zero():
    # first observation has no predecessor -> no flow
    assert ofi_increment(None, (0.45, 50, 0.55, 70)) == 0.0


def test_bid_price_up_adds_full_new_size():
    prev = (0.45, 50, 0.55, 70)
    cur = (0.46, 20, 0.55, 70)        # bid up; ask unchanged
    assert ofi_increment(prev, cur) == 20.0


def test_bid_same_price_size_grows():
    prev = (0.45, 50, 0.55, 70)
    cur = (0.45, 80, 0.55, 70)        # +30 bid depth
    assert ofi_increment(prev, cur) == 30.0


def test_bid_price_down_removes_old_size():
    prev = (0.45, 50, 0.55, 70)
    cur = (0.44, 90, 0.55, 70)        # bid retreated -> -Q_b
    assert ofi_increment(prev, cur) == -50.0


def test_ask_undercut_is_bearish():
    prev = (0.45, 50, 0.55, 70)
    cur = (0.45, 50, 0.54, 12)        # ask undercut -> e_a=+12 -> OFI=-12
    assert ofi_increment(prev, cur) == -12.0


def test_ask_lifted_is_bullish():
    prev = (0.45, 50, 0.55, 70)
    cur = (0.45, 50, 0.56, 5)         # ask retreated up -> e_a=-70 -> OFI=+70
    assert ofi_increment(prev, cur) == 70.0


def test_ask_depth_added_is_bearish():
    prev = (0.45, 50, 0.55, 70)
    cur = (0.45, 50, 0.55, 90)        # +20 ask depth -> OFI=-20
    assert ofi_increment(prev, cur) == -20.0


def test_combined_bid_up_ask_down():
    prev = (0.45, 50, 0.55, 70)
    cur = (0.46, 20, 0.54, 12)        # e_b=20, e_a=12 -> OFI=8
    assert ofi_increment(prev, cur) == 8.0


def test_missing_side_contributes_zero():
    # ask side empty in cur -> ask flow term is 0, bid term still counts
    prev = (0.45, 50, 0.55, 70)
    cur = (0.46, 20, None, 0)
    assert ofi_increment(prev, cur) == 20.0


def test_accumulator_sums_increments_over_window():
    acc = OFIAccumulator()
    assert acc.update((0.45, 50, 0.55, 70)) == 0.0   # first -> 0
    acc.update((0.46, 20, 0.55, 70))                 # +20
    acc.update((0.46, 20, 0.54, 12))                 # -12
    assert acc.total == 8.0
