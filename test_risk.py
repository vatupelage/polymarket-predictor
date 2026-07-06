"""Tests for the real-balance stop-loss math (live_trader/risk.py).

The kill-switch protects real money, so its arithmetic is unit-tested in
isolation from any network/RPC. We verify: open-trade capital is added back
(cost already debited, payout not yet landed), the trip fires at exactly the
limit, and the limit sign is normalised."""

from live_trader import risk


def test_flat_when_nothing_happened():
    assert risk.realized_real_pnl(100.0, 100.0, 0, 5.0) == 0.0


def test_open_trade_cost_is_added_back():
    # balance down $5 purely because one trade's $5 cost was debited; its
    # outcome is still pending -> realized PnL is flat, not -$5.
    assert risk.realized_real_pnl(100.0, 95.0, 1, 5.0) == 0.0


def test_two_open_trades_added_back():
    assert risk.realized_real_pnl(100.0, 90.0, 2, 5.0) == 0.0


def test_real_loss_after_settlement():
    # one trade settled as a loss: balance down $5, no open trades -> -$5.
    assert risk.realized_real_pnl(100.0, 95.0, 0, 5.0) == -5.0


def test_trip_fires_exactly_at_limit():
    tripped, pnl = risk.stop_loss_tripped(100.0, 50.0, 0, 5.0, 50.0)
    assert tripped is True
    assert pnl == -50.0


def test_no_trip_just_above_limit():
    tripped, pnl = risk.stop_loss_tripped(100.0, 51.0, 0, 5.0, 50.0)
    assert tripped is False
    assert pnl == -49.0


def test_open_trade_keeps_us_alive_near_limit():
    # stable shows -$55 but $5 of that is one open trade's cost -> real -$50.
    tripped, pnl = risk.stop_loss_tripped(100.0, 45.0, 1, 5.0, 50.0)
    assert tripped is True
    assert pnl == -50.0


def test_limit_sign_normalised():
    # caller may pass -50 or 50; both mean "halt at 50 down".
    a, _ = risk.stop_loss_tripped(100.0, 49.0, 0, 5.0, 50.0)
    b, _ = risk.stop_loss_tripped(100.0, 49.0, 0, 5.0, -50.0)
    assert a is True and b is True
