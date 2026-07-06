# tests/test_db_decision.py
from live_trader.db_decision import db_decision


def _kw(**over):
    base = dict(p_up=0.62, top_ask_up=0.42, top_ask_down=0.60,
                target_ask=0.45, max_ask=0.50, min_conf=0.10, fee_buffer=0.01)
    base.update(over)
    return base


def test_picks_up_when_p_up_above_half():
    d = db_decision(**_kw())
    assert d["side"] == "UP"
    assert abs(d["p_side"] - 0.62) < 1e-9
    assert d["ask"] == 0.42


def test_eligible_and_buy_now_when_below_target():
    d = db_decision(**_kw(top_ask_up=0.44))  # 0.44 <= target 0.45, edge 0.18 > buffer
    assert d["eligible"] is True
    assert d["buy_now"] is True


def test_eligible_not_buy_now_between_target_and_ceiling():
    d = db_decision(**_kw(top_ask_up=0.48))  # 0.45 < 0.48 < 0.50
    assert d["eligible"] is True
    assert d["buy_now"] is False


def test_not_eligible_at_or_above_ceiling():
    d = db_decision(**_kw(top_ask_up=0.50))  # not < max_ask
    assert d["eligible"] is False
    assert d["buy_now"] is False


def test_not_eligible_when_edge_below_buffer():
    # p_up 0.505 -> p_side 0.505, ask 0.50 would be > max anyway; use ask 0.49
    d = db_decision(**_kw(p_up=0.495, top_ask_up=0.49, top_ask_down=0.49))
    # side flips to DOWN (p_up<0.5); p_side=0.505, ask_down=0.49, edge=0.015>buffer,
    # but conf=|0.495-0.5|*2=0.01 < min_conf 0.10 -> not eligible
    assert d["side"] == "DOWN"
    assert d["eligible"] is False


def test_down_side_uses_down_ask():
    d = db_decision(**_kw(p_up=0.30, top_ask_up=0.80, top_ask_down=0.22))
    assert d["side"] == "DOWN"
    assert abs(d["p_side"] - 0.70) < 1e-9
    assert d["ask"] == 0.22
    assert d["eligible"] is True   # 0.22<0.50, edge 0.48>buffer, conf 0.40>min
    assert d["buy_now"] is True    # 0.22<=0.45


def test_missing_ask_not_eligible():
    d = db_decision(**_kw(top_ask_up=None))
    assert d["eligible"] is False
    assert d["buy_now"] is False
