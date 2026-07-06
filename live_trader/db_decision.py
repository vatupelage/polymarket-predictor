# live_trader/db_decision.py
"""Pure decision gate for the dollar-bar PTB contrarian strategy.

Given the model's calibrated P(close>PTB) and the live PM asks, choose the
predicted side and decide whether to buy. We only fire when the predicted side's
ask is below the ceiling AND the model's probability beats the ask by more than
the fee buffer AND model confidence clears a floor. `buy_now` is True once the
ask has dipped to the aspirational target; the caller adds the deadline rule.
"""
from __future__ import annotations


def db_decision(*, p_up, top_ask_up, top_ask_down, target_ask, max_ask,
                min_conf, fee_buffer):
    side = "UP" if p_up >= 0.5 else "DOWN"
    p_side = p_up if side == "UP" else (1.0 - p_up)
    ask = top_ask_up if side == "UP" else top_ask_down
    conf = abs(p_up - 0.5) * 2.0
    edge = (p_side - ask) if ask is not None else None
    eligible = (
        ask is not None
        and ask < max_ask
        and edge is not None and edge > fee_buffer
        and conf >= min_conf
    )
    buy_now = bool(eligible and ask <= target_ask)
    return {
        "side": side, "p_side": p_side, "conf": conf, "ask": ask,
        "edge": edge, "eligible": bool(eligible), "buy_now": buy_now,
    }


def entry_decision(ask, secs_to_close, max_ask, deadline_s):
    """Patient cheap-entry gate for the dbmodel bot. Side is already chosen by the
    model; this decides WHEN to act on the chosen token's current `ask`.

    Returns:
      "enter" - ask is known and <= max_ask (buy now, via FOK limit at max_ask)
      "skip"  - past the deadline with no cheap entry (give up this window)
      "wait"  - otherwise (keep polling)

    With max_ask >= 1.0 any real ask triggers "enter" on the first poll, so the
    standard bot's immediate entry-at-decision behavior is preserved.
    """
    if ask is not None and ask <= max_ask:
        return "enter"
    if secs_to_close <= deadline_s:
        return "skip"
    return "wait"
