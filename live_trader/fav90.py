"""Fav90 late-confirm strategy — pure decision helpers (no network, no I/O).

Strategy: at ~90s before a 5-min window closes, buy the FAVORITE side (the side
whose top ask is highest) when its ask sits in a near-certain band (~0.90) AND the
Polymarket order book shows real fillable depth at that ask. Hold to resolution.

These functions are intentionally pure so they can be unit-tested against fixture
books without a live client. bot.py wires them into the live poll loop.
"""
from __future__ import annotations


def pick_favorite(top_ask_up, top_ask_down):
    """Return (side, fav_ask) for the favorite (higher top ask), or (None, None)
    if either ask is missing. Ties resolve to UP."""
    if top_ask_up is None or top_ask_down is None:
        return None, None
    if top_ask_up >= top_ask_down:
        return "UP", top_ask_up
    return "DOWN", top_ask_down


def depth_within_tick(asks, tick=0.01):
    """Sum of resting share size on the ask side at/within `tick` of the best ask.

    `asks` is the order-book ask ladder as [[price, size], ...] sorted low->high
    (the shape returned by PolymarketClient.get_full_book). Returns 0.0 for an
    empty/None ladder. This measures how many shares we could buy at ~the quoted
    price before walking the book — i.e. whether the quote is genuinely fillable.
    """
    if not asks:
        return 0.0
    best = asks[0][0]
    total = 0.0
    for price, size in asks:
        if price <= best + tick + 1e-9:
            total += size
        else:
            break
    return total


def fav90_decision(*, secs_to_close, top_ask_up, top_ask_down, fav_asks,
                   entry_max_s, min_ask, max_ask, min_depth, tick=0.01):
    """Decide whether to fire a fav90 entry on this poll. Pure.

    fav_asks: the favorite token's ask ladder [[price,size],...] low->high
              (caller fetches it for the side `pick_favorite` chose).
    Returns a dict describing every gate so the caller can log all polls:
      {side, fav_ask, depth, timing_ok, price_ok, depth_ok, fire}
    `fire` is True iff all three gates pass.
    """
    side, fav_ask = pick_favorite(top_ask_up, top_ask_down)
    timing_ok = secs_to_close is not None and secs_to_close <= entry_max_s
    price_ok = fav_ask is not None and (min_ask <= fav_ask <= max_ask)
    depth = depth_within_tick(fav_asks, tick=tick)
    depth_ok = depth >= min_depth
    return {
        "side": side,
        "fav_ask": fav_ask,
        "depth": round(depth, 2),
        "timing_ok": bool(timing_ok),
        "price_ok": bool(price_ok),
        "depth_ok": bool(depth_ok),
        "fire": bool(timing_ok and price_ok and depth_ok and side is not None),
    }
