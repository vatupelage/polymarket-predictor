"""Dependency-free pagination helper for the Polymarket positions API.

Kept in its own module (no web3/httpx imports) so the loop logic is unit-testable
in isolation. Used by PolymarketBotClient.sweep_orphan_winners to fetch ALL
redeemable positions — the data-api truncates a single request (~100 rows), and a
wallet with a big old-loser backlog would otherwise hide fresh small winners past
the cap.
"""


def paginate_all(fetch_page, page_size: int = 500, max_pages: int = 40) -> list:
    """Accumulate every row across pages.

    `fetch_page(offset)` returns the page starting at `offset`. Stops when a page
    is empty, not a list, or shorter than `page_size` (the last page), or when
    `max_pages` is reached (defensive bound against a misbehaving API).
    """
    out: list = []
    for _ in range(max_pages):
        batch = fetch_page(len(out))
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        if len(batch) < page_size:
            break
    return out
