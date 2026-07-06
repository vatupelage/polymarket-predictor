"""Single-purpose $1 V2 market BUY test.

Picks the most liquid currently-open BTC 5m market, queries its order book,
attempts a $1 market BUY on the cheaper side (whichever has top_ask < 0.50,
so a $1 stake actually fills meaningfully), and prints the raw CLOB response.

No bot, no scheduling, no filters. Just: build order → sign → POST → print.

Run:
    python3 test_v2_trade.py
"""

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_trader.config import load_config

from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import (
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    MarketOrderArgs,
    OrderType,
)


def find_active_btc_5m_market():
    """Return (slug, token_yes, token_no) for the currently-open BTC 5m market."""
    now = int(time.time())
    for delta in (300, 600, 900):  # current window, next, next-next
        slug = f"btc-updown-5m-{(now // 300 + delta // 300) * 300}"
        try:
            req = urllib.request.Request(
                f"https://gamma-api.polymarket.com/events?slug={slug}",
                headers={"User-Agent": "Mozilla/5.0"},
            )
            data = json.load(urllib.request.urlopen(req, timeout=8))
            if not data:
                continue
            m = data[0]["markets"][0]
            if m.get("closed"):
                continue
            tids = json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"]
            return slug, str(tids[0]), str(tids[1])
        except Exception:
            continue
    raise RuntimeError("could not find an active BTC 5m market")


def main():
    dotenv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    cfg = load_config(dotenv_path=dotenv)

    slug, t_yes, t_no = find_active_btc_5m_market()
    print(f"  market: {slug}")
    print(f"  YES: {t_yes}")
    print(f"  NO:  {t_no}")

    client = ClobClient(
        host=cfg.clob_host,
        chain_id=cfg.chain_id,
        key=cfg.private_key,
        creds=ApiCreds(
            api_key=cfg.api_key,
            api_secret=cfg.api_secret,
            api_passphrase=cfg.api_passphrase,
        ),
        signature_type=cfg.signature_type,
        funder=cfg.funder_address,
    )
    print(f"  signer: {client.signer.address()}")
    print(f"  sig_type: {cfg.signature_type}  funder: {cfg.funder_address}")

    # CLOB-cached balance (often stale right after on-chain approvals)
    print()
    print("  CLOB cached COLLATERAL state:")
    bal = client.get_balance_allowance(BalanceAllowanceParams(
        asset_type=AssetType.COLLATERAL, signature_type=cfg.signature_type))
    print(f"    {bal}")

    # Pick the cheaper outcome by querying both books
    def best_ask(tid):
        b = client.get_order_book(tid)
        asks = (b.get("asks") if isinstance(b, dict) else getattr(b, "asks", None)) or []
        prices = []
        for a in asks:
            try:
                p = float(a["price"] if isinstance(a, dict) else a.price)
                s = float(a["size"] if isinstance(a, dict) else a.size)
                if s > 0:
                    prices.append(p)
            except Exception:
                continue
        return min(prices) if prices else None

    a_yes, a_no = best_ask(t_yes), best_ask(t_no)
    print()
    print(f"  top ask YES: {a_yes}")
    print(f"  top ask NO:  {a_no}")
    if a_yes is None and a_no is None:
        print("  no asks on either side; abort.")
        return
    pick_token, pick_label, pick_ask = (
        (t_yes, "YES", a_yes) if (a_no is None or (a_yes is not None and a_yes <= a_no))
        else (t_no, "NO", a_no)
    )
    print(f"  picking {pick_label} at ~{pick_ask}")

    # Build & post $1 market BUY (FOK)
    args = MarketOrderArgs(token_id=pick_token, amount=1.0, side="BUY")
    print()
    print(f"  building order: token={pick_token[:12]}…  amount=$1.00  side=BUY")
    signed = client.create_market_order(args)
    print(f"  signed order built; posting…")

    try:
        resp = client.post_order(signed, OrderType.FAK)
        print()
        print("  CLOB response:")
        print(f"    {resp}")
    except Exception as e:
        print()
        print(f"  POST failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
