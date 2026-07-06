"""Tell Polymarket's CLOB to re-read on-chain allowances.

After granting V2 approvals on-chain, the CLOB's internal balance/allowance
cache is still stale. This script calls update_balance_allowance() for the
collateral (USDC.e) and conditional-token sides so the next order submission
sees the new state.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_trader.config import load_config  # noqa: E402

from py_clob_client_v2.client import ClobClient
from py_clob_client_v2.clob_types import ApiCreds, AssetType, BalanceAllowanceParams


def main():
    dotenv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    cfg = load_config(dotenv_path=dotenv)

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

    print("  Refreshing collateral (USDC.e) cache...")
    r1 = client.update_balance_allowance(BalanceAllowanceParams(
        asset_type=AssetType.COLLATERAL,
        signature_type=cfg.signature_type,
    ))
    print(f"    response: {r1}")

    print()
    print("  Reading current cached allowance (post-refresh):")
    cur = client.get_balance_allowance(BalanceAllowanceParams(
        asset_type=AssetType.COLLATERAL,
        signature_type=cfg.signature_type,
    ))
    print(f"    {cur}")


if __name__ == "__main__":
    main()
