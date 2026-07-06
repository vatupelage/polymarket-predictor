"""Redeem resolved Polymarket positions held on the EOA directly to USDC.e.

Because our bot uses EOA signature mode (type 0), winning shares land on the
raw signer address, not the Polymarket proxy wallet. The Polymarket UI only
shows proxy-wallet positions, so these can't be redeemed via the UI — call
the CTF contract directly.

Even on V2 (pUSD-denominated trading), the underlying CTF collateral for
existing markets is still USDC.e — pUSD only sits on the exchange/onramp
layer. Redemption flows through the original collateral, so pass USDC.e
to redeemPositions. Payout lands as USDC.e on the EOA; wrap to pUSD via
the Onramp if you want it back in trading collateral.

Usage:
  python -m live_trader.redeem <slug>

Safe to re-run: if the position was already redeemed, the tx reverts harmlessly.
"""

import json
import os
import sys

import requests
from web3 import Web3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live_trader.config import load_config  # noqa: E402


CTF = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
USDCE = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
ZERO_BYTES32 = "0x" + "00" * 32
RPC = "https://polygon.drpc.org"

REDEEM_ABI = [
    {"name": "redeemPositions", "type": "function", "stateMutability": "nonpayable",
     "inputs": [
         {"name": "collateralToken", "type": "address"},
         {"name": "parentCollectionId", "type": "bytes32"},
         {"name": "conditionId", "type": "bytes32"},
         {"name": "indexSets", "type": "uint256[]"},
     ], "outputs": []},
]


def main():
    if len(sys.argv) < 2:
        print("usage: python -m live_trader.redeem <slug>")
        sys.exit(1)
    slug = sys.argv[1]

    dotenv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    cfg = load_config(dotenv_path=dotenv)

    r = requests.get(f"https://gamma-api.polymarket.com/events?slug={slug}", timeout=8).json()
    mkt = r[0]["markets"][0]
    condition_id = mkt["conditionId"]
    if not mkt.get("closed", False):
        print(f"Market not yet closed: {slug}")
        sys.exit(1)
    print(f"Slug: {slug}\n  conditionId: {condition_id}\n  outcomePrices: {mkt.get('outcomePrices')}")

    w3 = Web3(Web3.HTTPProvider(RPC))
    acct = w3.eth.account.from_key(cfg.private_key)
    ctf = w3.eth.contract(address=CTF, abi=REDEEM_ABI)

    tx = ctf.functions.redeemPositions(
        USDCE, ZERO_BYTES32, condition_id, [1, 2]
    ).build_transaction({
        "from": acct.address,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "chainId": 137,
        "gas": 250_000,
        "gasPrice": w3.eth.gas_price,
    })
    signed = acct.sign_transaction(tx)
    h = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"  tx: {h.hex()}")
    rcpt = w3.eth.wait_for_transaction_receipt(h, timeout=180)
    print(f"  status: {'OK' if rcpt.status == 1 else 'FAILED'}  gasUsed: {rcpt.gasUsed}")

    # verify by reading USDC.e balance (CTF redeems pay out in original collateral)
    bal = w3.eth.contract(address=USDCE, abi=[
        {"name":"balanceOf","type":"function","stateMutability":"view",
         "inputs":[{"name":"a","type":"address"}],
         "outputs":[{"name":"","type":"uint256"}]}
    ]).functions.balanceOf(acct.address).call()
    print(f"  USDC.e after redeem: {bal / 1e6:.6f}")


if __name__ == "__main__":
    main()
