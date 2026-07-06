"""One-shot ERC20 approvals for Polymarket trading.

Approves USDC.e spending for the three Polymarket exchange contracts, and
also sets CTF (ConditionalTokens) approval-for-all so SELL/redeem works
later. Safe to run multiple times — if already at max, the on-chain tx
still succeeds (it just rewrites the same allowance).

Run once per EOA after funding the wallet with USDC.e and POL.
"""

import os
import sys

from web3 import Web3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live_trader.config import load_config  # noqa: E402


USDCE = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
CTF = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")

EXCHANGES = {
    "CTF Exchange":          "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    "Neg-Risk CTF Exchange": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
    "Neg-Risk Adapter":      "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296",
}

ERC20_ABI = [
    {"name":"approve","type":"function","stateMutability":"nonpayable",
     "inputs":[{"name":"s","type":"address"},{"name":"v","type":"uint256"}],
     "outputs":[{"name":"","type":"bool"}]},
    {"name":"allowance","type":"function","stateMutability":"view",
     "inputs":[{"name":"o","type":"address"},{"name":"s","type":"address"}],
     "outputs":[{"name":"","type":"uint256"}]},
]

CTF_ABI = [
    {"name":"setApprovalForAll","type":"function","stateMutability":"nonpayable",
     "inputs":[{"name":"op","type":"address"},{"name":"ap","type":"bool"}],
     "outputs":[]},
    {"name":"isApprovedForAll","type":"function","stateMutability":"view",
     "inputs":[{"name":"o","type":"address"},{"name":"op","type":"address"}],
     "outputs":[{"name":"","type":"bool"}]},
]

MAX_UINT = 2**256 - 1
RPC = "https://1rpc.io/matic"


def main():
    dotenv = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    cfg = load_config(dotenv_path=dotenv)
    w3 = Web3(Web3.HTTPProvider(RPC))
    acct = w3.eth.account.from_key(cfg.private_key)
    owner = acct.address
    print(f"Signer: {owner}")
    print(f"POL:    {w3.from_wei(w3.eth.get_balance(owner), 'ether')}")

    usdce = w3.eth.contract(address=USDCE, abi=ERC20_ABI)
    ctf = w3.eth.contract(address=CTF, abi=CTF_ABI)

    bal = usdce.functions.allowance(owner, owner).call()  # dummy, just to confirm contract
    print(f"USDC.e contract OK (self-allowance: {bal})")

    nonce = w3.eth.get_transaction_count(owner)
    gas_price = w3.eth.gas_price

    for name, addr in EXCHANGES.items():
        spender = Web3.to_checksum_address(addr)
        current = usdce.functions.allowance(owner, spender).call()
        if current >= MAX_UINT // 2:
            print(f"  [skip] USDC.e -> {name}: already approved ({current})")
        else:
            tx = usdce.functions.approve(spender, MAX_UINT).build_transaction({
                "from": owner, "nonce": nonce, "chainId": 137,
                "gas": 80_000, "gasPrice": gas_price,
            })
            signed = acct.sign_transaction(tx)
            h = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  USDC.e.approve({name}) -> {h.hex()}")
            w3.eth.wait_for_transaction_receipt(h, timeout=120)
            nonce += 1

        already = ctf.functions.isApprovedForAll(owner, spender).call()
        if already:
            print(f"  [skip] CTF setApprovalForAll -> {name}: already true")
        else:
            tx = ctf.functions.setApprovalForAll(spender, True).build_transaction({
                "from": owner, "nonce": nonce, "chainId": 137,
                "gas": 100_000, "gasPrice": gas_price,
            })
            signed = acct.sign_transaction(tx)
            h = w3.eth.send_raw_transaction(signed.raw_transaction)
            print(f"  CTF.setApprovalForAll({name}) -> {h.hex()}")
            w3.eth.wait_for_transaction_receipt(h, timeout=120)
            nonce += 1

    print("All approvals set.")


if __name__ == "__main__":
    main()
