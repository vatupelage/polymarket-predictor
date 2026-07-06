"""One-shot approvals for Polymarket CLOB V2 exchanges.

After Polymarket's V2 cutover (April 28 2026), the new exchange contracts
need fresh ERC-20 and ERC-1155 approvals. This script grants them.

What it does:
  - Reads POLY_PRIVATE_KEY from .env
  - Checks current allowances against both V2 exchanges
  - Sends only the approvals that are missing (idempotent — safe to re-run)
  - Verifies new state on-chain

Approvals granted (4 max):
  1. USDC.e.approve(V2_Binary,  MAX_UINT256)
  2. CTF.setApprovalForAll(V2_Binary,  true)
  3. USDC.e.approve(V2_NegRisk, MAX_UINT256)
  4. CTF.setApprovalForAll(V2_NegRisk, true)

Run from the predictor directory:
    python3 approve_v2.py
"""

import os
import sys
import time

from web3 import Web3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_trader.config import load_config  # noqa: E402

# Polygon mainnet contract addresses
USDCE  = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
CTF    = Web3.to_checksum_address("0x4D97DCd97eC945f40cF65F87097ACe5EA0476045")
EX_BIN = Web3.to_checksum_address("0xE111180000d2663C0091e4f400237545B87B996B")
EX_NR  = Web3.to_checksum_address("0xe2222d279d744050d28e00520010520000310F59")

MAX_UINT256 = (1 << 256) - 1

# RPCs in fallback order — 1rpc.io rate-limited last time, so try drpc first
RPCS = [
    "https://polygon.drpc.org",
    "https://polygon.llamarpc.com",
    "https://polygon-rpc.com",
    "https://1rpc.io/matic",
]

ERC20_ABI = [
    {"name": "allowance", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"},
                {"name": "spender", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "balanceOf", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "a", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"name": "approve", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "spender", "type": "address"},
                {"name": "amount", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}]},
]

ERC1155_ABI = [
    {"name": "isApprovedForAll", "type": "function", "stateMutability": "view",
     "inputs": [{"name": "account", "type": "address"},
                {"name": "operator", "type": "address"}],
     "outputs": [{"name": "", "type": "bool"}]},
    {"name": "setApprovalForAll", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "operator", "type": "address"},
                {"name": "approved", "type": "bool"}],
     "outputs": []},
]


def connect():
    for rpc in RPCS:
        try:
            w = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 8}))
            if w.is_connected() and w.eth.block_number > 0:
                print(f"  RPC: {rpc}")
                return w
        except Exception:
            continue
    raise RuntimeError("no Polygon RPC reachable")


def send(w3, acct, contract_fn, label):
    """Build, sign, send, wait. Returns receipt."""
    nonce = w3.eth.get_transaction_count(acct.address)
    gas_price = w3.eth.gas_price
    tx = contract_fn.build_transaction({
        "from": acct.address,
        "nonce": nonce,
        "chainId": 137,
        "gas": 120_000,
        "gasPrice": gas_price,
    })
    signed = acct.sign_transaction(tx)
    h = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"  -> {label}: tx {h.hex()}")
    rcpt = w3.eth.wait_for_transaction_receipt(h, timeout=180)
    status = "OK" if rcpt.status == 1 else "FAILED"
    print(f"     status={status}  gasUsed={rcpt.gasUsed}  block={rcpt.blockNumber}")
    if rcpt.status != 1:
        raise RuntimeError(f"{label} reverted")
    return rcpt


def main():
    dotenv = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    cfg = load_config(dotenv_path=dotenv)

    w3 = connect()
    acct = w3.eth.account.from_key(cfg.private_key)
    me = acct.address
    print(f"  Wallet: {me}")

    matic_bal = w3.eth.get_balance(me) / 1e18
    print(f"  MATIC balance: {matic_bal:.4f}")
    if matic_bal < 0.05:
        print("  WARNING: low MATIC for gas; you may need to top up before running.")

    usdc = w3.eth.contract(address=USDCE, abi=ERC20_ABI)
    ctf  = w3.eth.contract(address=CTF, abi=ERC1155_ABI)

    usdc_bal = usdc.functions.balanceOf(me).call() / 1e6
    print(f"  USDC.e balance: {usdc_bal:.6f}")

    # current allowance state
    targets = [("V2_Binary", EX_BIN), ("V2_NegRisk", EX_NR)]
    print()
    print("  Current allowances:")
    state = {}
    for name, addr in targets:
        usdc_allow = usdc.functions.allowance(me, addr).call()
        ctf_allow  = ctf.functions.isApprovedForAll(me, addr).call()
        state[addr] = (usdc_allow, ctf_allow)
        usdc_str = "MAX" if usdc_allow >= MAX_UINT256 // 2 else f"{usdc_allow/1e6:.6f} USDC"
        print(f"    {name:11} ({addr})")
        print(f"      USDC.e allowance: {usdc_str}")
        print(f"      CTF setApprovalForAll: {ctf_allow}")

    # send only what's missing
    todo = []
    for name, addr in targets:
        usdc_allow, ctf_allow = state[addr]
        if usdc_allow < MAX_UINT256 // 2:
            todo.append(("USDC.e.approve",      name, usdc.functions.approve(addr, MAX_UINT256)))
        if not ctf_allow:
            todo.append(("CTF.setApprovalForAll", name, ctf.functions.setApprovalForAll(addr, True)))

    if not todo:
        print()
        print("  All approvals already in place. Nothing to do.")
        return

    print()
    print(f"  Sending {len(todo)} approval transaction(s):")
    for fn_name, target_name, fn in todo:
        send(w3, acct, fn, f"{fn_name} -> {target_name}")
        # short pause so the next get_transaction_count picks up the new nonce
        time.sleep(1.5)

    # verify
    print()
    print("  Verifying new state:")
    for name, addr in targets:
        usdc_allow = usdc.functions.allowance(me, addr).call()
        ctf_allow  = ctf.functions.isApprovedForAll(me, addr).call()
        ok_usdc = usdc_allow >= MAX_UINT256 // 2
        ok_ctf  = ctf_allow
        print(f"    {name:11}: USDC.e={'MAX' if ok_usdc else 'NOT-MAX'}  CTF={ok_ctf}  "
              f"{'✓' if ok_usdc and ok_ctf else 'X'}")


if __name__ == "__main__":
    main()
