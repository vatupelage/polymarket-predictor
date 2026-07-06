"""V2 collateral setup: wrap USDC.e -> pUSD, approve V2 exchanges to spend pUSD.

V2 uses pUSD (Polymarket USD), not USDC.e, as the collateral token. API-only
traders must wrap their USDC.e via the Collateral Onramp before they can place
V2 orders.

Steps (all idempotent — safe to re-run):
  1. Approve USDC.e -> CollateralOnramp (so it can pull funds for wrap)
  2. Call Onramp.wrap(USDC.e, our_addr, WRAP_AMOUNT) to mint pUSD
  3. Approve pUSD -> V2 Binary Exchange (MAX)
  4. Approve pUSD -> V2 NegRisk Exchange (MAX)
  5. Tell Polymarket's CLOB to re-read on-chain state
  6. Print final pUSD balance and CLOB-cached collateral state
"""

import os
import sys
import time

from web3 import Web3

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_trader.config import load_config  # noqa: E402

# Polygon mainnet
USDCE   = Web3.to_checksum_address("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174")
PUSD    = Web3.to_checksum_address("0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB")
ONRAMP  = Web3.to_checksum_address("0x93070a847efEf7F70739046A929D47a521F5B8ee")
EX_BIN  = Web3.to_checksum_address("0xE111180000d2663C0091e4f400237545B87B996B")
EX_NR   = Web3.to_checksum_address("0xe2222d279d744050d28e00520010520000310F59")

MAX_UINT256 = (1 << 256) - 1
WRAP_AMOUNT_USDC = 20.0   # how much USDC.e to wrap -> pUSD on this run

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

ONRAMP_ABI = [
    {"name": "wrap", "type": "function", "stateMutability": "nonpayable",
     "inputs": [{"name": "_asset", "type": "address"},
                {"name": "_to", "type": "address"},
                {"name": "_amount", "type": "uint256"}],
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


def send(w3, acct, contract_fn, label, gas=200_000):
    nonce = w3.eth.get_transaction_count(acct.address)
    tx = contract_fn.build_transaction({
        "from": acct.address, "nonce": nonce, "chainId": 137,
        "gas": gas, "gasPrice": w3.eth.gas_price,
    })
    signed = acct.sign_transaction(tx)
    h = w3.eth.send_raw_transaction(signed.raw_transaction)
    print(f"    -> {label}: tx {h.hex()}")
    rcpt = w3.eth.wait_for_transaction_receipt(h, timeout=180)
    status = "OK" if rcpt.status == 1 else "FAILED"
    print(f"       status={status}  gasUsed={rcpt.gasUsed}  block={rcpt.blockNumber}")
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
    print(f"  MATIC:  {w3.eth.get_balance(me)/1e18:.4f}")

    usdce  = w3.eth.contract(address=USDCE,  abi=ERC20_ABI)
    pusd   = w3.eth.contract(address=PUSD,   abi=ERC20_ABI)
    onramp = w3.eth.contract(address=ONRAMP, abi=ONRAMP_ABI)

    usdce_bal_raw = usdce.functions.balanceOf(me).call()
    pusd_bal_raw  = pusd.functions.balanceOf(me).call()
    print(f"  USDC.e: {usdce_bal_raw/1e6:.6f}")
    print(f"  pUSD:   {pusd_bal_raw/1e6:.6f}")

    wrap_amt_raw = int(WRAP_AMOUNT_USDC * 1e6)
    if usdce_bal_raw < wrap_amt_raw:
        print(f"\n  Insufficient USDC.e to wrap ${WRAP_AMOUNT_USDC}; have {usdce_bal_raw/1e6:.6f}")
        return

    # 1. USDC.e -> Onramp approval
    print("\n  [1/5] USDC.e -> Onramp allowance")
    cur = usdce.functions.allowance(me, ONRAMP).call()
    if cur < wrap_amt_raw:
        send(w3, acct, usdce.functions.approve(ONRAMP, MAX_UINT256),
             "USDC.e.approve(Onramp, MAX)")
        time.sleep(1.5)
    else:
        print(f"    already at {cur/1e6:.2f}; skipping")

    # 2. wrap USDC.e -> pUSD
    print(f"\n  [2/5] wrap ${WRAP_AMOUNT_USDC} USDC.e -> pUSD")
    send(w3, acct, onramp.functions.wrap(USDCE, me, wrap_amt_raw),
         f"Onramp.wrap(USDC.e, me, {WRAP_AMOUNT_USDC})", gas=400_000)
    time.sleep(1.5)

    # 3+4. approve pUSD to both V2 exchanges
    for label, addr in (("V2_Binary", EX_BIN), ("V2_NegRisk", EX_NR)):
        print(f"\n  [3-4] pUSD -> {label} allowance")
        cur = pusd.functions.allowance(me, addr).call()
        if cur < MAX_UINT256 // 2:
            send(w3, acct, pusd.functions.approve(addr, MAX_UINT256),
                 f"pUSD.approve({label}, MAX)")
            time.sleep(1.5)
        else:
            print(f"    already MAX; skipping")

    # 5. tell CLOB to re-read on-chain state
    print(f"\n  [5/5] CLOB cache refresh")
    try:
        from py_clob_client_v2.client import ClobClient
        from py_clob_client_v2.clob_types import ApiCreds, AssetType, BalanceAllowanceParams
        client = ClobClient(
            host=cfg.clob_host, chain_id=cfg.chain_id, key=cfg.private_key,
            creds=ApiCreds(api_key=cfg.api_key, api_secret=cfg.api_secret,
                           api_passphrase=cfg.api_passphrase),
            signature_type=cfg.signature_type, funder=cfg.funder_address,
        )
        client.update_balance_allowance(BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL, signature_type=cfg.signature_type))
        time.sleep(3)
        cur = client.get_balance_allowance(BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL, signature_type=cfg.signature_type))
        print(f"    CLOB cached state: {cur}")
    except Exception as e:
        print(f"    refresh error (CLOB may auto-refresh on next order): {e}")

    # final on-chain state
    print(f"\n  Final on-chain state:")
    print(f"    USDC.e balance: {usdce.functions.balanceOf(me).call()/1e6:.6f}")
    print(f"    pUSD balance:   {pusd.functions.balanceOf(me).call()/1e6:.6f}")
    for label, addr in (("V2_Binary", EX_BIN), ("V2_NegRisk", EX_NR)):
        a = pusd.functions.allowance(me, addr).call()
        print(f"    pUSD allowance to {label}: {'MAX' if a >= MAX_UINT256//2 else a/1e6}")


if __name__ == "__main__":
    main()
