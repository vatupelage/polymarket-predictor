"""Real-money risk controls for the live dbmodel bot.

The dbmodel runner bypasses the directional bot's `on_prediction` stop-loss, and
that stop reads an *optimistic* PnL counter (the order response's `makingAmount`
under-reports the true wallet debit by ~3%). So the only trustworthy kill-switch
is the wallet itself: snapshot pUSD+USDC.e at launch and halt when the real
drawdown reaches the limit.

`realized_real_pnl` / `stop_loss_tripped` are pure (unit-tested in test_risk.py).
`fetch_stable_balance` does the on-chain read (integration-only)."""

import time

from web3 import Web3

# Polygon collateral tokens, both 6-decimal. pUSD = V2 trading collateral;
# USDC.e = underlying CTF payout token (held transiently between redeem and wrap).
USDCE = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
PUSD = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"

# Public Polygon RPCs, tried in order. The bot's usual drpc endpoint 400s on
# rapid eth_call bursts, so we keep a fallback list for the once-per-window read.
_RPCS = ["https://polygon-bor-rpc.publicnode.com", "https://polygon-rpc.com",
         "https://rpc.ankr.com/polygon", "https://polygon.drpc.org"]
_ERC20 = [{"name": "balanceOf", "type": "function", "stateMutability": "view",
           "inputs": [{"name": "a", "type": "address"}],
           "outputs": [{"name": "", "type": "uint256"}]}]


def realized_real_pnl(start_stable, current_stable, open_trades, stake):
    """Realized PnL in real dollars.

    `current_stable - start_stable` already has every placed trade's cost
    debited but only *settled* trades' payouts credited. Open trades therefore
    drag it down by their cost with no offsetting payout yet, so we add that
    capital back. `stake` is a slight under-estimate of true per-trade cost
    (~3% fee on top), which makes the result marginally too negative -> the stop
    trips a hair EARLY, never late. Conservative on purpose."""
    return (current_stable - start_stable) + open_trades * stake


def stop_loss_tripped(start_stable, current_stable, open_trades, stake, limit):
    """(tripped, real_pnl). `limit` is the max tolerated drawdown in USD; its
    sign is normalised so -50 and 50 both mean 'halt at $50 down'."""
    pnl = realized_real_pnl(start_stable, current_stable, open_trades, stake)
    return (pnl <= -abs(limit)), pnl


def _mkw3(rpcs=_RPCS):
    for url in rpcs:
        try:
            w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 12}))
            if w3.eth.chain_id == 137:
                return w3
        except Exception:
            time.sleep(0.4)
    raise RuntimeError("no working Polygon RPC")


def fetch_stable_balance(address, w3=None):
    """pUSD + USDC.e balance (USD) for `address`. Raises if no RPC works — the
    caller MUST treat a failure to read as 'do not run without a stop-loss'."""
    w3 = w3 or _mkw3()
    addr = Web3.to_checksum_address(address)

    def bal(tok):
        c = w3.eth.contract(address=Web3.to_checksum_address(tok), abi=_ERC20)
        return c.functions.balanceOf(addr).call() / 1e6

    return bal(PUSD) + bal(USDCE)
