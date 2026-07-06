"""Centralized serialized redeemer for the shared-wallet multi-market cohort.

Why this exists: Polymarket BUYS/SELLS are off-chain signed CLOB orders, so many
bots can trade on ONE wallet without nonce contention. The only on-chain txs the
wallet sends are redeemPositions (claim winnings) + the pUSD wrap, at resolution.
Each PolymarketBotClient serializes its own redeem+wrap via an instance _tx_lock,
but that lock does NOT span the N separate bot processes sharing the wallet, so
concurrent redeems hit Polygon's "in-flight transaction limit reached for
delegated accounts". Solution: the bots set BOT_DBMODEL_DELEGATE_REDEEM=true and
never redeem; THIS single daemon owns all settlement, so one _tx_lock + one
tx-sender = no collisions.

Loop: sweep_orphan_winners(force=True, wrap_each=False) — redeems each redeemable
winner one tx at a time, waiting for each receipt — then one USDC.e->pUSD wrap.
Idempotent; safe to restart. Run as systemd redeemer.service with the cohort
wallet's real key and BOT_DRY_RUN=false.
"""
import os
import sys
import time
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from live_trader.config import load_config
from live_trader.polymarket import PolymarketBotClient

POLL_S = int(os.environ.get("REDEEMER_POLL_S", "60"))


def main():
    cfg = load_config(dotenv_path=os.path.join(HERE, ".env"))
    client = PolymarketBotClient(cfg)
    print(f"[REDEEMER] start wallet={cfg.funder_address} dry_run={cfg.dry_run} "
          f"poll={POLL_S}s", flush=True)
    if cfg.dry_run:
        print("[REDEEMER] NOTE: dry_run=true — wrap is a no-op and there are no "
              "real positions; running as an idle scan (set BOT_DRY_RUN=false at "
              "go-live).", flush=True)
    while True:
        ts = datetime.datetime.now(datetime.UTC).strftime("%H:%M:%S")
        try:
            r = client.sweep_orphan_winners(force=True, wrap_each=False)
            if r.get("redeemed") or r.get("failed"):
                print(f"[REDEEMER {ts}] scanned={r['scanned']} winners={r['winners']} "
                      f"redeemed={r['redeemed']} failed={r['failed']} "
                      f"skipped={r['skipped']} ~${r['value']:.2f}", flush=True)
                # one sweep of all freshly-redeemed USDC.e -> pUSD (fewer txs than
                # wrapping after every redeem). No-op under dry_run / no balance.
                client.wrap_usdce()
        except Exception as e:
            print(f"[REDEEMER {ts}] error {type(e).__name__}: {e}", flush=True)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
