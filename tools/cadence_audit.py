"""Flag dbot dollar-bar thresholds whose live bar cadence drifts out of the
~20-25s target band. Pulls Binance 24h quote volume per symbol.
    python tools/cadence_audit.py
"""
import requests

BOTS = [("btc_5m","BTCUSDT",250000),("eth_15m","ETHUSDT",150000),("bnb_15m","BNBUSDT",125000)]

def sec_per_bar(quote_vol_24h_usd, threshold_usd):
    return threshold_usd / (quote_vol_24h_usd / 86400.0)

def audit(threshold, vol24h, target_lo=18, target_hi=30):
    spb = sec_per_bar(vol24h, threshold)
    return spb, (target_lo <= spb <= target_hi)

def main():
    for lbl, pair, thr in BOTS:
        qv = float(requests.get("https://api.binance.com/api/v3/ticker/24hr",
                                params={"symbol": pair}, timeout=15).json()["quoteVolume"])
        spb, ok = audit(thr, qv)
        print(f"  {lbl:8s} thr=${thr:,} vol=${qv/1e9:.2f}B -> {spb:5.0f}s/bar  {'OK' if ok else 'RETRAIN'}")

if __name__ == "__main__":
    main()
