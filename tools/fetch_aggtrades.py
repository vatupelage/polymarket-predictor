"""Download Binance monthly aggTrade dumps for ANY symbol and store a compact
parquet of (ts_ms, price, qty). Usage:
    python tools/fetch_aggtrades.py SYMBOL START_YYYY-MM END_YYYY-MM OUT.parquet
e.g.
    python tools/fetch_aggtrades.py ETHUSDT 2026-04 2026-05 data/eth_aggtrades.parquet

Generalized from fetch_binance_aggtrades.py (which is hardcoded to BTCUSDT).
Memory-safe: explicit float32 price/qty + int64 ts, downcast immediately so the
concatenated frame stays small (~7GB RAM box). Skips months that 404."""
import io
import sys
import zipfile
import urllib.request
import urllib.error

import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/aggTrades"


def months(start: str, end: str):
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            m, y = 1, y + 1


def fetch_month(symbol: str, ym: str):
    url = f"{BASE}/{symbol}/{symbol}-aggTrades-{ym}.zip"
    print(f"  downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=300) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"    {ym}: 404 (not published yet) — skipping")
            return None
        raise
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = z.namelist()[0]
        with z.open(name) as fh:
            df = pd.read_csv(
                fh, header=None,
                names=["aggId", "price", "qty", "firstId", "lastId",
                       "ts", "isBuyerMaker", "isBestMatch"],
                usecols=["price", "qty", "ts"],
                dtype={"price": "float32", "qty": "float32", "ts": "int64"},
            )
    # Some 2025+ dumps store ts in microseconds; normalize to ms.
    if df["ts"].iloc[0] > 10**14:
        df["ts"] = df["ts"] // 1000
    print(f"    {ym}: {len(df):,} trades")
    return df[["ts", "price", "qty"]]


def main():
    if len(sys.argv) != 5:
        print("usage: fetch_aggtrades.py SYMBOL START_YYYY-MM END_YYYY-MM OUT.parquet")
        sys.exit(1)
    symbol, start, end, out = sys.argv[1].upper(), sys.argv[2], sys.argv[3], sys.argv[4]
    frames = []
    for ym in months(start, end):
        df = fetch_month(symbol, ym)
        if df is not None:
            frames.append(df)
    if not frames:
        print("  no data fetched — nothing written")
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True).sort_values("ts").reset_index(drop=True)
    df.to_parquet(out, index=False, compression="snappy")
    print(f"  wrote {len(df):,} trades to {out} "
          f"({pd.to_datetime(df['ts'].min(), unit='ms')} .. "
          f"{pd.to_datetime(df['ts'].max(), unit='ms')})")


if __name__ == "__main__":
    main()
