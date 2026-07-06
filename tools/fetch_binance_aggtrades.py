"""Download Binance BTCUSDT monthly aggTrade dumps and store a compact parquet
of (ts_ms, price, qty). Usage:
    python tools/fetch_binance_aggtrades.py 2026-03 2026-06 data/aggtrades.parquet
Downloads months in [start, end] inclusive (YYYY-MM).

Memory-safe: parses each month with explicit float32 price/qty + int64 ts (the
training box has ~7GB RAM), downcasting immediately so the concatenated frame
stays small. Skips months that 404 (not yet published)."""
import io
import sys
import zipfile
import urllib.request
import urllib.error

import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT"


def months(start: str, end: str):
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    y, m = sy, sm
    while (y, m) <= (ey, em):
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            m, y = 1, y + 1


def fetch_month(ym: str):
    url = f"{BASE}/BTCUSDT-aggTrades-{ym}.zip"
    print(f"  downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=180) as resp:
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
    if len(sys.argv) != 4:
        print("usage: fetch_binance_aggtrades.py START_YYYY-MM END_YYYY-MM OUT.parquet")
        sys.exit(1)
    start, end, out = sys.argv[1], sys.argv[2], sys.argv[3]
    frames = []
    for ym in months(start, end):
        df = fetch_month(ym)
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
