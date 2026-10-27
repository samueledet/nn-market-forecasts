"""Download daily adjusted closing prices from Yahoo Finance.

Usage:
    python get_prices.py ^N225 ^GSPC --start 2000-01-01 --end 2025-12-31

Each ticker is saved to data/<ticker>.csv with columns date, close.
"""
import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone

import pandas as pd

URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
       "?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit")


def fetch(ticker, start, end):
    """Return a Series of daily adjusted closes indexed by date."""
    p1 = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp())
    p2 = int(datetime.fromisoformat(end).replace(tzinfo=timezone.utc).timestamp())
    req = urllib.request.Request(URL.format(ticker=urllib.request.quote(ticker), p1=p1, p2=p2),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.load(resp)["chart"]["result"][0]
    offset = res["meta"].get("gmtoffset", 0)             # exchange-local trading dates
    dates = pd.to_datetime(pd.Series(res["timestamp"]) + offset, unit="s").dt.normalize()
    ind = res["indicators"]
    close = ind["adjclose"][0]["adjclose"] if "adjclose" in ind else ind["quote"][0]["close"]
    s = pd.Series(close, index=pd.DatetimeIndex(dates), name=ticker, dtype=float).dropna()
    return s[~s.index.duplicated(keep="last")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for t in args.tickers:
        s = fetch(t, args.start, args.end)
        path = os.path.join(args.out, t.replace("^", "") + ".csv")
        s.rename("close").rename_axis("date").to_csv(path)
        print(f"{t}: {len(s)} days, {s.index[0].date()} to {s.index[-1].date()} -> {path}")


if __name__ == "__main__":
    main()
