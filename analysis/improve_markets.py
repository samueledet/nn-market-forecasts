"""Replicate the S&P 500 design of improve_us.py in 10 other markets.

Usage (after downloading data/ and data_us/ as in the README):
    python improve_markets.py            # all 10 markets
    python improve_markets.py N225 BVSP  # a subset

The design, model settings and trading rules are those of improve_us.py, fixed before these markets were
examined, so the 10 markets act as an out-of-sample check on the US result.

Features for market m on its trading day t
    Own index: momentum over 1, 5, 20, 60, 120 and 250 days; gaps to the 50- and 200-day moving averages;
    drawdown from the 250-day high; realised volatility over 5, 22 and 66 days and their ratio.
    Global (from the last US close strictly before day t, so known before m's close): S&P 500 1-day and
    20-day return, VIX level, 5-day change and variance risk premium (VIX minus S&P 500 realised
    volatility), 10-year and 3-month US yields, term spread, 20-day yield change, and 20-day returns of
    long Treasuries, corporate bonds, gold and the dollar index.
Target, models, strategies and evaluation: as in improve_us.py. Own-market data start in 2005, so with
one-year lookbacks the first training year is 2006.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import improve_us as us

HERE = os.path.dirname(os.path.abspath(__file__))
MARKETS = {
    "N225": ("Japan", "Developed"), "FTSE": ("United Kingdom", "Developed"),
    "GDAXI": ("Germany", "Developed"), "AXJO": ("Australia", "Developed"),
    "BVSP": ("Brazil", "Emerging"), "MXX": ("Mexico", "Emerging"), "BSESN": ("India", "Emerging"),
    "000001.SS": ("China", "Emerging"), "J200.JO": ("South Africa", "Emerging"),
    "JKSE": ("Indonesia", "Emerging"),
}


def us_global():
    """Global variables on the US calendar, as known at each US close."""
    spy = us.series("SPY")
    idx = spy.index
    get = lambda n: us.series(n).reindex(idx).ffill()
    g = pd.DataFrame(index=idx)
    r = np.log(spy).diff()
    g["spx_1d"] = r
    g["spx_20"] = np.log(spy / spy.shift(20))
    vix = get("VIX")
    g["vix"] = np.log(vix)
    g["vix_chg5"] = np.log(vix / vix.shift(5))
    g["vrp"] = np.log(vix / 100) - np.log(np.sqrt((r**2).rolling(22).mean() * 252))
    tnx, irx = get("TNX"), get("IRX")
    g["y10"], g["y3m"], g["term"] = tnx, irx, tnx - irx
    g["y10_chg20"] = tnx - tnx.shift(20)
    for n in ["TLT", "LQD", "GLD", "DX-Y.NYB"]:
        s = get(n)
        g[n.lower().split("-")[0] + "_20"] = np.log(s / s.shift(20))
    return g


def build_market(name, g):
    px = pd.read_csv(os.path.join(HERE, "data", name + ".csv"), index_col=0, parse_dates=True)["close"]
    r = np.log(px).diff()
    f = pd.DataFrame(index=px.index)
    for k in [1, 5, 20, 60, 120, 250]:
        f[f"mom_{k}"] = np.log(px / px.shift(k))
    f["ma50_gap"] = np.log(px / px.rolling(50).mean())
    f["ma200_gap"] = np.log(px / px.rolling(200).mean())
    f["drawdown_250"] = np.log(px / px.rolling(250).max())
    r2 = r**2
    for k in [5, 22, 66]:
        f[f"rv_{k}"] = np.log(np.sqrt(r2.rolling(k).mean() * 252))
    f["rv_ratio"] = f["rv_5"] - f["rv_66"]
    # last US close strictly before each local trading day
    gl = pd.merge_asof(pd.DataFrame({"date": f.index}), g.rename_axis("date").reset_index(),
                       on="date", allow_exact_matches=False).set_index("date")
    f = f.join(gl)
    f["r"] = r
    f["r_next"] = r.shift(-1)
    fwd = np.log(px.shift(-us.H) / px)
    f["up_h"] = (fwd > 0).astype(float).where(fwd.notna())
    f["har_d"] = np.log(r2 + 1e-8)
    f["har_w"] = np.log(r2.rolling(5).mean() + 1e-8)
    f["har_m"] = np.log(r2.rolling(22).mean() + 1e-8)
    f["logrv_next5"] = np.log(sum(r2.shift(-k) for k in range(1, 6)) / 5 + 1e-8)
    return f.dropna(subset=[c for c in f.columns if c not in ("up_h", "logrv_next5", "r_next")])


def run(name):
    us.THREADS = 1
    f = build_market(name, us_global())
    P = us.walk_forward(f)
    P.to_csv(os.path.join(HERE, "results", f"improved_predictions_{name}.csv"), float_format="%.6g")
    S = us.evaluate(P)
    S.insert(0, "market", MARKETS[name][0])
    S.insert(1, "group", MARKETS[name][1])
    sub = []
    for a, b in [(2013, 2019), (2020, 2025)]:
        part = us.evaluate(P[(P.index.year >= a) & (P.index.year <= b)])[["strategy", "Sharpe"]]
        sub.append(part.set_index("strategy")["Sharpe"].rename(f"Sharpe {a}-{b}"))
    S = S.join(pd.concat(sub, axis=1), on="strategy")
    print(name, "done", flush=True)
    return S


def main(names):
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        out = pd.concat(ex.map(run, names), ignore_index=True)
    out.to_csv(os.path.join(HERE, "results", "improved_markets_summary.csv"), index=False, float_format="%.4f")
    best = out[out["strategy"].isin(["Buy and hold", "Volatility target (HAR)", "Volatility target x timing: lstm",
                                     "Volatility target x timing: ensemble", "Timing: ensemble"])]
    with pd.option_context("display.width", 250):
        print(best[["market", "strategy", "annual return", "volatility", "Sharpe", "max drawdown",
                    "return at buy-and-hold volatility", "P(Sharpe gain <= 0)", "Sharpe 2013-2019",
                    "Sharpe 2020-2025"]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main(sys.argv[1:] or list(MARKETS))
