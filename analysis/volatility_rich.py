"""Volatility forecasting with the rich feature set, 11 markets, and volatility targeting.

Usage (after downloading data/ and data_us/ as in the README):
    python volatility_rich.py

Same walk-forward design and features as improve_us.py (S&P 500) and improve_markets.py (10 markets).
Target: log of the average squared daily return over the next 5 trading days.
Models
    Naive        last month's realised variance
    HAR          Corsi (2009): log variance over the last day, week and month
    GARCH(1,1)   normal shocks, refitted each year
    HAR-X        linear regression on the HAR terms plus all rich features
    Trees        gradient-boosted regression trees on the same inputs
    LSTM         60-day window of the same inputs
    Average      mean of HAR-X, trees and LSTM
Use: volatility targeting with each forecast, w = min(15% / forecast volatility, 1.5), 5 basis points per
unit traded. Log forecasts are converted to variances with a bias factor estimated on training data.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import improve_markets as im
import improve_us as us
from forecast_markets import garch_fit, garch_forecast

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"SPY": ("United States", "Developed"), **im.MARKETS}
HORIZON = 5
MODELS = ["naive", "har", "garch", "harx", "trees", "lstm", "average"]


def lstm_reg(Xtr, ytr, Xte, seed):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    class Net(torch.nn.Module):
        def __init__(self, n):
            super().__init__()
            self.rnn = torch.nn.LSTM(n, 32, batch_first=True)
            self.drop = torch.nn.Dropout(0.3)
            self.out = torch.nn.Linear(32, 1)

        def forward(self, x):
            h, _ = self.rnn(x)
            return self.out(self.drop(h[:, -1])).squeeze(-1)

    ym, ys = ytr.mean(), ytr.std()
    net = Net(Xtr.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(10):
        net.train()
        for b in torch.randperm(len(Xt), generator=g).split(128):
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(Xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return (net(torch.tensor(Xte, dtype=torch.float32)).numpy() * ys + ym,
                net(Xt).numpy() * ys + ym)


def frame(name):
    return us.build() if name == "SPY" else im.build_market(name, im.us_global())


def run(name):
    f = frame(name)
    feats = [c for c in f.columns if c not in ("r", "r_next", "up_h", "logrv_next5")]   # HAR terms included
    har_cols = ["har_d", "har_w", "har_m"]
    X = f[feats].values
    y = f["logrv_next5"].values
    years = f.index.year.values
    out = []
    for ty in us.TEST_YEARS:
        te = np.where(years == ty)[0]
        tr = np.where(years < ty)[0]
        tr = tr[(tr < te[0] - HORIZON) & (tr >= us.WINDOW - 1)]
        mu, sd = X[:te[0]].mean(0), X[:te[0]].std(0) + 1e-12
        Z = (X - mu) / sd
        o = pd.DataFrame(index=f.index[te])
        bias = {}

        def fit_lin(cols_idx, key):
            m = LinearRegression().fit(Z[tr][:, cols_idx], y[tr])
            o[key] = m.predict(Z[te][:, cols_idx])
            bias[key] = np.log(np.mean(np.exp(y[tr] - m.predict(Z[tr][:, cols_idx]))))

        o["naive"] = f["har_m"].values[te]
        bias["naive"] = np.log(np.mean(np.exp(y[tr] - f["har_m"].values[tr])))
        fit_lin([feats.index(c) for c in har_cols], "har")
        fit_lin(list(range(len(feats))), "harx")
        th = garch_fit(f["r"].values[:te[0]])
        o["garch"] = garch_forecast(f["r"].values, th, te)
        bias["garch"] = 0.0
        gbt = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                            min_samples_leaf=100, random_state=0).fit(Z[tr], y[tr])
        o["trees"] = gbt.predict(Z[te])
        bias["trees"] = np.log(np.mean(np.exp(y[tr] - gbt.predict(Z[tr]))))
        seq = lambda ix: np.stack([Z[i - us.WINDOW + 1:i + 1] for i in ix])
        fits = [lstm_reg(seq(tr), y[tr], seq(te), s) for s in us.SEEDS]
        o["lstm"] = np.mean([p for p, _ in fits], 0)
        bias["lstm"] = np.log(np.mean(np.exp(y[tr] - np.mean([q for _, q in fits], 0))))
        o["average"] = o[["harx", "trees", "lstm"]].mean(1)
        bias["average"] = np.mean([bias["harx"], bias["trees"], bias["lstm"]])
        for k, v in bias.items():
            o["bias_" + k] = v
        o["y"] = y[te]
        o["r_next"] = f["r_next"].values[te]
        out.append(o)
    P = pd.concat(out).dropna(subset=["r_next"])
    P.to_csv(os.path.join(HERE, "results", f"vol_rich_predictions_{name}.csv"), float_format="%.6g")
    print(name, "done", flush=True)
    return name, P


def summarise(name, P):
    market, group = NAMES[name]
    ok = P["y"].notna()
    mse = {m: np.mean((P.loc[ok, "y"] - P.loc[ok, m]) ** 2) for m in MODELS}
    row = {"market": market, "group": group}
    for m in MODELS[1:]:
        row[f"R2 vs naive {m}"] = 1 - mse[m] / mse["naive"]
    r = P["r_next"].values
    bh = us.perf(r)
    row.update({"bh Sharpe": bh["Sharpe"], "bh maxDD": bh["max drawdown"], "bh return": bh["annual return"]})
    for m in ["har", "harx", "lstm", "average"]:
        vol = np.sqrt(np.exp(P[m].values + P["bias_" + m].values) * 252)
        w = np.minimum(us.TARGET_VOL / vol, us.MAX_LEVERAGE)
        s = us.run_strategy(w, r)
        p = us.perf(s)
        row.update({f"vt {m} Sharpe": p["Sharpe"], f"vt {m} maxDD": p["max drawdown"], f"vt {m} vol": p["volatility"],
                    f"vt {m} P(gain<=0)": us.block_bootstrap_pvalue(s, r)})
    return row


def main():
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(run, list(NAMES)))
    S = pd.DataFrame([summarise(n, P) for n, P in res])
    S.to_csv(os.path.join(HERE, "results", "vol_rich_summary.csv"), index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(S.set_index("market").T.round(3).to_string())


if __name__ == "__main__":
    main()
