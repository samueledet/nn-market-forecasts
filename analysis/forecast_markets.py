"""What can recurrent networks forecast in equity markets? Direction against volatility, 11 markets.

Usage:
    python get_prices.py ^GSPC ^N225 ^FTSE ^GDAXI ^AXJO ^BVSP ^MXX ^BSESN 000001.SS ^J200.JO ^JKSE \
        --start 2005-01-01 --end 2026-01-01
    python forecast_markets.py

Design (walk-forward): for each test year 2013-2025, every model is trained on all data from 2005 up to
the end of the previous year and then forecasts each day of the test year. Nothing from the test year is
used for training or for standardising inputs.

Task 1, direction: will the index rise tomorrow?
    Always up | logistic regression | LSTM
Task 2, volatility: log of the average squared daily return over the next 5 trading days.
    Naive (last 22 days' realised variance) | HAR (Corsi, 2009) | GARCH(1,1) | LSTM
Use: volatility targeting, holding w_t = min(15% / forecast volatility, 1.5) of the index, paying 5 basis
points per unit of turnover, compared with buy and hold.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
MARKETS = {
    "GSPC": ("United States", "Developed"), "N225": ("Japan", "Developed"),
    "FTSE": ("United Kingdom", "Developed"), "GDAXI": ("Germany", "Developed"),
    "AXJO": ("Australia", "Developed"), "BVSP": ("Brazil", "Emerging"), "MXX": ("Mexico", "Emerging"),
    "BSESN": ("India", "Emerging"), "000001.SS": ("China", "Emerging"),
    "J200.JO": ("South Africa", "Emerging"), "JKSE": ("Indonesia", "Emerging"),
}
TEST_YEARS = range(2013, 2026)
WINDOW = 22          # one month of daily history for the LSTM
HORIZON = 5          # volatility target: next 5 trading days
SEEDS = [0, 1, 2]
EPOCHS = 15
HIDDEN = 32
COST = 0.0005
TARGET_VOL = 0.15
MAX_LEVERAGE = 1.5


# ---------------------------------------------------------------- features

def build(name):
    s = pd.read_csv(os.path.join(HERE, "data", name + ".csv"), index_col=0, parse_dates=True)["close"]
    r = np.log(s).diff().dropna()
    df = pd.DataFrame({"r": r})
    r2 = r**2
    df["rv_d"] = np.log(r2 + 1e-8)                                  # HAR components (log)
    df["rv_w"] = np.log(r2.rolling(5).mean() + 1e-8)
    df["rv_m"] = np.log(r2.rolling(22).mean() + 1e-8)
    df["abs_r"] = r.abs()
    df["up_next"] = (r.shift(-1) > 0).astype(float)
    df["r_next"] = r.shift(-1)
    fwd = sum(r2.shift(-k) for k in range(1, HORIZON + 1)) / HORIZON
    df["logrv_next"] = np.log(fwd + 1e-8)
    return df.dropna()


SEQ_COLS = ["r", "abs_r", "rv_d", "rv_w", "rv_m"]


def sequences(Z, idx):
    """Windows of the last WINDOW rows ending at each position in idx."""
    return np.stack([Z[i - WINDOW + 1:i + 1] for i in idx])


# ---------------------------------------------------------------- models

def lstm_fit_predict(Xtr, ytr, Xte, task, seed):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = torch.nn.LSTM(Xtr.shape[2], HIDDEN, batch_first=True)
            self.drop = torch.nn.Dropout(0.2)
            self.out = torch.nn.Linear(HIDDEN, 1)

        def forward(self, x):
            h, _ = self.rnn(x)
            return self.out(self.drop(h[:, -1])).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss() if task == "direction" else torch.nn.MSELoss()
    ym, ys = (0.0, 1.0) if task == "direction" else (ytr.mean(), ytr.std())
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(EPOCHS):
        net.train()
        for b in torch.randperm(len(Xt), generator=g).split(64):
            opt.zero_grad()
            lossf(net(Xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.tensor(Xte, dtype=torch.float32)).numpy()
        if task == "direction":
            return 1 / (1 + np.exp(-out))
        fit_tr = net(Xt).numpy() * ys + ym
    # log forecasts understate the variance (Jensen); scale factor estimated on training data
    bias = np.log(np.mean(np.exp(ytr - fit_tr)))
    return out * ys + ym, bias


def garch_fit(r):
    """GARCH(1,1) with normal shocks; returns (c, omega, alpha, beta)."""
    def var_path(th):
        c, w, a, b = th
        e2 = (r - c) ** 2
        const = w / (1 - b)
        y = lfilter([a], [1, -b], e2[:-1], zi=[b * (r.var() - const)])[0]
        return np.concatenate([[r.var()], y + const])

    def nll(th):
        c, w, a, b = th
        if w <= 0 or a < 0 or b < 0 or a + b >= 1:
            return np.inf
        return -norm.logpdf(r, c, np.sqrt(var_path(th))).sum()

    th = np.array([r.mean(), 0.02 * r.var(), 0.08, 0.9])
    for _ in range(2):
        th = minimize(nll, th, method="Nelder-Mead", options={"maxiter": 5000, "xatol": 1e-9, "fatol": 1e-9}).x
    return th


def garch_forecast(r_all, th, idx):
    """Average variance over the next HORIZON days, from information up to each day in idx."""
    c, w, a, b = th
    e2 = (r_all - c) ** 2
    const = w / (1 - b)
    h = np.concatenate([[r_all.var()], lfilter([a], [1, -b], e2[:-1], zi=[b * (r_all.var() - const)])[0] + const])
    h_next = w + a * e2 + b * h                              # variance for day t+1, known at t
    lr = w / (1 - a - b)
    k = np.arange(HORIZON)
    path = lr + np.outer(h_next - lr, (a + b) ** k)         # t+1 ... t+HORIZON
    return np.log(path.mean(1)[idx])


# ---------------------------------------------------------------- one market

def run_market(name):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    df = build(name)
    years = df.index.year.values
    preds = []
    for ty in TEST_YEARS:
        tr = np.where((years < ty) & (np.arange(len(df)) >= WINDOW - 1))[0]
        te = np.where(years == ty)[0]
        # targets that peek into the test year are dropped from training
        tr = tr[tr < te[0] - HORIZON]
        mu, sd = df[SEQ_COLS].values[:te[0]].mean(0), df[SEQ_COLS].values[:te[0]].std(0)
        Z = (df[SEQ_COLS].values - mu) / sd
        Xtr, Xte = sequences(Z, tr), sequences(Z, te)
        out = pd.DataFrame(index=df.index[te])
        out["market"] = name
        # direction
        yd = df["up_next"].values
        out["up_next"] = yd[te]
        out["p_logit"] = LogisticRegression(max_iter=1000).fit(Z[tr], yd[tr]).predict_proba(Z[te])[:, 1]
        out["p_lstm"] = np.mean([lstm_fit_predict(Xtr, yd[tr], Xte, "direction", s) for s in SEEDS], 0)
        # volatility
        yv = df["logrv_next"].values
        out["logrv_next"] = yv[te]
        out["v_naive"] = df["rv_m"].values[te]
        H = df[["rv_d", "rv_w", "rv_m"]].values
        har = LinearRegression().fit(H[tr], yv[tr])
        out["v_har"] = har.predict(H[te])
        out["bias_har"] = np.log(np.mean(np.exp(yv[tr] - har.predict(H[tr]))))
        th = garch_fit(df["r"].values[:te[0]])
        out["v_garch"] = garch_forecast(df["r"].values, th, te)
        fits = [lstm_fit_predict(Xtr, yv[tr], Xte, "volatility", s) for s in SEEDS]
        out["v_lstm"] = np.mean([f[0] for f in fits], 0)
        out["bias_lstm"] = np.mean([f[1] for f in fits])
        out["r_next"] = df["r_next"].values[te]
        preds.append(out)
        print(name, ty, flush=True)
    return pd.concat(preds)


# ---------------------------------------------------------------- evaluation

def perf(r):
    ann = r.mean() * 252
    vol = r.std(ddof=1) * np.sqrt(252)
    w = np.exp(np.cumsum(r))
    return {"return": np.expm1(ann), "volatility": vol, "Sharpe": ann / vol,
            "max drawdown": (w / np.maximum.accumulate(w) - 1).min()}


def evaluate(P):
    rows = []
    for name, d in P.groupby("market"):
        market, group = MARKETS[name]
        base = {"market": market, "group": group, "days": len(d)}
        y = d["up_next"].values
        row = {**base, "acc always up": y.mean(),
               "acc logistic": ((d["p_logit"] > .5) == y).mean(),
               "acc lstm": ((d["p_lstm"] > .5) == y).mean()}
        # direction trading rule (long or cash) with costs
        sig = (d["p_lstm"].values > .5).astype(float)
        strat = sig * d["r_next"].values - np.abs(np.diff(np.concatenate([[0], sig]))) * COST
        row["direction rule Sharpe"] = perf(strat)["Sharpe"]
        row["buy and hold Sharpe"] = perf(d["r_next"].values)["Sharpe"]
        # volatility: out-of-sample R^2 of log realised variance, relative to the naive forecast
        yv = d["logrv_next"].values
        mse = {m: np.mean((yv - d["v_" + m].values) ** 2) for m in ["naive", "har", "garch", "lstm"]}
        for m in ["har", "garch", "lstm"]:
            row[f"vol R2 vs naive {m}"] = 1 - mse[m] / mse["naive"]
        for m in ["naive", "har", "garch", "lstm"]:
            row[f"vol MSE {m}"] = mse[m]
        # volatility targeting
        for m in ["har", "garch", "lstm"]:
            # GARCH forecasts the variance itself; HAR and LSTM forecast its log, so add the bias term
            logvar = d["v_" + m].values + (d["bias_" + m].values if m != "garch" else 0)
            vol_f = np.sqrt(np.exp(logvar) * 252)
            w = np.minimum(TARGET_VOL / vol_f, MAX_LEVERAGE)
            turn = np.abs(np.diff(np.concatenate([[w[0]], w])))
            p = perf(w * d["r_next"].values - turn * COST)
            row[f"voltarget {m} Sharpe"] = p["Sharpe"]
            row[f"voltarget {m} maxDD"] = p["max drawdown"]
            row[f"voltarget {m} vol"] = p["volatility"]
        bh = perf(d["r_next"].values)
        row["buy and hold maxDD"] = bh["max drawdown"]
        row["buy and hold vol"] = bh["volatility"]
        rows.append(row)
    return pd.DataFrame(rows)


def main(names):
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        P = pd.concat(ex.map(run_market, names))
    P.to_csv(os.path.join(HERE, "results", "walk_forward_predictions.csv"), float_format="%.6g")
    S = evaluate(P)
    S.to_csv(os.path.join(HERE, "results", "walk_forward_summary.csv"), index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(S.set_index("market").T.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:] or list(MARKETS))
