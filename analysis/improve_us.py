"""Can better features, targets and position sizing beat buy and hold on the S&P 500?

Usage:
    python get_prices.py SPY ^VIX ^TNX ^IRX TLT LQD GLD DX-Y.NYB ^FTSE ^GDAXI ^N225 ^HSI \
        --start 2004-01-01 --end 2026-01-01 --out data_us
    python improve_us.py

Every choice below (features, model settings, trading rules) is fixed in advance and none is tuned on
the test period. Walk-forward: for each test year 2013-2025, models are trained on all earlier data
(from 2005) and predict every day of the test year. Training rows whose 20-day target overlaps the test
year are dropped.

Features (known at the close of day t)
    SPY momentum over 1, 5, 20, 60, 120 and 250 days; gaps to the 50- and 200-day moving averages;
    drawdown from the 250-day high; realised volatility over 5, 22 and 66 days and their ratio;
    VIX level, its 5-day change and the variance risk premium (VIX minus realised volatility);
    10-year yield, 3-month yield, term spread and 20-day yield change;
    20-day returns of long Treasuries (TLT), corporate bonds (LQD), gold (GLD) and the dollar index;
    same-day returns of the FTSE 100, DAX, Nikkei 225 and Hang Seng (closed before New York).
    VIX, yields and the dollar index settle after the equity close, so they enter with a one-day lag.
Target
    Will SPY's return over the next 20 trading days be positive?
Models
    Logistic regression (L2), gradient-boosted trees, LSTM (60-day window), and their average.
Strategies (5 basis points per unit of turnover, daily rebalancing)
    Buy and hold
    Volatility target: w = min(15% / forecast volatility, 1.5), volatility forecast from a HAR model
    Timing: fully invested when P(up) > 0.5, otherwise half invested
    Volatility target x timing
Evaluation
    Return, volatility, Sharpe ratio, maximum drawdown, and return scaled to buy-and-hold volatility.
    A stationary block bootstrap gives the probability that the Sharpe ratio gain over buy and hold is
    no greater than zero.
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_us")
TEST_YEARS = range(2013, 2026)
H = 20
WINDOW = 60
SEEDS = [0, 1, 2]
THREADS = 4
COST = 0.0005
TARGET_VOL = 0.15
MAX_LEVERAGE = 1.5


def series(name):
    return pd.read_csv(os.path.join(DATA, name + ".csv"), index_col=0, parse_dates=True)["close"]


def build():
    spy = series("SPY")
    idx = spy.index
    get = lambda n: series(n).reindex(idx).ffill()
    r = np.log(spy).diff()
    f = pd.DataFrame(index=idx)
    for k in [1, 5, 20, 60, 120, 250]:
        f[f"mom_{k}"] = np.log(spy / spy.shift(k))
    f["ma50_gap"] = np.log(spy / spy.rolling(50).mean())
    f["ma200_gap"] = np.log(spy / spy.rolling(200).mean())
    f["drawdown_250"] = np.log(spy / spy.rolling(250).max())
    r2 = r**2
    for k in [5, 22, 66]:
        f[f"rv_{k}"] = np.log(np.sqrt(r2.rolling(k).mean() * 252))
    f["rv_ratio"] = f["rv_5"] - f["rv_66"]
    vix = get("VIX").shift(1)
    f["vix"] = np.log(vix)
    f["vix_chg5"] = np.log(vix / vix.shift(5))
    f["vrp"] = np.log(vix / 100) - f["rv_22"]
    tnx, irx = get("TNX").shift(1), get("IRX").shift(1)
    f["y10"] = tnx
    f["y3m"] = irx
    f["term"] = tnx - irx
    f["y10_chg20"] = tnx - tnx.shift(20)
    for n in ["TLT", "LQD", "GLD"]:
        s = get(n)
        f[f"{n.lower()}_20"] = np.log(s / s.shift(20))
    dxy = get("DX-Y.NYB").shift(1)
    f["dxy_20"] = np.log(dxy / dxy.shift(20))
    for n in ["FTSE", "GDAXI", "N225", "HSI"]:
        s = series(n)
        f[f"{n.lower()}_1d"] = np.log(s).diff().reindex(idx).fillna(0)   # closed day: no news
    # targets and HAR inputs
    f["r"] = r
    f["r_next"] = r.shift(-1)
    fwd = np.log(spy.shift(-H) / spy)
    f["up_h"] = (fwd > 0).astype(float).where(fwd.notna())
    f["har_d"], f["har_w"], f["har_m"] = np.log(r2 + 1e-8), np.log(r2.rolling(5).mean() + 1e-8), np.log(r2.rolling(22).mean() + 1e-8)
    f["logrv_next5"] = np.log(sum(r2.shift(-k) for k in range(1, 6)) / 5 + 1e-8)
    f = f.loc["2005-01-01":]
    return f.dropna(subset=[c for c in f.columns if c not in ("up_h", "logrv_next5", "r_next")])


FEATURES = None


def lstm_probs(Xtr_seq, ytr, Xte_seq, seed):
    import torch
    torch.set_num_threads(THREADS)
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

    net = Net(Xtr_seq.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xtr_seq, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(10):
        net.train()
        for b in torch.randperm(len(Xt), generator=g).split(128):
            opt.zero_grad()
            lossf(net(Xt[b]), yt[b]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xte_seq, dtype=torch.float32))).numpy()


def walk_forward(f):
    feats = [c for c in f.columns if c not in ("r", "r_next", "up_h", "har_d", "har_w", "har_m", "logrv_next5")]
    X = f[feats].values
    years = f.index.year.values
    out = []
    for ty in TEST_YEARS:
        te = np.where(years == ty)[0]
        tr = np.where(years < ty)[0]
        tr = tr[tr < te[0] - H]                         # purge targets overlapping the test year
        tr = tr[tr >= WINDOW - 1]
        mu, sd = X[:te[0]].mean(0), X[:te[0]].std(0) + 1e-12
        Z = (X - mu) / sd
        y = f["up_h"].values
        o = pd.DataFrame(index=f.index[te])
        o["p_logit"] = LogisticRegression(C=0.1, max_iter=2000).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
        hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                             l2_regularization=1.0, min_samples_leaf=100, random_state=0)
        o["p_gbt"] = hgb.fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
        seq = lambda ix: np.stack([Z[i - WINDOW + 1:i + 1] for i in ix])
        o["p_lstm"] = np.mean([lstm_probs(seq(tr), y[tr], seq(te), s) for s in SEEDS], 0)
        o["p_ensemble"] = o[["p_logit", "p_gbt", "p_lstm"]].mean(1)
        # HAR volatility forecast (5-day), with bias correction estimated on training data
        Hx = f[["har_d", "har_w", "har_m"]].values
        yv = f["logrv_next5"].values
        trv = tr[tr < te[0] - 5]
        har = LinearRegression().fit(Hx[trv], yv[trv])
        bias = np.log(np.mean(np.exp(yv[trv] - har.predict(Hx[trv]))))
        o["vol_har"] = np.sqrt(np.exp(har.predict(Hx[te]) + bias) * 252)
        o["up_h"] = y[te]
        o["r_next"] = f["r_next"].values[te]
        out.append(o)
        print(ty, flush=True)
    return pd.concat(out).dropna(subset=["r_next"])


def perf(r):
    ann = r.mean() * 252
    vol = r.std(ddof=1) * np.sqrt(252)
    w = np.exp(np.cumsum(r))
    return {"annual return": np.expm1(ann), "volatility": vol, "Sharpe": ann / vol,
            "max drawdown": (w / np.maximum.accumulate(w) - 1).min()}


def run_strategy(w, r):
    turn = np.abs(np.diff(np.concatenate([[w[0]], w])))
    return w * r - turn * COST


def block_bootstrap_pvalue(a, b, n_boot=2000, block=20, seed=0):
    """Share of bootstrap samples in which Sharpe(a) - Sharpe(b) <= 0 (stationary block bootstrap)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    sh = lambda x: x.mean() / x.std(ddof=1)
    count = 0
    for _ in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = 0
        while i < n:
            start = rng.integers(n)
            length = min(rng.geometric(1 / block), n - i)
            idx[i:i + length] = (start + np.arange(length)) % n
            i += length
        count += sh(a[idx]) - sh(b[idx]) <= 0
    return count / n_boot


def evaluate(P):
    r = P["r_next"].values
    bh = r - 0.0                                        # buy and hold: no rebalancing cost
    rows = []
    bh_vol = r.std(ddof=1) * np.sqrt(252)

    def add(name, strat, acc=np.nan):
        p = perf(strat)
        scaled = strat * bh_vol / (strat.std(ddof=1) * np.sqrt(252))
        rows.append({"strategy": name, "20-day accuracy": acc, **p,
                     "return at buy-and-hold volatility": perf(scaled)["annual return"],
                     "P(Sharpe gain <= 0)": np.nan if name == "Buy and hold" else block_bootstrap_pvalue(strat, bh)})

    add("Buy and hold", bh, P["up_h"].mean())
    w_vt = np.minimum(TARGET_VOL / P["vol_har"].values, MAX_LEVERAGE)
    add("Volatility target (HAR)", run_strategy(w_vt, r))
    for m in ["logit", "gbt", "lstm", "ensemble"]:
        p = P["p_" + m].values
        ok = P["up_h"].notna().values
        acc = ((p[ok] > .5) == P["up_h"].values[ok]).mean()
        tilt = np.where(p > .5, 1.0, 0.5)
        add(f"Timing: {m}", run_strategy(tilt, r), acc)
        add(f"Volatility target x timing: {m}", run_strategy(w_vt * tilt, r), acc)
    return pd.DataFrame(rows)


def main():
    f = build()
    P = walk_forward(f)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    P.to_csv(os.path.join(HERE, "results", "us_improved_predictions.csv"), float_format="%.6g")
    S = evaluate(P)
    S.to_csv(os.path.join(HERE, "results", "us_improved_summary.csv"), index=False, float_format="%.4f")
    print(S.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    # sub-periods
    for a, b in [(2013, 2019), (2020, 2025)]:
        sub = P[(P.index.year >= a) & (P.index.year <= b)]
        print(f"\n{a}-{b}")
        print(evaluate(sub)[["strategy", "annual return", "volatility", "Sharpe", "max drawdown"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
