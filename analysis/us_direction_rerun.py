"""Re-run of the 2017 essay with clean data and an out-of-sample design.

Usage:
    python get_prices.py SPY ^FCHI ^FTSE ^GDAXI ^DJI ^IXIC ^HSI XOM GE MSFT PG JNJ \
        --start 2008-10-01 --end 2026-01-01
    python us_direction_rerun.py

Differences from the essay:
  * adjusted closing prices from Yahoo Finance; non-US markets that are closed on a US trading
    day contribute a zero return rather than an averaged price;
  * the target is whether SPY rises on the NEXT trading day, using only information available at
    today's close (European and Asian markets close before New York);
  * train and test sets are split in time order, never shuffled;
  * networks are compared with two baselines: always predicting "up", and logistic regression;
  * each network is trained with 5 random seeds and the results averaged;
  * the trading rule pays 5 basis points per trade.
The script also re-creates the essay's original label (today's return above yesterday's, random
split) on the clean data, to show where the essay's high accuracy came from.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
TICKERS = ["SPY", "FCHI", "FTSE", "GDAXI", "DJI", "IXIC", "HSI", "XOM", "GE", "MSFT", "PG", "JNJ"]
WINDOW = 10           # days of history fed to the networks
SEEDS = [0, 1, 2, 3, 4]
COST = 0.0005         # 5 basis points per trade
# architecture: (hidden units, epochs) chosen in the essay
NETS = {"Simple RNN": ("rnn", 75, 40), "LSTM": ("lstm", 25, 80), "GRU": ("gru", 50, 40)}
# (label, training period, test period)
DESIGNS = [("Essay period, time-ordered split", ("2009-01-01", "2015-08-31"), ("2015-09-01", "2017-04-01")),
           ("New data: train 2009-2019, test 2020-2025", ("2009-01-01", "2019-12-31"), ("2020-01-01", "2025-12-31"))]

torch.set_num_threads(4)


def load():
    px = {}
    for t in TICKERS:
        s = pd.read_csv(os.path.join(HERE, "data", t + ".csv"), index_col=0, parse_dates=True)["close"]
        px[t] = s
    px = pd.DataFrame(px)
    px = px[px["SPY"].notna()].ffill()            # US trading days; closed markets carry yesterday's price
    R = np.log(px).diff().dropna()
    R["target"] = (R["SPY"].shift(-1) > 0).astype(int)   # does SPY rise tomorrow?
    R["next_ret"] = R["SPY"].shift(-1)
    return R.iloc[:-1]


def windows(X, length):
    """Stack the last `length` rows of X for every date (rows before that are dropped)."""
    return np.stack([X[i - length + 1:i + 1] for i in range(length - 1, len(X))])


class Net(torch.nn.Module):
    def __init__(self, kind, n_in, hidden):
        super().__init__()
        cell = {"rnn": torch.nn.RNN, "lstm": torch.nn.LSTM, "gru": torch.nn.GRU}[kind]
        self.rnn = cell(n_in, hidden, batch_first=True)
        self.drop = torch.nn.Dropout(0.2)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.rnn(x)
        return self.out(self.drop(h[:, -1])).squeeze(-1)


def train_net(kind, hidden, epochs, Xtr, ytr, Xte, seed):
    torch.manual_seed(seed)
    net = Net(kind, Xtr.shape[2], hidden)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(Xt), generator=g).split(32):
            opt.zero_grad()
            lossf(net(Xt[idx]), yt[idx]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.tensor(Xte, dtype=torch.float32))).numpy()


def trading(signal, next_ret):
    """Hold SPY on days the signal is 1, cash otherwise; pay COST each time the position changes."""
    pos = signal.astype(float)
    trades = np.abs(np.diff(np.concatenate([[0.0], pos])))
    strat = pos * next_ret - trades * COST
    def stats(r):
        ann = r.mean() * 252
        vol = r.std(ddof=1) * np.sqrt(252)
        wealth = np.exp(np.cumsum(r))
        mdd = (wealth / np.maximum.accumulate(wealth) - 1).min()
        return {"annual return": np.expm1(ann), "volatility": vol, "Sharpe": ann / vol, "max drawdown": mdd}
    return stats(strat), stats(next_ret), np.cumsum(strat), np.cumsum(next_ret)


def run_design(R, label, train, test):
    feats = R[TICKERS].values
    dates = R.index
    tr = (dates >= train[0]) & (dates <= train[1])
    te = (dates >= test[0]) & (dates <= test[1])
    mu, sd = feats[tr].mean(0), feats[tr].std(0)                 # standardise with training data only
    Z = (feats - mu) / sd
    W = windows(Z, WINDOW)
    keep = np.arange(WINDOW - 1, len(Z))
    y, nxt = R["target"].values[keep], R["next_ret"].values[keep]
    tr, te = tr[keep], te[keep]
    rows, curves = [], {}

    rows.append({"design": label, "model": "Always up", "accuracy": y[te].mean(), "accuracy sd": 0.0})
    logit = LogisticRegression(max_iter=1000).fit(Z[keep][tr], y[tr])
    p_log = logit.predict_proba(Z[keep][te])[:, 1]
    rows.append({"design": label, "model": "Logistic regression", "accuracy": ((p_log > .5) == y[te]).mean(),
                 "accuracy sd": 0.0})
    for name, (kind, hidden, epochs) in NETS.items():
        probs = [train_net(kind, hidden, epochs, W[tr], y[tr], W[te], s) for s in SEEDS]
        accs = [((p > .5) == y[te]).mean() for p in probs]
        rows.append({"design": label, "model": name, "accuracy": np.mean(accs), "accuracy sd": np.std(accs)})
        signal = (np.mean(probs, axis=0) > .5).astype(int)          # seed-averaged probability
        st, bh, cs, cb = trading(signal, nxt[te])
        rows[-1].update({f"strategy {k}": v for k, v in st.items()})
        rows[-1].update({"share of days invested": signal.mean()})
        curves[name] = cs
    rows[0].update({f"strategy {k}": v for k, v in bh.items()})     # always up = buy and hold
    rows[0].update({"share of days invested": 1.0})
    curves["Buy and hold"] = cb
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
    return rows, (dates[keep][te], curves)


def essay_label_check(R):
    """The essay's label and split on clean data, with logistic regression instead of a network."""
    sub = R.loc["2009-01-01":"2017-04-01"]
    X = sub[["SPY", "JNJ", "IXIC"]].values.copy()
    X = np.column_stack([np.roll(X[:, 0], 1), X[:, 1], X[:, 2]])[1:]   # SPY t-1, JNJ t, NASDAQ t
    X[1:] = X[1:] - X[:-1]
    y = sub["SPY"].values[1:]
    y = (np.diff(np.concatenate([[0], y])) >= 0).astype(int)              # today's return above yesterday's
    Xtr, Xte, ytr, yte = train_test_split(X[1:], y[1:], test_size=0.2, random_state=0)
    acc = LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte, yte)
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(len(sub) + 1)
    yn = (np.diff(noise[1:]) >= 0).astype(int)
    Xn = np.column_stack([np.diff(noise[:-1]), rng.standard_normal(len(yn)), rng.standard_normal(len(yn))])
    Xtr, Xte, ytr, yte = train_test_split(Xn, yn, test_size=0.2, random_state=0)
    acc_noise = LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte, yte)
    return pd.DataFrame([{"check": "Essay label and random split, clean data, logistic regression", "accuracy": acc},
                         {"check": "Same label on pure random noise", "accuracy": acc_noise}])


def main():
    R = load()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    check = essay_label_check(R)
    print(check.to_string(index=False), flush=True)
    check.to_csv(os.path.join(HERE, "results", "essay_label_check.csv"), index=False, float_format="%.4f")
    allrows = []
    for label, train, test in DESIGNS:
        rows, (d, curves) = run_design(R, label, train, test)
        allrows += rows
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, c in curves.items():
            ax.plot(d, np.expm1(c) * 100, lw=2.2 if name == "Buy and hold" else 1.4,
                    color="#222222" if name == "Buy and hold" else None, label=name)
        ax.set_ylabel("Cumulative return (%)")
        ax.set_title(f"{label}: trading rule after costs vs buy and hold")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "figures", "trading_" + ("essay_period" if "Essay" in label else "2020_2025") + ".png"), dpi=130)
        plt.close(fig)
    pd.DataFrame(allrows).to_csv(os.path.join(HERE, "results", "out_of_sample.csv"), index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
