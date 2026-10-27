"""Figures for the README from the saved walk-forward results."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
COST, TARGET_VOL, MAX_LEVERAGE = 0.0005, 0.15, 1.5


def volatility_r2():
    s = pd.read_csv(os.path.join(RES, "walk_forward_summary.csv")).sort_values("vol R2 vs naive lstm")
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, (m, c, lab) in enumerate([("garch", "#9aa0a6", "GARCH(1,1)"), ("har", "#1f6feb", "HAR"),
                                     ("lstm", "#d1242f", "LSTM")]):
        ax.barh(y + (k - 1) * 0.27, s[f"vol R2 vs naive {m}"] * 100, height=0.25, color=c, label=lab)
    ax.set_yticks(y, s["market"])
    ax.axvline(0, color="#222222", lw=0.8)
    ax.set_xlabel("Reduction in squared forecast error against last month's volatility (%)")
    ax.set_title("Next-week volatility forecasts, walk-forward 2013-2025")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "volatility_forecasts.png"), dpi=130)
    plt.close(fig)


def us_strategy():
    P = pd.read_csv(os.path.join(RES, "us_improved_predictions.csv"), index_col=0, parse_dates=True).dropna(subset=["r_next"])
    r = P["r_next"].values
    w = np.minimum(TARGET_VOL / P["vol_har"].values, MAX_LEVERAGE) * np.where(P["p_lstm"].values > .5, 1.0, 0.5)
    strat = w * r - np.abs(np.diff(np.concatenate([[w[0]], w]))) * COST
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for x, c, lab in [(r, "#222222", "Buy and hold"), (strat, "#d1242f", "Volatility target x LSTM timing")]:
        wealth = np.exp(np.cumsum(x))
        a1.plot(P.index, wealth, color=c, lw=1.6, label=lab)
        a2.plot(P.index, (wealth / np.maximum.accumulate(wealth) - 1) * 100, color=c, lw=1.2)
    a1.set_ylabel("Growth of $1")
    a1.set_title("S&P 500 (SPY), walk-forward 2013-2025, after costs")
    a1.legend(frameon=False)
    a2.set_ylabel("Drawdown (%)")
    for a in (a1, a2):
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "us_strategy.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    volatility_r2()
    us_strategy()
