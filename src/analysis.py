from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

TREATED = ["BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
CONTROL = ["NIFTY"]
EVENT = pd.Timestamp("2024-11-20")
LOTSIZE_EVENT = pd.Timestamp("2024-11-20")


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta, y - X @ beta


def hac_se(X, resid, L):
    n, k = X.shape
    g = X * resid[:, None]
    S = g.T @ g
    for lag in range(1, min(L, n - 1) + 1):
        w = 1.0 - lag / (L + 1)
        A = g[lag:].T @ g[:-lag]
        S += w * (A + A.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv
    d = np.diag(V)
    return np.sqrt(np.where(d > 0, d, np.nan))


def reg(y, Xcols, names, L=10, label=""):
    X = np.column_stack([np.ones(len(y))] + Xcols)
    beta, resid = ols(X, y)
    se = hac_se(X, resid, L)
    r2 = 1 - resid @ resid / ((y - y.mean()) ** 2).sum()
    print(f"\n  {label}   n={len(y):,}  R2={r2:.4f}  (NW L={L})")
    print(f"    {'term':<26}{'coef':>12}{'se':>10}{'t':>8}")
    for nm, b, s in zip(["const"] + names, beta, se):
        t = b / s if s and np.isfinite(s) else np.nan
        print(f"    {nm:<26}{b:>12.4f}{s:>10.4f}{t:>8.2f}")
    return beta, se


def main() -> int:
    pp = pd.read_parquet(PROC / "participant_panel.parquet")
    ix = pd.read_parquet(PROC / "index_panel.parquet")

    w = pp.pivot_table(index="date", columns="client_type",
                       values="net_options", aggfunc="first") / 1e6
    w = w.rename(columns={c: f"net_{c.lower()}" for c in w.columns})
    gross = pp[pp.client_type == "TOTAL"].set_index("date")["gross_options"] / 1e6
    w["gross_total"] = gross
    for c in ["net_client", "net_fii", "net_dii", "net_pro"]:
        if c in w:
            w[c + "_scaled"] = w[c] / w["gross_total"]

    nifty = ix[ix["index"] == "NIFTY"].set_index("date").sort_index()
    d = w.join(nifty[["ret", "park", "rv5", "rv21", "absret"]], how="inner").dropna(
        subset=["net_client", "park"])
    d = d.sort_index()
    print(f"merged daily sample: n={len(d):,}  "
          f"{d.index.min():%Y-%m-%d} .. {d.index.max():%Y-%m-%d}")

    d["rv_fwd5"] = d["ret"].shift(-1).rolling(5).std().shift(-4) * np.sqrt(252)
    d["park_fwd1"] = d["park"].shift(-1)

    print("\n" + "=" * 78)
    print("T1  DOES PARTICIPANT NET OPTIONALITY FORECAST VOLATILITY?  (descriptive)")
    print("=" * 78)
    sub = d.dropna(subset=["park_fwd1", "net_client_scaled", "park"])
    y = sub["park_fwd1"].to_numpy(float)
    reg(y,
        [sub["net_client_scaled"].to_numpy(float), sub["park"].to_numpy(float)],
        ["net_client/gross", "park_t"], label="next-day Parkinson vol")

    sub2 = d.dropna(subset=["rv_fwd5", "net_client_scaled", "rv5"])
    reg(sub2["rv_fwd5"].to_numpy(float),
        [sub2["net_client_scaled"].to_numpy(float), sub2["rv5"].to_numpy(float)],
        ["net_client/gross", "rv5_t"], L=15, label="next-5d realised vol")

    print("\n" + "=" * 78)
    print("T2  PLACEBO: does volatility forecast positioning? (reverse direction)")
    print("=" * 78)
    sub3 = d.dropna(subset=["net_client_scaled", "park"]).copy()
    sub3["nc_fwd"] = sub3["net_client_scaled"].shift(-1)
    sub3 = sub3.dropna(subset=["nc_fwd"])
    reg(sub3["nc_fwd"].to_numpy(float),
        [sub3["park"].to_numpy(float), sub3["net_client_scaled"].to_numpy(float)],
        ["park_t", "net_client_t"], label="next-day net client positioning")

    print("\n" + "=" * 78)
    print("T3  SEBI EVENT: weekly options withdrawn from 3 indices, NIFTY retains")
    print("=" * 78)
    p = ix[ix["index"].isin(TREATED + CONTROL)].copy()
    p = p.dropna(subset=["park"])
    p["treated"] = p["index"].isin(TREATED).astype(int)
    p["post"] = (p["date"] >= EVENT).astype(int)
    win = p[(p["date"] >= EVENT - pd.Timedelta(days=365)) &
            (p["date"] <= EVENT + pd.Timedelta(days=365))].copy()
    print(f"  window {win['date'].min():%Y-%m-%d} .. {win['date'].max():%Y-%m-%d}"
          f"   obs={len(win):,}")
    print("\n  mean annualised Parkinson vol:")
    tab = win.pivot_table(index="index", columns="post", values="park", aggfunc="mean")
    tab.columns = ["pre", "post"]
    tab["diff"] = tab["post"] - tab["pre"]
    print(tab.round(4).to_string())
    tr = tab.loc[[i for i in TREATED if i in tab.index], "diff"].mean()
    ct = tab.loc[[i for i in CONTROL if i in tab.index], "diff"].mean()
    print(f"\n  DiD (mean treated change - control change) = {tr - ct:+.4f} "
          f"annualised vol points")

    win["lpark"] = np.log(win["park"].clip(lower=1e-6))
    idx_d = pd.get_dummies(win["index"], prefix="ix", drop_first=True).astype(float)
    yy = win["lpark"].to_numpy(float)
    XX = np.column_stack([np.ones(len(win)),
                          (win["treated"] * win["post"]).to_numpy(float),
                          win["post"].to_numpy(float),
                          idx_d.to_numpy(float)])
    beta, resid = ols(XX, yy)
    XtX_inv = np.linalg.inv(XX.T @ XX)
    S = np.zeros((XX.shape[1], XX.shape[1]))
    for g in win["index"].unique():
        m = (win["index"] == g).to_numpy()
        u = (XX[m] * resid[m, None]).sum(axis=0)
        S += np.outer(u, u)
    G = win["index"].nunique()
    V = XtX_inv @ (S * G / max(G - 1, 1)) @ XtX_inv
    se = np.sqrt(np.diag(V))
    print(f"\n  log-vol DiD: treated x post = {beta[1]:+.4f} "
          f"(cluster-by-index se {se[1]:.4f}, t={beta[1]/se[1]:+.2f}, G={G})")
    print("  NOTE: only 4 clusters -- cluster-robust inference is unreliable here.")

    pre = p[(p["date"] >= EVENT - pd.Timedelta(days=365)) & (p["date"] < EVENT)].copy()
    pre["t"] = (pre["date"] - pre["date"].min()).dt.days
    pre["lpark"] = np.log(pre["park"].clip(lower=1e-6))
    Xp = np.column_stack([np.ones(len(pre)), pre["t"] * pre["treated"],
                          pre["t"], pre["treated"]])
    bp, rp = ols(Xp, pre["lpark"].to_numpy(float))
    sep = hac_se(Xp, rp, 20)
    print(f"  PRE-TREND (treated x time) = {bp[1]:+.6f} per day "
          f"(NW se {sep[1]:.6f}, t={bp[1]/sep[1]:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
