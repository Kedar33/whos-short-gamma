from __future__ import annotations

import csv
import glob
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

H = 21


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b, y - X @ b


def hac_se(X, resid, L):
    g = X * resid[:, None]
    S = g.T @ g
    for lag in range(1, min(L, len(X) - 1) + 1):
        w = 1.0 - lag / (L + 1)
        A = g[lag:].T @ g[:-lag]
        S += w * (A + A.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    V = XtX_inv @ S @ XtX_inv
    d = np.diag(V)
    return np.sqrt(np.where(d > 0, d, np.nan))


def load_vix() -> pd.Series:
    rows = {}
    for f in sorted(glob.glob(str(RAW / "idx_*.csv"))):
        if os.path.getsize(f) == 0:
            continue
        try:
            txt = open(f, encoding="utf8", errors="replace").read()
            for r in csv.DictReader(io.StringIO(txt)):
                if (r.get("Index Name") or "").strip().upper() == "INDIA VIX":
                    d = pd.to_datetime(r["Index Date"], dayfirst=True, errors="coerce")
                    v = r.get("Closing Index Value", "")
                    if pd.notna(d) and v not in ("", "-", None):
                        rows[d] = float(str(v).replace(",", ""))
        except Exception:
            continue
    return pd.Series(rows).sort_index()


def main() -> int:
    vix = load_vix()
    print(f"India VIX observations: {len(vix):,}  "
          f"{vix.index.min():%Y-%m-%d} .. {vix.index.max():%Y-%m-%d}")
    print(f"  VIX level: mean={vix.mean():.2f} median={vix.median():.2f} "
          f"min={vix.min():.2f} max={vix.max():.2f}")

    ix = pd.read_parquet(PROC / "index_panel.parquet")
    nifty = ix[ix["index"] == "NIFTY"].set_index("date").sort_index()
    d = pd.DataFrame({"vix": vix}).join(nifty[["ret", "close"]], how="inner").dropna()

    r = d["ret"].to_numpy(float)
    n = len(r)
    fwd = np.full(n, np.nan)
    for i in range(n - H):
        seg = r[i + 1: i + 1 + H]
        if len(seg) == H:
            fwd[i] = seg.std(ddof=1) * np.sqrt(252) * 100
    d["rv_fwd"] = fwd
    d = d.dropna(subset=["rv_fwd"])
    d["vrp"] = d["vix"] - d["rv_fwd"]
    d["vrp_var"] = d["vix"] ** 2 - d["rv_fwd"] ** 2

    print(f"\nmatched sample n={len(d):,}  "
          f"{d.index.min():%Y-%m-%d} .. {d.index.max():%Y-%m-%d}")
    print(f"  mean implied (VIX)          = {d['vix'].mean():7.3f}")
    print(f"  mean forward realised vol   = {d['rv_fwd'].mean():7.3f}")
    print(f"  mean VRP (vol points)       = {d['vrp'].mean():+7.3f}")
    print(f"  median VRP                  = {d['vrp'].median():+7.3f}")
    print(f"  share of days VRP > 0       = {(d['vrp']>0).mean():7.1%}")

    for L, lbl in [(H, f"NW L={H}"), (2 * H, f"NW L={2*H}")]:
        X = np.ones((len(d), 1))
        b, res = ols(X, d["vrp"].to_numpy(float))
        se = hac_se(X, res, L)[0]
        print(f"  mean VRP {lbl:<10}: {b[0]:+.3f}  se={se:.3f}  t={b[0]/se:+.2f}")

    print("\n=== VRP BY YEAR (vol points, IV - subsequent RV) ===")
    d["yr"] = d.index.year
    g = d.groupby("yr").agg(n=("vrp", "size"), vix=("vix", "mean"),
                            rv=("rv_fwd", "mean"), vrp=("vrp", "mean"),
                            pos=("vrp", lambda x: (x > 0).mean()))
    print(g.round(3).to_string())

    ex = d[(d.index < "2020-02-01") | (d.index > "2020-06-30")]
    X = np.ones((len(ex), 1))
    b, res = ols(X, ex["vrp"].to_numpy(float))
    se = hac_se(X, res, 2 * H)[0]
    print(f"\n  excluding Feb-Jun 2020 (COVID): n={len(ex):,}  "
          f"mean VRP={b[0]:+.3f}  se={se:.3f}  t={b[0]/se:+.2f}")

    post = d[d.index >= "2024-11-20"]
    if len(post) > 50:
        X = np.ones((len(post), 1))
        b, res = ols(X, post["vrp"].to_numpy(float))
        se = hac_se(X, res, 2 * H)[0]
        print(f"  post-SEBI (>=2024-11-20)      : n={len(post):,}  "
              f"mean VRP={b[0]:+.3f}  se={se:.3f}  t={b[0]/se:+.2f}")

    pp = pd.read_parquet(PROC / "participant_panel.parquet")
    w = pp.pivot_table(index="date", columns="client_type",
                       values="net_options", aggfunc="first")
    gross = pp[pp.client_type == "TOTAL"].set_index("date")["gross_options"]
    if "Client" in w:
        nc = (w["Client"] / gross).rename("net_client_scaled")
        dd = d.join(nc, how="inner").dropna(subset=["net_client_scaled"])
        X = np.column_stack([np.ones(len(dd)),
                             dd["net_client_scaled"].to_numpy(float)])
        b, res = ols(X, dd["vrp"].to_numpy(float))
        se = hac_se(X, res, 2 * H)
        print(f"\n=== VRP on retail net optionality (n={len(dd):,}) ===")
        print(f"  const            = {b[0]:+.3f}  se={se[0]:.3f}  t={b[0]/se[0]:+.2f}")
        print(f"  net_client/gross = {b[1]:+.3f}  se={se[1]:.3f}  t={b[1]/se[1]:+.2f}")

    d.to_parquet(PROC / "vrp.parquet")
    print(f"\nwrote {(PROC/'vrp.parquet').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
