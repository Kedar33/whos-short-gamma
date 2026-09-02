from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
EVENT = pd.Timestamp("2024-11-20")


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
    Xi = np.linalg.inv(X.T @ X)
    V = Xi @ S @ Xi
    d = np.diag(V)
    return np.sqrt(np.where(d > 0, d, np.nan))


def main() -> int:
    pp = pd.read_parquet(PROC / "participant_panel.parquet")
    net = pp.pivot_table(index="date", columns="client_type",
                         values="net_options", aggfunc="first")
    gross_by = pp.pivot_table(index="date", columns="client_type",
                              values="gross_options", aggfunc="first")
    tot = gross_by["TOTAL"]

    d = pd.DataFrame(index=net.index).sort_index()
    for ct in ["Client", "FII", "DII", "Pro"]:
        if ct in net:
            d[ct] = net[ct] / tot
    d["mktsize"] = tot / 1e6
    d = d.dropna()

    print(f"sample n={len(d):,}  {d.index.min():%Y-%m-%d} .. {d.index.max():%Y-%m-%d}")
    print(f"event: {EVENT:%Y-%m-%d} (weekly expiries withdrawn; lot size tripled; "
          f"+2% ELM)")

    for win in (120, 250):
        pre = d[(d.index < EVENT) & (d.index >= EVENT - pd.Timedelta(days=win * 2))]
        post = d[(d.index >= EVENT) & (d.index <= EVENT + pd.Timedelta(days=win * 2))]
        print(f"\n=== +/-{win} trading-day-equivalent window "
              f"(n_pre={len(pre)}, n_post={len(post)}) ===")
        print(f"{'participant':<12}{'pre':>10}{'post':>10}{'change':>10}"
              f"{'t (NW)':>10}")
        for ct in ["Client", "FII", "DII", "Pro"]:
            if ct not in d:
                continue
            sub = pd.concat([pre[[ct]], post[[ct]]])
            X = np.column_stack([np.ones(len(sub)),
                                 (sub.index >= EVENT).astype(float)])
            b, r = ols(X, sub[ct].to_numpy(float))
            se = hac_se(X, r, 20)
            t = b[1] / se[1] if np.isfinite(se[1]) and se[1] > 0 else np.nan
            print(f"{ct:<12}{pre[ct].mean():>10.4f}{post[ct].mean():>10.4f}"
                  f"{b[1]:>+10.4f}{t:>10.2f}")

    print("\n=== market size (gross index-option OI, millions of contracts) ===")
    for win in (250,):
        pre = d[(d.index < EVENT) & (d.index >= EVENT - pd.Timedelta(days=win * 2))]
        post = d[(d.index >= EVENT) & (d.index <= EVENT + pd.Timedelta(days=win * 2))]
        print(f"  pre  mean = {pre['mktsize'].mean():.3f}")
        print(f"  post mean = {post['mktsize'].mean():.3f}")
        print(f"  change    = {post['mktsize'].mean()-pre['mktsize'].mean():+.3f} "
              f"({(post['mktsize'].mean()/pre['mktsize'].mean()-1):+.1%})")

    print("\n=== quarterly path of net optionality share ===")
    q = d.resample("QE").mean()
    print(q.round(4).to_string())

    print("\n=== gross positions by participant (millions), around event ===")
    g = gross_by.copy() / 1e6
    pre = g[(g.index < EVENT) & (g.index >= EVENT - pd.Timedelta(days=500))]
    post = g[(g.index >= EVENT) & (g.index <= EVENT + pd.Timedelta(days=500))]
    for ct in ["Client", "FII", "DII", "Pro", "TOTAL"]:
        if ct in g:
            print(f"  {ct:<8} pre={pre[ct].mean():8.3f}  post={post[ct].mean():8.3f}"
                  f"  {(post[ct].mean()/pre[ct].mean()-1):+8.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
