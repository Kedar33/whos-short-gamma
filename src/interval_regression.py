from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamma_identification import load_participants, PARTICIPANTS
from gamma_panel import totals
from flip_level import endpoints
from fast_cells import load_fast, bucket_fast

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RNG = np.random.default_rng(20260901)

WINDOWS = [(date(2024, 12, 1), date(2025, 10, 31)),
           (date(2026, 2, 1), date(2026, 5, 31))]


def daily_gamma_intervals(participant="Pro"):
    rows = []
    p = PARTICIPANTS.index(participant)
    for lo_d, hi_d in WINDOWS:
        d = lo_d
        while d <= hi_d:
            if d.weekday() >= 5:
                d += timedelta(days=1); continue
            cells, raw_tot = load_fast(d)
            part = load_participants(d)
            if cells is None or not part:
                d += timedelta(days=1); continue
            tot_lo = tot_hi = 0.0
            ok = True
            for wing in ("C", "P"):
                g, oi = bucket_fast(cells, wing)
                if g is None or len(g) < 5:
                    ok = False; break
                L, S = totals(part, wing, float(oi.sum()), raw_tot[wing])
                if L is None:
                    ok = False; break
                a, b = endpoints(g, oi, L[p], S[p])
                tot_lo += a; tot_hi += b
            if ok:
                rows.append({"date": pd.Timestamp(d), "lo": tot_lo, "hi": tot_hi})
            if len(rows) and len(rows) % 25 == 0:
                print(f"    {len(rows)} days built ...", flush=True)
            d += timedelta(days=1)
    return pd.DataFrame(rows)


def ols_slope(x, y):
    xm, ym = x - x.mean(), y - y.mean()
    den = xm @ xm
    return (xm @ ym) / den if den > 1e-18 else np.nan


def identified_set_beta(lo, hi, y, n_random=4000, n_restarts=40):
    n = len(y)
    vals = []

    def push(g):
        b = ols_slope(g, y)
        if np.isfinite(b):
            vals.append(b)
        return b

    for g in (0.5 * (lo + hi), lo.copy(), hi.copy()):
        push(g)

    for direction in (+1, -1):
        for _ in range(n_restarts):
            g = np.where(RNG.random(n) < 0.5, lo, hi).astype(float)
            cur = ols_slope(g, y)
            for _sweep in range(3):
                improved = False
                for t in RNG.permutation(n):
                    old = g[t]
                    for cand in (lo[t], hi[t]):
                        if cand == old:
                            continue
                        g[t] = cand
                        b = ols_slope(g, y)
                        if np.isfinite(b) and direction * b > direction * cur + 1e-15:
                            cur = b; improved = True; break
                        g[t] = old
                if not improved:
                    break
            push(g)

    for _ in range(n_random):
        u = RNG.random(n)
        push(lo + u * (hi - lo))

    v = np.array(vals)
    return float(v.min()), float(v.max()), v


def main() -> int:
    print("building daily gamma intervals (Theorem 2 closed form) ...", flush=True)
    gi = daily_gamma_intervals("Pro")
    if len(gi) < 30:
        print(f"insufficient days ({len(gi)})"); return 1
    print(f"  days with a valid interval: {len(gi)}")

    ix = pd.read_parquet(PROC / "index_panel.parquet")
    nifty = ix[ix["index"] == "NIFTY"][["date", "ret", "park"]].copy()
    nifty["date"] = pd.to_datetime(nifty["date"]).dt.tz_localize(None)
    gi["date"] = pd.to_datetime(gi["date"]).dt.tz_localize(None)
    df = gi.merge(nifty, on="date", how="inner").dropna(subset=["park"])
    df = df.sort_values("date").reset_index(drop=True)
    df["park_next"] = df["park"].shift(-1)
    df = df.dropna(subset=["park_next"]).reset_index(drop=True)
    print(f"  merged with NIFTY volatility: n = {len(df)}")

    lo = df["lo"].to_numpy(float) / 1e6
    hi = df["hi"].to_numpy(float) / 1e6
    y = df["park_next"].to_numpy(float)

    print(f"\n  gamma interval width: median {np.median(hi-lo):,.1f} mn")
    print(f"  every interval contains zero: {bool(np.all((lo<0)&(hi>0)))}")

    b_lo, b_hi, v = identified_set_beta(lo, hi, y)
    b_mid = ols_slope(0.5 * (lo + hi), y)

    print("\n" + "=" * 72)
    print("IDENTIFIED SET FOR beta IN  RV_{t+1} = alpha + beta * G_t + eps")
    print("=" * 72)
    print(f"  midpoint-plug-in estimate (what a researcher would report): "
          f"{b_mid:+.6f}")
    print(f"  identified set (inner approximation): "
          f"[{b_lo:+.6f}, {b_hi:+.6f}]")
    print(f"  contains zero: {b_lo < 0 < b_hi}")
    print(f"  selections evaluated: {len(v):,}")
    print(f"  share of selections giving beta > 0: {(v>0).mean():.1%}")

    sd_g = np.median(hi - lo) / 4.0
    print(f"\n  a one-standard-deviation move in G shifts next-day volatility by")
    print(f"    between {b_lo*sd_g:+.4f} and {b_hi*sd_g:+.4f} annualised vol points")
    print(f"    (midpoint plug-in would say {b_mid*sd_g:+.4f})")

    out = {"n": int(len(df)), "beta_mid": b_mid, "beta_lo": b_lo,
           "beta_hi": b_hi, "contains_zero": bool(b_lo < 0 < b_hi),
           "n_selections": int(len(v)),
           "share_positive": float((v > 0).mean())}
    (PROC / "interval_regression.json").write_text(json.dumps(out, indent=2))
    df.to_csv(PROC / "gamma_vol_daily.csv", index=False)
    print("\nwrote data/processed/interval_regression.json")
    print("STATUS: inner approximation; the true identified set is at least "
          "this wide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
