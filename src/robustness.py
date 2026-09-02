from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamma_identification import (load_cells, load_participants, bs_gamma,
                                  PARTICIPANTS)
from gamma_panel import totals, raw_contract_total
from flip_level import endpoints

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20260831)

RESOLUTIONS = {
    "coarse (6 bins)": np.array([-np.inf, -.10, -.04, 0, .04, .10, np.inf]),
    "medium (10 bins)": np.array([-np.inf, -.15, -.10, -.05, -.02, 0,
                                  .02, .05, .10, .15, np.inf]),
    "fine (14 bins)": np.array([-np.inf, -.15, -.10, -.07, -.05, -.03, -.015, 0,
                                .015, .03, .05, .07, .10, .15, np.inf]),
}


def bucket_with(cells, bins):
    agg = {}
    for c in cells:
        m = np.log(c["K"] / c["S"])
        mb = int(np.digitize(m, bins)) - 1
        key = (c["sym"], mb)
        a = agg.setdefault(key, {"oi": 0.0, "gsum": 0.0})
        a["oi"] += c["oi"]
        a["gsum"] += c["gnotional"] * c["oi"]
    g = np.array([a["gsum"] / a["oi"] for a in agg.values() if a["oi"] > 0])
    oi = np.array([a["oi"] for a in agg.values() if a["oi"] > 0])
    return g, oi


def perturb_iv(cells, pct):
    out = []
    for c in cells:
        if not c.get("iv") or c["iv"] <= 0:
            continue
        sig = c["iv"] * (1.0 + RNG.uniform(-pct, pct))
        g = bs_gamma(c["S"], c["K"], c["T"], sig)
        d = dict(c); d["gnotional"] = g * c["S"] ** 2 / 100.0
        out.append(d)
    return out


def main() -> int:
    for d in (date(2025, 6, 17), date(2026, 4, 16)):
        cells = load_cells(d)
        part = load_participants(d)
        if not cells or not part:
            print(f"{d}: unavailable"); continue
        sub = [c for c in cells if c["cp"] == "CE"]
        oi_tot = sum(c["oi"] for c in sub)
        L, S = totals(part, "C", oi_tot, raw_contract_total(d, "C"))
        if L is None:
            print(f"{d}: gate failed"); continue
        p = PARTICIPANTS.index("Pro")
        print(f"\n=== {d}   Pro call wing   Lbar={L[p]:,.0f} Sbar={S[p]:,.0f} ===")

        print("\n  ITEM 3 -- width vs bucket resolution (coarser must be narrower)")
        print(f"    {'resolution':<20}{'cells':>7}{'lo':>14}{'hi':>14}{'width':>14}{'0?':>5}")
        widths = []
        for nm, bins in RESOLUTIONS.items():
            g, oi = bucket_with(sub, bins)
            lo, hi = endpoints(g, oi, L[p], S[p])
            widths.append(hi - lo)
            print(f"    {nm:<20}{len(g):>7}{lo/1e6:>14.2f}{hi/1e6:>14.2f}"
                  f"{(hi-lo)/1e6:>14.2f}{'YES' if lo<0<hi else 'no':>5}")
        mono = all(widths[i] <= widths[i + 1] + 1e-9 for i in range(len(widths) - 1))
        print(f"    monotone in resolution: {mono}  "
              f"(coarse width / fine width = {widths[0]/widths[-1]:.3f})")

        print("\n  ITEM 4 -- IV perturbation +/-20% (endpoints depend on RANK)")
        g0, oi0 = bucket_with(sub, RESOLUTIONS["fine (14 bins)"])
        lo0, hi0 = endpoints(g0, oi0, L[p], S[p])
        los, his = [], []
        for _ in range(50):
            pc = perturb_iv(sub, 0.20)
            g1, oi1 = bucket_with(pc, RESOLUTIONS["fine (14 bins)"])
            lo1, hi1 = endpoints(g1, oi1, L[p], S[p])
            los.append(lo1); his.append(hi1)
        los, his = np.array(los), np.array(his)
        print(f"    baseline           : [{lo0/1e6:>10.2f}, {hi0/1e6:>10.2f}]")
        print(f"    perturbed  median  : [{np.median(los)/1e6:>10.2f}, "
              f"{np.median(his)/1e6:>10.2f}]")
        print(f"    perturbed  5-95pct : lo [{np.percentile(los,5)/1e6:.2f}, "
              f"{np.percentile(los,95)/1e6:.2f}]   "
              f"hi [{np.percentile(his,5)/1e6:.2f}, {np.percentile(his,95)/1e6:.2f}]")
        rel = np.median(np.abs(his - hi0)) / abs(hi0) * 100
        print(f"    median |change| in upper endpoint: {rel:.2f}% of baseline")
        print(f"    spans zero in {int(((los<0)&(his>0)).sum())}/50 perturbed draws")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
