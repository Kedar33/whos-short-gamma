from __future__ import annotations

import csv
import io
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nse
from gamma_identification import (load_cells, load_participants, col,
                                  PARTICIPANTS)

ROOT = Path(__file__).resolve().parents[1]
MONEY_BINS = np.array([-np.inf, -.15, -.10, -.07, -.05, -.03, -.015, 0,
                       .015, .03, .05, .07, .10, .15, np.inf])
TENOR_BINS = np.array([0, 7, 21, 60, 1e9]) / 365.0


def bucketise(cells, wing):
    sub = [c for c in cells if (c["cp"] == "CE") == (wing == "C")]
    agg = {}
    for c in sub:
        m = math.log(c["K"] / c["S"])
        mb = int(np.digitize(m, MONEY_BINS)) - 1
        tb = int(np.digitize(c["T"], TENOR_BINS)) - 1
        key = (c["sym"], tb, mb)
        a = agg.setdefault(key, {"oi": 0.0, "gsum": 0.0})
        a["oi"] += c["oi"]
        a["gsum"] += c["gnotional"] * c["oi"]
    out = []
    for k, a in agg.items():
        if a["oi"] <= 0:
            continue
        out.append({"key": k, "oi": a["oi"], "g": a["gsum"] / a["oi"]})
    return out


RECONCILE_TOL = 0.02


def raw_contract_total(d, wing):
    b = nse.fo_bhav(d)
    if not b:
        return None
    want = "CE" if wing == "C" else "PE"
    tot = 0.0
    for r in csv.DictReader(io.StringIO(b.decode("utf8", "replace"))):
        if r.get("FinInstrmTp") != "IDO" or r.get("OptnTp") != want:
            continue
        try:
            oi = float(r["OpnIntrst"]); lot = float(r.get("NewBrdLotQty") or 0)
        except (ValueError, TypeError, KeyError):
            continue
        if lot > 0:
            tot += oi / lot
    return tot


def totals(part, wing, scale_to, raw_total=None):
    nm = "Option Index Call" if wing == "C" else "Option Index Put"
    L = np.array([col(part[p], f"{nm} Long") or 0.0 for p in PARTICIPANTS])
    S = np.array([col(part[p], f"{nm} Short") or 0.0 for p in PARTICIPANTS])
    if L.sum() <= 0 or S.sum() <= 0:
        return None, None
    basis = raw_total if raw_total is not None else scale_to
    if abs(basis / L.sum() - 1.0) > RECONCILE_TOL:
        return None, None
    return L * scale_to / L.sum(), S * scale_to / S.sum()


def vol_totals(d, wing):
    raw = nse.participant_vol(d)
    if not raw:
        return None
    lines = [l for l in raw.decode("utf8", "replace").splitlines() if l.strip()]
    if len(lines) < 3:
        return None
    hdr = [h.strip().strip('"') for h in lines[1].split(",")]
    rec = {}
    for line in lines[2:]:
        p = [x.strip().strip('"') for x in line.split(",")]
        d2 = {}
        for h, v in zip(hdr[1:], p[1:]):
            try:
                d2[h.strip()] = float(v) if v not in ("", "-") else 0.0
            except ValueError:
                d2[h.strip()] = 0.0
        rec[p[0]] = d2
    nm = "Option Index Call" if wing == "C" else "Option Index Put"
    out = []
    for p in PARTICIPANTS:
        if p not in rec:
            return None
        lo = col(rec[p], f"{nm} Long") or 0.0
        sh = col(rec[p], f"{nm} Short") or 0.0
        out.append(lo + sh)
    return np.array(out)


def single_period(buckets, Lbar, Sbar):
    n, P = len(buckets), len(PARTICIPANTS)
    oi = np.array([b["oi"] for b in buckets])
    g = np.array([b["g"] for b in buckets])
    N = 2 * P * n
    A = lil_matrix((2 * n + 2 * P, N)); b = np.zeros(2 * n + 2 * P)
    r = 0
    for i in range(n):
        for p in range(P):
            A[r, p * n + i] = 1.0
        b[r] = oi[i]; r += 1
        for p in range(P):
            A[r, P * n + p * n + i] = 1.0
        b[r] = oi[i]; r += 1
    for p in range(P):
        for i in range(n):
            A[r, p * n + i] = 1.0
        b[r] = Lbar[p]; r += 1
        for i in range(n):
            A[r, P * n + p * n + i] = 1.0
        b[r] = Sbar[p]; r += 1
    A = A.tocsr()
    res = {}
    for p, nm in enumerate(PARTICIPANTS):
        c = np.zeros(N)
        c[p * n:(p + 1) * n] = g
        c[P * n + p * n:P * n + (p + 1) * n] = -g
        lo = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        hi = linprog(-c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
        if lo.success and hi.success:
            res[nm] = (float(c @ lo.x), float(c @ hi.x))
    return res


def two_period(bk0, bk1, L0, S0, L1, S1, vol1):
    keys = sorted(set(b["key"] for b in bk0) | set(b["key"] for b in bk1))
    if len(keys) < 5:
        return None
    m0 = {b["key"]: b for b in bk0}; m1 = {b["key"]: b for b in bk1}
    n, P = len(keys), len(PARTICIPANTS)
    oi0 = np.array([m0[k]["oi"] if k in m0 else 0.0 for k in keys])
    oi1 = np.array([m1[k]["oi"] if k in m1 else 0.0 for k in keys])
    g1 = np.array([m1[k]["g"] if k in m1 else 0.0 for k in keys])
    L0 = L0 * oi0.sum() / max(L0.sum(), 1e-9); S0 = S0 * oi0.sum() / max(S0.sum(), 1e-9)
    L1 = L1 * oi1.sum() / max(L1.sum(), 1e-9); S1 = S1 * oi1.sum() / max(S1.sum(), 1e-9)

    B = P * n
    N = 6 * B
    def o(blk): return blk * B
    rows, bvals = [], []
    A = lil_matrix((2 * n * 2 + 4 * P + 4 * B + P, N)); bv = np.zeros(A.shape[0])
    Aub = lil_matrix((4 * B + P, N)); bub = np.zeros(4 * B + P)
    r = 0
    for (blkL, blkS, oi) in ((0, 1, oi0), (2, 3, oi1)):
        for i in range(n):
            for p in range(P):
                A[r, o(blkL) + p * n + i] = 1.0
            bv[r] = oi[i]; r += 1
            for p in range(P):
                A[r, o(blkS) + p * n + i] = 1.0
            bv[r] = oi[i]; r += 1
    for (blk, tot) in ((0, L0), (1, S0), (2, L1), (3, S1)):
        for p in range(P):
            for i in range(n):
                A[r, o(blk) + p * n + i] = 1.0
            bv[r] = tot[p]; r += 1
    A = A[:r].tocsr(); bv = bv[:r]
    ru = 0
    for p in range(P):
        for i in range(n):
            j = p * n + i
            for (blk_new, blk_old, blk_d) in ((2, 0, 4), (3, 1, 5)):
                Aub[ru, o(blk_new) + j] = 1.0
                Aub[ru, o(blk_old) + j] = -1.0
                Aub[ru, o(blk_d) + j] = -1.0
                bub[ru] = 0.0; ru += 1
                Aub[ru, o(blk_new) + j] = -1.0
                Aub[ru, o(blk_old) + j] = 1.0
                Aub[ru, o(blk_d) + j] = -1.0
                bub[ru] = 0.0; ru += 1
    for p in range(P):
        for i in range(n):
            Aub[ru, o(4) + p * n + i] = 1.0
            Aub[ru, o(5) + p * n + i] = 1.0
        bub[ru] = 2.0 * vol1[p]; ru += 1
    Aub = Aub[:ru].tocsr(); bub = bub[:ru]

    res = {}
    for p, nm in enumerate(PARTICIPANTS):
        c = np.zeros(N)
        c[o(2) + p * n:o(2) + (p + 1) * n] = g1
        c[o(3) + p * n:o(3) + (p + 1) * n] = -g1
        lo = linprog(c, A_eq=A, b_eq=bv, A_ub=Aub, b_ub=bub,
                     bounds=(0, None), method="highs")
        hi = linprog(-c, A_eq=A, b_eq=bv, A_ub=Aub, b_ub=bub,
                     bounds=(0, None), method="highs")
        if lo.success and hi.success:
            res[nm] = (float(c @ lo.x), float(c @ hi.x))
    return res


def next_bday(d):
    d2 = d + timedelta(days=1)
    while d2.weekday() >= 5:
        d2 += timedelta(days=1)
    return d2


def main() -> int:
    windows = [(date(2024, 12, 1), date(2025, 10, 31)),
               (date(2026, 2, 1), date(2026, 5, 31))]
    pairs = []
    for lo, hi in windows:
        d = lo
        while d <= hi:
            if d.weekday() < 5:
                pairs.append((d, next_bday(d)))
            d += timedelta(days=11)
    print(f"panel: {len(pairs)} date pairs\n")
    print(f"{'date':>12}{'wing':>5}{'part':>8}{'1P lo':>13}{'1P hi':>13}"
          f"{'1P width':>13}{'2P width':>13}{'shrink':>8}{'0?':>4}")
    print("-" * 92)
    out = []
    for d0, d1 in pairs:
        c0, c1 = load_cells(d0), load_cells(d1)
        p0, p1 = load_participants(d0), load_participants(d1)
        if not (c0 and c1 and p0 and p1):
            continue
        for wing in ("C", "P"):
            b0, b1 = bucketise(c0, wing), bucketise(c1, wing)
            if len(b0) < 5 or len(b1) < 5:
                continue
            L0, S0 = totals(p0, wing, sum(b["oi"] for b in b0),
                            raw_contract_total(d0, wing))
            L1, S1 = totals(p1, wing, sum(b["oi"] for b in b1),
                            raw_contract_total(d1, wing))
            v1 = vol_totals(d1, wing)
            if L0 is None or L1 is None:
                continue
            r1 = single_period(b1, L1, S1)
            r2 = two_period(b0, b1, L0, S0, L1, S1, v1) if v1 is not None else None
            for nm in PARTICIPANTS:
                if nm not in r1:
                    continue
                lo, hi = r1[nm]; w1 = hi - lo
                w2 = (r2[nm][1] - r2[nm][0]) if (r2 and nm in r2) else np.nan
                sh = (1 - w2 / w1) * 100 if (w1 > 0 and np.isfinite(w2)) else np.nan
                z = "YES" if lo < 0 < hi else ""
                z2 = (r2 and nm in r2 and r2[nm][0] < 0 < r2[nm][1])
                print(f"{str(d1):>12}{wing:>5}{nm:>8}{lo/1e6:>13.1f}{hi/1e6:>13.1f}"
                      f"{w1/1e6:>13.1f}{w2/1e6:>13.1f}{sh:>7.1f}%{z:>4}")
                out.append({"date": str(d1), "wing": wing, "participant": nm,
                            "lo1": lo, "hi1": hi, "w1": w1, "w2": float(w2),
                            "shrink_pct": float(sh),
                            "spans_zero_1p": bool(lo < 0 < hi),
                            "spans_zero_2p": bool(z2)})
    if out:
        (ROOT / "data" / "processed" / "gamma_panel.json").write_text(
            json.dumps(out, indent=2))
        core = [o for o in out if o["participant"] in ("Client", "FII", "Pro")]
        s1 = sum(o["spans_zero_1p"] for o in core)
        solved = [o for o in core if np.isfinite(o["w2"])]
        s2 = sum(o["spans_zero_2p"] for o in solved)
        shr = np.array([o["shrink_pct"] for o in solved
                        if np.isfinite(o["shrink_pct"])])
        print("\n" + "=" * 92)
        print(f"core participant-wing-days (Client/FII/Pro): {len(core)}")
        print(f"  spans zero, market+totals only : {s1} / {len(core)} "
              f"({s1/len(core):.1%})")
        print(f"  two-period LP solved           : {len(solved)} / {len(core)} "
              f"({len(solved)/len(core):.1%})   [failures NOT counted as identified]")
        if solved:
            print(f"  spans zero, + turnover limits  : {s2} / {len(solved)} "
                  f"({s2/len(solved):.1%})  [conditional on solving]")
        if len(shr):
            print(f"  median width shrinkage from turnover: {np.median(shr):.2f}%"
                  f"   max {shr.max():.2f}%")
        print("wrote data/processed/gamma_panel.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
