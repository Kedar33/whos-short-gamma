from __future__ import annotations

import csv
import io
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nse

ROOT = Path(__file__).resolve().parents[1]
INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
PARTICIPANTS = ["Client", "DII", "FII", "Pro"]


def bs_gamma(S, K, T, sigma, r=0.065):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm.pdf(d1) / (S * sigma * math.sqrt(T))


def implied_vol(price, S, K, T, cp, r=0.065):
    if T <= 0 or price <= 0:
        return None
    def bs(sig):
        d1 = (math.log(S / K) + (r + 0.5 * sig ** 2) * T) / (sig * math.sqrt(T))
        d2 = d1 - sig * math.sqrt(T)
        if cp == "CE":
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    lo, hi = 1e-3, 5.0
    if bs(hi) < price or bs(lo) > price:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs(mid) < price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load_cells(d: date):
    raw = nse.fo_bhav(d)
    if not raw:
        return None
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf8", "replace"))))
    cells = []
    for r in rows:
        if r.get("FinInstrmTp") != "IDO":
            continue
        if r.get("TckrSymb") not in INDEX_SYMBOLS:
            continue
        try:
            K = float(r["StrkPric"]); S = float(r["UndrlygPric"])
            oi_units = float(r["OpnIntrst"]); px = float(r["ClsPric"])
            lot = float(r.get("NewBrdLotQty") or 0)
            xp = datetime.strptime(r["XpryDt"], "%Y-%m-%d").date()
        except (ValueError, KeyError, TypeError):
            continue
        if lot <= 0:
            continue
        oi = oi_units / lot
        if oi <= 0 or S <= 0 or px <= 0:
            continue
        T = (xp - d).days / 365.0
        if T <= 0:
            T = 0.5 / 365.0
        cp = r.get("OptnTp")
        if cp not in ("CE", "PE"):
            continue
        sig = implied_vol(px, S, K, T, cp)
        if sig is None:
            continue
        g = bs_gamma(S, K, T, sig)
        cells.append({"sym": r["TckrSymb"], "cp": cp, "K": K, "S": S,
                      "oi": oi, "gamma": g, "T": T, "iv": sig,
                      "gnotional": g * S * S / 100.0})
    return cells


def load_participants(d: date):
    raw = nse.participant_oi(d)
    if not raw:
        return None
    lines = [l for l in raw.decode("utf8", "replace").splitlines() if l.strip()]
    if len(lines) < 3:
        return None
    hdr = [h.strip().strip('"') for h in lines[1].split(",")]
    out = {}
    for line in lines[2:]:
        p = [x.strip().strip('"') for x in line.split(",")]
        rec = {}
        for h, v in zip(hdr[1:], p[1:]):
            try:
                rec[h.strip()] = float(v) if v not in ("", "-") else 0.0
            except ValueError:
                rec[h.strip()] = 0.0
        out[p[0]] = rec
    return out


def col(rec, name):
    for k in rec:
        if k.replace(" ", "").lower() == name.replace(" ", "").lower():
            return rec[k]
    return None


def identified_interval(cells, part, wing):
    sub = [c for c in cells if (c["cp"] == "CE") == (wing == "C")]
    if not sub:
        return None
    n = len(sub)
    P = len(PARTICIPANTS)
    oi = np.array([c["oi"] for c in sub])
    gn = np.array([c["gnotional"] for c in sub])

    nm = "Option Index Call" if wing == "C" else "Option Index Put"
    Lbar = np.array([col(part[p], f"{nm} Long") or 0.0 for p in PARTICIPANTS])
    Sbar = np.array([col(part[p], f"{nm} Short") or 0.0 for p in PARTICIPANTS])
    tot = oi.sum()
    if tot <= 0 or Lbar.sum() <= 0:
        return None
    Lbar = Lbar * tot / Lbar.sum()
    Sbar = Sbar * tot / Sbar.sum()

    N = 2 * P * n
    def Li(p, i): return p * n + i
    def Si(p, i): return P * n + p * n + i

    A_eq, b_eq = [], []
    for i in range(n):
        r1 = np.zeros(N); r2 = np.zeros(N)
        for p in range(P):
            r1[Li(p, i)] = 1.0
            r2[Si(p, i)] = 1.0
        A_eq.append(r1); b_eq.append(oi[i])
        A_eq.append(r2); b_eq.append(oi[i])
    for p in range(P):
        r1 = np.zeros(N); r2 = np.zeros(N)
        for i in range(n):
            r1[Li(p, i)] = 1.0
            r2[Si(p, i)] = 1.0
        A_eq.append(r1); b_eq.append(Lbar[p])
        A_eq.append(r2); b_eq.append(Sbar[p])

    A_eq = np.array(A_eq); b_eq = np.array(b_eq)
    res = {}
    for p, pname in enumerate(PARTICIPANTS):
        c = np.zeros(N)
        for i in range(n):
            c[Li(p, i)] = gn[i]
            c[Si(p, i)] = -gn[i]
        lo = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        hi = linprog(-c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        if lo.success and hi.success:
            res[pname] = (float(c @ lo.x), float(c @ hi.x))
    return res, n


def main() -> int:
    dates = [date(2025, 6, 2), date(2025, 6, 3), date(2025, 6, 4)]
    print("IDENTIFIED SET FOR PARTICIPANT GAMMA EXPOSURE (NSE index options)")
    print("units: gamma-notional per 1% move, in contract-equivalents (millions)\n")
    allout = []
    for d in dates:
        cells = load_cells(d)
        part = load_participants(d)
        if not cells or not part:
            print(f"{d}: data unavailable"); continue
        print(f"=== {d}   cells={len(cells):,}  "
              f"symbols={sorted({c['sym'] for c in cells})} ===")
        for wing in ("C", "P"):
            r = identified_interval(cells, part, wing)
            if not r:
                continue
            res, n = r
            print(f"  wing={wing}  cells={n:,}")
            for p, (lo, hi) in res.items():
                span = "SPANS ZERO" if lo < 0 < hi else ""
                print(f"    {p:<7} [{lo/1e6:>12.2f}, {hi/1e6:>12.2f}]  "
                      f"width={ (hi-lo)/1e6:>10.2f}  {span}")
                allout.append({"date": str(d), "wing": wing, "participant": p,
                               "lo": lo, "hi": hi, "spans_zero": bool(lo < 0 < hi)})
        print()
    if allout:
        (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)
        (ROOT / "data" / "processed" / "gamma_identification.json").write_text(
            json.dumps(allout, indent=2))
        sz = sum(1 for a in allout if a["spans_zero"])
        print(f"participant-wing-days whose gamma interval SPANS ZERO: "
              f"{sz} / {len(allout)}")
        print("wrote data/processed/gamma_identification.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
