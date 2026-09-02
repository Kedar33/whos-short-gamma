from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamma_identification import load_cells, load_participants, PARTICIPANTS
from gamma_panel import totals, raw_contract_total, bucketise
from flip_level import endpoints

ROOT = Path(__file__).resolve().parents[1]


def disclosure_curve(g, oi, Lbar, Sbar, ks, rule="oi"):
    if rule == "gamma":
        order = np.argsort(-(g * oi))
    else:
        order = np.argsort(-oi)
    tot = oi.sum()
    out = []
    for k in ks:
        k = min(k, len(oi))
        disc = order[:k]
        und = order[k:]
        if len(und) == 0:
            out.append((k, 0.0, 0.0, 0.0, False, 1.0))
            continue
        oi_disc = oi[disc].sum()
        share = oi_disc / tot
        L_res = Lbar * (1.0 - share)
        S_res = Sbar * (1.0 - share)
        D = (g[disc] * (Lbar - Sbar) * oi[disc] / tot).sum()
        lo_r, hi_r = endpoints(g[und], oi[und], L_res, S_res)
        lo, hi = D + lo_r, D + hi_r
        out.append((k, lo, hi, hi - lo, bool(lo < 0 < hi), share))
    return out


def main() -> int:
    dates = [date(2025, 6, 17), date(2025, 9, 25), date(2026, 4, 16)]
    allrows = []
    for d in dates:
        cells = load_cells(d)
        part = load_participants(d)
        if not cells or not part:
            print(f"{d}: unavailable"); continue
        for wing in ("C", "P"):
            sub = [c for c in cells if (c["cp"] == "CE") == (wing == "C")]
            if len(sub) < 20:
                continue
            oi_tot = sum(c["oi"] for c in sub)
            L, S = totals(part, wing, oi_tot, raw_contract_total(d, wing))
            if L is None:
                continue
            p = PARTICIPANTS.index("Pro")
            g = np.array([c["gnotional"] for c in sub])
            oi = np.array([c["oi"] for c in sub])
            n = len(g)
            ks = [0, 2, 5, 10, 25, 50, 100, 200, 400, int(0.9 * n)]
            ks = sorted(set(min(k, n) for k in ks))
            print(f"\n=== {d}  wing={wing}  Pro  cells={n}  "
                  f"Lbar={L[p]:,.0f} Sbar={S[p]:,.0f} ===")
            summary = {}
            for rule in ("oi", "gamma"):
                curve = disclosure_curve(g, oi, L[p], S[p], ks, rule=rule)
                w0 = curve[0][3] if curve else 1.0
                kstar = None
                print(f"  --- disclosure ordered by "
                      f"{'OPEN INTEREST' if rule=='oi' else 'GAMMA CONTRIBUTION'} ---")
                print(f"  {'k':>6}{'% of OI':>9}{'width':>13}"
                      f"{'width/k=0':>11}{'spans 0':>9}")
                for k, lo, hi, w, z, share in curve:
                    print(f"  {k:>6}{share*100:>8.1f}%{w/1e6:>13.2f}"
                          f"{w/w0 if w0 else 0:>11.3f}{'YES' if z else 'no':>9}")
                    if kstar is None and not z:
                        kstar = (k, share)
                    allrows.append({"date": str(d), "wing": wing, "rule": rule,
                                    "k": k, "oi_share": share, "lo": lo,
                                    "hi": hi, "width": w, "spans_zero": z})
                summary[rule] = kstar
                if kstar:
                    print(f"  -> IDENTIFIED at k={kstar[0]} cells "
                          f"({kstar[1]*100:.1f}% of OI)")
                else:
                    print(f"  -> NOT identified even at k={curve[-1][0]} "
                          f"of {n} cells")
            a, b = summary.get("oi"), summary.get("gamma")
            if a and b:
                print(f"  ==> gamma-ordered disclosure needs k={b[0]} vs "
                      f"OI-ordered k={a[0]}  ({a[0]/max(b[0],1):.1f}x fewer cells)")
    if allrows:
        (ROOT / "data" / "processed" / "disclosure.json").write_text(
            json.dumps(allrows, indent=2))
        print("\nwrote data/processed/disclosure.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
