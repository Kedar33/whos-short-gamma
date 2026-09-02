from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamma_identification import load_participants, PARTICIPANTS
from gamma_panel import totals, next_bday
from flip_level import endpoints
from fast_cells import load_fast, bucket_fast

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
WINDOWS = [(date(2024, 12, 1), date(2025, 10, 31)),
           (date(2026, 2, 1), date(2026, 5, 31))]
CORE = ("Client", "FII", "Pro")


def recompute():
    out = {}
    for lo_d, hi_d in WINDOWS:
        d = lo_d
        while d <= hi_d:
            if d.weekday() >= 5:
                d += timedelta(days=11); continue
            d1 = next_bday(d)
            cells, raw_tot = load_fast(d1)
            part = load_participants(d1)
            if cells is not None and part:
                for wing in ("C", "P"):
                    g, oi = bucket_fast(cells, wing)
                    if g is None or len(g) < 5:
                        continue
                    L, S = totals(part, wing, float(oi.sum()), raw_tot[wing])
                    if L is None:
                        continue
                    for nm in CORE:
                        p = PARTICIPANTS.index(nm)
                        a, b = endpoints(g, oi, L[p], S[p])
                        out[(str(d1), wing, nm)] = (a, b)
            d += timedelta(days=11)
    return out


def main() -> int:
    f = PROC / "gamma_panel.json"
    if not f.exists():
        print("run src/gamma_panel.py first"); return 1
    lp = {(r["date"], r["wing"], r["participant"]): (r["lo1"], r["hi1"])
          for r in json.loads(f.read_text())
          if r["participant"] in CORE}
    print(f"LP panel rows (core): {len(lp)}")

    print("recomputing by Theorem 2 closed form ...", flush=True)
    cf = recompute()
    print(f"closed-form rows     : {len(cf)}")

    common = sorted(set(lp) & set(cf))
    only_lp = sorted(set(lp) - set(cf))
    only_cf = sorted(set(cf) - set(lp))
    print(f"\nrows in both         : {len(common)}")
    print(f"only in LP panel     : {len(only_lp)}")
    print(f"only in closed form  : {len(only_cf)}")

    if not common:
        print("NO OVERLAP -- cannot verify"); return 1

    rel_lo, rel_hi, sign_agree = [], [], 0
    worst = None
    for k in common:
        a1, b1 = lp[k]
        a2, b2 = cf[k]
        scale = max(abs(a1), abs(b1), 1.0)
        rl, rh = abs(a1 - a2) / scale, abs(b1 - b2) / scale
        rel_lo.append(rl); rel_hi.append(rh)
        if (a1 < 0 < b1) == (a2 < 0 < b2):
            sign_agree += 1
        if worst is None or max(rl, rh) > worst[1]:
            worst = (k, max(rl, rh), (a1, b1), (a2, b2))

    rel_lo, rel_hi = np.array(rel_lo), np.array(rel_hi)
    print(f"\nrelative difference in endpoints (scaled by interval magnitude):")
    print(f"  lower endpoint: median {np.median(rel_lo):.2e}  max {rel_lo.max():.2e}")
    print(f"  upper endpoint: median {np.median(rel_hi):.2e}  max {rel_hi.max():.2e}")
    print(f"\n'contains zero' verdict agrees on {sign_agree}/{len(common)} rows "
          f"({sign_agree/len(common):.1%})")
    if worst:
        k, m, v1, v2 = worst
        print(f"\nlargest discrepancy: {k}  rel={m:.2e}")
        print(f"  LP          : [{v1[0]:,.1f}, {v1[1]:,.1f}]")
        print(f"  closed form : [{v2[0]:,.1f}, {v2[1]:,.1f}]")

    both_zero = sum(1 for k in common if lp[k][0] < 0 < lp[k][1]
                    and cf[k][0] < 0 < cf[k][1])
    print(f"\ncontains zero under BOTH implementations: {both_zero}/{len(common)}")
    ok = (sign_agree == len(common)) and (max(rel_lo.max(), rel_hi.max()) < 0.02)
    print("\nVERDICT: " + ("REPLICATED" if ok else "DISCREPANCY -- investigate"))
    (PROC / "replication.json").write_text(json.dumps({
        "rows_compared": len(common), "sign_agree": sign_agree,
        "median_rel_lo": float(np.median(rel_lo)),
        "median_rel_hi": float(np.median(rel_hi)),
        "max_rel": float(max(rel_lo.max(), rel_hi.max())),
        "both_contain_zero": both_zero, "replicated": bool(ok)}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
