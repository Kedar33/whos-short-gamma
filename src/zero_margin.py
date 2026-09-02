from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def main() -> int:
    rows = [r for r in json.loads((PROC / "gamma_panel.json").read_text())
            if r["participant"] in ("Client", "FII", "Pro")]
    lo = np.array([r["lo1"] for r in rows])
    hi = np.array([r["hi1"] for r in rows])
    w = hi - lo
    pos = (0.0 - lo) / w
    margin = np.minimum(pos, 1 - pos)

    print(f"panel rows (Client/FII/Pro): {len(rows)}")
    print(f"rows with lo < 0 < hi      : {(lo < 0).sum() & (hi > 0).sum()}"
          f"  ({int(((lo<0)&(hi>0)).sum())})")

    print("\nWHERE DOES ZERO SIT INSIDE THE INTERVAL? (0 = lower end, 1 = upper)")
    for q in (0, 5, 25, 50, 75, 95, 100):
        print(f"  p{q:<3}: {np.percentile(pos, q):.4f}")

    print("\nMARGIN = distance from zero to the NEAREST endpoint, as a fraction")
    print("of the interval width.  0.5 is dead centre; 0 would be fragile.")
    for q in (0, 1, 5, 25, 50):
        print(f"  p{q:<3}: {np.percentile(margin, q):.4f}")
    print(f"  minimum across all {len(rows)} rows: {margin.min():.4f}")

    shift_needed = np.minimum(-lo, hi) / w
    print("\nHOW BIG AN ERROR WOULD FLIP THE RESULT?")
    print("A systematic SHIFT of the whole interval is what could remove zero.")
    print("Required shift, as a fraction of the interval width:")
    for q in (0, 1, 5, 50):
        print(f"  p{q:<3}: {np.percentile(shift_needed, q):.4f}")
    print(f"  smallest required shift anywhere: {shift_needed.min():.4f} "
          f"of the width")

    print("\nABSOLUTE SCALE (millions of gamma-notional units)")
    print(f"  median |lo|          : {np.median(np.abs(lo))/1e6:,.1f}")
    print(f"  median  hi           : {np.median(hi)/1e6:,.1f}")
    print(f"  median width         : {np.median(w)/1e6:,.1f}")
    print(f"  median distance 0->lo: {np.median(-lo)/1e6:,.1f}")
    print(f"  median distance 0->hi: {np.median(hi)/1e6:,.1f}")

    out = {"n_rows": len(rows),
           "n_contain_zero": int(((lo < 0) & (hi > 0)).sum()),
           "margin_min": float(margin.min()),
           "margin_median": float(np.median(margin)),
           "pos_median": float(np.median(pos))}
    (PROC / "zero_margin.json").write_text(json.dumps(out, indent=2))
    print("\nwrote data/processed/zero_margin.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
