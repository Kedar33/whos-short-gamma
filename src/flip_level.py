from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gamma_identification import (load_cells, load_participants, col,
                                  bs_gamma, PARTICIPANTS)
from gamma_panel import bucketise, totals

ROOT = Path(__file__).resolve().parents[1]


def greedy_fill(g, oi, total, descending):
    order = np.argsort(-g if descending else g)
    x = np.zeros_like(g, dtype=float)
    left = total
    for i in order:
        take = min(oi[i], left)
        x[i] = take
        left -= take
        if left <= 1e-12:
            break
    return x


def endpoints(g, oi, Lbar, Sbar):
    hi = g @ greedy_fill(g, oi, Lbar, True) - g @ greedy_fill(g, oi, Sbar, False)
    lo = g @ greedy_fill(g, oi, Lbar, False) - g @ greedy_fill(g, oi, Sbar, True)
    return lo, hi


def revalue(cells, S_new):
    out = []
    for c in cells:
        sig = c.get("iv")
        if sig is None or sig <= 0:
            continue
        g = bs_gamma(S_new, c["K"], c["T"], sig)
        out.append({"key": (c["sym"], c["cp"], c["K"]), "oi": c["oi"],
                    "g": g * S_new * S_new / 100.0})
    return out


def main() -> int:
    dates = [date(2025, 6, 2), date(2025, 9, 15)]
    allout = []
    for d in dates:
        cells = load_cells(d)
        part = load_participants(d)
        if not cells or not part:
            print(f"{d}: unavailable"); continue
        S0 = np.median([c["S"] for c in cells if c["sym"] == "NIFTY"])
        print(f"\n=== {d}   NIFTY spot ~ {S0:,.0f} ===")
        for wing in ("C", "P"):
            sub = [c for c in cells if (c["cp"] == "CE") == (wing == "C")]
            oi_tot = sum(c["oi"] for c in sub)
            L, S_ = totals(part, wing, oi_tot)
            if L is None:
                print(f"  wing={wing}: reconciliation gate failed"); continue
            p = PARTICIPANTS.index("Pro")
            print(f"  wing={wing}  Pro Lbar={L[p]:,.0f}  Sbar={S_[p]:,.0f}")
            print(f"    {'spot':>10}{'gamma lo':>16}{'gamma hi':>16}{'sign':>10}")
            determined = []
            for mult in (0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10):
                Sx = S0 * mult
                rev = revalue(sub, Sx)
                if not rev:
                    continue
                g = np.array([r["g"] for r in rev])
                oi = np.array([r["oi"] for r in rev])
                lo, hi = endpoints(g, oi, L[p], S_[p])
                sign = "SPANS 0" if lo < 0 < hi else ("+" if lo > 0 else "-")
                if lo > 0 or hi < 0:
                    determined.append(mult)
                print(f"    {Sx:>10,.0f}{lo/1e6:>16.2f}{hi/1e6:>16.2f}{sign:>10}")
                allout.append({"date": str(d), "wing": wing, "spot_mult": mult,
                               "spot": Sx, "lo": lo, "hi": hi,
                               "determined": bool(lo > 0 or hi < 0)})
            print(f"    -> spot levels with a DETERMINED sign: "
                  f"{determined if determined else 'NONE'}")
    if allout:
        (ROOT / "data" / "processed" / "flip_level.json").write_text(
            json.dumps(allout, indent=2))
        det = sum(a["determined"] for a in allout)
        print(f"\n{'='*60}")
        print(f"grid points with a determined gamma sign: {det} / {len(allout)}")
        if det == 0:
            print("=> THE GAMMA FLIP LEVEL DOES NOT EXIST as an identified object:")
            print("   the sign is undetermined at EVERY spot level on the grid.")
        print("wrote data/processed/flip_level.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
