from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flip_level import endpoints

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(20260901)


def draw_market(n_cells, gamma_spread, seed_rng):
    g = np.exp(seed_rng.normal(0.0, gamma_spread, size=n_cells))
    oi = seed_rng.lognormal(mean=6.0, sigma=1.2, size=n_cells)
    return g, oi


def draw_truth(n_cells, oi, imbalance, seed_rng, n_participants=4):
    tot = oi.sum()
    base = seed_rng.uniform(0.10, 0.35)
    s0 = base * tot
    l0 = min(imbalance * s0, 0.9 * tot)
    rest_l = seed_rng.dirichlet(np.ones(n_participants - 1)) * (tot - l0)
    rest_s = seed_rng.dirichlet(np.ones(n_participants - 1)) * (tot - s0)
    Lbar = np.concatenate([[l0], rest_l])
    Sbar = np.concatenate([[s0], rest_s])

    share_l = seed_rng.dirichlet(np.ones(len(oi)))
    share_s = seed_rng.dirichlet(np.ones(len(oi)))
    L0 = np.minimum(share_l * Lbar[0], oi)
    S0 = np.minimum(share_s * Sbar[0], oi)
    for _ in range(50):
        L0 *= Lbar[0] / max(L0.sum(), 1e-12); L0 = np.minimum(L0, oi)
        S0 *= Sbar[0] / max(S0.sum(), 1e-12); S0 = np.minimum(S0, oi)
    return Lbar, Sbar, L0, S0


def main() -> int:
    print("SIMULATED DATA -- known truth. No number here enters an empirical table.\n")

    print("(A) COVERAGE: does the identified interval contain the true gamma?")
    cover = 0
    trials = 400
    for _ in range(trials):
        n = int(RNG.integers(20, 200))
        g, oi = draw_market(n, RNG.uniform(0.3, 1.5), RNG)
        Lbar, Sbar, L0, S0 = draw_truth(n, oi, RNG.uniform(0.5, 2.0), RNG)
        true_G = float(g @ (L0 - S0))
        lo, hi = endpoints(g, oi, L0.sum(), S0.sum())
        if lo - 1e-6 <= true_G <= hi + 1e-6:
            cover += 1
    print(f"    truth inside the interval: {cover}/{trials} "
          f"({cover/trials:.1%})   [must be 100%]")

    print("\n(B) WHEN IS THE SIGN IDENTIFIED?  varying book imbalance")
    print(f"    {'L/S ratio':>10}{'trials':>8}{'sign identified':>17}"
          f"{'median width/|G_true|':>24}")
    rows = []
    for imb in (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0):
        ident = 0; widths = []
        T = 300
        for _ in range(T):
            n = int(RNG.integers(30, 150))
            g, oi = draw_market(n, 0.8, RNG)
            Lbar, Sbar, L0, S0 = draw_truth(n, oi, imb, RNG)
            if Lbar[0] <= 0 or Sbar[0] <= 0:
                continue
            true_G = float(g @ (L0 - S0))
            lo, hi = endpoints(g, oi, Lbar[0], Sbar[0])
            if not (lo < 0 < hi):
                ident += 1
            if abs(true_G) > 1e-9:
                widths.append((hi - lo) / abs(true_G))
        med = float(np.median(widths)) if widths else float("nan")
        print(f"    {imb:>10.2f}{T:>8}{ident/T:>16.1%}{med:>24.1f}")
        rows.append({"imbalance": imb, "share_identified": ident / T,
                     "median_rel_width": med})

    print("\n(C) HOW DOES WIDTH SCALE?")
    print(f"    {'n cells':>9}{'gamma spread':>14}{'median width/total OI-gamma':>30}")
    for n in (25, 50, 100, 400):
        for sp in (0.4, 0.8, 1.6):
            w = []
            for _ in range(120):
                g, oi = draw_market(n, sp, RNG)
                Lbar, Sbar, _, _ = draw_truth(n, oi, 1.0, RNG)
                lo, hi = endpoints(g, oi, Lbar[0], Sbar[0])
                denom = float(g @ oi)
                if denom > 0:
                    w.append((hi - lo) / denom)
            print(f"    {n:>9}{sp:>14.1f}{np.median(w):>30.4f}")

    out = ROOT / "data" / "processed" / "simulation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"coverage": cover / trials,
                               "imbalance_curve": rows}, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    print("\nREADING: the sign becomes identified only once one side of the book")
    print("exceeds the other by a wide margin.  Observed NSE books are close to")
    print("balanced, which is why the empirical criterion fails on every day.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
