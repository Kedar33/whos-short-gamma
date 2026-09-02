import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RNG = np.random.default_rng(31082026)


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


def closed_form(g, oi, Lbar, Sbar):
    hi = g @ greedy_fill(g, oi, Lbar, True) - g @ greedy_fill(g, oi, Sbar, False)
    lo = g @ greedy_fill(g, oi, Lbar, False) - g @ greedy_fill(g, oi, Sbar, True)
    return lo, hi


def lp_endpoints(g, oi, Lbars, Sbars, p=0):
    n, P = len(g), len(Lbars)
    N = 2 * P * n
    A, b = [], []
    for i in range(n):
        r = np.zeros(N)
        for q in range(P):
            r[q * n + i] = 1.0
        A.append(r); b.append(oi[i])
        r = np.zeros(N)
        for q in range(P):
            r[P * n + q * n + i] = 1.0
        A.append(r); b.append(oi[i])
    for q in range(P):
        r = np.zeros(N); r[q * n:(q + 1) * n] = 1.0
        A.append(r); b.append(Lbars[q])
        r = np.zeros(N); r[P * n + q * n:P * n + (q + 1) * n] = 1.0
        A.append(r); b.append(Sbars[q])
    A, b = np.array(A), np.array(b)
    c = np.zeros(N)
    c[p * n:(p + 1) * n] = g
    c[P * n + p * n:P * n + (p + 1) * n] = -g
    lo = linprog(c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    hi = linprog(-c, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    assert lo.success and hi.success, "LP infeasible"
    return float(c @ lo.x), float(c @ hi.x)


def random_instance(n=8, P=3, zero_short_for_p0=False):
    g = RNG.uniform(0.05, 3.0, size=n)
    oi = RNG.uniform(50, 1000, size=n)
    tot = oi.sum()
    wl = RNG.dirichlet(np.ones(P)); ws = RNG.dirichlet(np.ones(P))
    Lbars = wl * tot
    Sbars = ws * tot
    if zero_short_for_p0:
        Sbars[0] = 0.0
        Sbars[1:] = RNG.dirichlet(np.ones(P - 1)) * tot
    return g, oi, Lbars, Sbars


def test_theorem2_closed_form_matches_lp():
    for _ in range(25):
        g, oi, Lbars, Sbars = random_instance()
        lo_lp, hi_lp = lp_endpoints(g, oi, Lbars, Sbars, p=0)
        lo_cf, hi_cf = closed_form(g, oi, Lbars[0], Sbars[0])
        assert hi_cf == pytest.approx(hi_lp, rel=1e-6, abs=1e-6)
        assert lo_cf == pytest.approx(lo_lp, rel=1e-6, abs=1e-6)


def test_theorem3_criterion_is_exact():
    checked = 0
    for _ in range(600):
        g, oi, Lbars, Sbars = random_instance(
            n=int(RNG.integers(3, 14)), P=int(RNG.integers(2, 5)))
        if Lbars[0] <= 0 or Sbars[0] <= 0:
            continue
        L_up = g @ greedy_fill(g, oi, Lbars[0], True)
        L_dn = g @ greedy_fill(g, oi, Lbars[0], False)
        S_up = g @ greedy_fill(g, oi, Sbars[0], True)
        S_dn = g @ greedy_fill(g, oi, Sbars[0], False)
        criterion = (L_dn < S_up) and (L_up > S_dn)
        lo, hi = L_dn - S_up, L_up - S_dn
        assert criterion == (lo < 0 < hi), (
            f"criterion mismatch: lo={lo}, hi={hi}, crit={criterion}")
        checked += 1
    assert checked > 300, "too few admissible draws"


def test_balanced_books_imply_unidentified_sign():
    for _ in range(100):
        n = int(RNG.integers(4, 12))
        g = RNG.uniform(0.05, 3.0, size=n)
        oi = RNG.uniform(50, 1000, size=n)
        tot = 0.4 * oi.sum()
        lo, hi = closed_form(g, oi, tot, tot)
        assert lo < 0 < hi, f"balanced book should be unidentified: [{lo},{hi}]"


def test_A4_boundary_zero_shorts_gives_determinate_sign():
    for _ in range(25):
        g, oi, Lbars, Sbars = random_instance(zero_short_for_p0=True)
        lo, hi = closed_form(g, oi, Lbars[0], Sbars[0])
        assert lo >= -1e-9, "with zero shorts the lower endpoint must be >= 0"
        assert hi > 0
        assert not (lo < 0 < hi), "sign should be determinate when (A4) fails"


def test_two_period_lp_is_feasible_on_union_cells():
    from gamma_panel import two_period
    for _ in range(12):
        n = int(RNG.integers(6, 12))
        P = len(PARTICIPANTS_LOCAL)
        oi0 = RNG.uniform(100, 2000, size=n)
        oi1 = oi0 * RNG.uniform(0.7, 1.4, size=n)
        oi0[RNG.integers(0, n)] = 0.0
        oi1[RNG.integers(0, n)] = 0.0
        g = RNG.uniform(0.05, 3.0, size=n)
        bk0 = [{"key": i, "oi": oi0[i], "g": g[i]} for i in range(n) if oi0[i] > 0]
        bk1 = [{"key": i, "oi": oi1[i], "g": g[i]} for i in range(n) if oi1[i] > 0]
        w0 = RNG.dirichlet(np.ones(P)); w1 = RNG.dirichlet(np.ones(P))
        L0 = w0 * oi0.sum(); S0 = RNG.dirichlet(np.ones(P)) * oi0.sum()
        L1 = w1 * oi1.sum(); S1 = RNG.dirichlet(np.ones(P)) * oi1.sum()
        vol = np.full(P, 4.0 * max(oi0.sum(), oi1.sum()))
        res = two_period(bk0, bk1, L0, S0, L1, S1, vol)
        assert res is not None, "two-period LP infeasible with a generous budget"
        assert set(res) == set(PARTICIPANTS_LOCAL)


PARTICIPANTS_LOCAL = ["Client", "DII", "FII", "Pro"]


def test_gamma_positive_but_underflows_in_far_wings():
    from gamma_identification import bs_gamma
    near, under = 0, 0
    for K in (0.5, 0.9, 1.0, 1.1, 2.0):
        for T in (1 / 365, 0.1, 1.0):
            for sig in (0.05, 0.2, 0.8):
                v = bs_gamma(1.0, K, T, sig)
                assert v >= 0
                if v > 0:
                    near += 1
                else:
                    under += 1
    assert near > 0
    assert bs_gamma(1.0, 1.0, 0.1, 0.2) > 0
    assert bs_gamma(1.0, 0.5, 1 / 365, 0.05) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
