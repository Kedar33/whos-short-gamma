from __future__ import annotations

import csv
import io
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nse

INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYNXT50"}
SQRT2 = np.sqrt(2.0)


def _ncdf(x):
    from scipy.special import erf
    return 0.5 * (1.0 + erf(x / SQRT2))


def _npdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _bs(S, K, T, sig, r, is_call):
    sq = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * T) / sq
    d2 = d1 - sq
    disc = np.exp(-r * T)
    call = S * _ncdf(d1) - K * disc * _ncdf(d2)
    put = K * disc * _ncdf(-d2) - S * _ncdf(-d1)
    return np.where(is_call, call, put)


def implied_vol_vec(px, S, K, T, is_call, r=0.065, iters=64):
    lo = np.full_like(px, 1e-3)
    hi = np.full_like(px, 5.0)
    f_lo = _bs(S, K, T, lo, r, is_call) - px
    f_hi = _bs(S, K, T, hi, r, is_call) - px
    bracket = (f_lo <= 0) & (f_hi >= 0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = _bs(S, K, T, mid, r, is_call) - px
        go_up = f_mid < 0
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    sig = 0.5 * (lo + hi)
    return np.where(bracket, sig, np.nan)


def bs_gamma_vec(S, K, T, sig, r=0.065):
    sq = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig * sig) * T) / sq
    return _npdf(d1) / (S * sq)


def load_fast(d: date):
    raw = nse.fo_bhav(d)
    if not raw:
        return None, None
    sym, cp, K, S, px, oi, T = [], [], [], [], [], [], []
    raw_tot = {"C": 0.0, "P": 0.0}
    for r in csv.DictReader(io.StringIO(raw.decode("utf8", "replace"))):
        if r.get("FinInstrmTp") != "IDO":
            continue
        t = r.get("TckrSymb")
        if t not in INDEX_SYMBOLS:
            continue
        o = r.get("OptnTp")
        if o not in ("CE", "PE"):
            continue
        try:
            lot = float(r.get("NewBrdLotQty") or 0)
            if lot <= 0:
                continue
            n_ct = float(r["OpnIntrst"]) / lot
            raw_tot["C" if o == "CE" else "P"] += n_ct
            k = float(r["StrkPric"]); s = float(r["UndrlygPric"])
            p = float(r["ClsPric"])
            xp = datetime.strptime(r["XpryDt"], "%Y-%m-%d").date()
        except (ValueError, KeyError, TypeError):
            continue
        if n_ct <= 0 or s <= 0 or p <= 0:
            continue
        tt = (xp - d).days / 365.0
        if tt <= 0:
            tt = 0.5 / 365.0
        sym.append(t); cp.append(o); K.append(k); S.append(s)
        px.append(p); oi.append(n_ct); T.append(tt)

    if not K:
        return None, None
    K = np.array(K); S = np.array(S); px = np.array(px)
    oi = np.array(oi); T = np.array(T)
    is_call = np.array([c == "CE" for c in cp])
    with np.errstate(all="ignore"):
        sig = implied_vol_vec(px, S, K, T, is_call)
        gam = bs_gamma_vec(S, K, T, sig)
    ok = np.isfinite(sig) & np.isfinite(gam) & (gam >= 0)
    cells = {"sym": np.array(sym)[ok], "is_call": is_call[ok], "K": K[ok],
             "S": S[ok], "oi": oi[ok], "T": T[ok], "iv": sig[ok],
             "gnotional": (gam * S * S / 100.0)[ok]}
    return cells, raw_tot


MONEY_BINS = np.array([-np.inf, -.15, -.10, -.07, -.05, -.03, -.015, 0,
                       .015, .03, .05, .07, .10, .15, np.inf])
TENOR_BINS = np.array([0, 7, 21, 60, 1e9]) / 365.0


def bucket_fast(cells, wing):
    m = cells["is_call"] if wing == "C" else ~cells["is_call"]
    if not m.any():
        return None, None
    K, S, oi, T = cells["K"][m], cells["S"][m], cells["oi"][m], cells["T"][m]
    g = cells["gnotional"][m]
    sym = cells["sym"][m]
    mb = np.digitize(np.log(K / S), MONEY_BINS)
    tb = np.digitize(T, TENOR_BINS)
    keys = np.char.add(np.char.add(sym.astype(str), mb.astype(str)),
                       tb.astype(str))
    uniq, inv = np.unique(keys, return_inverse=True)
    oi_sum = np.bincount(inv, weights=oi, minlength=len(uniq))
    g_sum = np.bincount(inv, weights=g * oi, minlength=len(uniq))
    ok = oi_sum > 0
    return g_sum[ok] / oi_sum[ok], oi_sum[ok]
