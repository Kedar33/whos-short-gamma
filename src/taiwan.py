from __future__ import annotations

import csv
import io
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_cells import implied_vol_vec, bs_gamma_vec
from flip_level import endpoints

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "taifex"
PROC = ROOT / "data" / "processed"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
      "Referer": "https://www.taifex.com.tw/"}

DEALER = "---"
TRUST = "--"
FOREIGN = "-----"
PARTICIPANTS_TW = [DEALER, TRUST, FOREIGN]


def _post(url: str, payload: dict, cache: Path) -> str | None:
    RAW.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return cache.read_text(encoding="utf8", errors="replace")
    try:
        req = urllib.request.Request(
            url, data=urllib.parse.urlencode(payload).encode(), headers=UA)
        with urllib.request.urlopen(req, timeout=90) as r:
            b = r.read()
    except Exception:
        return None
    txt = b.decode("big5", "replace")
    cache.write_text(txt, encoding="utf8")
    return txt


def participants(d: date):
    ds = d.strftime("%Y/%m/%d")
    txt = _post("https://www.taifex.com.tw/cht/3/callsAndPutsDateDown",
                {"down_type": "1", "queryStartDate": ds, "queryEndDate": ds,
                 "commodityId": "TXO"},
                RAW / f"tw_part_{d:%Y%m%d}.csv")
    if not txt:
        return None
    out = {}
    for row in csv.DictReader(io.StringIO(txt)):
        cp = (row.get("----") or "").strip()
        who = (row.get("---") or "").strip()
        if cp not in ("CALL", "PUT") or who not in PARTICIPANTS_TW:
            continue
        try:
            lo = float(row["-------"])
            sh = float(row["-------"])
        except (KeyError, ValueError, TypeError):
            continue
        out[("C" if cp == "CALL" else "P", who)] = (lo, sh)
    return out or None


def chain(d: date):
    ds = d.strftime("%Y/%m/%d")
    txt = _post("https://www.taifex.com.tw/cht/3/dlOptDataDown",
                {"down_type": "1", "commodity_id": "TXO",
                 "queryStartDate": ds, "queryEndDate": ds},
                RAW / f"tw_chain_{d:%Y%m%d}.csv")
    if not txt:
        return None
    rows = []
    for r in csv.DictReader(io.StringIO(txt)):
        if (r.get("----") or "").strip() != "--":
            continue
        if (r.get("--") or "").strip() != "TXO":
            continue
        try:
            K = float(r["---"])
            oi = float(r["------"])
            settle = float(r["---"])
        except (KeyError, ValueError, TypeError):
            continue
        cp = (r.get("---") or "").strip()
        exp = (r.get("----(--)") or "").strip()
        if oi <= 0 or settle <= 0 or cp not in ("--", "--"):
            continue
        rows.append({"K": K, "oi": oi, "px": settle,
                     "is_call": cp == "--", "exp": exp})
    return rows or None


def _expiry_date(tag: str, d: date):
    tag = tag.strip()
    try:
        y, m = int(tag[:4]), int(tag[4:6])
    except (ValueError, IndexError):
        return None
    if "W" in tag:
        try:
            wk = int(tag.split("W")[1])
        except (ValueError, IndexError):
            wk = 1
        first = date(y, m, 1)
        w = first + timedelta(days=(2 - first.weekday()) % 7)
        return w + timedelta(weeks=wk - 1)
    first = date(y, m, 1)
    w = first + timedelta(days=(2 - first.weekday()) % 7)
    return w + timedelta(weeks=2)


def forward_from_parity(rows):
    byk = {}
    for r in rows:
        byk.setdefault(r["K"], {})["C" if r["is_call"] else "P"] = r["px"]
    best, bk = None, None
    for k, d2 in byk.items():
        if "C" in d2 and "P" in d2:
            g = abs(d2["C"] - d2["P"])
            if best is None or g < best:
                best, bk = g, k
    if bk is None:
        return None
    return bk + (byk[bk]["C"] - byk[bk]["P"])


def analyse(d: date):
    part = participants(d)
    rows = chain(d)
    if not part or not rows:
        return None
    S = forward_from_parity(rows)
    if not S or S <= 0:
        return None
    out = []
    for wing in ("C", "P"):
        sub = [r for r in rows if r["is_call"] == (wing == "C")]
        if len(sub) < 20:
            continue
        K = np.array([r["K"] for r in sub], float)
        oi = np.array([r["oi"] for r in sub], float)
        px = np.array([r["px"] for r in sub], float)
        T = np.array([max(((_expiry_date(r["exp"], d) or d) - d).days, 1) / 365.0
                      for r in sub], float)
        isc = np.full(len(sub), wing == "C")
        Sv = np.full(len(sub), S)
        with np.errstate(all="ignore"):
            sig = implied_vol_vec(px, Sv, K, T, isc)
            gam = bs_gamma_vec(Sv, K, T, sig)
        ok = np.isfinite(sig) & np.isfinite(gam) & (gam > 0)
        if ok.sum() < 20:
            continue
        g = (gam * Sv * Sv / 100.0)[ok]
        o = oi[ok]
        mb = np.digitize(np.log(K[ok] / S),
                         np.array([-np.inf, -.15, -.10, -.07, -.05, -.03,
                                   -.015, 0, .015, .03, .05, .07, .10, .15,
                                   np.inf]))
        uniq, inv = np.unique(mb, return_inverse=True)
        oi_s = np.bincount(inv, weights=o, minlength=len(uniq))
        g_s = np.bincount(inv, weights=g * o, minlength=len(uniq))
        keep = oi_s > 0
        gb, ob = g_s[keep] / oi_s[keep], oi_s[keep]
        tot = ob.sum()
        for who in PARTICIPANTS_TW:
            if (wing, who) not in part:
                continue
            L, Sh = part[(wing, who)]
            if L <= 0 or Sh <= 0:
                continue
            scale = tot / max(sum(part[(wing, w2)][0] for w2 in PARTICIPANTS_TW), 1)
            lo, hi = endpoints(gb, ob, L * scale, Sh * scale)
            out.append({"date": str(d), "wing": wing, "investor": who,
                        "L": L, "S": Sh, "ratio": L / Sh,
                        "lo": lo, "hi": hi, "spans_zero": bool(lo < 0 < hi)})
    return out


def main() -> int:
    dates = []
    d = date(2025, 1, 15)
    while d <= date(2026, 6, 30):
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=21)
    print(f"TAIWAN (TAIFEX TXO) replication -- {len(dates)} sample dates\n")
    print(f"{'date':>12}{'wing':>5}{'investor':>10}{'L/S':>7}"
          f"{'lo':>14}{'hi':>14}{'0?':>5}")
    print("-" * 70)
    allrows = []
    for dt in dates:
        res = analyse(dt)
        if not res:
            continue
        for r in res:
            nm = {DEALER: "Dealer", TRUST: "Trust", FOREIGN: "Foreign"}[r["investor"]]
            print(f"{r['date']:>12}{r['wing']:>5}{nm:>10}{r['ratio']:>7.2f}"
                  f"{r['lo']/1e6:>14.1f}{r['hi']/1e6:>14.1f}"
                  f"{'YES' if r['spans_zero'] else 'no':>5}")
            allrows.append(r)
    if allrows:
        PROC.mkdir(parents=True, exist_ok=True)
        (PROC / "taiwan.json").write_text(json.dumps(allrows, indent=2))
        n = len(allrows)
        z = sum(r["spans_zero"] for r in allrows)
        rat = np.array([r["ratio"] for r in allrows])
        print("\n" + "=" * 70)
        print(f"observations                : {n}")
        print(f"identified set contains zero: {z} ({z/n:.1%})")
        print(f"L/S ratio  median {np.median(rat):.2f}  "
              f"min {rat.min():.2f}  max {rat.max():.2f}")
        print(f"share of ratios below 2.0   : {(rat < 2.0).mean():.1%}"
              f"   [simulation: 0% identified below 2.0]")
        d_only = [r for r in allrows if r["investor"] == DEALER]
        if d_only:
            zz = sum(r["spans_zero"] for r in d_only)
            print(f"\nDealers only (market-maker analogue): "
                  f"{zz}/{len(d_only)} contain zero")
        print("\nwrote data/processed/taiwan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
