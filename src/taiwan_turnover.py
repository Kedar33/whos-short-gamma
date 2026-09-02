from __future__ import annotations

import csv
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taiwan import _post, RAW, PARTICIPANTS_TW, DEALER, TRUST, FOREIGN

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def vol_and_oi(d: date):
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
            vb = float(row["------"]); vs = float(row["------"])
            ob = float(row["-------"]); os_ = float(row["-------"])
        except (KeyError, ValueError, TypeError):
            continue
        out[("C" if cp == "CALL" else "P", who)] = (vb + vs, ob + os_)
    return out or None


def main() -> int:
    dates = []
    d = date(2025, 1, 15)
    while d <= date(2026, 6, 30):
        if d.weekday() < 5:
            dates.append(d)
        d += timedelta(days=21)

    rows = []
    for dt in dates:
        r = vol_and_oi(dt)
        if not r:
            continue
        for (wing, who), (vol, oi) in r.items():
            if oi > 0:
                rows.append({"date": str(dt), "wing": wing, "investor": who,
                             "vol": vol, "oi": oi, "ratio": vol / oi})
    if not rows:
        print("no data"); return 1

    print("TURNOVER / OPEN INTEREST, by investor type (TAIFEX TXO)\n")
    print(f"{'investor':>12}{'n':>5}{'median':>10}{'p10':>9}{'p90':>9}"
          f"{'max':>9}")
    print("-" * 54)
    names = {DEALER: "Dealer", TRUST: "Trust", FOREIGN: "Foreign"}
    for who in PARTICIPANTS_TW:
        v = np.array([r["ratio"] for r in rows if r["investor"] == who])
        if len(v) == 0:
            continue
        print(f"{names[who]:>12}{len(v):>5}{np.median(v):>10.2f}"
              f"{np.percentile(v,10):>9.2f}{np.percentile(v,90):>9.2f}"
              f"{v.max():>9.2f}")

    dealer = np.array([r["ratio"] for r in rows if r["investor"] == DEALER])
    print("\n" + "=" * 54)
    print("COMPARISON WITH INDIA")
    print(f"  India, Pro (proprietary)  : 18.7 - 23.6  x")
    print(f"  Taiwan, Dealer            : {np.median(dealer):.2f} x (median)")
    print(f"  ratio of ratios           : "
          f"{18.7/max(np.median(dealer),1e-9):.1f}x lower in Taiwan")
    print("\nBITE TEST: the turnover budget is 2 x volume; it can bind only if")
    print("2 x volume is comparable to the position, i.e. ratio not >> 1.")
    tight = (dealer < 5).mean()
    print(f"  share of Dealer observations with Vol/OI < 5 : {tight:.1%}")
    print(f"  share with Vol/OI < 2                        : "
          f"{(dealer < 2).mean():.1%}")
    verdict = ("JOINT CONSTRAINT PLAUSIBLY BINDS -- the methodological "
               "extension has empirical content here"
               if np.median(dealer) < 5 else
               "joint constraint likely empty here too")
    print(f"\nVERDICT: {verdict}")
    (PROC / "taiwan_turnover.json").write_text(json.dumps(
        {"median_dealer_ratio": float(np.median(dealer)),
         "share_below_5": float(tight), "n": len(dealer)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
