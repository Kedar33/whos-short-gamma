from __future__ import annotations

import csv
import io
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

START = date(2019, 4, 1)
END = date(2026, 8, 28)

WANT = {
    "NIFTY 50": "NIFTY",
    "NIFTY BANK": "BANKNIFTY",
    "NIFTY FINANCIAL SERVICES": "FINNIFTY",
    "NIFTY MIDCAP SELECT": "MIDCPNIFTY",
    "NIFTY NEXT 50": "NIFTYNEXT50",
    "NIFTY MIDCAP 100": "MIDCAP100",
    "NIFTY IT": "NIFTYIT",
    "NIFTY AUTO": "NIFTYAUTO",
    "NIFTY FMCG": "NIFTYFMCG",
    "NIFTY PHARMA": "NIFTYPHARMA",
}


def parse(raw: bytes, d: date) -> list[dict]:
    txt = raw.decode("utf8", "replace")
    rd = csv.DictReader(io.StringIO(txt))
    out = []
    for r in rd:
        name = (r.get("Index Name") or r.get("indexName") or "").strip().upper()
        if name not in WANT:
            continue
        def g(*keys):
            for k in keys:
                if k in r and r[k] not in ("", "-", None):
                    try:
                        return float(str(r[k]).replace(",", ""))
                    except ValueError:
                        pass
            return np.nan
        out.append({
            "date": d.isoformat(), "index": WANT[name],
            "open": g("Open Index Value", "openIndexValue"),
            "high": g("High Index Value", "highIndexValue"),
            "low": g("Low Index Value", "lowIndexValue"),
            "close": g("Closing Index Value", "closingIndexValue"),
            "turnover": g("Turnover (Rs. Cr.)", "turnoverRsCr"),
        })
    return out


def main() -> int:
    rows, miss = [], 0
    days = list(nse.business_days(START, END))
    print(f"scanning {len(days):,} business days")
    for i, d in enumerate(days):
        b = nse.index_close(d)
        if not b:
            miss += 1
            continue
        rows.extend(parse(b, d))
        if (i + 1) % 250 == 0:
            print(f"  ...{i+1:,}/{len(days):,}  rows={len(rows):,} miss={miss:,}",
                  flush=True)
    if not rows:
        print("no index data retrieved")
        return 1

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"]).sort_values(["index", "date"])
    df["ret"] = df.groupby("index")["close"].pct_change()
    hl = np.log(df["high"] / df["low"])
    df["park"] = np.sqrt((hl ** 2) / (4 * np.log(2))) * np.sqrt(252)
    df["absret"] = df["ret"].abs()
    for w in (5, 21):
        df[f"rv{w}"] = (df.groupby("index")["ret"]
                          .transform(lambda x: x.rolling(w).std() * np.sqrt(252)))

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "index_panel.parquet", index=False)

    print(f"\ndays with data {df['date'].nunique():,}   missing {miss:,}")
    print(f"range {df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}")
    print("\ncoverage by index:")
    for ix, g in df.groupby("index"):
        print(f"  {ix:<12} n={len(g):>5}  {g['date'].min():%Y-%m-%d} .. "
              f"{g['date'].max():%Y-%m-%d}  ann.vol={g['ret'].std()*np.sqrt(252):.3f}")
    print(f"\nwrote {(OUT/'index_panel.parquet').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
