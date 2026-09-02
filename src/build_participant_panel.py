from __future__ import annotations

import csv
import io
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

START = date(2019, 4, 1)
END = date(2026, 8, 28)

COLS = ["Future Index Long", "Future Index Short", "Future Stock Long",
        "Future Stock Short", "Option Index Call Long", "Option Index Put Long",
        "Option Index Call Short", "Option Index Put Short",
        "Option Stock Call Long", "Option Stock Put Long",
        "Option Stock Call Short", "Option Stock Put Short",
        "Total Long Contracts", "Total Short Contracts"]


def parse_participant(raw: bytes, d: date) -> list[dict]:
    txt = raw.decode("utf8", "replace")
    lines = [l for l in txt.splitlines() if l.strip()]
    if len(lines) < 3:
        return []
    hdr = [h.strip().strip('"') for h in lines[1].split(",")]
    out = []
    for line in lines[2:]:
        parts = [p.strip().strip('"') for p in line.split(",")]
        if len(parts) < 3:
            continue
        rec = {"date": d.isoformat(), "client_type": parts[0]}
        for h, v in zip(hdr[1:], parts[1:]):
            key = h.strip()
            try:
                rec[key] = float(v) if v not in ("", "-") else 0.0
            except ValueError:
                rec[key] = 0.0
        out.append(rec)
    return out


def main() -> int:
    rows, miss = [], 0
    days = list(nse.business_days(START, END))
    print(f"scanning {len(days):,} business days {START} -> {END}")
    for i, d in enumerate(days):
        b = nse.participant_oi(d)
        if not b:
            miss += 1
            continue
        rows.extend(parse_participant(b, d))
        if (i + 1) % 250 == 0:
            print(f"  ...{i+1:,}/{len(days):,} days, {len(rows):,} rows, "
                  f"{miss:,} missing", flush=True)

    if not rows:
        print("no data retrieved")
        return 1
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])

    ren = {c: c.strip() for c in df.columns}
    df = df.rename(columns=ren)

    def col(name):
        for c in df.columns:
            if c.replace(" ", "").lower() == name.replace(" ", "").lower():
                return c
        return None

    cl = col("Option Index Call Long"); pl = col("Option Index Put Long")
    cs = col("Option Index Call Short"); ps = col("Option Index Put Short")
    for c in (cl, pl, cs, ps):
        if c is None:
            print("FATAL: expected option columns missing", list(df.columns))
            return 1

    df["opt_long"] = df[cl] + df[pl]
    df["opt_short"] = df[cs] + df[ps]
    df["net_options"] = df["opt_long"] - df["opt_short"]
    df["gross_options"] = df["opt_long"] + df["opt_short"]
    df["net_calls"] = df[cl] - df[cs]
    df["net_puts"] = df[pl] - df[ps]

    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "participant_panel.parquet", index=False)

    n_days = df["date"].nunique()
    print(f"\ndays with data : {n_days:,}   missing/holiday : {miss:,}")
    print(f"range          : {df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}")
    print(f"client types   : {sorted(df.client_type.unique())}")

    print("\n=== NET INDEX-OPTION POSITION by client type (contracts, millions) ===")
    print(f"{'type':<10}{'mean':>12}{'median':>12}{'p5':>12}{'p95':>12}"
          f"{'%days net short':>17}")
    for ct in ["Client", "FII", "DII", "Pro"]:
        s = df[df.client_type == ct]["net_options"]
        if s.empty:
            continue
        print(f"{ct:<10}{s.mean()/1e6:>12.3f}{s.median()/1e6:>12.3f}"
              f"{s.quantile(.05)/1e6:>12.3f}{s.quantile(.95)/1e6:>12.3f}"
              f"{(s < 0).mean():>16.1%}")

    print("\n=== by year: median net index-option position (millions of contracts) ===")
    df["yr"] = df["date"].dt.year
    piv = df.pivot_table(index="yr", columns="client_type",
                         values="net_options", aggfunc="median") / 1e6
    print(piv.round(3).to_string())
    print(f"\nwrote {(OUT/'participant_panel.parquet').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
