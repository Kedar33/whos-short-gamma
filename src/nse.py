from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifest.jsonl"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
      "Accept": "*/*", "Accept-Language": "en-US,en;q=0.9"}

PARTICIPANT_OI = "https://archives.nseindia.com/content/nsccl/fao_participant_oi_{d}.csv"
PARTICIPANT_VOL = "https://archives.nseindia.com/content/nsccl/fao_participant_vol_{d}.csv"
INDEX_CLOSE = "https://archives.nseindia.com/content/indices/ind_close_all_{d}.csv"
FO_UDIFF = ("https://archives.nseindia.com/content/fo/"
            "BhavCopy_NSE_FO_0_0_0_{ymd}_F_0000.csv.zip")


def _manifest(url: str, path: Path, payload: bytes, cached: bool) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", encoding="utf8") as fh:
        fh.write(json.dumps({
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "url": url,
            "file": str(path.relative_to(ROOT)),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "from_cache": cached,
        }) + "\n")


def _log_miss(url: str, code) -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    with (ROOT / "logs" / "missing.log").open("a", encoding="utf8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat()}\t{code}\t{url}\n")


def fetch(url: str, cache_name: str, retries: int = 3) -> bytes | None:
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / cache_name
    if p.exists():
        b = p.read_bytes()
        _manifest(url, p, b, cached=True)
        return b if b else None
    delay = 0.4
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r:
                b = r.read()
            p.write_bytes(b)
            _manifest(url, p, b, cached=False)
            return b
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                _log_miss(url, e.code)
                p.write_bytes(b"")
                return None
            if attempt == retries - 1:
                _log_miss(url, e.code)
                return None
            time.sleep(delay); delay *= 2
        except Exception as e:
            if attempt == retries - 1:
                _log_miss(url, type(e).__name__)
                return None
            time.sleep(delay); delay *= 2
    return None


def ddmmyyyy(d: date) -> str:
    return d.strftime("%d%m%Y")


def participant_oi(d: date) -> bytes | None:
    return fetch(PARTICIPANT_OI.format(d=ddmmyyyy(d)), f"poi_{ddmmyyyy(d)}.csv")


def participant_vol(d: date) -> bytes | None:
    return fetch(PARTICIPANT_VOL.format(d=ddmmyyyy(d)), f"pvol_{ddmmyyyy(d)}.csv")


def index_close(d: date) -> bytes | None:
    return fetch(INDEX_CLOSE.format(d=ddmmyyyy(d)), f"idx_{ddmmyyyy(d)}.csv")


def fo_bhav(d: date) -> bytes | None:
    ymd = d.strftime("%Y%m%d")
    b = fetch(FO_UDIFF.format(ymd=ymd), f"fo_{ymd}.csv.zip")
    if not b:
        return None
    try:
        z = zipfile.ZipFile(io.BytesIO(b))
        return z.read(z.namelist()[0])
    except Exception:
        return None


def business_days(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)
