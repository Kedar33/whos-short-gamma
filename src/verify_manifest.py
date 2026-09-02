from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.jsonl"


def main() -> int:
    if not MANIFEST.exists():
        print("no manifest; nothing to verify")
        return 0
    expected: dict[str, set[str]] = defaultdict(set)
    urls: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        expected[r["file"]].add(r["sha256"])
        urls[r["file"]] = r["url"]

    ok = missing = mismatch = unstable = 0
    for rel, hashes in sorted(expected.items()):
        p = ROOT / rel.replace("\\", "/")
        if len(hashes) > 1:
            unstable += 1
            print(f"UNSTABLE  {rel}: {len(hashes)} distinct hashes recorded")
            continue
        if not p.exists():
            missing += 1
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h == next(iter(hashes)):
            ok += 1
        else:
            mismatch += 1
            print(f"MISMATCH  {rel}\n          expected {next(iter(hashes))}"
                  f"\n          found    {h}\n          source   {urls[rel]}")

    total = len(expected)
    print(f"\nmanifested files : {total:,}")
    print(f"  verified       : {ok:,}")
    print(f"  missing locally: {missing:,}  (re-downloadable; cache is optional)")
    print(f"  hash mismatch  : {mismatch:,}")
    print(f"  unstable hash  : {unstable:,}")
    if mismatch or unstable:
        print("\nINTEGRITY FAILURE")
        return 1
    print("\nintegrity OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
