import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import fast_cells
import gamma_identification as gi
from gamma_panel import raw_contract_total

TEST_DATE = date(2025, 6, 17)


@pytest.fixture(scope="module")
def both():
    fast, raw_tot = fast_cells.load_fast(TEST_DATE)
    slow = gi.load_cells(TEST_DATE)
    if fast is None or not slow:
        pytest.skip("bhavcopy unavailable offline")
    return fast, raw_tot, slow


def test_raw_contract_total_matches(both):
    _, raw_tot, _ = both
    for wing in ("C", "P"):
        assert raw_tot[wing] == pytest.approx(
            raw_contract_total(TEST_DATE, wing), rel=1e-9)


def test_cell_count_close(both):
    fast, _, slow = both
    assert abs(len(fast["K"]) - len(slow)) / max(len(slow), 1) < 0.01


def test_total_gamma_notional_matches(both):
    fast, _, slow = both
    for wing, want_call in (("C", True), ("P", False)):
        m = fast["is_call"] if want_call else ~fast["is_call"]
        f = float((fast["gnotional"][m] * fast["oi"][m]).sum())
        s = sum(c["gnotional"] * c["oi"] for c in slow
                if (c["cp"] == "CE") == want_call)
        assert f == pytest.approx(s, rel=2e-3), f"wing {wing}: {f} vs {s}"


def test_implied_vols_agree(both):
    fast, _, slow = both
    key_fast = {(str(fast["sym"][i]), bool(fast["is_call"][i]),
                 round(float(fast["K"][i]), 2),
                 round(float(fast["T"][i]), 6)): float(fast["iv"][i])
                for i in range(len(fast["K"]))}
    diffs = []
    for c in slow:
        k = (c["sym"], c["cp"] == "CE", round(c["K"], 2), round(c["T"], 6))
        if k in key_fast and c.get("iv"):
            diffs.append(abs(key_fast[k] - c["iv"]))
    assert len(diffs) > 100, "too few overlapping contracts to compare"
    assert np.median(diffs) < 1e-4, f"median IV gap {np.median(diffs)}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
