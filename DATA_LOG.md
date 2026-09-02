# DATA_LOG

Every source is free, public and unauthenticated. **No API key is used anywhere
in this project. No paid data source is used.** No data is simulated; the only
synthetic numbers appear in `tests/test_theory.py`, which are labelled as
random instances for theorem verification and never enter the paper.

Retrieval session: **31 August 2026**.

## 1. Manifest

Every download is recorded in `data/manifest.jsonl`, one JSON object per
retrieval, with the full query URL, UTC retrieval timestamp, byte count,
SHA-256 of the raw payload, and whether the payload was served from cache.

| | |
|---|---|
| Manifest records | 4,979 |
| Distinct URLs | 3,886 |
| Raw files cached | 4,089 |
| Raw bytes cached | 151.3 MB |

Example record:

```json
{"retrieved_utc": "2026-08-31T11:12:05+00:00",
 "url": "https://archives.nseindia.com/content/nsccl/fao_participant_oi_01042019.csv",
 "file": "data\\raw\\poi_01042019.csv", "bytes": 902,
 "sha256": "99807537b8a4574bc914dd54a003067cfb16236c60b5ff74615b5f429643811e",
 "from_cache": false}
```

## 2. Sources

| Source | URL pattern | Content |
|---|---|---|
| Participant-wise OI | `archives.nseindia.com/content/nsccl/fao_participant_oi_DDMMYYYY.csv` | Daily index/stock option and futures OI split into Client / DII / FII / Pro, long and short |
| Participant-wise volume | `.../fao_participant_vol_DDMMYYYY.csv` | Same split, contracts traded |
| F&O bhavcopy (UDiFF) | `archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_YYYYMMDD_F_0000.csv.zip` | Per-contract strike, expiry, type, OHLC, settlement, underlying price, OI, volume, **board lot** |
| Index closes | `archives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv` | Daily index OHLC incl. India VIX |

Coverage verified live: participant files resolve from **April 2019** through
**August 2026**; 1,829 trading days retrieved, 106 dates absent (weekends and
exchange holidays), recorded in `logs/missing.log`. Absent dates are logged and
skipped, never interpolated.

## 3. The units problem — and how it was found

**bhavcopy `OpnIntrst` is denominated in units/shares; the participant file is
in contracts.** The two differ by a factor of **70–120** that drifts over time
with the symbol mix, because SEBI's 2024 reforms changed lot sizes at different
dates for different indices. Observed board lots in the sample:

| Symbol | Lot sizes observed |
|---|---|
| NIFTY | 25, 50, 65, 75 |
| BANKNIFTY | 30, 35 |
| FINNIFTY | 60, 65 |
| MIDCPNIFTY | 120, 140 |
| NIFTYNXT50 | 25 |

Converting via the bhavcopy's own `NewBrdLotQty` field reconciles the two
universes **exactly**:

| Date | Calls ratio | Puts ratio |
|---|---|---|
| 2025-06-02 | 1.000 | 1.000 |
| 2025-09-15 | 1.000 | 1.000 |
| 2025-06-16 | 1.000 | 1.000 |
| 2026-06-16 | 1.881 | 1.937 (fails) |

## 4. Filter cascade, with counts

Applied in order, to each date:

1. **Index options only** — `FinInstrmTp == "IDO"`, symbols in
   {NIFTY, BANKNIFTY, FINNIFTY, MIDCPNIFTY, NIFTYNXT50}.
2. **Board lot present and positive** — required for the units conversion;
   rows without it are dropped.
3. **Reconciliation gate** — the date enters only if the *raw* bhavcopy
   contract total matches the participant-file total to within **2%**.
   Failing dates are **dropped, never rescaled**, because a mismatch means the
   two sources describe different instrument sets.
   *Gate outcome:* passes for Dec-2024 → Oct-2025 and Feb-2026 → May-2026;
   fails for all sampled dates before Nov-2024, and for scattered later dates
   (2025-12-16, 2026-06-16).
4. **Black–Scholes invertibility** — implied volatility must solve within
   $[10^{-3}, 5]$; positive price, positive OI, positive underlying.
   *Attrition:* removes **0.5%–4.4% of open interest**, concentrated in deep
   ITM/OTM wings where the settlement price sits at or below intrinsic.
5. **Moneyness/tenor bucketing** — cells aggregated by (symbol, tenor bucket,
   log-moneyness bin). Bucketing *removes* allocation freedom and can only
   *shrink* the identified set, so it is conservative for our claim.

## 5. Final sample

| | |
|---|---|
| Dates sampled | biweekly within gate-passing windows |
| Core participant-wing-days (Client/FII/Pro) | **96** |
| Two-period (turnover) specification solved | 60 of 96 |

DII is reported but excluded from core statistics: its short open interest is
identically zero, which violates Assumption 4 of the paper and makes its sign
trivially determined. This is the theory's own boundary case, and the data's
only exception — an internal consistency check, not a filter of convenience.

   verified available (daily/monthly, Oct 1993 – Mar 2023) but are **not used**
   in this paper.
