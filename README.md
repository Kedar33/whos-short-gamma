# Who's Short Gamma? Nobody Knows

Sharp bounds on dealer gamma exposure, and on its effect on volatility, from
participant-level options disclosure in India and Taiwan.

**Headline.** Exchange disclosure fixes each participant's total long and short
open interest but not its allocation across strikes, so gamma exposure is
bounded rather than measured. Whether the bounds exclude zero turns on one
observable: how balanced the participant's book is. Indian and Taiwanese
proprietary desks run long-to-short ratios near **1.2**, and at that balance the
sign is undetermined on **96 of 96** participant-wing-days in India and **50 of
50** dealer observations in Taiwan. The exceptions in both markets are investors
holding one-sided books, which is where the arithmetic says the sign should be
recoverable. The "gamma flip level" is undetermined at **every** spot level
examined; observable turnover flows shrink the interval by **0.00%**.

---

## Reproducing

```bash
conda env create -f environment.yml && conda activate india-options
make all
```

`make all` runs, in order:

| Step | Script | Output |
|---|---|---|
| Theory verification | `pytest tests/ -v` | 6 tests, incl. counterexample search |
| Participant panel | `src/build_participant_panel.py` | `data/processed/participant_panel.parquet` |
| Identification panel | `src/gamma_panel.py` | `data/processed/gamma_panel.json` |
| Flip level | `src/flip_level.py` | `data/processed/flip_level.json` |
| Robustness | `src/robustness.py` | bucket + IV perturbation tables |
| Disclosure design | `src/disclosure.py` | `data/processed/disclosure.json` |
| Paper | `cd paper && pdflatex main && bibtex main && pdflatex main` | `paper/main.pdf` |

Raw data is cached under `data/raw/` and hashed into `data/manifest.jsonl`.
Deleting the cache triggers re-download from NSE; results are unchanged because
every file is content-addressed by SHA-256.

## Layout

```
theory/setup.md      working notes: bounds, proofs, assumptions, the retraction
src/                 ingestion, identification LPs, robustness, disclosure
tests/test_theory.py numerical verification + adversarial counterexample search
paper/               main.tex, refs.bib
DATA_LOG.md          sources, hashes, filter cascade with counts, known gaps
OBJECTIONS.md        the ten strongest objections, answered or conceded
```

## Claim status

Every claim carries a tag.

| Claim | Status |
|---|---|
| Identified set is a compact interval, endpoints attained | proved |
| Closed-form endpoints by greedy fill | proved (verified against simplex to 1e-6) |
| Sign is recoverable exactly when one block dominates under every allocation | proved (follows by inspection of the endpoints; verified exact over 300+ instances) |
| Book-balance threshold for identification lies between ratios of 2 and 3 | simulated (300 markets per row, `simulation/known_truth.py`) |
| Disclosure thresholds (88–100% of OI) | numerical (benchmark-dependent; see the caveat in the paper) |

## Corrections made during this project

Recorded because the audit trail is part of the argument.

1. **A claimed result was retracted.** An earlier draft argued by symmetry that
   the interval *always* contains zero. A counterexample search refuted it
   within 400 random draws: with a long-to-short ratio of 17.9 the interval is
   entirely positive. Whether the sign is identified is a question about the
   data, not a fact about the geometry, which is what makes the empirical work
   necessary. See `theory/setup.md` and Remark 1 in the paper.
2. **A refuted conjecture.** Ordering disclosure by gamma contribution rather
   than open interest was predicted to identify the sign far sooner. It does
   not. Reported in `src/disclosure.py` and in the paper.
3. **Four data/code bugs**, all caught by checks rather than by inspection:
   silent counting of failed LPs as "identified"; two-period LP infeasibility
   from common-cell rescaling; a units mismatch between bhavcopy (units) and
   the participant file (contracts); and a reconciliation gate that measured
   Black–Scholes attrition instead of universe mismatch. Three of the four
   distorted a headline number before being fixed.

## Data

Free and unauthenticated public exchange archives only --- NSE (India) and
TAIFEX (Taiwan). No API key, no paid source. No exchange data is redistributed
here; everything is fetched at run time and hash-verified against
`data/manifest.jsonl` by `make check`. See `DATA_LOG.md` for URLs, timestamps,
the filter cascade with counts, and the documented gaps.

## Licence

Dual, as is usual for a research repository:

- **Code** (`src/`, `tests/`, `simulation/`, `Makefile`) --- MIT
- **Paper and text** (`paper/`, `theory/`, `*.md`) --- CC BY 4.0, matching the
  preprint

See `LICENSE`.

## Citation

> Sahu, K. (2026). *Who's Short Gamma? Nobody Knows.* Working paper.

If you use the identification machinery, the closed-form endpoints in
`src/flip_level.py` and the interval-regression bounds in
`src/interval_regression.py` are the reusable pieces.
