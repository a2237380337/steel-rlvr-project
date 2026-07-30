# Data Card: Low-frequency Steel Leveling

## Summary

This project uses a private historical hot-rolling dataset associated with the
author's CAC-submitted KD-Steel study. The dataset is not redistributed. The
repository publishes only the processing code, the split protocol, aggregate
counts, hashes and aggregate model metrics.

The task predicts one continuous leveling value for each of three rolling
passes. Each source record is converted into three time-ordered prompts so that
a pass may use only measurements that would already be available at that point.

## Access and license

- Source owner/access: private industrial research data available locally to
  the author.
- Redistribution: prohibited in this repository.
- Public users must provide their own legally obtained data with the required
  columns, or use synthetic data for code-path tests.
- This repository's MIT license does not grant any right to the private data.

## Audited dataset contract

| Item | Count |
|---|---:|
| Raw Excel rows | 34,318 |
| Excluded field-template sentinel rows | 1 |
| Valid source records | 34,317 |
| Real steel grades | 63 |
| Common grades (frequency at least 50) | 44 |
| Rare held-out grades (frequency below 50) | 19 |
| Train source records | 27,182 |
| Validation source records | 6,796 |
| Test source records | 339 |
| Train/validation/test pass samples | 81,546 / 20,388 / 1,017 |

The excluded row contained `Slab_ID` as its slab identifier and `Steel_Grade`
as its category value. It was classified as a template record from field
semantics, not from model error. Removing it left the train and validation
JSONL files byte-identical:

- train SHA256:
  `ef60984c117b8f5b6118e3e2272e160864e95254c20b44919e6d8af18fa0abc2`
- validation SHA256:
  `7268251f77d0fe159602ba8a88c6b4511ea43538706388fd9086b7ce007561c1`
- final test SHA256:
  `6922b193e3f7818549699862d04ecae2bbd9bc011b545b13049109ec75153a49`

## Split and leakage controls

1. Grades with fewer than 50 source records are held out as unseen test
   categories; none of their records enters training or validation.
2. Common-grade source records are split 8:2 with seed 42. All three pass
   prompts from one source record stay in the same split.
3. Train-only labels determine target medians, MAD scales, 1st/99th percentile
   physical bounds and tail-aware grade frequencies.
4. Test labels do not affect SFT, reward computation, normalization,
   checkpoint selection or physical bounds.
5. Pass1/Pass2/Pass3 schemas explicitly prevent later-pass measurements from
   appearing in an earlier-pass prompt.
6. Formal data preparation fails unless the audited counts match. Non-paper
   synthetic data requires the explicit `--allow-contract-mismatch` flag.

## Published fields

The code names the semantic process fields needed to reproduce feature
construction and temporal availability. The repository does not publish:

- individual orders, slab identifiers or timestamps;
- row-level process measurements or targets;
- generated prompts, predictions or failure examples containing process values;
- processed JSONL files or source spreadsheets.

## Known limitations

- Historical observational data cannot establish closed-loop control safety or
  intervention effects.
- The test set contains only 339 source records across 19 rare grades.
- Grade names alone do not provide robust material semantics to an unseen
  language model.
- A single production line and time window may not represent other mills,
  equipment states or operating policies.

## Reproduction without the private data

The reward, parsing, metric, split and contract logic can be checked with:

```bash
python -m pip install -e ".[dev]"
ruff check src tests
pytest -q
```

The unit tests construct synthetic frames in temporary directories and do not
load private production records.
