# Formal Experiment Provenance

This sanitized index contains no row-level prompt, target, prediction or production value. Timestamps are UTC.

## Training runs

| Run | Session | Start | End | Samples | Peak MiB | Status |
|---|---|---|---|---:|---:|---|
| LoRA-SFT | `sft-sft_main-48350392257a` | 2026-07-29 17:33:24 | 2026-07-29 18:36:17 | 3,000 | 3511.09 | completed |
| DPO | `dpo-dpo_baseline-60bd25026618` | 2026-08-01 08:17:46 | 2026-08-01 09:07:09 | 3,000 | 6620.30 | completed |
| Frequency-aware DPO | `dpo-dpo_tail_aware-e01200fe962d` | 2026-08-01 09:07:33 | 2026-08-01 09:57:28 | 3,000 | 6620.30 | completed |

DPO training source-tree SHA256:

```text
9a7f65fce87aaacf9b4d06a64c678534f7859d982bedf04f7ba6ae88a345ea12
```

Config SHA256:

```text
dpo_baseline.yaml     7a313239038fd11cb6789e62ed41dff0236f8b84ac9bab66499663d1b0c1a30f
dpo_tail_aware.yaml   14f1e70106bfc3ee6d0de8221dba98496758834039536be08d7d299b0b04eda9
```

Preference artifacts:

```text
preferences_train.jsonl       bfeb77c489fbfedce0a47a1ae67954260d61e27e40b20fb7952459e1be84524b
preferences_validation.jsonl  e16120c7346725b83ea8b70287becb7c63456b6293ea238d911ecc8ec97bf84a
uniform ordered train IDs     8b8c3cada5e26c6378033bdd5f009631f38c04b1390591afd0c30fc6ff126d7e
weighted ordered train IDs    69c1b096e979d55d5e9f70ab5abe651df91f5d0bbdf1dfcde398afc36c5da1d2
shared validation IDs         7ce1ab00193da0b06468cfb3de2fd2eb49a279ab094afdfe18fc360629bb445c
```

The train-ID hashes intentionally differ because frequency-aware sampling changes the selected pair distribution. Pass quotas, pair count, seed and all optimizer settings remain controlled. Validation IDs and order are identical.

Adapter SHA256:

```text
DPO                  07a1c3fe2daebd4193f0e18b526fd95337491ec4f2cbad5c2fdde339e450ebfa
Frequency-aware DPO  d052e8e7ba34bae9193bf9e856d30a0c4b3477da1f039ff9d95c182258079a59
```

## Final evaluation

All four evaluations used:

```text
base revision          a9a407bcae463285164cc9133995c515379cebe5
test.jsonl              6922b193e3f7818549699862d04ecae2bbd9bc011b545b13049109ec75153a49
ordered sample IDs      16688d7338105327588dcafe04f738d2b5d59041203a86c1994d8630e8113139
evaluation source tree  041ade8ebf1837af5d01412005cbbf956fcb883dddacb4f6ef8ec31c43419f85
sample count            1,017
```

`results/formal_evaluation_matrix.json` is generated only after all summaries agree on these identifiers.

## Git boundary

The DPO runs were executed from commit `7f9e6276ddcbcee0326433275932632632309ec1` with the new DPO implementation still uncommitted. The manifests record `dirty=true` and the exact source-tree hash; this release does not claim that the old commit contains the new code. Documentation and release-audit edits after evaluation may change the current whole-tree hash without changing recorded model outputs.
