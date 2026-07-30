# Formal Experiment Provenance

This file is a sanitized index of the private run manifests. It contains no
row-level prompt, target, prediction or production value.

## Training runs

All timestamps are UTC. The training source-tree SHA256 covers project-owned
Python, tests, YAML configs, shell entry points and dependency files.

| Run | Session | Start | End | Samples | Peak MiB | Status |
|---|---|---|---|---:|---:|---|
| LoRA-SFT | `sft-sft_main-48350392257a` | 2026-07-29 17:33:24 | 2026-07-29 18:36:17 | 3,000 | 3511.09 | completed |
| Dr. GRPO | `grpo-grpo_baseline-7d2e61f14f2a` | 2026-07-29 18:36:56 | 2026-07-29 18:48:36 | 1,500 | 2422.67 | completed |
| Tail-aware Dr. GRPO | `grpo-grpo_tail_aware-ca05be7026f3` | 2026-07-29 18:49:07 | 2026-07-29 19:00:32 | 1,500 | 2391.78 | completed |

Shared training source-tree SHA256:

```text
f583735af7384814b7fe0312427d9f4abaf65ed68a76088236d91404eb277e4b
```

Config SHA256:

```text
sft_main.yaml          a5c9fdd84e0ea5da00962d0ff14d1a7a0f27099dd17444601beab0eadbf58ed3
grpo_baseline.yaml     03d2ae99e1841f384a5a79c37d57cf2f3816e55c7a5e05029229510ace433037
grpo_tail_aware.yaml   25c14289b576d98857e3feb79de2924af2c14df2a56025ce2068c2ff3718d1fd
```

Data and ordered-selection SHA256:

```text
train.jsonl            ef60984c117b8f5b6118e3e2272e160864e95254c20b44919e6d8af18fa0abc2
validation.jsonl       7268251f77d0fe159602ba8a88c6b4511ea43538706388fd9086b7ce007561c1
SFT ordered train IDs  544da8e199c0073ac4a7862ca84f77aeaa914e297a80e09ade83055d9fae7cef
RL ordered train IDs   43fffe5d7da0d0cddc90a128ebc1dd0064f6186437dfb5759a904e06a5397cb9
```

The identical RL ordered-ID hash is evidence that ordinary and tail-aware
Dr. GRPO used the same 1,500 examples in the same order.

## Final evaluation

All four evaluations used:

```text
base revision          a9a407bcae463285164cc9133995c515379cebe5
test.jsonl              6922b193e3f7818549699862d04ecae2bbd9bc011b545b13049109ec75153a49
ordered sample IDs      16688d7338105327588dcafe04f738d2b5d59041203a86c1994d8630e8113139
evaluation source tree  ff61ca60d7a37395ae88a75692861d744990fa72b63602c1f0efc60838f2e3a2
sample count            1,017
```

`results/formal_evaluation_matrix.json` was accepted only after all four
summaries agreed on these identifiers.

## Git history boundary

The formal runs occurred while this project directory was still inside an
uncommitted parent worktree. Their manifests therefore honestly record a null
run commit and a dirty worktree; no historical commit was fabricated later.

The independent public Git history begins with the reviewed release package
after training and evaluation. The exact historical run identities remain the
config, source-tree, data and ordered-ID hashes above. Later release-only files
such as the data card, model card and release audit intentionally change the
current whole-tree hash without changing the recorded model outputs.
