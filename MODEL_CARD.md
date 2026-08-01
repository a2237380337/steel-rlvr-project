# Model Card: Qwen3.5 Steel DPO Adapters

## Model overview

This repository trains LoRA adapters on `Qwen/Qwen3.5-0.8B-Base` revision `a9a407bcae463285164cc9133995c515379cebe5` for structured historical steel-leveling prediction.

The comparison contains the frozen base model, a 3,000-sample LoRA-SFT adapter, a uniform-sampling DPO adapter, and a frequency-aware DPO adapter. Both DPO policies start from the same SFT checkpoint and use an explicit frozen copy of that SFT adapter as reference.

No weights or row-level industrial data are redistributed.

## Training

| Item | Value |
|---|---|
| Base model | Qwen3.5-0.8B-Base |
| Precision / attention | BF16 / SDPA |
| SFT | 3,000 samples, 300 steps, LoRA rank 16, alpha 32 |
| Preference pool | 81,546 train / 20,388 validation pairs |
| Each DPO run | 3,000 train / 768 validation pairs, 200 steps |
| DPO | sigmoid loss, beta 0.1, effective batch 16 |
| Negative margins | 0.5/1.0/1.5 × train-only MAD |
| Hardware | one AMD Radeon RX 7900 XT (19.94 GiB) |
| Runtime | WSL2 Ubuntu 22.04, ROCm/HIP 7.2 |

The frequency-aware run samples pairs using `clip(sqrt(200 / n_g_train), 1, 2)`. Test labels and held-out-grade frequencies are not used.

## Evaluation

All models use the same 1,017 prompts from 19 training-unseen steel grades. Test SHA256: `6922b193e3f7818549699862d04ecae2bbd9bc011b545b13049109ec75153a49`.

| Method | Overall MAE | R² | Macro-MAE | Worst-group | Strict JSON |
|---|---:|---:|---:|---:|---:|
| Qwen3.5 Base | 0.6864 | -0.2313 | **0.6895** | 1.6783 | 100.00% |
| LoRA-SFT | **0.6486** | **-0.0255** | 0.7029 | **1.3750** | **100.00%** |
| SFT + DPO | 0.6488 | -0.0263 | 0.7028 | **1.3750** | 99.12% |
| SFT + frequency-aware DPO | 0.6488 | -0.0263 | 0.7028 | **1.3750** | 99.02% |

All outputs contained a finite numeric value and none violated the train-derived physical interval. DPO did not improve MAE over SFT. The two DPO weight files differ, but greedy decoding produced identical numeric predictions on all 1,017 test prompts.

## Intended use

- Offline study of SFT and preference optimization for structured numeric output.
- Reproduction with independently licensed data under the same leakage controls.
- Analysis of why preference accuracy may not translate to regression accuracy.

## Limitations

- All overall R² values remain negative.
- SFT and DPO Macro-MAE are worse than the frozen base model.
- Only one seed and one GPU run are reported; no significance claim is made.
- Fixed-offset negatives were too easy to change greedy unseen-grade predictions materially.
- The CAC-submitted KD-Steel model remains stronger and is only a same-scenario external baseline.
- Historical test performance is not evidence of safe online control.

Traceability is provided by `results/formal_evaluation_matrix.json`, `reports/result_card.generated.md`, `reports/main_experiment_report.md`, and `reports/provenance.md`.
