# Model Card: Qwen3.5 Steel RLVR Adapters

## Model overview

This repository trains LoRA adapters on
`Qwen/Qwen3.5-0.8B-Base` at revision
`a9a407bcae463285164cc9133995c515379cebe5` for structured, historical
steel-leveling prediction.

The reported comparison contains:

1. the frozen base model;
2. a 3,000-sample, 300-step LoRA-SFT adapter;
3. a 1,500-sample, 200-step Dr. GRPO adapter initialized from SFT;
4. a second 1,500-sample, 200-step adapter that changes only the numeric reward
   by a train-frequency tail weight.

No model weights are redistributed because the adapters were trained on private
industrial prompts. The repository contains code, fixed configs and aggregate
evaluation results.

## Intended use

- Research on verifiable-reward post-training for structured numeric output.
- Offline comparison of SFT, Dr. GRPO and frequency-aware reward weighting.
- Reproduction with an independently licensed dataset following the same
  schema and split controls.

## Out-of-scope use

- Online rolling-mill control, actuator commands or safety-critical decisions.
- Claiming that the model estimates causal intervention effects.
- Use on another production line without new validation and engineering review.
- Inferring proprietary process values from the published aggregate results.

## Training

| Item | Value |
|---|---|
| Base model | Qwen3.5-0.8B-Base |
| Precision / attention | BF16 / SDPA |
| Adaptation | LoRA rank 16, alpha 32 |
| SFT | 3,000 samples, 300 steps |
| Each RL run | 1,500 samples, 200 steps, 4 completions |
| RL loss | Dr. GRPO, `scale_rewards=false`, `beta=0` |
| Max completion length | 128 tokens |
| Hardware | one AMD Radeon RX 7900 XT (19.94 GiB) |
| Runtime | WSL2 Ubuntu 22.04, ROCm/HIP 7.2 |

The tail-aware numeric reward is:

```text
w_g = clip(sqrt(200 / n_g_train), 1, 2)
r = w_g * exp(-abs(prediction - target) / train_MAD)
    + 0.05 * strict_json_reward
    + 0.05 * train_range_reward
```

`n_g_train`, `train_MAD` and the physical range are calculated only from the
training split.

## Evaluation

All four models use the same 1,017 prompts from 19 training-unseen steel grades.
The test-file SHA256 is
`6922b193e3f7818549699862d04ecae2bbd9bc011b545b13049109ec75153a49`.

| Method | Overall MAE | R² | Macro-MAE | Worst-group MAE |
|---|---:|---:|---:|---:|
| Qwen3.5 Base | 0.6864 | -0.2313 | 0.6895 | 1.6783 |
| LoRA-SFT | 0.6485 | -0.0256 | 0.7026 | 1.3750 |
| SFT + Dr. GRPO | 0.6470 | -0.0186 | 0.7011 | 1.4000 |
| SFT + Tail-aware Dr. GRPO | **0.6438** | **-0.0133** | **0.6967** | **1.3750** |

All outputs were finite strict JSON and none violated the train-derived
physical interval. Tail-aware Dr. GRPO improved overall, macro and worst-group
MAE by 0.50%, 0.63% and 1.79% relative to ordinary Dr. GRPO.

## Limitations

- All overall R² values remain negative.
- Tail-aware Macro-MAE is still 1.05% worse than the frozen base model.
- Only one seed and one GPU run are reported; no statistical significance is
  claimed.
- The CAC-submitted KD-Steel model remains substantially stronger. Its numbers
  are an external same-scenario baseline, not a reproduction by this code.
- Historical test performance is not evidence of safe online control.

## Traceability

- Full aggregate matrix: `results/formal_evaluation_matrix.json`
- Human-readable result card: `reports/result_card.generated.md`
- Experiment report: `reports/main_experiment_report.md`
- Protocol and audit amendment: `docs/experiment_protocol.md`
- Data access and leakage controls: `DATA_CARD.md`

The code and documentation are MIT-licensed. Use of the Qwen base model remains
subject to its own model license and terms.
