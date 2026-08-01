from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def test_sft_main_protocol_is_fixed_and_rocm_safe() -> None:
    config = _config("sft_main.yaml")
    assert config["model_name_or_path"] == "Qwen/Qwen3.5-0.8B-Base"
    assert config["model_revision"] != "main"
    assert config["mixed_precision"] == "bf16"
    assert config["max_samples"] == 3000
    assert config["max_steps"] == 300
    assert config["lora_r"] == 16
    assert config["lora_alpha"] == 32
    assert config["train_file"] != config["validation_file"]


def test_dpo_baseline_and_tail_configs_only_change_output_and_sampling() -> None:
    baseline = _config("dpo_baseline.yaml")
    tail_aware = _config("dpo_tail_aware.yaml")
    assert baseline["output_dir"] != tail_aware["output_dir"]
    assert baseline["tail_aware_sampling"] is False
    assert tail_aware["tail_aware_sampling"] is True
    controlled_baseline = {
        key: value
        for key, value in baseline.items()
        if key not in {"output_dir", "tail_aware_sampling"}
    }
    controlled_tail = {
        key: value
        for key, value in tail_aware.items()
        if key not in {"output_dir", "tail_aware_sampling"}
    }
    assert controlled_baseline == controlled_tail
    assert baseline["loss_type"] == "sigmoid"
    assert baseline["beta"] == 0.1
    assert baseline["max_samples"] == 3000
    assert baseline["max_steps"] == 200
    assert baseline["validation_max_samples"] == 768


def test_formal_evaluations_share_one_test_contract() -> None:
    configs = [
        _config("eval_base_dpo_study.yaml"),
        _config("eval_sft_dpo_study.yaml"),
        _config("eval_dpo_baseline.yaml"),
        _config("eval_dpo_tail_aware.yaml"),
    ]
    controlled = [
        "split_file",
        "data_report_file",
        "max_samples",
        "max_new_tokens",
        "balanced_by_pass",
        "seed",
    ]
    assert all(
        {key: config[key] for key in controlled}
        == {key: configs[0][key] for key in controlled}
        for config in configs[1:]
    )
    assert configs[0]["max_samples"] == 1017
