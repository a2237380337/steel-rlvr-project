"""Fail fast when every RLVR group has zero reward variance."""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback


class RewardHealthCallback(TrainerCallback):
    def __init__(self, patience: int) -> None:
        if patience <= 0:
            raise ValueError("reward_zero_std_patience must be positive")
        self.patience = patience
        self.consecutive_zero_std = 0

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        fraction = (logs or {}).get("frac_reward_zero_std")
        if fraction is None:
            return
        self.consecutive_zero_std = self.consecutive_zero_std + 1 if float(fraction) >= 1.0 else 0
        if self.consecutive_zero_std >= self.patience:
            raise RuntimeError(
                "RLVR reward health check failed: all groups had zero reward variance "
                f"for {self.consecutive_zero_std} consecutive logged steps"
            )
