"""RSL-RL runner extensions required by the migrated backflip task."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner


class BackflipOnPolicyRunner(OnPolicyRunner):
    """Preserve Gym-era action-noise bounds and checkpoint compatibility."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self._min_action_std = float(train_cfg.get("min_action_std", 0.35))
        self._max_action_std = float(train_cfg.get("max_action_std", 1.50))
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self._rsl_update = self.alg.update
        self.alg.update = self._update_with_bounded_std

    def _update_with_bounded_std(self):
        losses = self._rsl_update()
        with torch.no_grad():
            if hasattr(self.alg.policy, "std"):
                self.alg.policy.std.clamp_(self._min_action_std, self._max_action_std)
            elif hasattr(self.alg.policy, "log_std"):
                self.alg.policy.log_std.clamp_(
                    torch.log(torch.tensor(self._min_action_std, device=self.alg.policy.log_std.device)),
                    torch.log(torch.tensor(self._max_action_std, device=self.alg.policy.log_std.device)),
                )
        return losses

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        """Load either an Isaac Lab RSL-RL checkpoint or the original Gym checkpoint."""
        checkpoint = torch.load(
            path,
            weights_only=False,
            map_location=map_location if map_location is not None else self.device,
        )
        if "iter" in checkpoint and "infos" in checkpoint:
            return super().load(path, load_optimizer=load_optimizer, map_location=map_location)

        # The Gym AsymmetricActorCritic and Isaac Lab ActorCritic have exactly
        # the same parameter names and tensor shapes for this task.
        self.alg.policy.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]

        iteration = int(checkpoint.get("iteration", 0))
        if iteration == 0:
            match = re.search(r"model_(\d+)\.pt$", Path(path).name)
            if match:
                iteration = int(match.group(1))
        self.current_learning_iteration = iteration
        print(f"[INFO] Loaded legacy Isaac Gym checkpoint at iteration {iteration}: {path}")
        return checkpoint.get("infos", {})
