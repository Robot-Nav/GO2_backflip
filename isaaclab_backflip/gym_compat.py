"""Isaac Lab adapter for the original Isaac Gym backflip PPO implementation.

The original ``rl.Backflip`` code only depends on a small vector-environment
contract.  This adapter keeps Isaac Lab responsible for simulation, rewards,
and resets while presenting that contract without changing any tensors.
"""

from __future__ import annotations

import torch


class GymBackflipVecEnvAdapter:
    """Expose a DirectRLEnv through the original BackflipRunner API."""

    def __init__(self, env):
        self.env = env
        base_env = env.unwrapped
        self.num_envs = base_env.num_envs
        self.device = base_env.device
        self.num_obs = base_env.cfg.observation_space
        self.num_privileged_obs = base_env.cfg.state_space
        self.num_actions = base_env.cfg.action_space
        self.max_episode_length = base_env.max_episode_length

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self.unwrapped.episode_length_buf = value

    @staticmethod
    def _observations(observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            "obs": observations["policy"],
            "privileged_info": observations["critic"],
        }

    def reset(self):
        observations, _ = self.env.reset()
        return self._observations(observations)

    def get_observations(self):
        return self._observations(self.unwrapped._get_observations())

    def step(self, actions: torch.Tensor):
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        # The original PPO only needs ``time_outs`` for value bootstrapping.
        # Map Isaac Lab's episodic log payload to the original runner name so
        # its TensorBoard/console logging remains available as well.
        infos = dict(extras)
        infos["time_outs"] = truncated
        if "log" in extras:
            infos["episode"] = extras["log"]
        return self._observations(observations), rewards, dones, infos

    def close(self):
        self.env.close()
