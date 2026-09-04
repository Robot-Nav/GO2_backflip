"""Gym registration for the direct Go2 backflip task."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Go2-Backflip-Direct-v0",
    entry_point=f"{__name__}.go2_backflip_env:Go2BackflipEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_backflip_env_cfg:Go2BackflipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2BackflipPPORunnerCfg",
    },
)

