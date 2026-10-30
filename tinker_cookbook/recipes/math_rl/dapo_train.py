import asyncio
from typing import Any

import chz
from tinker.types import LossFnType

from tinker_cookbook.recipes.math_rl.train import CLIConfig as MathRLCLIConfig
from tinker_cookbook.recipes.math_rl.train import cli_main


@chz.chz
class DAPOConfig(MathRLCLIConfig):
    """DAPO preset for math_rl: clip-higher PPO + dynamic group filtering.

    References:
        - DAPO paper: https://arxiv.org/abs/2503.14476
    """

    loss_fn: LossFnType = "ppo"
    # DAPO recommends asymmetric clipping: tight low side, looser high side
    # so positive-advantage tokens with rising ratios can keep contributing.
    loss_fn_config: dict[str, Any] | None = chz.field(
        default_factory=lambda: {
            "clip_low_threshold": 0.8,
            "clip_high_threshold": 1.28,
        }
    )
    remove_constant_reward_groups: bool = True


if __name__ == "__main__":
    cli_config = chz.entrypoint(DAPOConfig)
    asyncio.run(cli_main(cli_config))
