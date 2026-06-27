"""RL post-training for the Cog agent app.

The trainer never imports the agent's loop. It stands up an OpenAI-compatible sampling
proxy (``proxy.py``) backed by the policy being trained, points the production app at it,
and triggers rollouts through the app's HTTP endpoints. It then grades each rollout's
final program against the held-out expected output (``grading.py`` / ``tasks.py``) and
runs GRPO on the tokens the proxy captured (``train.py``).
"""

from .grading import shaped_reward
from .tasks import EVAL_FAMILIES, TRAIN_FAMILIES, CogTask, build_tasks

__all__ = ["CogTask", "shaped_reward", "build_tasks", "TRAIN_FAMILIES", "EVAL_FAMILIES"]
