"""Knowledge distillation from teacher to student models.

Three distillation approaches are available:

**On-policy** (:mod:`~tinker_cookbook.distillation.train_on_policy`):
    The student generates its own rollouts. Teacher logprobs are computed on
    the student's sampled tokens and used as a KL penalty on the RL advantages.

**Off-policy** (:mod:`~tinker_cookbook.distillation.train_off_policy`):
    The student trains on fixed data (e.g., an SFT mix). At each token position,
    the teacher's top-K distribution serves as soft targets for cross-entropy.

**SDFT** (:mod:`~tinker_cookbook.distillation.sdft`):
    Self-Distillation Fine-Tuning. The teacher sees both question and golden
    answer as a demonstration; the student sees only the question and generates
    on-policy.

Each submodule exposes a ``Config`` dataclass and an async ``main()`` entry
point. Access them via the submodule to avoid naming conflicts::

    from tinker_cookbook.distillation import train_on_policy

    config = train_on_policy.Config(
        model_name="Qwen/Qwen3.6-35B-A3B",
        ...,
    )
    await train_on_policy.main(config)
"""

from tinker_cookbook.distillation import sdft, train_off_policy, train_on_policy

__all__ = [
    "train_on_policy",
    "train_off_policy",
    "sdft",
]
