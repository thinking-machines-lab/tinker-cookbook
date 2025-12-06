#!/usr/bin/env python
"""Fake training script for xmux demos"""

import random
import time
from typing import Any

from pydantic import BaseModel


class Config(BaseModel):
    duration: int = 60
    failure_rate: float = 0.2
    model: str = "unknown"
    lr: float = 0.001


def fake_train_model(config_dict: dict[str, Any]):
    """Simulate a training job with configurable duration and failure rate"""
    config = Config.model_validate(config_dict)
    assert isinstance(config, Config)

    # Determine if this run will fail
    will_fail = random.random() < config.failure_rate

    print("Starting fake training job...")
    print(f"Model: {config.model}")
    print(f"Learning rate: {config.lr}")
    print(f"Duration: {config.duration}s")
    print(f"Config: {Config.model_dump_json(config)}")
    print("-" * 50)

    # Simulate training with periodic output
    start_time = time.time()
    loss: float = 2.0  # Initialize loss in case loop doesn't execute
    for epoch in range(1, config.duration // 5 + 1):
        if (epoch - 1) * 5 >= config.duration:
            break

        # Simulate loss decreasing over time (with some noise)
        base_loss = 2.0 * (0.95**epoch)
        loss = base_loss + random.uniform(-0.1, 0.1)

        elapsed = int(time.time() - start_time)
        print(f"[Epoch {epoch:3d}] [{elapsed:3d}s] loss={loss:.4f} lr={config.lr}")

        # Random events
        if random.random() < 0.1:
            print(f"[Epoch {epoch:3d}] Validation: accuracy={random.uniform(0.7, 0.95):.3f}")

        # Fail midway if designated to fail
        if will_fail and epoch > config.duration // 10:
            print("\nERROR: Training failed due to simulated error!")
            print("Exception: Fake convergence issue detected")
            raise Exception("Fake convergence issue detected")

        time.sleep(5)

    # Success
    print("\nTraining completed successfully!")
    print(f"Final loss: {loss:.4f}")
    print(f"Total time: {int(time.time() - start_time)}s")
    return 0


def main(config: dict[str, Any]):
    """Entry point that xmux will call"""
    # For compatibility with how xmux calls this
    return fake_train_model(config)


if __name__ == "__main__":
    # For testing standalone
    test_config = {"model": "test-model", "lr": 0.01, "duration": 30, "failure_rate": 0.1}
    exit(main(test_config))
