#!/usr/bin/env python
"""Fake training script for xmux demos"""

import json
import random
import time

from pydantic import BaseModel


class Config(BaseModel):
    """Config structure that matches what xmux expects"""

    log_relpath: str
    entrypoint: str
    entrypoint_config: dict[str, object]
    num_gpus: int = 0


def fake_train_model(config: dict[str, object]):
    """Simulate a training job with configurable duration and failure rate"""
    # Extract parameters
    duration: int = config.get("duration", 60)  # type: ignore
    failure_rate: float = config.get("failure_rate", 0.2)  # type: ignore
    model: str = config.get("model", "unknown")  # type: ignore
    lr: float = config.get("lr", 0.001)  # type: ignore

    # Determine if this run will fail
    will_fail = random.random() < failure_rate

    print("Starting fake training job...")
    print(f"Model: {model}")
    print(f"Learning rate: {lr}")
    print(f"Duration: {duration}s")
    print(f"Config: {json.dumps(config, indent=2)}")
    print("-" * 50)

    # Simulate training with periodic output
    start_time = time.time()
    loss: float = 2.0  # Initialize loss in case loop doesn't execute
    for epoch in range(1, duration // 5 + 1):
        if (epoch - 1) * 5 >= duration:
            break

        # Simulate loss decreasing over time (with some noise)
        base_loss = 2.0 * (0.95**epoch)
        loss = base_loss + random.uniform(-0.1, 0.1)

        elapsed = int(time.time() - start_time)
        print(f"[Epoch {epoch:3d}] [{elapsed:3d}s] loss={loss:.4f} lr={lr}")

        # Random events
        if random.random() < 0.1:
            print(f"[Epoch {epoch:3d}] Validation: accuracy={random.uniform(0.7, 0.95):.3f}")

        # Fail midway if designated to fail
        if will_fail and epoch > duration // 10:
            print("\nERROR: Training failed due to simulated error!")
            print("Exception: Fake convergence issue detected")
            raise Exception("Fake convergence issue detected")

        time.sleep(5)

    # Success
    print("\nTraining completed successfully!")
    print(f"Final loss: {loss:.4f}")
    print(f"Total time: {int(time.time() - start_time)}s")
    return 0


def main(config: dict[str, object]):
    """Entry point that xmux will call"""
    # For compatibility with how xmux calls this
    return fake_train_model(config)


if __name__ == "__main__":
    # For testing standalone
    test_config = {"model": "test-model", "lr": 0.01, "duration": 30, "failure_rate": 0.1}
    exit(main({"entrypoint_config": test_config, "log_relpath": "test/standalone"}))
