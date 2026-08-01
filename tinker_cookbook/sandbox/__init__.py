"""
Code execution backends for sandboxed code evaluation.

The sandbox/ directory provides thin wrappers around different sandbox backends:
- SandboxFusionClient: HTTP-based sandbox using SandboxFusion Docker container
- ModalSandbox: Cloud sandbox using Modal's infrastructure
- FystashSandbox / FystashSandboxPool: Warm Firecracker rooms via Fystash
"""

from enum import StrEnum

from tinker_cookbook.sandbox.fystash_pool import FystashSandboxPool
from tinker_cookbook.sandbox.fystash_sandbox import FystashSandbox, fystash_sandbox_factory
from tinker_cookbook.sandbox.sandbox_interface import (
    SandboxInterface,
    SandboxResult,
    SandboxTerminatedError,
)
from tinker_cookbook.sandbox.sandboxfusion import SandboxFusionClient


class SandboxBackend(StrEnum):
    SANDBOXFUSION = "sandboxfusion"
    MODAL = "modal"
    FYSTASH = "fystash"


__all__ = [
    "FystashSandbox",
    "FystashSandboxPool",
    "SandboxBackend",
    "SandboxFusionClient",
    "SandboxInterface",
    "SandboxResult",
    "SandboxTerminatedError",
    "fystash_sandbox_factory",
]
