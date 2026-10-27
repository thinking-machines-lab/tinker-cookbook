"""
Code execution backends for sandboxed code evaluation.

The sandbox/ directory provides thin wrappers around different sandbox backends:
- SandboxFusionClient: HTTP-based sandbox using SandboxFusion Docker container
- ModalSandbox: Cloud sandbox using Modal's infrastructure
- SailSandbox: Cloud sandbox using Sail's Sailboxes
"""

from enum import StrEnum
from typing import Literal

from tinker_cookbook.sandbox.sandbox_interface import (
    SandboxInterface,
    SandboxResult,
    SandboxTerminatedError,
)
from tinker_cookbook.sandbox.sandboxfusion import SandboxFusionClient


class SandboxBackend(StrEnum):
    SANDBOXFUSION = "sandboxfusion"
    MODAL = "modal"
    SAIL = "sail"


CloudSandboxBackend = Literal["modal", "sail"]
"""The backends that implement :class:`SandboxInterface`. SandboxFusion only runs code snippets."""


__all__ = [
    "CloudSandboxBackend",
    "SandboxBackend",
    "SandboxFusionClient",
    "SandboxInterface",
    "SandboxResult",
    "SandboxTerminatedError",
]
