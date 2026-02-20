"""
Code execution backends for sandboxed code evaluation.

The sandbox/ directory provides thin wrappers around different sandbox backends:
- SandboxFusionClient: HTTP-based sandbox using SandboxFusion Docker container
- ModalSandbox: Cloud sandbox using Modal's infrastructure
"""

from enum import StrEnum

from tinker_cookbook.sandbox.sandbox_protocol import Sandbox
from tinker_cookbook.sandbox.sandboxfusion import SandboxFusionClient


class SandboxBackend(StrEnum):
    SANDBOXFUSION = "sandboxfusion"
    MODAL = "modal"


__all__ = [
    "Sandbox",
    "SandboxBackend",
    "SandboxFusionClient",
]
