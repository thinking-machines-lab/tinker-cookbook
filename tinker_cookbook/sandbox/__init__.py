"""
Code execution backends for sandboxed code evaluation.

The sandbox/ directory provides thin wrappers around different sandbox backends:
- SandboxFusionClient: HTTP-based sandbox using SandboxFusion Docker container
- ModalSandbox: Cloud sandbox using Modal's infrastructure
"""

from enum import StrEnum

from tinker_cookbook.sandbox.sandboxfusion import SandboxFusionClient


class SandboxBackend(StrEnum):
    SANDBOXFUSION = "sandboxfusion"
    MODAL = "modal"


__all__ = [
    "SandboxBackend",
    "SandboxFusionClient",
]


# ModalSandbox and ModalSandboxPool are lazily imported to avoid requiring modal as a dependency
def get_modal_sandbox_pool(**kwargs):
    """Factory to create a ModalSandboxPool without importing modal at module load."""
    from tinker_cookbook.sandbox.modal_sandbox import ModalSandboxPool

    return ModalSandboxPool(**kwargs)
