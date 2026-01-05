"""
Code execution backends for sandboxed code evaluation.

The sandbox/ directory provides thin wrappers around different sandbox backends:
- SandboxFusionClient: HTTP-based sandbox using SandboxFusion Docker container
- ModalSandbox: Cloud sandbox using Modal's infrastructure
"""

from typing import Literal

from tinker_cookbook.sandbox.sandboxfusion import SandboxFusionClient

SandboxBackend = Literal["sandboxfusion", "modal"]

__all__ = [
    "SandboxBackend",
    "SandboxFusionClient",
]


# ModalSandbox is lazily imported to avoid requiring modal as a dependency
def get_modal_sandbox(**kwargs):
    """Factory to avoid importing modal at module load."""
    from tinker_cookbook.sandbox.modal_sandbox import ModalSandbox

    return ModalSandbox(**kwargs)
