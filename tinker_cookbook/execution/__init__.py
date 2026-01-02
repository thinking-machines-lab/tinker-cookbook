"""
Code execution backends for sandboxed code evaluation.

This package provides thin wrappers around different sandbox backends:
- SandboxFusionClient: HTTP-based sandbox using SandboxFusion Docker container
- ModalSandbox: Cloud sandbox using Modal's infrastructure

Each wrapper exposes its native API. Dataset code (e.g., code_rl, bigcodebench)
is responsible for using these primitives appropriately for their test formats.
"""

from tinker_cookbook.execution.sandboxfusion import SandboxFusionClient

__all__ = [
    "SandboxFusionClient",
]


# ModalSandbox is lazily imported to avoid requiring modal as a dependency
def get_modal_sandbox(**kwargs):
    """Factory to avoid importing modal at module load."""
    from tinker_cookbook.execution.modal_sandbox import ModalSandbox

    return ModalSandbox(**kwargs)
