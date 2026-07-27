"""
Code execution backends for sandboxed code evaluation.

The sandbox/ directory provides thin wrappers around different sandbox backends:
- SandboxFusionClient: HTTP-based sandbox using SandboxFusion Docker container
- ModalSandbox: Cloud sandbox using Modal's infrastructure
- HyperbrowserSandbox: Cloud sandbox using Hyperbrowser's infrastructure
"""

import os
from enum import StrEnum

from tinker_cookbook.sandbox.sandbox_interface import (
    SandboxInterface,
    SandboxResult,
    SandboxTerminatedError,
)
from tinker_cookbook.sandbox.sandboxfusion import SandboxFusionClient


class SandboxBackend(StrEnum):
    SANDBOXFUSION = "sandboxfusion"
    MODAL = "modal"
    HYPERBROWSER = "hyperbrowser"


SANDBOX_BACKEND_ENV_VAR = "TINKER_SANDBOX_BACKEND"
"""Environment variable that overrides the default cloud sandbox backend."""

CLOUD_BACKENDS = (SandboxBackend.MODAL, SandboxBackend.HYPERBROWSER)
"""Backends that provide persistent per-episode sandboxes (not just code grading)."""


def resolve_backend(
    backend: SandboxBackend | None = None,
    default: SandboxBackend = SandboxBackend.MODAL,
) -> SandboxBackend:
    """Resolve which sandbox backend to use.

    Precedence: the explicit argument, then ``TINKER_SANDBOX_BACKEND``, then
    ``default``. The env var lets existing scripts switch backends without code
    changes.
    """
    if backend is not None:
        return backend
    raw = os.getenv(SANDBOX_BACKEND_ENV_VAR)
    if not raw:
        return default
    try:
        return SandboxBackend(raw.strip().lower())
    except ValueError:
        valid = ", ".join(b.value for b in SandboxBackend)
        raise ValueError(
            f"Invalid {SANDBOX_BACKEND_ENV_VAR}={raw!r}. Expected one of: {valid}."
        ) from None


__all__ = [
    "CLOUD_BACKENDS",
    "SANDBOX_BACKEND_ENV_VAR",
    "SandboxBackend",
    "SandboxFusionClient",
    "SandboxInterface",
    "SandboxResult",
    "SandboxTerminatedError",
    "resolve_backend",
]
