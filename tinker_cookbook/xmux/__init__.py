"""xmux - TMUX-based experiment launcher for ML sweeps"""

from .core import JobSpec, SwarmConfig, launch_swarm

__version__ = "0.1.0"
__all__ = ["JobSpec", "SwarmConfig", "launch_swarm"]
