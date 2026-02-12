"""Type definitions for Tau2Env components."""

from dataclasses import dataclass, field
from enum import Enum


class AskSonnetMode(Enum):
    """Mode for handling ask_sonnet interactions."""
    DIRECT_INJECTION = "direct"      # Sonnet's response is used directly as tau2 action
    CONDITIONING = "conditioning"    # Sonnet's response is advice; policy decides what to do


class ExplorationMode(Enum):
    """Mode for ask_sonnet exploration during RL training."""
    EPSILON_GREEDY = "epsilon"       # Random forcing with probability epsilon
    RAO_BLACKWELL = "rao_blackwell"  # Force on assistant turn == rollout_idx (structured exploration)


class ActionType(Enum):
    """Type of action parsed from model output."""
    TOOL_CALL = "tool_call"          # Regular tool call
    ASK_SONNET = "ask_sonnet"        # Special ask_sonnet tool call
    TEXT = "text"                    # Plain text response


class ObservationType(Enum):
    """Type of observation from tau2 gym."""
    USER_MESSAGE = "user"            # User message (starts with "user: ")
    TOOL_RESULT = "tool"             # Tool result (starts with "tool: ")
    OTHER = "other"                  # Unknown format


@dataclass
class ParsedAction:
    """Structured representation of a parsed model action."""
    raw_content: str                          # Original content string
    action_type: ActionType                   # Type of action
    tool_name: str | None = None              # Tool name if tool call
    tool_args: dict | None = None             # Tool arguments if tool call
    parse_success: bool = True                # Whether parsing succeeded
    original_message: dict = field(default_factory=dict)  # Original message dict from renderer


@dataclass
class Tau2StepResult:
    """Structured result from tau2 gym step."""
    obs_type: ObservationType                 # Type of observation
    obs_content: str                          # Observation content (without prefix)
    raw_obs: str                              # Raw observation string
    reward: float                             # Reward from step
    terminated: bool                          # Episode terminated
    truncated: bool                           # Episode truncated
    info: dict = field(default_factory=dict)  # Additional info


@dataclass
class ExternalLLMConfig:
    """Configuration for external LLM (e.g., Sonnet)."""
    model: str                                # Model name (e.g., "claude-sonnet-4-5-20250929")
    temperature: float = 0.0                  # Sampling temperature
    max_tokens: int = 1024                    # Max tokens to generate
