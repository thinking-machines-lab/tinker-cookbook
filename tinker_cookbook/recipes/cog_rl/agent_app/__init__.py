"""Standalone Cog coding-agent app (chat agent + HTTP endpoints + UI).

A self-contained agent application: it writes programs in Cog, runs them through the
reference interpreter, and self-corrects on interpreter errors. It is configured like any
OpenAI app (``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``) and imports nothing from
tinker-cookbook, so it stands in for an agent already running in production.

You post-train a model for this app by pointing it at a sampling proxy (see
``../training/``) via ``OPENAI_BASE_URL`` and triggering rollouts through its HTTP
endpoints — never by importing this loop into the trainer.
"""

from .agent import AgentResult, CogAgent
from .cog_lang import run_cog
from .program import extract_program
from .prompts import COG_SYSTEM_PROMPT

__all__ = ["CogAgent", "AgentResult", "extract_program", "COG_SYSTEM_PROMPT", "run_cog"]
