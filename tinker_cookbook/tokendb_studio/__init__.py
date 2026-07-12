"""Token DB studio: the analysis app over the token DB persistence layer.

The web viewer/chat server (:mod:`~tinker_cookbook.tokendb_studio.serve`),
the chat agent and its stores (:mod:`~tinker_cookbook.tokendb_studio.agent`),
and the provider-agnostic LLM client
(:mod:`~tinker_cookbook.tokendb_studio.llm`). Depends one-way on
:mod:`tinker_cookbook.tokendb` (through its public API and the
``TokenStoreBackend`` protocol); the persistence layer never imports this
package.

Requires the ``tokendb-studio`` extra::

    pip install 'tinker-cookbook[tokendb-studio]'
    python -m tinker_cookbook.tokendb_studio.serve log_path=~/runs/my-run
"""

from typing import TYPE_CHECKING, Any

from tinker_cookbook.tokendb_studio.agent import (
    ChatStore,
    RegistryToolbox,
    RunToolbox,
    TurnManager,
    VisualStore,
    run_chat_turn,
)
from tinker_cookbook.tokendb_studio.agent_prompt import build_system_prompt, format_schema_card
from tinker_cookbook.tokendb_studio.llm import LLMClient, LLMConfig

if TYPE_CHECKING:
    from tinker_cookbook.tokendb_studio.serve import Config, build_app, run

# The serve exports are lazy so `python -m tinker_cookbook.tokendb_studio.serve`
# does not trigger runpy's "found in sys.modules" warning.
_SERVE_EXPORTS = {"Config", "build_app", "run"}


def __getattr__(name: str) -> Any:
    if name in _SERVE_EXPORTS:
        from tinker_cookbook.tokendb_studio import serve

        return getattr(serve, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ChatStore",
    "LLMClient",
    "LLMConfig",
    "RegistryToolbox",
    "RunToolbox",
    "TurnManager",
    "VisualStore",
    "Config",
    "build_app",
    "build_system_prompt",
    "format_schema_card",
    "run",
    "run_chat_turn",
]
