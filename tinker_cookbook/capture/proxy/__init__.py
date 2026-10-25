"""Anthropic/OpenAI-compatible chat proxy for capturing black-box harnesses.

See ``tinker_cookbook/capture/proxy/README.md`` for the design overview.
"""

from tinker_cookbook.capture.proxy.address import ADDRESS_KEY_MAP, parse_address
from tinker_cookbook.capture.proxy.app import ProxyDeps, make_app

__all__ = ["ADDRESS_KEY_MAP", "ProxyDeps", "make_app", "parse_address"]
