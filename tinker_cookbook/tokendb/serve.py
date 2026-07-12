"""Deprecated forwarder: the viewer moved to :mod:`tinker_cookbook.tokendb_studio.serve`."""

import warnings

from tinker_cookbook.tokendb_studio.serve import Config, build_app, run

__all__ = ["Config", "build_app", "run"]

warnings.warn(
    "tinker_cookbook.tokendb.serve has moved to tinker_cookbook.tokendb_studio.serve",
    DeprecationWarning,
    stacklevel=2,
)

if __name__ == "__main__":
    import chz

    chz.nested_entrypoint(run)
