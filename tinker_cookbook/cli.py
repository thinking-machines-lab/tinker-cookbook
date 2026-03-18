"""CLI entry point for tinker-cookbook.

Provides the ``tinker-cookbook`` (and ``tcb`` shortcut) console commands.

Architecture
------------
This module is a thin dispatcher. It does NOT replace ``chz`` for recipe
configuration -- recipes already have rich ``chz``-based CLIs. Instead it
provides:

- ``tinker-cookbook run <recipe> [chz args...]``  -- discover and launch recipes
- ``tinker-cookbook weights download|merge|publish`` -- weight lifecycle
- ``tinker-cookbook chat [chz args...]``  -- interactive chat
- ``tinker-cookbook viz [chz args...]``   -- dataset visualization

For ``run``, ``chat``, and ``viz``, extra arguments are forwarded to the
underlying ``chz`` entrypoint, so users get the same CLI experience they
would with ``python -m``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Recipe discovery
# ---------------------------------------------------------------------------

_RECIPES_PACKAGE = "tinker_cookbook.recipes"


def _find_recipes() -> dict[str, str]:
    """Return a mapping of recipe name -> module path for runnable recipes.

    A recipe is "runnable" if it contains a ``train.py`` with a ``__main__``
    guard, or is a top-level ``.py`` file in the recipes package.
    """
    recipes: dict[str, str] = {}
    try:
        pkg = importlib.import_module(_RECIPES_PACKAGE)
    except ImportError:
        return recipes

    # Support both regular packages (__file__) and namespace packages (__path__)
    if pkg.__file__:
        pkg_path = Path(pkg.__file__).parent
    elif hasattr(pkg, "__path__") and pkg.__path__:
        pkg_path = Path(next(iter(pkg.__path__)))
    else:
        return recipes

    # Top-level .py files (sl_basic, rl_basic, sl_loop, rl_loop)
    for item in sorted(pkg_path.iterdir()):
        if item.suffix == ".py" and item.name != "__init__.py":
            name = item.stem
            recipes[name] = f"{_RECIPES_PACKAGE}.{name}"

    # Sub-directories with a train.py (handles namespace packages without __init__.py)
    for item in sorted(pkg_path.iterdir()):
        if item.is_dir() and not item.name.startswith(("_", ".")):
            train_path = item / "train.py"
            if train_path.exists():
                recipes[item.name] = f"{_RECIPES_PACKAGE}.{item.name}.train"

    return recipes


def _list_recipes() -> str:
    """Format a human-readable list of available recipes."""
    recipes = _find_recipes()
    if not recipes:
        return "  (no recipes found)"
    lines = []
    for name, module in sorted(recipes.items()):
        lines.append(f"  {name:<30s} ({module})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


def _cmd_run(args: list[str]) -> None:
    """Run a recipe by name, forwarding remaining args to its chz CLI."""
    if not args or args[0] in ("-h", "--help"):
        recipes_list = _list_recipes()
        print(
            f"Usage: tinker-cookbook run <recipe> [args...]\n"
            f"\n"
            f"Run a training recipe. Extra arguments are forwarded to the\n"
            f"recipe's chz-based CLI.\n"
            f"\n"
            f"Available recipes:\n"
            f"{recipes_list}\n"
            f"\n"
            f"Examples:\n"
            f"  tinker-cookbook run sl_basic\n"
            f"  tinker-cookbook run math_rl --model_name Qwen/Qwen3-8B\n"
            f"  tinker-cookbook run chat_sl --help"
        )
        return

    recipe_name = args[0]
    rest = args[1:]

    recipes = _find_recipes()
    if recipe_name not in recipes:
        print(f"Error: unknown recipe {recipe_name!r}\n", file=sys.stderr)
        print("Available recipes:", file=sys.stderr)
        print(_list_recipes(), file=sys.stderr)
        sys.exit(1)

    module_path = recipes[recipe_name]

    # Run as ``python -m <module>`` so the recipe's ``if __name__ == "__main__"``
    # block executes with chz parsing the remaining argv.
    import runpy

    sys.argv = [module_path] + rest
    runpy.run_module(module_path, run_name="__main__", alter_sys=True)


# ---------------------------------------------------------------------------
# Subcommand: weights
# ---------------------------------------------------------------------------


def _cmd_weights(args: list[str]) -> None:
    """Weight lifecycle commands: download, merge, publish."""
    parser = argparse.ArgumentParser(
        prog="tinker-cookbook weights",
        description="Download, merge, and publish model weights.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    # -- download --
    dl = sub.add_parser("download", help="Download checkpoint weights from Tinker storage.")
    dl.add_argument(
        "tinker_path",
        help='Tinker checkpoint path (e.g. "tinker://<run_id>/sampler_weights/final").',
    )
    dl.add_argument("output_dir", help="Local directory for the extracted checkpoint.")
    dl.add_argument("--base-url", default=None, help="Custom Tinker service URL.")

    # -- merge --
    mg = sub.add_parser("merge", help="Merge a LoRA adapter into a HuggingFace model.")
    mg.add_argument("--base-model", required=True, help="HuggingFace model name or local path.")
    mg.add_argument("--adapter-path", required=True, help="Path to the Tinker adapter directory.")
    mg.add_argument("--output-path", required=True, help="Directory for the merged model.")
    mg.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for loading the base model (default: bfloat16).",
    )
    mg.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=None,
        help="Trust remote code when loading HF models.",
    )

    # -- publish --
    pub = sub.add_parser("publish", help="Publish a model directory to HuggingFace Hub.")
    pub.add_argument("model_path", help="Local path to the model directory.")
    pub.add_argument("repo_id", help='HuggingFace repo ID (e.g. "user/my-model").')
    pub.add_argument(
        "--public", action="store_true", help="Make the repository public (default: private)."
    )
    pub.add_argument("--token", default=None, help="HuggingFace API token.")

    parsed = parser.parse_args(args)

    if parsed.action == "download":
        from tinker_cookbook.weights import download

        result = download(
            tinker_path=parsed.tinker_path,
            output_dir=parsed.output_dir,
            base_url=parsed.base_url,
        )
        print(f"Downloaded to: {result}")

    elif parsed.action == "merge":
        import logging

        from tinker_cookbook.weights import build_hf_model

        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        build_hf_model(
            base_model=parsed.base_model,
            adapter_path=parsed.adapter_path,
            output_path=parsed.output_path,
            dtype=parsed.dtype,
            trust_remote_code=parsed.trust_remote_code,
        )

    elif parsed.action == "publish":
        from tinker_cookbook.weights import publish_to_hf_hub

        url = publish_to_hf_hub(
            model_path=parsed.model_path,
            repo_id=parsed.repo_id,
            private=not parsed.public,
            token=parsed.token,
        )
        print(f"Published to: {url}")


# ---------------------------------------------------------------------------
# Subcommand: chat
# ---------------------------------------------------------------------------


def _cmd_chat(args: list[str]) -> None:
    """Launch the interactive chat CLI, forwarding args to chz."""
    import runpy

    module = "tinker_cookbook.chat_app.tinker_chat_cli"
    sys.argv = [module] + args
    runpy.run_module(module, run_name="__main__", alter_sys=True)


# ---------------------------------------------------------------------------
# Subcommand: viz
# ---------------------------------------------------------------------------


def _cmd_viz(args: list[str]) -> None:
    """Launch the dataset visualizer, forwarding args to chz."""
    import runpy

    module = "tinker_cookbook.supervised.viz_sft_dataset"
    sys.argv = [module] + args
    runpy.run_module(module, run_name="__main__", alter_sys=True)


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

_COMMANDS: dict[str, tuple[str, object]] = {
    "run": ("Run a training recipe", _cmd_run),
    "weights": ("Download, merge, or publish model weights", _cmd_weights),
    "chat": ("Interactive chat with a Tinker model", _cmd_chat),
    "viz": ("Visualize a supervised dataset", _cmd_viz),
}


def main() -> None:
    """Entry point for ``tinker-cookbook`` / ``tcb`` console commands."""
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        _print_help()
        return

    if sys.argv[1] in ("--version", "-V"):
        from tinker_cookbook import __version__

        print(f"tinker-cookbook {__version__}")
        return

    cmd_name = sys.argv[1]
    if cmd_name not in _COMMANDS:
        print(f"Error: unknown command {cmd_name!r}\n", file=sys.stderr)
        _print_help(file=sys.stderr)
        sys.exit(1)

    _, handler = _COMMANDS[cmd_name]
    handler(sys.argv[2:])  # type: ignore[operator]


def _print_help(file: object = None) -> None:
    """Print top-level usage."""
    import io

    out: io.TextIOBase = file or sys.stdout  # type: ignore[assignment]
    lines = [
        "Usage: tinker-cookbook <command> [args...]",
        "",
        "Commands:",
    ]
    for name, (desc, _) in _COMMANDS.items():
        lines.append(f"  {name:<12s} {desc}")
    lines += [
        "",
        "Options:",
        "  -h, --help     Show this help message",
        "  -V, --version  Show version",
        "",
        "Run 'tinker-cookbook <command> --help' for more information on a command.",
    ]
    print("\n".join(lines), file=out)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
