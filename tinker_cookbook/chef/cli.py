"""CLI entry point for Tinker Chef.

Usage::

    tinker-chef serve /path/to/log_dir
    tinker-chef serve /path/to/log_dir --port 8150 --host 0.0.0.0
"""

import argparse
import logging
import os
import sys

# The chef, in all their glory
BANNER = r"""
      ___________
     /           \
    |  TINKER     |
    |    CHEF     |
     \___________/
         | |
     .---' '---.
    /  ^     ^  \
   |  (o)   (o)  |
   |      <      |
   |    \___/    |
    \           /
     '---___---'
      /  | |  \
     /  /   \  \
    |__|     |__|
"""

BANNER_MINI = r"""
  [ TINKER CHEF ]
     (o   o)
      \__/
"""


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tinker-chef",
        description="Tinker Chef — training visualization dashboard for tinker-cookbook",
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the ASCII art banner",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the dashboard server")
    serve_parser.add_argument(
        "log_dirs",
        nargs="+",
        help="One or more paths to training run directories or parent directories",
    )
    serve_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8150,
        help="Port to bind to (default: 8150)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    serve_parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "serve":
        _run_serve(args)


def _show_banner(no_banner: bool) -> None:
    """Print the ASCII art banner unless suppressed."""
    if no_banner or os.environ.get("TINKER_CHEF_NO_BANNER"):
        return

    # Use mini banner if terminal is narrow
    try:
        cols = os.get_terminal_size().columns
    except OSError:
        cols = 80

    banner = BANNER_MINI if cols < 50 else BANNER
    print(banner)


def _run_serve(args: argparse.Namespace) -> None:
    """Start the Tinker Chef server."""
    try:
        import uvicorn
    except ImportError:
        print(
            "Error: uvicorn is not installed. Install chef dependencies with:\n"
            "  uv pip install 'tinker_cookbook[chef]'",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from tinker_cookbook.chef.app import create_app

    log_dirs = args.log_dirs
    app = create_app(log_dirs[0] if len(log_dirs) == 1 else log_dirs)

    _show_banner(args.no_banner)

    for d in log_dirs:
        print(f"  Serving: {d}")
    print(f"  Dashboard:  http://{args.host}:{args.port}")
    print(f"  API docs:   http://{args.host}:{args.port}/docs")
    print()

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
