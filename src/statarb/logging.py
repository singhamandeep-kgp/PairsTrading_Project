"""
Logging configuration — called from `cli/` only, never from library code.

WHY THIS MODULE EXISTS
----------------------
Five library modules used to call `logging.basicConfig(level=INFO, ...)` at import
time. That is wrong for a library, for three reasons:

* `basicConfig` mutates the ROOT logger, so importing one module silently changes
  logging for the entire host process — including code that has nothing to do with
  this package;
* it is first-import-wins, so which of the five configurations you got depended on
  import order;
* a library cannot know whether its caller wants INFO on stderr. That is the
  application's decision, and `cli/` is the application.

Library modules now do the correct thing: `log = logging.getLogger(__name__)`,
emit records, and configure nothing.
"""

from __future__ import annotations

import logging
import os
import sys

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"

_configured = False


def configure_logging(
    level: int | str | None = None,
    *,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
    force: bool = False,
) -> None:
    """Configure root logging for an application entry point.

    Idempotent: calling it twice is a no-op unless `force=True`. That matters
    because a CLI subcommand may be invoked from another, and double-configuring
    duplicates every log line.

    Level precedence: explicit argument, then $STATARB_LOG_LEVEL, then INFO.
    """
    global _configured
    if _configured and not force:
        return

    resolved = level if level is not None else os.environ.get("STATARB_LOG_LEVEL", "INFO")
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    if force:
        for existing in list(root.handlers):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved)

    # These emit one record per pair per rebalance at INFO, which buries anything
    # useful. Quietened here rather than in the libraries themselves, so the
    # choice is visible in one place and easy to override.
    logging.getLogger("statarb.research.cointegration").setLevel(logging.WARNING)
    logging.getLogger("statarb.research.spread").setLevel(logging.WARNING)

    _configured = True
