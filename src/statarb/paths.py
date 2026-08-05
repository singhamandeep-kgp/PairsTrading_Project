"""
Filesystem locations — the ONLY module that knows where data lives.

Before this existed, the data root was computed independently in
`curate/schema.py` and `ingest/factset/config.py`, both at import time, both
defaulting to the repository root. That is how 1.4 GB of licensed vendor data
ended up inside a git working tree.

Two rules follow from that, and they are the whole point of this module:

1. **One definition.** Anything needing a path calls `locations()`. Nothing else
   reads `FACTSET_DEST_ROOT`.

2. **Resolved on call, never at import.** Import-time path resolution is what
   makes a module untestable: tests then have to monkeypatch the environment
   *before* the import, and the ordering games that follow eventually produce a
   test that passes for the wrong reason. It is also the exact defect that makes
   the legacy `settings.py` unimportable.

An explicit `root` argument always wins, which is what lets tests point at a
temporary directory without touching the environment at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

ENV_VAR = "FACTSET_DEST_ROOT"

# Outside the repository, deliberately. See the module docstring.
DEFAULT_ROOT = Path.home() / "statarb-data"


@dataclass(frozen=True)
class Locations:
    """Every path the project uses, derived from one root."""

    root: Path

    # -- raw landing zone (stage 1 output; immutable) -----------------------
    @cached_property
    def raw(self) -> Path:
        return self.root / "raw"

    @cached_property
    def raw_prices(self) -> Path:
        return self.raw / "prices"

    @cached_property
    def manifest(self) -> Path:
        return self.root / "_manifest"

    @cached_property
    def discovery_output(self) -> Path:
        return self.root / "discovery_output"

    # -- curated Delta tables (stage 2 output) -----------------------------
    @cached_property
    def curated(self) -> Path:
        return self.root / "curated"

    @cached_property
    def prices(self) -> Path:
        return self.curated / "prices"

    @cached_property
    def security(self) -> Path:
        return self.curated / "dim_security"

    @cached_property
    def corp_actions(self) -> Path:
        return self.curated / "corporate_actions"

    @cached_property
    def calendar(self) -> Path:
        return self.curated / "calendar"

    def raw_table(self, name: str) -> Path:
        """Path to a named raw Parquet file, e.g. `raw_table("splits")`."""
        return self.raw / f"{name}.parquet"

    def exists(self) -> bool:
        """Whether a curated warehouse is present at this root."""
        return self.prices.exists()

    def describe(self) -> str:
        return (
            f"root={self.root}  "
            f"raw={'ok' if self.raw.exists() else 'missing'}  "
            f"curated={'ok' if self.curated.exists() else 'missing'}"
        )


def locations(root: str | Path | None = None) -> Locations:
    """Resolve data locations.

    Precedence: explicit argument, then $FACTSET_DEST_ROOT, then ~/statarb-data.

    The explicit argument exists for tests and for anyone running against a
    second warehouse; it means no test ever has to mutate the environment.
    """
    if root is not None:
        resolved = Path(root)
    else:
        resolved = Path(os.environ.get(ENV_VAR) or DEFAULT_ROOT)
    return Locations(root=resolved.expanduser())
