#!/usr/bin/env python
"""
Assert that source files are trackable and data directories are not.

WHY THIS IS A DEDICATED CHECK
-----------------------------
A gitignore pattern whose only slash is a trailing one matches at EVERY depth.
So a bare `curated/`, intended for the top-level data directory, also matched
`src/statarb/curated/` — and silently gitignored the entire stage-2 source
package. 27.7 KB of the most-tested code in the repository could not be
committed, and nothing surfaced an error; `git add` simply did nothing.

A bare `src/` had the same latent bug waiting for the whole source tree.

That failure mode is invisible in review: the .gitignore reads sensibly, and the
consequence only shows up as "why is this file not in the repo?" weeks later. So
it gets an executable assertion in both directions.
"""

from __future__ import annotations

import subprocess
import sys

# Paths that MUST be trackable. Some do not exist yet on purpose: they are the
# places a future file would land, and the check must fail before someone
# discovers the hard way that the directory was ignored all along.
MUST_BE_TRACKABLE = [
    "src/statarb/__init__.py",
    "src/statarb/paths.py",
    "src/statarb/curate/build.py",
    "src/statarb/curate/factors.py",
    "src/statarb/curate/schema.py",
    "src/statarb/data/api.py",
    "src/statarb/ingest/factset/queries.py",
    "src/statarb/research/spread.py",
    "src/statarb/backtest/engine.py",
    "tests/conftest.py",
    "tests/fixtures/synthetic.py",
    "tests/unit/test_factors.py",
    "notebooks/example.ipynb",       # notebooks are committed (outputs stripped)
    "tests/data/reference.csv",      # small reference CSVs are committed
    "docs/adr/0001-example.md",
    "tools/check_no_secrets.py",
]

# Paths that MUST be ignored: licensed vendor data and secrets.
MUST_BE_IGNORED = [
    ".env",
    "raw/prices/part-0.parquet",
    "curated/prices/part-0.parquet",
    "_manifest/manifest.jsonl",
    "discovery_output/Q1.csv",
    "sid_map.parquet",
    "data/anything.parquet",
]


def _is_ignored(path: str) -> bool:
    """git check-ignore exits 0 when the path IS ignored."""
    return subprocess.run(
        ["git", "check-ignore", "-q", path],
        capture_output=True,
    ).returncode == 0


def main() -> int:
    problems = []

    for p in MUST_BE_TRACKABLE:
        if _is_ignored(p):
            rule = subprocess.run(
                ["git", "check-ignore", "-v", p],
                capture_output=True, text=True,
            ).stdout.strip()
            problems.append(
                f"SOURCE IS GITIGNORED: {p}\n"
                f"      matched by: {rule}\n"
                f"      Anchor that pattern with a leading slash (/name/) so it "
                f"only matches at the repository root."
            )

    for p in MUST_BE_IGNORED:
        if not _is_ignored(p):
            problems.append(
                f"DATA OR SECRET IS NOT IGNORED: {p}\n"
                f"      Licensed vendor data and credentials must never be "
                f"committable. Add an anchored rule to .gitignore."
            )

    if problems:
        print("gitignore check failed:\n", file=sys.stderr)
        for prob in problems:
            print(f"  {prob}\n", file=sys.stderr)
        return 1

    print(
        f"gitignore ok: {len(MUST_BE_TRACKABLE)} source paths trackable, "
        f"{len(MUST_BE_IGNORED)} data/secret paths ignored"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
