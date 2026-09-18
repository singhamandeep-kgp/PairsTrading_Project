"""
`statarb` — the single command-line entry point.

    statarb --help
    statarb info                     # where is the data, what is in it
    statarb discover [Q1 Q2 ...]     # probe the vendor schema (office only)
    statarb ingest --smoke 1         # stage 1: vendor -> raw Parquet (office only)
    statarb ingest --full-refresh    # re-download everything, ignoring the manifest
    statarb verify                   # reconcile raw against the server oracle
    statarb curate                   # stage 2: raw -> curated Delta tables
    statarb demo                     # prove the DataAPI works end to end

WHY A SINGLE ENTRY POINT
------------------------
Every stage used to be a bare script run from a specific working directory. That
is not a stylistic problem: it is how `queries.py`'s bare `import config` survived
undetected, because the bug is invisible when the module's own directory happens
to be on sys.path. A console script forces the package to be importable properly.

Subcommands that need vendor access are marked "office only" and import their
heavy dependencies (arrow-odbc, pyodbc) LAZILY, so `statarb --help`,
`statarb info` and `statarb demo` all work on a machine with no ODBC driver.
"""

from __future__ import annotations

import argparse
import sys

from ..logging import configure_logging


def _cmd_info(args: argparse.Namespace) -> int:
    """Report where the data is and what is in it."""
    from ..paths import locations

    loc = locations(args.root)
    print(f"data root : {loc.root}")
    print(f"  raw     : {'present' if loc.raw.exists() else 'MISSING'}")
    print(f"  curated : {'present' if loc.curated.exists() else 'MISSING'}")

    if not loc.exists():
        print("\nNo curated warehouse found. Run `statarb curate` (needs raw/), or")
        print("point FACTSET_DEST_ROOT at an existing warehouse.")
        return 1

    from ..data.api import DataAPI

    print()
    print(DataAPI(root=args.root).summary())
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from . import demo

    return demo.main()


def _cmd_curate(args: argparse.Namespace) -> int:
    from ..curate import build

    return build.main()


def _cmd_verify(args: argparse.Namespace) -> int:
    from ..ingest.factset import verify

    return verify.main()


def _cmd_discover(args: argparse.Namespace) -> int:
    from ..ingest.factset import run_discovery

    return run_discovery.main(args.queries)


def _cmd_ingest(args: argparse.Namespace) -> int:
    from ..ingest.factset import extract_raw_factset

    argv: list[str] = []
    if args.smoke is not None:
        argv += ["--smoke", str(args.smoke)]
    elif args.execute:
        argv.append("--execute")
    if args.full_refresh:
        argv.append("--full-refresh")
    return extract_raw_factset.main(argv)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="statarb",
        description="US equity statistical-arbitrage research platform.",
    )
    p.add_argument("--root", default=None, metavar="DIR",
                   help="data root (default: $FACTSET_DEST_ROOT, else ~/statarb-data)")
    p.add_argument("--log-level", default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("info", help="show data locations and warehouse contents")
    s.set_defaults(func=_cmd_info)

    s = sub.add_parser("demo", help="prove the DataAPI works end to end")
    s.set_defaults(func=_cmd_demo)

    s = sub.add_parser("curate", help="stage 2: raw Parquet -> curated Delta tables")
    s.set_defaults(func=_cmd_curate)

    s = sub.add_parser("verify", help="reconcile raw extract against the server oracle")
    s.set_defaults(func=_cmd_verify)

    s = sub.add_parser("discover", help="probe the vendor schema (office network only)")
    s.add_argument("queries", nargs="*",
                   help="query name filters, e.g. Q1 Q2 (default: all)")
    s.set_defaults(func=_cmd_discover)

    s = sub.add_parser("ingest", help="stage 1: vendor -> raw Parquet (office network only)")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--execute", action="store_true", help="run the full extraction")
    g.add_argument("--smoke", type=int, metavar="N",
                   help="run only N price shards; do this before a full run")
    s.add_argument("--full-refresh", action="store_true",
                   help="re-extract every artifact, ignoring manifest completion "
                        "state. Each shard spans the full date range, so a plain "
                        "re-run skips new data; refresh is the update strategy. "
                        "The manifest is kept as history. Implies a real run; "
                        "combinable with --smoke to prove the path on N shards")
    s.set_defaults(func=_cmd_ingest)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The application configures logging; libraries never do. See statarb.logging.
    configure_logging(args.log_level)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
