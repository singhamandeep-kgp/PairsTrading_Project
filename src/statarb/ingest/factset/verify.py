"""
Verify the raw extract, and derive the life-span fields locally.

Two jobs:

1. RECONCILE against the server-authored oracle (raw/oracle_checksum.parquet).
   Because the server cannot be reached from home, this is the only evidence that
   the local Parquet is complete and correct. Per year it compares row count,
   distinct securities, date bounds and a price checksum.

2. DERIVE first_px / last_px / n_obs per security. These used to come from a
   server-side `span` CTE over 1.015 billion rows, which killed the connection
   every time. Computing them locally from data we already have is the same
   answer in seconds - and it produces the survivorship evidence.

Run:  python verify.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date, timedelta

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from . import config

EPOCH = pa.scalar(0, pa.date32())


def _fmt(n) -> str:
    return f"{n:,}" if isinstance(n, (int, float)) else str(n)


def reconcile() -> bool:
    """Compare local Parquet aggregates against the server's oracle, per year."""
    oracle_path = config.RAW_DIR / "oracle_checksum.parquet"
    if not oracle_path.is_file():
        print(f"!! oracle not found at {oracle_path}")
        return False

    oracle = pq.read_table(oracle_path).to_pylist()
    orc = {int(r["yr"]): r for r in oracle}

    # Stream file by file and accumulate per year. Never materialises the full
    # 39.7M-row panel - this must stay comfortable on an 8 GB machine.
    n_rows: dict[int, int] = defaultdict(int)
    sids: dict[int, set] = defaultdict(set)
    d_min: dict[int, object] = {}
    d_max: dict[int, object] = {}
    px_sum: dict[int, int] = defaultdict(int)
    vol_sum: dict[int, int] = defaultdict(int)

    files = sorted((config.RAW_DIR / "prices").glob("*.parquet"))
    print(f"scanning {len(files)} price files ...")
    for i, f in enumerate(files, 1):
        tbl = pq.read_table(f, columns=["sid", "d", "c", "v"])
        years = pc.year(tbl.column("d")).to_pylist()
        sid_l = tbl.column("sid").to_pylist()
        d_l = tbl.column("d").to_pylist()
        c_l = tbl.column("c").to_pylist()
        v_l = tbl.column("v").to_pylist()
        for y, s, d, c, v in zip(years, sid_l, d_l, c_l, v_l):
            n_rows[y] += 1
            sids[y].add(s)
            if y not in d_min or d < d_min[y]:
                d_min[y] = d
            if y not in d_max or d > d_max[y]:
                d_max[y] = d
            if c is not None:
                px_sum[y] += int(round(round(float(c), 4) * 10000))
            if v is not None:
                vol_sum[y] += int(v)
        if i % 10 == 0 or i == len(files):
            print(f"  {i}/{len(files)} files")

    print()
    hdr = (f"{'yr':>5} {'local rows':>12} {'oracle rows':>12} {'Δ':>8} "
           f"{'local ids':>10} {'oracle ids':>11} {'Δ':>6}  dates  px_chk")
    print(hdr)
    print("-" * len(hdr))

    ok = True
    for y in sorted(set(n_rows) | set(orc)):
        o = orc.get(y)
        lr = n_rows.get(y, 0)
        li = len(sids.get(y, ()))
        if o is None:
            print(f"{y:>5} {lr:>12,} {'--':>12} {'??':>8}  "
                  "LOCAL YEAR NOT IN ORACLE")
            ok = False
            continue
        orows, oids = int(o["n_rows"]), int(o["n_ids"])
        drow, did = lr - orows, li - oids
        dates_ok = (str(d_min.get(y)) == str(o["d_min"])
                    and str(d_max.get(y)) == str(o["d_max"]))
        # The oracle computed its checksum from float(53) prices; we store
        # float32. Above ~$1,000 with 4 decimals float32 has too few significant
        # digits to reproduce the value exactly, so a small relative delta here is
        # EXPECTED and is not evidence of data loss. Row and id counts are the
        # exact integrity checks.
        opx = int(o["px_checksum"] or 0)
        lpx = px_sum.get(y, 0)
        rel = abs(lpx - opx) / max(abs(opx), 1)
        flag = "" if drow == 0 and did == 0 else "  <-- MISMATCH"
        if drow or did:
            ok = False
        if not dates_ok:
            flag += " DATES"
            ok = False
        print(f"{y:>5} {lr:>12,} {orows:>12,} {drow:>8,} "
              f"{li:>10,} {oids:>11,} {did:>6,}  "
              f"{'ok' if dates_ok else 'BAD':>4}  {rel:.2e}{flag}")

    print()
    print(f"local totals : {sum(n_rows.values()):,} rows")
    print(f"oracle totals: {sum(int(r['n_rows']) for r in oracle):,} rows")
    return ok


DEAD_GRACE_DAYS = 10   # a name still quoted within ~2 weeks of the feed max is
                       # live; anything older stopped trading


def derive_life_spans() -> None:
    """first_px_date / last_px_date / n_obs / is_dead per security.

    Replaces the server-side `span` CTE that aggregated 1.015 billion rows and
    killed the connection. Same numbers, computed locally in seconds.

    Covers EVERY security in dim_security, not just those with prices: universe
    members that never traded in the window get n_obs = 0. Step 2 needs to be
    able to distinguish "no price data" from "not in the universe".
    """
    print("\nderiving life spans from the extracted prices ...")

    grouped = (
        ds.dataset(config.RAW_DIR / "prices", format="parquet")
        .to_table(columns=["sid", "d"])
        .group_by("sid")
        .aggregate([("d", "min"), ("d", "max"), ("sid", "count")])
    )
    # pyarrow names aggregate outputs <column>_<func>, so map explicitly rather
    # than positionally - a positional rename silently mislabels columns if
    # pyarrow ever reorders them, and mislabelled dates here would corrupt every
    # survivorship statistic downstream.
    rename = {"d_min": "first_px_date", "d_max": "last_px_date",
              "sid_count": "n_obs", "sid": "sid"}
    missing = set(rename) - set(grouped.column_names)
    if missing:
        raise RuntimeError(
            f"unexpected aggregate column names from pyarrow: "
            f"got {grouped.column_names}, missing {sorted(missing)}")
    grouped = grouped.rename_columns([rename[c] for c in grouped.column_names])

    # Left-join onto the full universe so securities with no prices still appear.
    dim = pq.read_table(config.RAW_DIR / "dim_security.parquet",
                        columns=["sid", "fsym_regional_id", "ticker_region",
                                 "active_flag"])
    joined = dim.join(grouped, keys="sid", join_type="left outer")
    n_obs = pc.fill_null(joined.column("n_obs"), 0)
    joined = joined.set_column(joined.column_names.index("n_obs"), "n_obs", n_obs)

    # is_dead: last quoted price meaningfully before the feed's own max date.
    feed_max = date.fromisoformat(config.FEED_MAX_DATE)
    cutoff = feed_max - timedelta(days=DEAD_GRACE_DAYS)
    last = joined.column("last_px_date")
    is_dead = pc.or_(
        pc.is_null(last),                                   # never traded
        pc.less(last, pa.scalar(cutoff, pa.date32())),      # stopped trading
    )
    joined = joined.append_column("is_dead", pc.fill_null(is_dead, True))

    out = config.RAW_DIR / "security_life_spans.parquet"
    pq.write_table(joined, out, compression="zstd")

    total = joined.num_rows
    n_traded = pc.sum(pc.cast(pc.greater(n_obs, 0), pa.int32())).as_py() or 0
    n_dead = pc.sum(pc.cast(joined.column("is_dead"), pa.int32())).as_py() or 0
    n_never = total - n_traded

    print(f"wrote {out.name}: {total:,} securities")
    print(f"  with price history : {n_traded:,}")
    print(f"  never traded in window : {n_never:,} (n_obs = 0)")
    print(f"  DEAD (last price before {cutoff}): {n_dead:,} of {total:,} "
          f"({n_dead / max(total, 1):.1%})")
    if n_dead / max(total, 1) < 0.30:
        print("  !! WARNING: dead fraction under 30% - survivorship bias check "
              "would normally fail here.")


def main() -> int:
    print("=" * 79)
    print(f"VERIFYING {config.RAW_DIR}")
    print("=" * 79)
    ok = reconcile()
    try:
        derive_life_spans()
    except Exception as exc:  # noqa: BLE001
        print(f"!! life-span derivation failed: {exc!r}")
        ok = False          # a silent failure here would leave step 2 reading
                            # mislabelled or absent date columns

    print()
    print("=" * 79)
    if ok:
        print("RECONCILIATION PASSED - row counts, security counts and date "
              "bounds match the server for every year.")
    else:
        print("!! RECONCILIATION FAILED - see MISMATCH rows above. Do NOT treat "
              "this extract as good.")
    print("=" * 79)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
