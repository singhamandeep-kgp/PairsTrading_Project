# US Equity Statistical Arbitrage

A survivorship-bias-free US equity research platform and pairs-trading pipeline,
built on 39.7 million daily price observations from FactSet covering 1995–2026.

Every result is **point-in-time correct** and **reproducible to a pinned data
version** — the two properties that separate a backtest you can defend from one
that merely looks good.

---

## Status

| Stage | What it does | Status |
|---|---|---|
| **1 · Ingest** | FactSet SQL Server → raw Parquet. Sharded, resumable, checksummed, reconciled against a server-authored oracle. | **shipped** |
| **2 · Curate** | raw → Delta Lake. Corporate-action factors, adjust-on-read, ACID, time travel. | **shipped** |
| **3 · Research** | PCA → OPTICS clustering → cointegration → rolling walk-forward backtest. | in progress |

**Scale:** 16,367 securities · 39,677,253 price rows · 1995–2026 · 0.54 GB raw

**The number that matters:** of the 4,774 securities investable on 2005-06-01,
**2,927 (61%) no longer exist** — and they are all still in the universe. A
"currently listed" screen would silently delete every one of them.

---

## Quickstart

No FactSet access, credentials, or vendor relationship required. The test suite
builds a complete synthetic Delta warehouse in a temporary directory.

```bash
pip install -e ".[dev]"
pytest                    # 47 tests, synthetic data, ~3s
statarb --help
```

With a real warehouse:

```bash
statarb info              # what's in the database
statarb demo              # prove the read path works end to end
statarb curate            # rebuild curated Delta tables from raw
```

---

## Architecture

```
FactSet SQL Server  (AWS ap-south-1, shared production instance)
      │  arrow-odbc streaming · 41 sid-range shards · MAXDOP 1 · 3 workers
      │  universe-first + server-side CAST: 121 GB → 2.65 GB on the wire (45x)
      ▼
raw/  Parquet + manifest (sha256 + query_hash + universe_hash per partition)
      │  511 MB · immutable · never modified after extraction
      ▼
curated/  Delta Lake · prices | dim_security | corporate_actions | calendar
      │  float32 prices, int64 volume, cumulative adjustment factors
      │  ACID appends · time travel · 10-year retention
      ▼
DataAPI  ◄── the only thing strategy code imports. No file paths leak past here.
      │
      ▼
research/ (PCA, OPTICS, Engle-Granger)  →  backtest/ (rolling, walk-forward)
```

---

## Why it is built this way

Every one of these was measured, not assumed.

| Naive approach | Why it loses |
|---|---|
| `pd.read_sql` for the fact table | 14× slower **and ~10 GB peak RAM** on an 8 GB machine. The memory is the disqualifier, not the speed. |
| Pull the table, filter in pandas | 121 GB on the wire vs 2.65 GB. Hours vs minutes. |
| `WHERE active_flag = 1` | Silently removes **68.6%** of the universe, non-randomly — precisely the failures. The wrongness looks like alpha. |
| Store adjusted prices | Tomorrow's split invalidates 25 years of history, undetectably. Store raw + factor instead. |
| `float32` for volume | Silent corruption above 2²⁴ = 16,777,216 shares. Poisons every liquidity screen. |
| Partition by symbol | 16k directories of ~30 KB: footer overhead, dictionary reset per file, 16k file opens per cross-section. |
| `ORDER BY` in the extraction query | Tempdb sort spill on a **shared production server**. Sort locally instead. |
| N shards on an unindexed column | N full scans of a 1-billion-row table — slower than one pass, and it hammers a box other people depend on. |

The full 27-row version, with the arithmetic, is in
[docs/DATABASE_DESIGN.md](docs/DATABASE_DESIGN.md). Contested decisions have
[ADRs](docs/adr/).

---

## The point-in-time guarantee

`get_universe_as_of(date)` returns securities that were investable **on that
date**, including ones that later died. It never filters on `is_dead` or
`active_flag`, and liquidity screens use only data at or before the date.

That single method is where look-ahead bias would re-enter after all the work done
to keep it out — so it is tested adversarially rather than asserted:

> A security is planted whose average daily volume steps from **\$0.2m to \$50m on
> the day after the as-of date**. A correct trailing window must exclude it. A
> centred window, a forward-looking window, or an off-by-one on the `<= as_of`
> bound all admit it — and all three are ordinary mistakes that no reviewer would
> catch by reading the code.

Both failure modes were **mutation-verified**: injecting `.filter(~is_dead)` makes
the survivorship test fail, and widening the liquidity window by one day makes the
look-ahead test fail.

→ [`tests/integration/test_data_api_point_in_time.py`](tests/integration/test_data_api_point_in_time.py)

---

## Repository layout

```
src/statarb/
  paths.py          the single definition of where data lives
  logging.py        configure_logging(); called only from cli/
  ingest/
    factset/        stage 1: SQL Server → raw Parquet (arrow-odbc, manifest)
    crsp/           second vendor, for delisting-return cross-validation
  curate/
    factors.py      cumulative adjustment arithmetic (pure, 17 tests)
    schema.py       explicit pyarrow schemas + Delta retention
    build.py        raw → curated Delta tables
  data/api.py       DataAPI — the read interface
  research/         PCA, OPTICS, cointegration, spread, half-life
  backtest/         signals, P&L, rolling engine
  cli/              one `statarb` entry point with subcommands
tests/
  fixtures/         synthetic warehouse generator (12 planted securities)
  unit/  integration/  office/     (office = needs live FactSet, deselected)
tools/              secret and gitignore gates, used by pre-commit and CI
docs/               design record, vendor findings, ADRs
```

---

## Data access and licensing

The code is MIT. **The data is not.**

Market data is licensed from FactSet and CRSP, is not redistributed here, and
appears nowhere in this repository or its git history. Running against those feeds
requires your own vendor licence. `FACTSET_DEST_ROOT` points at a location
**outside** the repo; credentials live in `.env` (gitignored) — see
[.env.example](.env.example).

The synthetic fixture exists so the entire read path is testable with no vendor
relationship at all.

---

## Known limitations

Stated plainly, because a reader who finds them independently will trust nothing
else here.

- **No delisting returns.** FactSet ships none; CRSP is required for that field.
  Its absence systematically inflates backtest returns.
- **Sector data is a snapshot, not point-in-time** (`rbics_v1` is empty in this
  feed). Mitigated by clustering on statistical factors rather than vendor labels.
- **Volume is accurate to ±1,000 shares** (~8e-6 relative error).
- **Pre-2001 mean reversion is overstated** — sixteenths-era tick sizes inflate
  bid-ask bounce, which is precisely the signal being traded.
- **`ingest/factset/extractor.py` is largely untested**: it needs a live ODBC
  connection. Excluded from coverage openly rather than padding the number.

Full list, with what's planned: [ROADMAP.md](ROADMAP.md)

---

## What CI can and cannot verify

**Can**, with no vendor access: the entire read path end to end on synthetic data
(raw → curate → Delta → DataAPI → point-in-time invariants); that the package
imports with core dependencies only; schema round-trip contracts; that
`safe_connection_summary()` never leaks the password; that source files are
trackable and data paths are ignored; and cross-platform path handling — the Linux
runner is the only thing that will ever catch a Windows-only assumption.

**Cannot:** ODBC driver behaviour, real throughput, vendor data correctness, or
the oracle reconciliation. Those live in `tests/office/` behind a marker,
deselected by default. Distinguishing the two honestly is the point.
