# Roadmap and Known Limitations

**Last reviewed:** 2026-08-05

This is the **single** list of open items. It replaces three separate lists that
had drifted into contradicting each other — one still said "the Delta database
does not exist yet" while 895 MB of Delta tables and a working `DataAPI` sat in
the tree. The design documents in `docs/` now record *what was decided and why*;
anything still open lives here.

---

## Now

| Item | Why it matters |
|---|---|
| **Stage 3: the backtester** | The data platform is done and the strategy math is present, but nothing yet runs end to end and produces an equity curve. This is the only item that turns the repo from well-engineered plumbing into a quant result. |
| **`--full-refresh` flag on ingest** | Re-downloading currently needs manual manifest deletion. Each of the 41 shards covers the full date range, so a re-run skips everything as already-complete — including new data. Since a full extraction is only 9.3 minutes, full refresh is the *correct* update strategy; it just needs a flag. |
| **Rotate the FactSet password** | It sat in plaintext in a `.py` docstring. Never committed (verified with `git log --all -S`), but exposed on disk. Standing rule: any credential that has appeared in a file other than `.env` gets rotated. |

## Next

| Item | Why it matters |
|---|---|
| **CRSP delisting-return join** | FactSet ships no delisting return (see Known Limitations). CRSP does. This is the highest-value data work remaining, because its absence is what silently inflates backtest returns. |
| **Consolidate the six panel loaders** | `DataAPI.get_panel` supersedes all six. Deferred deliberately — see [ADR 0001](docs/adr/0001-duplicate-panel-loaders.md). Should happen as part of stage 3, when the strategy moves onto the DataAPI anyway. |
| **Port the strategy onto `DataAPI`** | The legacy modules still read pickles from `os.getcwd()`. They work, but they cannot be tested and they bypass every point-in-time guarantee the API provides. |
| **`fp_basic_prices` OHLC verification against a second source** | Open/high/low are extracted but only close has been reconciled against the server oracle. |
| **Fix `@lru_cache` on `ModelSpread.load_log_prices_for_sector`** | Caches on `self`, so every instance and its DataFrames leak for the process lifetime. A behaviour change, so out of scope for the structural cleanup. |
| **Coverage on `research/` and `backtest/`** | Currently ~0%. The pure functions (half-life, hedge ratio, Engle-Granger) have analytically known answers and are the cheapest real coverage available. |

## Known limitations

These are properties of the data or the design, not bugs. Stated plainly because
a reader who discovers them independently will trust nothing else in the repo.

**No delisting returns.** FactSet's liquidation dividend types total 374 records
globally across 40+ years, against many thousands of US delistings. There is no
CRSP-style `dlret`. A position in a company that goes to zero simply stops having
P&L, which **systematically inflates backtest returns** — and the inflation looks
exactly like alpha. CRSP remains necessary for this one field.

**Sector classification is a snapshot, not point-in-time.** FactSet's `rbics_v1`
tables are present but **empty** in this feed, and both remaining sector tables
carry no validity dates. Applying today's sector to 2005 is look-ahead bias, and
it matters here because sector would define the clustering universe. Mitigation:
cluster on statistical factor loadings instead of vendor sector labels, which
removes the dependency entirely rather than working around it.

**Volume is accurate to ±1,000 shares.** FactSet ships `p_volume` as a fractional
value in thousands (AAPL 2024-01-02 = `82488.672`). The extraction cast it to
`BIGINT` server-side to save wire bytes, truncating the fraction. Relative error
~8e-6 — irrelevant for liquidity screens, recorded so nobody later assumes
exactness. Fixable only by re-extracting, which is not worth 9 minutes of office
time for 8e-6.

**Pre-2001 mean reversion is overstated.** US markets quoted in sixteenths until
decimalisation in 2001, so bid-ask bounce is structurally larger before then —
and bid-ask bounce is precisely the signal a pairs strategy trades. Expect
flattering 1995–2000 half-lives. Treat that window as a separate regime rather
than pooling it.

**27,169 `SHARE` securities have a NULL `p_sec_type_code`.** They are absent from
`fp_sec_coverage`. Believed to have no price coverage either, so the INNER JOIN
drops them harmlessly, but this has not been confirmed.

**`ingest/factset/extractor.py` is largely untested** and excluded from coverage.
It needs a live ODBC connection to a shared production server. Excluded openly
rather than padding the number; the pure parts (`_split_batches`, `sha256_file`,
`Manifest` round-trip, `price_shards`) are testable and should be covered.

**Extraction requires the office network.** The FactSet server is reachable only
from there, so the raw lake cannot be rebuilt from home. This is why extraction
is resumable, checksummed, and verified against a server-authored oracle *before
leaving the office*.

## Won't do

| Item | Reason |
|---|---|
| **`fgp_v1.fgp_global_prices` as the price source** | Richer (VWAP, turnover, trade count, period returns) but querying AAPL's `-R` id returns **zero rows** — it keys differently and its US coverage is unconfirmed. Not a drop-in. Revisit only if VWAP becomes necessary. |
| **Intraday data** | The strategy rebalances monthly and trades a daily spread. Intraday would multiply storage by ~1000x for no signal. |
| **Spark / Dask / Ray** | At ~2 GB, one laptop's cores beat any distributed scheduler. The honest threshold for distributed compute is "does not fit on one machine's disk"; this is ~100x away. |
| **Committing the data to git** | Licensed and non-redistributable, and git stores every binary revision forever. Code and manifests only. |
| **`asyncio` anywhere** | Two I/O patterns exist: local NVMe (already saturated by the page cache and Polars' threaded reader) and a handful of long SQL queries (threads are simpler, and the drivers are synchronous). No win condition. |

---

## Recently completed

- Stage 1: FactSet → raw Parquet. 39,677,253 rows, 16,367 securities, 1995–2026,
  reconciled row-for-row against a server-authored oracle for all 32 years.
- Stage 2: raw → curated Delta Lake with adjust-on-read factors, plus `DataAPI`.
- The split convention pinned against six real corporate actions.
- Repository restructure: `src/` layout, one `statarb` package, real dependency
  declarations, CI, and the secret/gitignore gates.
