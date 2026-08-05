# ADR 0001 — Six panel loaders: renamed, not consolidated

**Status:** accepted · **Date:** 2026-08-05

## Context

"Load a sector price panel" is implemented **six** times in this repository:

| # | Location | Behaviour |
|---|---|---|
| 1 | `research/cointegration.py::load_log_prices_by_glob` | globs every `GICS_*.pkl`, returns the first file containing the permno |
| 2 | `research/spread.py::ModelSpread.load_log_prices_for_sector` | reads **one explicit** sector file |
| 3 | `backtest/adapters.py::load_log_prices_by_glob` | pass-through to #1 |
| 4 | `backtest/adapters.py` (`load_sector_prices`) | reads a pickle, no log transform |
| 5 | `backtest/portfolio.py` (inline `pd.read_pickle`) | inline, no function |
| 6 | `backtest/calendar_legacy.py` (inline) | inline, formerly **at import time** |

Before this decision, #1 and #2 were **both named `_load_log_price_series`**.

That is the part that mattered. They have *different semantics*: for a security
that appears in two sector files — which happens whenever a company is
reclassified — #1 returns whichever file sorts first and #2 returns the caller's
sector. They return **different price series, with no error and no warning.** A
backtest built on the wrong one is silently wrong, and no amount of reading a
single file reveals it.

All six are superseded by `DataAPI.get_panel()`, which dominates them on every
axis: no pickle (an arbitrary-code-execution format), no `os.getcwd()`
dependence, point-in-time correct, predicate-pushdown instead of whole-file
reads, and one definition instead of six.

## Decision

**Rename to disambiguate. Do not consolidate yet.**

- `_load_log_price_series` (glob) → **`load_log_prices_by_glob`**
- `_load_log_price_series` (sector) → **`load_log_prices_for_sector`**
- `adapters.optics_cluster` → **`optics_cluster_adapter`** (it shadowed
  `research.clustering.optics_cluster`, which is why it had to import the real
  one under an alias)

The duplication stays. The *ambiguity* goes.

## Consequences

**Good.** The silent-wrong-series trap is eliminated: the two loaders can no
longer be confused, and a reader can tell from the call site which semantics are
in play. Nothing was deleted, so the legacy strategy code — which is still the
only implementation of the PCA → OPTICS → cointegration pipeline — keeps working
untouched.

**Bad, and accepted.** Six implementations still exist. A reader will notice.
That is why this document exists: **documented, deliberate duplication reads as a
decision; undocumented duplication reads as an oversight.**

**Migration path.** When the strategy is ported onto `DataAPI` (stage 3), all six
collapse into `DataAPI.get_panel()` plus a three-line
`research/panel.py::log_price_panel()` so that "load a log price panel" exists
exactly once. The renames done here make that a mechanical change rather than an
archaeology exercise.

## Rejected alternatives

**Consolidate now.** Cleanest tree, and it was the original recommendation.
Rejected because the legacy modules are the only working implementations of the
strategy math, and rewriting their data access while also restructuring the
repository would mean two behaviour-changing edits landing in one unreviewable
diff. This cleanup was deliberately scoped to *structural* changes so that if
something breaks, the cause is unambiguous.

**Delete the legacy stack.** Genuinely tempting — it is unreachable from any
entry point and partly broken on pandas ≥ 2.0. Rejected because the owner needs
the strategy implementations for stage 3, and `git show` is a worse interface for
code you are about to actively use than a file in the tree.

**Leave the names alone.** Rejected. Two same-named functions with different
semantics is not untidiness, it is a live correctness hazard, and it is the one
item here that could invalidate a research result rather than merely annoy a
reader.

## Related

- Superseding implementation: `src/statarb/data/api.py` (`get_panel`)
- The `@lru_cache` on `ModelSpread.load_log_prices_for_sector` is a separate known
  defect (it caches on `self`, so every instance and its DataFrames leak for the
  process lifetime). Recorded in `ROADMAP.md`, not fixed here — it is a behaviour
  change.
