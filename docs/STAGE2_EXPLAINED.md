# Stage 2 Explained — in plain language

**Written:** 2026-08-04 · **Last reviewed:** 2026-08-05
**Status:** stage 2 is now **complete**. §2's status table was accurate when
written and is now historical — see §6 for what changed.

> Written for a reader who is not a database engineer. If you want the design
> argument and the measured tradeoffs instead, read
> [DATABASE_DESIGN.md](DATABASE_DESIGN.md).

---

## 1. What stage 2 is for

Stage 1 downloaded the data. **Stage 2 makes it usable.**

That sounds like housekeeping. It isn't. Here is the actual problem, using a real
row from your own database:

| Date | AAPL closing price |
|---|---|
| 2020-08-28 | **$499.23** |
| 2020-08-31 | **$129.04** |

Did Apple fall 74% in one day? No — it split four-for-one that morning. Every
shareholder woke up owning four times as many shares, each worth a quarter as
much. Nothing was lost. But a computer reading those two numbers calculates a
**−74.15% return**, and that fake crash goes straight into your backtest.

Apple has done this five times. Across 16,367 securities there are **22,507
splits** in your data. Without correcting for them your return series is
nonsense — not slightly wrong, *nonsense*.

Stage 2's job is to turn raw prices into a **continuous, comparable series** so
that a return means what it says. Everything else in stage 2 exists to make that
correction trustworthy and repeatable.

---

## 2. Honest status — what is and isn't done

| Piece | Status | What it does |
|---|---|---|
| Toolchain gate | ✅ done | Confirmed `deltalake` and `polars` install on your Python |
| `curated/factors.py` | ✅ done | **The split/dividend correction arithmetic** |
| `tests/test_factors.py` | ✅ done | 17 tests, incl. 6 real-world splits |
| Real-data check | ✅ done | AAPL now reads +3.39%, not −74.15% |
| `curated/schema.py` | ❌ not started | Column types for the clean database |
| `curated/build.py` | ❌ not started | Converts all 41 raw files into the clean database |
| **The Delta database itself** | ❌ **does not exist yet** | |
| `api/data_api.py` | ❌ not started | What your backtester will call |
| Validation suite | ❌ not started | The 11 data-quality checks |

**So: the correction logic is built and proven. The database it will write into is
not built yet.** Section 5 below describes how versioning *will* work — it is
design, not something you can go and look at today.

---

## 3. What was actually built, in plain terms

### The raw ingredients

Stage 1 gave you two separate things:

**Prices** — 39.7 million rows of "on this day, this stock closed at this price."
Raw, exactly as traded. `$499.23` then `$129.04`.

**Events** — a list of corporate actions with a "factor" attached. For AAPL:

| Date | Factor | What happened |
|---|---|---|
| 2014-06-09 | 0.142857 | 7-for-1 split (1÷7) |
| 2020-08-31 | 0.25 | 4-for-1 split (1÷4) |

Nothing joined those two lists together. That was the gap.

### The correction

To compare a 2013 Apple price with today's, you must shrink the old price by
**every** split that happened since. Apple split 7-for-1 and then 4-for-1, so a
2013 price must be divided by 28 — or equivalently multiplied by 1/28 = 0.0357.

So the rule is:

> **The number you multiply an old price by is the product of the factors for
> every event that happened *after* that date.**

Worked through for Apple:

| Price date | Splits still to come | Multiplier | $500 becomes |
|---|---|---|---|
| 2013-01-02 | 7-for-1 *and* 4-for-1 | 1/7 × 1/4 = **0.0357** | $17.86 |
| 2015-01-02 | only the 4-for-1 | **0.25** | $125.00 |
| 2021-01-04 | none | **1.0** | $500.00 |

That multiplier is what `factors.py` computes, for every stock on every day.

### The one subtle thing worth understanding

On the split day *itself*, FactSet already reports the new, smaller price
($129.04). So that day needs **no** correction — its multiplier is 1.0.

This is the difference between "events *after* this date" and "events *on or
after* this date." Get it wrong and every split gets applied one day too early,
quietly doubling or quartering exactly one day's return per split. It looks
completely normal on a chart.

The output proves it lands correctly — the multiplier steps from `0.25` to `1.0`
precisely on 2020-08-31:

```
2020-08-28   raw 499.23   multiplier 0.25   ->  124.81
2020-08-31   raw 129.04   multiplier 1.00   ->  129.04
                                     return = +3.39%   ✅
```

### Two different corrections, on purpose

- **Price return** — corrects for splits only. "What did the share price do?"
- **Total return** — also corrects for dividends. "What did I actually earn?"

Both are computed. Apple's total-return multiplier for August 2020 is `0.242493`
rather than `0.25` — the extra ~3% is the ~24 dividends paid since. Same
percentage move on the split day, different cumulative wealth. Both are correct
answers to different questions.

### Why prices are corrected on *reading*, not on *storing*

Your decision, and it pays off directly in §5. We store the **raw price** and the
**multiplier** as two separate columns, and do the multiplication only when data
is read.

The alternative — storing corrected prices — seems more efficient. It isn't. When
Apple splits again next year, every historical corrected price becomes wrong.
Storing raw + multiplier means only the multiplier column changes and every
historical price is instantly right again.

**Stale-but-plausible is the most dangerous state data can be in**, because
nothing looks broken.

---

## 4. Adding three more months of data

### Does stage 1 have to run again?

**Yes — stage 1 is the only thing that talks to FactSet.** New data can only come
from there.

### Does it re-download prices you already have?

**As the code stands today: it would download nothing at all.** That is a real
gap, and here is why.

Each of the 41 download jobs is defined by a *range of stocks*, and each covers
the **entire date range 1995→2050**. The manifest records "job 7 finished, here
is its checksum." On re-running, all 41 jobs are recorded as finished with
matching checksums, so all 41 are skipped — including the three new months.

There is currently **no date-based incremental path**. I should have flagged this
before now.

### The recommended fix is to not build one

Here is the useful engineering judgement: **the full download takes 9.3 minutes.**

Incremental updating is genuinely fiddly — you need watermarks, you must handle
data the vendor revises after the fact, and you must decide what to do when a
corporate action rewrites old history. Every one of those is a place for a subtle
bug that silently corrupts your database.

For a nine-minute job, that complexity buys nothing. So:

> **To add three months, re-download everything. Delete the manifest, run
> `--execute`, and wait nine minutes.**

This is not laziness, it is the correct trade. A full rebuild is **self-healing**:
it picks up new prices, revised old prices, and new corporate actions in one
pass, with no watermark logic to get wrong. Incremental complexity is only
justified when the full rebuild is expensive — the usual threshold is hours, not
minutes.

**What's missing:** a `--full-refresh` flag that clears the manifest for you, so
you aren't deleting files by hand. Small, and it belongs in the next batch of
work.

### So the update procedure will be

```
1.  (at the office)  extract_raw_factset.py --full-refresh --execute   ~9 min
2.                   verify.py                                          ~2 min
3.  (anywhere)       build.py                                           ~5 min
```

Step 3 is the only part that touches the Delta database, and it is where
versioning comes in.

---

## 5. Restatements and Delta versioning

First, terminology. A **restatement** is when the vendor changes data you already
downloaded — a price correction, or more commonly a newly reported corporate
action that rewrites a stock's entire adjusted history.

### How Delta stores versions

A Delta table is not a single file. It is a folder:

```
curated/prices/
├── _delta_log/
│   ├── 00000000000000000000.json     <- version 0: what the table was at creation
│   ├── 00000000000000000001.json     <- version 1: what changed
│   └── 00000000000000000002.json     <- version 2: what changed
├── year=2020/part-0000....parquet
├── year=2020/part-0001....parquet
└── ...
```

The `_delta_log` folder is the important part. Each JSON file is one **commit**,
and it records only *which data files were added and which were removed*.
Crucially: **"removed" does not mean deleted from disk.** The file stays; the log
simply stops pointing at it from that version onward.

So a version is a *list of files*, and reading version 1 means "read exactly the
files version 1's log says belong to the table."

### Where a restated value actually lives

Suppose Apple reports a new split, and we rebuild the 2020 partition.

| | Before | After |
|---|---|---|
| Version | 1 | 2 |
| `year=2020/part-0000.parquet` | in the table, holds the **old** value | marked removed, **still on disk** |
| `year=2020/part-0007.parquet` | doesn't exist | added, holds the **new** value |

So the answer to your question: **the old value stays in the old Parquet file, and
that file remains referenced by version 1's log.** The new value is in a new file
referenced by version 2. Nothing is overwritten in place — that is what makes
reading the past possible.

Which lets you do this:

```python
now    = DeltaTable("curated/prices")                      # newest
before = DeltaTable("curated/prices").load_as_version(1)   # as it was
```

…and diff them to see **exactly what the vendor changed.** Without versioning, a
restatement is invisible: your numbers quietly change and you cannot tell whether
it was the data or your code.

### ⚠ The trap you must configure around

Delta has a housekeeping command, `VACUUM`, that permanently deletes files no
longer referenced by the current version. **Its default retention is 7 days**, and
the transaction log's default retention is **30 days**.

Meaning: with default settings, **your ability to reproduce a three-month-old
backtest expires.** For a research database that defeats the whole point.

So when the table is created we will set long retention explicitly, and never run
`VACUUM` casually. Disk is the cheap resource here; the ability to prove a result
is the expensive one.

### Why our design makes restatements cheap

This is where your "store raw + multiplier" decision pays off.

A newly reported split changes a stock's **multiplier** for its whole history —
but its **raw prices are untouched**, because $499.23 really was the price on
2020-08-28 regardless of what has happened since.

- If we stored corrected prices: **every row** for that stock changes.
- Storing raw + multiplier: **only the multiplier column** changes.

Smaller rewrites, and a much clearer audit trail — you can see that a factor
changed rather than watching thousands of prices shift for no visible reason.

---

## 6. What has changed since this was written

**Everything in §2's "not started" column has since shipped.** This document was
written mid-build, and §5's description of Delta versioning was design rather than
something you could go and inspect. That is no longer true, so the gaps list is
replaced by what actually happened:

| Was listed as a gap | Now |
|---|---|
| "The Delta database does not exist yet" | **It exists.** `curated/` holds four Delta tables: 39,677,253 price rows across 32 files, plus `dim_security`, `corporate_actions` and `calendar`. |
| `schema.py` and `build.py` not written | Both shipped, at `src/statarb/curate/`. |
| Delta retention must be configured | Done: `delta.logRetentionDuration` and `deletedFileRetentionDuration` set to 10 years in `schema.DELTA_PROPERTIES`, so history does not silently expire at the 7-day default. |
| Column renaming lives in `build.py` | Done: `c` → `px_close`, `v` → `volume`. |
| Volume needs converting | Done: `round(v × 1000)` → `int64` whole shares. Accurate to ±1,000 shares, since the extraction had already truncated the fraction server-side. |
| No `--full-refresh` flag | **Still open.** See [../ROADMAP.md](../ROADMAP.md). |
| Sector is a snapshot | **Still true**, and now handled by design rather than worked around: cluster on statistical factor loadings, not vendor sector labels. |
| No delisting return | **Still true.** CRSP remains required for that one field. |

Also added since: `DataAPI` (the read interface), 47 tests including the
adversarial point-in-time suite, and the `statarb` CLI.

**Open items live in [../ROADMAP.md](../ROADMAP.md)** — the single list. This
document explains *how stage 2 works* in plain language; it is not a status board.
