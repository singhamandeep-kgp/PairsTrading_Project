# FactSet Discovery Findings — 2026-08-04

Run against the FactSet SQL Server (`FACTSET_SQL_SERVER` / `FACTSET_SQL_DATABASE`
from `.env`), via
`pairs_trading_pipeline/extraction/run_discovery.py`. Raw CSVs in
`discovery_output/`.

**All seven blockers are resolved.** Three of my prior assumptions were wrong;
they are corrected below.

---

## 1. OHLCV exists ✅

`fp_v2.fp_basic_prices` has exactly 8 columns:

| # | Column | Type | Bytes |
|---|---|---|---|
| 1 | `fsym_id` | `char(8)` | 8 |
| 2 | `p_date` | `date` | 3 |
| 3 | `currency` | `char(3)` | 3 |
| 4 | `p_price` | `float(53)` | 8 |
| 5 | `p_price_open` | `float(53)` | 8 |
| 6 | `p_price_high` | `float(53)` | 8 |
| 7 | `p_price_low` | `float(53)` | 8 |
| 8 | `p_volume` | `float(53)` | 8 |

Two consequences for wire cost:

- **`fsym_id` is `char(8)`, not `nvarchar`.** 8 bytes, not 40. So the `sid`
  substitution saves 4 bytes/row rather than up to 36 — still worth doing, but
  it is a smaller win than budgeted.
- **Every numeric is `float(53)`** (= float64, 8 bytes). Server-side
  `CAST(... AS real)` therefore saves **4 bytes × 5 columns = 20 bytes/row**,
  which is the single largest wire saving available. Confirmed worth doing.

Raw row ≈ 8+3+3+40 = **54 bytes**. Optimised (sid int + date + 5×real + bigint
volume) ≈ **35 bytes**. About **1.55×**, on top of the universe filter.

## 2. Split convention — PINNED ✅

`fp_v2.fp_basic_splits` → `fsym_id`, `p_split_date`, `p_split_factor`.
AAPL (`MH33D6-R`), against independently known real splits:

| `p_split_date` | Real split | `p_split_factor` |
|---|---|---|
| 1987-06-16 | 2-for-1 | **0.5** |
| 2000-06-21 | 2-for-1 | **0.5** |
| 2005-02-28 | 2-for-1 | **0.5** |
| 2014-06-09 | 7-for-1 | **0.142857** |
| 2020-08-31 | 4-for-1 | **0.25** |

**`p_split_factor` is the RECIPROCAL of the split ratio** — `1/ratio`, i.e.
old shares per new share.

So the adjustment is a **multiplication of prices strictly before the split
date**:

```
adj_close(t) = p_price(t) × Π p_split_factor  for all splits with date > t
```

A pre-2020 AAPL price of $500 × 0.25 = $125 in post-split terms. Correct
direction, verified against five independent real-world splits.

## 3. The adjustment factor exists — and it is better than expected ✅

`fgp_v1.fgp_ca_adj_factors` (3,261,280 rows) →
`fsym_id`, `effective_date`, `adj_factor_combined`, `div_spl_spin_adj_factor`.

Sampled on AAPL around the 2020 split:

| `effective_date` | `adj_factor_combined` | `div_spl_spin_adj_factor` | Event |
|---|---|---|---|
| 2020-08-31 | **0.25** | **0.25** | the 4-for-1 split |
| 2020-08-07 | *(null)* | 0.998200 | a dividend |
| 2020-05-08 | *(null)* | 0.997300 | a dividend |

The two columns mean different things, and the distinction is exactly what a
backtester needs:

- **`adj_factor_combined`** — populated only for **price-affecting capital
  events** (splits, spinoffs). Cumulative product gives a **price-return**
  adjusted series.
- **`div_spl_spin_adj_factor`** — populated for **every** event including
  dividends. On dividend dates it is `≈ (1 − div/price)`. Cumulative product
  gives a **total-return** adjusted series.

**Key point:** these are **per-event**, not cumulative. We build the cumulative
factor ourselves as a running product backwards from the present, which is why
this must live in versioned code with a unit test, not a notebook cell.

**This validates the decision to skip `fp_total_returns_daily`.** We can build
total returns from `p_price` × cumulative `div_spl_spin_adj_factor` without
pulling a second billion-row table. And `fp_total_returns_daily` remains
available for spot-check validation on a sample.

**Both schemas share the `-R` regional key** — AAPL is `MH33D6-R` in both
`fp_v2.fp_basic_splits` and `fgp_v1.fgp_ca_adj_factors`. That is a significant
simplification: no cross-schema identifier translation needed.

## 4. Index layout — the ideal case ✅

| Table | Index | Type | Keys |
|---|---|---|---|
| `fp_basic_prices` | PK | **CLUSTERED, unique** | **`fsym_id, p_date`** |
| `fp_basic_prices` | `idx_fbp_fsym_id` | nonclustered | `fsym_id` |
| `fp_basic_prices` | `idx_fbp_p_date` | nonclustered | `p_date` |
| `fp_basic_prices` | `idx_fp_basic_prices` | nonclustered | `fsym_id, p_date` |
| `fp_total_returns_daily` | PK | CLUSTERED | `fsym_id, p_date` |
| `own_sec_prices` | PK | CLUSTERED | `fsym_id, price_date` |

Two wins:

1. **`SHARD_MODE = "fsym_range"`** — sharding on `fsym_id` ranges is a clustered
   range **seek**, not a scan. The index trap is fully avoided.
2. **The sort is free.** Rows arrive already ordered by `(fsym_id, p_date)`,
   which is exactly the target order for the sid-major replica. So the step-2
   external merge sort for the primary layout is **unnecessary** — a saving of
   5–15 minutes and a large tempdir. And because the clustered index already
   provides that order, adding `ORDER BY fsym_id, p_date` is genuinely free
   here (no sort operator, no tempdb spill), which reverses my earlier blanket
   "never ORDER BY" caution *for this specific order only*.

## 5. Table sizes (via `sys.partitions`)

| Table | Rows |
|---|---|
| `fp_v2.fp_total_returns_daily` | 1,015,592,525 |
| `fp_v2.fp_basic_prices` | 1,015,109,748 |
| `fgp_v1.fgp_global_prices` | **682,474,432** |
| `ent_v1.ent_entity_mkt_val` | 135,255,743 |
| **`sym_v1.sym_coverage`** | **99,509,239** |
| `fp_v2.fp_basic_shares_hist` | 35,399,570 |
| `own_v5.own_sec_prices` | **24,227,110** |
| `sym_v1.sym_entity_sector` | 10,297,919 |
| `own_v5.own_float_hist` | 8,924,526 |
| `sym_v1.sym_entity_sector_rbics` | 7,736,040 |
| `fp_v2.fp_basic_dividends` | 7,009,339 |
| `fgp_v1.fgp_ca_events` | 3,763,451 |
| `fgp_v1.fgp_ca_adj_factors` | 3,261,280 |
| `fp_v2.fp_basic_splits` | 293,182 |
| `ref_v2.fp_sec_type_map` | 96 |

## 6. Corrections to earlier assumptions

### ⚠ Volume is in THOUSANDS, and it is FRACTIONAL

AAPL 2024-01-02: `p_volume = 82488.672`. Real AAPL volume that day was ~82.5
million shares. So **`p_volume` is in thousands of shares**, and being
`float(53)` it carries a fractional part.

**This invalidates my earlier `uint32` recommendation** — casting to an integer
would silently truncate. Options:

- `float32` in thousands — but `82488.672` needs 8 significant figures and
  float32 gives ~7, so it rounds to `82488.67`. Fine for a liquidity screen,
  lossy for exact reconciliation.
- **`int64` shares = `round(p_volume × 1000)`** — exact, semantically clean
  (`82,488,672` shares), compresses well under delta encoding. **Recommended.**

Note `int32` is not safe: it caps at 2.147e9 and a heavily-traded penny stock
can exceed that in shares.

### ⚠ `sym_coverage` is 99.5M rows, not ~10⁶

My universe-resolution cost estimate was off by two orders of magnitude. It is
the global multi-asset symbology table. The universe query still returns ~34k
rows, but the joins are heavier than budgeted — expect minutes, not seconds. The
`regional_flag = 1` + `currency = 'USD'` filters do most of the work.

### ⚠ `own_v5.own_sec_prices` is NOT a daily second source

24.2M rows across the whole global universe cannot be daily — it is almost
certainly month-end. So it is **not** the free daily cross-validation source I
described. It remains useful for `unadj_shares_outstanding`. For daily
cross-validation, use `fp_total_returns_daily` on a sample instead.

## 7. Point-in-time sector — mixed news

`sym_v1.sym_entity_sector` does **not** appear among tables carrying both
`start_date` and `end_date`, so it **is a current snapshot**. Using it for
history would inject look-ahead into the clustering universe.

**But a whole `rbics_v1` schema exists**, and its tables *are* point-in-time:
`rb_sec_entity_hist`, `rbics_bus_seg_item`, `rbics_bus_seg_report`,
`rbics_structure`. Also newly discovered: an **`ff_v3`** schema (FactSet
Fundamentals) with `ff_sec_entity_hist`.

**So strict PIT sector classification is achievable via RBICS** rather than the
FactSet sector map the other team uses. This needs one follow-up query to map
the RBICS structure, but the capability is there.

## 8. Bonus finds

- **`ref_v2.ref_calendar_dates`** (`ref_date`, `day_of_week`, **`eom_flag`**) and
  **`ref_v2.ref_calendar_holidays`** (`fref_exchange_code`, `holiday_date`,
  `holiday_name`, 174k rows). A proper per-exchange trading calendar — strictly
  better than `pandas_market_calendars`, and far better than the legacy code's
  approach of deriving the calendar from one stock's observed dates (which also
  encodes that stock's trading halts). `eom_flag` directly gives month-end
  rebalance dates.
- **`ref_v2.ca_event_type_map`** — 13 clean codes: `FSP` forward split, `RSP`
  reverse split, `SPL` split, `SPO` spin off, `DVC` dividend, `DVS` stock
  dividend, `BNS` bonus, `DSR` rights, `DRP`/`DVCD` reinvestment, `EXOS`
  exchange of securities.
- **`ref_v2.fp_div_type_map`** — 74 codes, and several matter for delisting:
  `73` liquidation distribution, `74` final liquidation distribution, `16`
  estimate of liquidation returns, `200`/`1055` spin off, `11` special dividend.
  **These are the closest thing to a CRSP-style delisting return in this feed**
  and should be captured explicitly.
- `sym_v1.sym_isin_hist` / `sym_cusip_hist` / `sym_sedol_hist` exist with
  validity windows — proper time-varying identifier history for CRSP
  reconciliation.

## 9. Permission limits

`VIEW DATABASE STATE` is **denied**, so no DMVs: `sys.dm_db_partition_stats`,
`sys.dm_db_index_usage_stats` and friends all fail with error 262/297.

Workaround already in place: `sys.partitions` is a catalog view rather than a
DMV and works fine. Consequence — we **cannot** check whether other users are
actively hammering the tables, so the conservative posture stands:
`N_WORKERS = 3`, `OPTION (MAXDOP 1)`.

---

---

# Round 2 findings

## 10. `fp_basic_prices` is UNADJUSTED ✅ — confirmed empirically

AAPL across its own 2020-08-31 4-for-1 split:

| `p_date` | `p_price` | `p_volume` (000s) |
|---|---|---|
| 2020-08-27 | **500.04** | 38,888 |
| 2020-08-28 | **499.23** | 46,907 |
| **2020-08-31** | **129.04** | 225,703 |
| 2020-09-01 | 134.18 | 152,470 |

Pre-split rows show ~$500, post-split ~$129. So the series is **raw as-traded**,
which is exactly what we want as the anchor. Volume is also unadjusted (~4×
higher share count after the split, as expected) and confirms the thousands unit.

## 11. ⚠ `rbics_v1` IS EMPTY — strict point-in-time sector is NOT achievable

Every `rbics_v1` table has **0 rows**: `rb_sec_entity_hist`, `rbics_coverage`,
`rbics_structure`, `rbics_sec_entity`, `rbics_bus_seg_item`,
`rbics_bus_seg_report`, `rbics_address`. The schema exists but was never
populated.

**This reverses finding 7.** Both remaining sector tables are snapshots:
`sym_entity_sector` (10.3M) and `sym_entity_sector_rbics` (7.7M — columns are
just `factset_entity_id`, `l2_id`, `focus_flag`, no dates).

So **there is no point-in-time sector classification in this feed.** Three ways
forward:

1. **Accept the snapshot and document the bias.** Simplest, honest, and the bias
   is measurable (count how many companies changed sector).
2. **Derive a time-varying proxy from `ff_v3.ff_segbus_af`** (8.4M rows,
   business segments by fiscal year). Heavy, but genuinely point-in-time.
3. **Design the look-ahead away.** The pipeline currently uses sector only to
   *partition* the universe before PCA. Drop the sector partition, run PCA on the
   whole universe, and let OPTICS discover clusters from factor loadings alone.
   **The look-ahead disappears entirely** because no forward-looking label is
   used, and arguably the clustering gets better — it finds statistical
   comovement rather than inheriting a vendor's taxonomy.

Option 3 is the strongest answer and costs nothing extra.

**Bonus:** `ff_v3` (FactSet Fundamentals) is fully populated and large —
`ff_basic_af`/`ff_advanced_af` ~2.9M, `ff_basic_qf` ~6.5M, `ff_basic_ltm` ~6.9M,
segment tables ~8.4M. Not needed now, but it means fundamental factors are
available later without another data-sourcing exercise.

## 12. ⚠ No delisting return in FactSet — CRSP stays necessary

Liquidation-type dividends exist but are far too rare to serve as delisting
returns:

| Code | Description | Count (global, all history) |
|---|---|---|
| 11 | Special dividend | 47,536 |
| 200 | Spinoff | 7,488 |
| 1055 | Spin Off | 2,849 |
| **73** | **Liquidation distribution** | **342** |
| **74** | **Final liquidation distribution** | **32** |

374 liquidation records across the entire global universe over 40+ years, against
many thousands of US delistings. **FactSet does not ship a CRSP-style `dlret`.**

So the CRSP overlay is confirmed **load-bearing, not optional** — it is the only
source for the one field whose absence systematically inflates backtest returns.
Spinoffs (`200`/`1055`, ~10k records) and `fgp_ca_events.dist_inst_fsym_id` do
give proper spinoff handling, which is worth having.

## 13. Universe filter — RESOLVED, and my exchange guesses were wrong

`fref_listing_exchange` for USD + `SHARE`, by `p_sec_type_code`:

| Exchange | `'10'` (Equity) | NULL code |
|---|---|---|
| **OTC** US OTC | **28,712** | 18,885 |
| **NAS** NASDAQ | **9,917** | 3,513 |
| **NYS** New York SE | **5,286** | 959 |
| **ASE** NYSE American | **1,138** | 697 |
| **PSE** NYSE Arca | 26 | 47 |
| CHI NYSE Texas | — | 42 |
| ZZUS US Unlisted Funds | 42 | 655 |

Corrections to my earlier config:

- **`"ARC"`, `"BAT"`, `"IEX"` are not real codes.** Cboe BZX/BYX/EDGA/EDGX, IEX,
  MEMX and LTSE exist in `fref_sec_exchange_map` but **never appear as listing
  exchanges** — they are trading venues. My guess was about half wrong.
- **`currency = 'USD'` does NOT imply US-listed.** USD-quoted `SHARE` securities
  appear on London (1,204), Moscow (1,231), Lima, SIX Swiss, Santiago, Buenos
  Aires, Shanghai, Guayaquil, Kazakhstan, Palestine, Toronto, Luxembourg,
  Amsterdam, BX Swiss. **The exchange filter is load-bearing** — currency alone
  would silently import foreign listings into a "US" universe.
- `SEC_TYPE_CODES = ('10',)` — `'10'` is "Equity". Excludes `'6C'` open-ended
  fund (87) and `'54'` composite unit (73), both of which otherwise carry
  `fref_security_type = 'SHARE'`.

### ⚠ Revised universe size — and an OTC decision for you

**Exchange-listed US common equity: 16,367 securities** (NAS + NYS + ASE + PSE
with code `'10'`). That is roughly **half my 34,000 estimate**, so the fact table
should land nearer **~41M rows and ~0.6 GB**, not 92M / 1.3 GB. Faster and
smaller than planned.

**Adding OTC would take it to ~45,000 securities — nearly 3×.** Config currently
sets `OTC_EXCHANGE_CODES = ()`, i.e. excluded, because OTC is pink sheets,
shells, and names that generally cannot be shorted or borrowed. For a pairs
strategy that is mostly untradeable signal. Flip it on only alongside a hard
liquidity screen. **This is a judgment call worth making deliberately.**

## 14. ⚠ `fgp_global_prices` is NOT a drop-in replacement

Its column list is far richer than `fp_basic_prices` — it adds `turnover`,
`trade_count`, `vwap`, `one_day_pct`, and a full ladder of period returns
(`wtd`/`mtd`/`qtd`/`ytd`/`1m`/`3m`/`6m`/`9m`/`1y`/`2y`/`3y`/`5y`/`10y`). That
looked like it would remove the need for a separate total-returns pull.

**But querying AAPL's `-R` id (`MH33D6-R`) against `fgp_global_prices` returned
zero rows.** So it is keyed differently (likely the `-L` listing id) or its
coverage excludes US names. **My earlier claim that `fgp_v1` and `fp_v2` share
the `-R` key was wrong** — I inferred it from `fgp_ca_adj_factors`, which *does*
use `-R` and *does* contain AAPL.

**Decision: `fp_v2.fp_basic_prices` stays the primary source** (unadjusted,
OHLCV, clustered index, confirmed US coverage), with `fgp_v1.fgp_ca_adj_factors`
for adjustment factors. `fgp_global_prices` is worth revisiting later for VWAP
and turnover, but it needs an identifier investigation first.

## 15. Corporate-action schemas — full column lists

**`fp_v2.fp_basic_splits`** — `fsym_id`, `p_split_date`, `p_split_factor`

**`fp_v2.fp_basic_dividends`** — `fsym_id`, **`p_divs_exdate`** (the correct date
for total-return maths), `p_divs_pd_id`, `p_divs_pd` (amount), `currency`,
`p_divs_paydatec`, `p_divs_recdatec`, `p_divs_s_spinoff`, `p_divs_s_pd`,
`p_divs_pd_type_code`, `p_divs_pd_ngflag_code`, `p_divs_pd_ngequiv`,
`p_divs_pd_tax_code`

**`fp_v2.fp_basic_shares_hist`** — `fsym_id`, `p_date`, `p_com_shs_out`
(35.4M rows; clean shares-outstanding history — better than the month-end
`own_v5` source)

**`fgp_v1.fgp_ca_adj_factors`** — `fsym_id`, `effective_date`,
`adj_factor_combined`, `div_spl_spin_adj_factor`

**`fgp_v1.fgp_ca_events`** (40 columns, very rich) — `ca_event_id`,
`ca_event_type_code`, `effective_date`, `announcement_date`, `pay_date`,
`record_date`, **`price_adj_factor`**, `dist_old_term`/`dist_new_term`/`dist_pct`
(split terms), net/gross/declared amounts both adjusted and unadjusted,
`trading_currency`, `declared_currency`, `div_type_code`, `dividend_spec_flag`,
`dividend_status`, `dividend_active_flag`, franking fields, and
**`dist_inst_fsym_id`** — the distributed instrument, which is what makes proper
spinoff handling possible.

---

## The adjustment arithmetic (now fully specified)

```
cum_split(t)    = Π p_split_factor        for splits with p_split_date  > t
cum_price(t)    = Π adj_factor_combined   for events with effective_date > t
cum_total(t)    = Π div_spl_spin_adj_factor for events with effective_date > t

px_split_adj(t) = p_price(t) × cum_price(t)      # price-return series
px_total_adj(t) = p_price(t) × cum_total(t)      # total-return series
ret_total(t)    = px_total_adj(t) / px_total_adj(t-1) − 1
```

Reconciliation identity to assert in tests: `cum_price` built from
`adj_factor_combined` must agree with `cum_split` built independently from
`fp_basic_splits`, since both should capture the same split events. Any
disagreement is a spinoff or capital change present in one source and not the
other — which is itself worth investigating rather than smoothing over.

Validation: `px_total_adj` returns must match `fp_total_returns_daily.one_day_pct`
on a sample of ~500 securities. We are not extracting that table in full, but a
sample is cheap and closes the loop.

## Remaining open items

**Moved to [../ROADMAP.md](../ROADMAP.md).** This document is a dated evidence log
— what was measured against the live server, on which date. Keeping a second
open-issues list here guaranteed it would drift out of step with the other two,
which is exactly what happened before they were consolidated.

For the record, the items that were open when this was written have since been
resolved as follows:

| Was open | Outcome |
|---|---|
| `fgp_global_prices` identifier | Not used. Querying AAPL's `-R` id returned zero rows, so it is not a drop-in for `fp_v2`. Recorded under "Won't do". |
| Q7 universe-count-by-year | Done. Curve traced ~6.3k (1997) → ~4.4k (2012) → ~5.4k (2021), matching the real US listing curve. |
| 27,169 `SHARE` rows with NULL `p_sec_type_code` | Still unconfirmed. Now in ROADMAP under Known limitations. |
| OTC decision | Excluded. `OTC_EXCHANGE_CODES = ()`; OTC would have nearly tripled the universe with names that cannot be borrowed or shorted. |
| Sector approach | Option 3: cluster on statistical factor loadings, removing the dependency on vendor sector labels entirely. |
