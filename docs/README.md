# Documentation index

**Last reviewed:** 2026-08-05

Four documents, each with a distinct job. They are deliberately not merged: a
design argument, an evidence log and a plain-language explainer are different
kinds of writing, and collapsing them would lose all three.

| Document | What it is | Read it when |
|---|---|---|
| **[DATABASE_DESIGN.md](DATABASE_DESIGN.md)** | The design record. Every technical decision with the naive alternative and measured arithmetic for why it loses. Written before implementation; deviations are recorded in the header rather than edited away. | You want to know *why* something is built this way, or you are about to change it. |
| **[DISCOVERY_FINDINGS.md](DISCOVERY_FINDINGS.md)** | A dated evidence log of what was actually measured against the live FactSet server. Append-only. Overturns three assumptions made in the design doc. | You need a fact about the vendor data — real column names, row counts, the split convention. |
| **[STAGE2_EXPLAINED.md](STAGE2_EXPLAINED.md)** | Plain-language explainer for a non-specialist: what the corporate-action problem is, and how Delta versioning handles restatements. | You want to understand the *problem* rather than the implementation. |
| **[adr/](adr/)** | Architecture Decision Records. One page each, only for genuinely contested decisions. | You are about to ask "why on earth did they do it like this?" |

**Open items are in [../ROADMAP.md](../ROADMAP.md)** — one list, at the repo root.

## A note on reading order

`DATABASE_DESIGN.md` was written first and is the longest. It is a *plan*, and in
several places reality disagreed with it — the estimated universe was twice the
real size, DuckDB was designed in and never used, and the point-in-time sector
tables turned out to be empty. Those corrections are in `DISCOVERY_FINDINGS.md`.

So: **design doc for the reasoning, findings doc for the facts.** Where they
conflict, the findings doc wins, and its header says so.

Every document carries a `Last reviewed` date. A document without one cannot be
calibrated by a reader, and stale documentation that looks current is worse than
no documentation.
