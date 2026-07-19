# Removed SUMMARY.md Entries — Phase 0 Record

**Date:** 2026-07-19
**Backup of the pre-change file:** `/tmp/SUMMARY.bak` (ephemeral — the authoritative record is this file plus git history)

## What happened

`SUMMARY.md` contained 42 navigation entries pointing at files that **do not exist anywhere on disk**. These were removed.

**No content was deleted.** Every removed line was a link to a nonexistent file. Verified with `check-links.py`, which reported 39 dead `SUMMARY.md` targets before the change and 0 after.

These entries are the residue of a **prior restructure that was planned and abandoned** — the same event that left the gaps in the folder numbering (`04`, `08`, `10`, `11`). Someone wrote the target navigation before writing the target content, then stopped.

## Why remove rather than leave them

GitBook renders `SUMMARY.md` as the published site's nav. Entries pointing at missing files produce dead links for readers. More importantly for the migration: while 39 known-dead entries sat in the file, **any new breakage introduced by moving folders would be indistinguishable from the pre-existing breakage**. Clearing them is what makes Phase 3 verifiable.

---

## Removed entries, by planned section

### `08-emerging-topics/` — never created (3 entries)

- `08-emerging-topics/README.md`
- `08-emerging-topics/experimentation-and-causal-inference/README.md`
- `08-emerging-topics/experimentation-and-causal-inference/01-experimentation-and-causal-inference.md`

> **Content status:** Causal inference and experimentation exist elsewhere and are still published — see `06-production-ml/system-design/14-ab-testing-experimentation.md`. Nothing was lost.

### `10-references/` — never created (23 entries)

Book notes: Alice in Differentiable Wonderland · Build an LLM From Scratch · Deep Learning: A Practitioner's Approach · Deep Learning with PyTorch · Dive Into Deep Learning · Grokking Deep Learning · ML Pocket Reference · Designing ML Systems · Keras to Kubernetes · ML Design Patterns · ML Engineering

Research papers: Computer Vision · LLM · Time Series · Classical ML · MLOps

Plus the six `README.md` index files for those subtrees and `10-references/resources.md`.

> **Content status:** This is the one genuine loss of *intent*. Book and paper notes were planned and never written. `06-production-ml/04-books.md` exists and lists several of the MLOps titles — its three `**Book notes:**` links into this tree were also dead and were removed. Reading lists belong in `18-resources/` in the target structure.

### `11-data-scientist/` — never created (8 entries)

- Overview · Interview Prep · EDA & Data Quality · SQL & Data Manipulation · Statistics & Probability · Metrics & Business Analytics · Causal Inference · Experiment Design & AB Testing

> **Content status:** Substantially covered by `07-interview-prep/ml/` (statistics, probability, canonical stats questions) and the EA interview material in sections 13–14. The DS-specific gap is **SQL as an interview subject**: SQL appears throughout the repo as *applied* usage — query blocks inside data-engineering, feature-store, and system-design files — but nowhere as drillable material (window functions, CTEs, join semantics, the "top-N per group" family). Worth adding if data-scientist roles are in scope; skip if the target is ML engineering.

### Orphaned singletons (4 entries)

- `01-foundations/05-information-theory.md` — *content exists* as §4 of `01-foundations/02-math-and-theory-foundations.md`
- `03-deep-learning/mcp.md`
- `03-deep-learning/components/13-distributed-training-and-parallelism.md` — planned; now tracked as Phase 1 work under `12-systems-and-scale/`
- `06-production-ml/system-design/11-distributed-training.md` — same topic, second dead slot

### Duplicate (1 entry)

- `05-llms/07-context-window-extension.md` was listed twice (lines 88 and 95). Confirmed pre-existing in the backup, not introduced during Phase 0. The nested occurrence was removed; the file remains published via line 88.

---

## Recovering any of this

```bash
git log --oneline -- SUMMARY.md
git show <commit-before-phase-0>:SUMMARY.md
```

The removed lines were link text only. To restore a section, write the files first, then re-add the nav entries — not the reverse. That ordering is what this cleanup was undoing.

---

## Carried forward as real work

Three items above are genuine gaps now tracked in `REPO-STRUCTURE.md`:

| Gap | Target | Phase |
| :--- | :--- | :--- |
| Distributed training (two dead slots) | `12-systems-and-scale/` | 1 |
| Book & paper notes | `18-resources/` | 3 |
| SQL for data-science interviews | `02-data/` | 1, *if DS roles are in scope* |
