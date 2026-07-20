# Repo Structure — Target State and Migration Plan

**Purpose:** This repo is an **interview-preparation knowledge base** for ML/DL engineering roles (SDE-2 / AI Engineer and above). Every structural decision below serves that goal: content is organized so you can find a concept, understand it deeply, and defend it under cross-examination.

**Status:** Phase 0 complete (2026-07-19). Phase 1 complete and re-scoped (2026-07-20). **Phase 3 complete (2026-07-20)** — repo now matches the target structure below. Phase 2 (file splits) pending.

---

## Design Principles

1. **One concept, one file, one home.** The repo's historical defect was misfiling — math content lived under `07-interview-prep/`, so `01-foundations/` carried a broken link for months. *Topic* determines folder; *audience* determines modality suffix, never location.
2. **Numbering is a sequence, not a label.** Gaps (`02`, `05`, `13–15` missing) make it impossible to tell a dead slot from unfinished work. Renumber on insert; never leave a hole.
3. **Modality is a filename suffix, not a folder.** Never build a `flashcards/` tree parallel to the content tree — parallel trees drift.
4. **Depth follows interview frequency, not personal interest.** The repo was 225k words of LLM content against 50k of classical ML. Real interviews invert that ratio.
5. **Cheatsheets link, they don't restate.** Any fact duplicated across modalities will eventually contradict itself. The deep-dive file is the single source of truth.

---

## Target Structure

```
00-meta/                    This file, link checker, prerequisite DAG, conventions
01-math-foundations/        Linear algebra, probability, statistics, optimization, information theory
02-data/                    EDA, preprocessing, feature engineering, leakage, validation, imbalance
03-classical-ml/            One file per algorithm family
04-evaluation/              Metrics, calibration, error analysis, experiment design, A/B testing
05-deep-learning-core/      Backprop, activations, init, normalization, regularization, optimizers
06-architectures/           MLP, CNN, RNN/LSTM, attention, transformers, GNN, SSM/Mamba
07-domains/                 NLP, CV, time series, speech, recsys, tabular
08-generative/              Autoencoders, VAE, GAN, diffusion, flows
09-reinforcement-learning/  Bandits, value/policy methods, RLHF bridge
10-llms/                    Tokenization, pretraining, scaling, fine-tuning, alignment, inference, MoE
11-llm-applications/        Prompting, RAG, agents, tool use, multimodal, evals
12-systems-and-scale/       Distributed training, compression, serving, hardware, cost
13-production-ml/           MLOps, deployment, monitoring, drift, governance, incidents
14-responsible-ai/          Fairness, interpretability, privacy, security
15-system-design/           Framework + case studies
16-interview-prep/          Question banks, derivations, mock scenarios, behavioral
17-projects/                Runnable end-to-end code
18-resources/               Papers, books, courses, reading paths
```

### Why this differs from the current layout

| Change | Reason |
| :--- | :--- |
| `04-evaluation/` promoted to top level | Currently scattered across classical-ml, production-ml, and interview-prep. Metrics govern LLM evals too — it isn't a classical-ML subtopic. |
| `06-architectures/` split from `05-deep-learning-core/` | Components (backprop, optimizers) and architectures (CNN, transformer) are different axes. The current `components/` folder mixes them. |
| `08-generative/`, `09-reinforcement-learning/` added | Diffusion, VAE/GAN, and RL are absent from the repo. Largest genuine content gap. |
| `07-domains/recsys` added | Missing entirely, and central to games-industry interviews. |
| `12-systems-and-scale/` split from production | Distributed training and serving are engineering-heavy enough to separate from MLOps process. |
| `14-responsible-ai/` added | Absent, and increasingly its own interview round. |
| `00-meta/` added | Nothing previously explained how to use the repo or in what order to read. |

---

## Per-Topic File Convention

Every leaf folder carries the same set, so navigation becomes muscle memory:

```
03-classical-ml/
  README.md                 Map, prerequisites, what lives here
  01-linear-regression.md
  02-logistic-regression.md
  03-decision-trees.md
  04-random-forests.md
  05-gradient-boosting.md    XGBoost vs LightGBM vs CatBoost mechanics
  06-svm.md                  Kernel trick, dual formulation
  07-naive-bayes.md
  08-knn.md
  09-clustering.md
  10-dimensionality-reduction.md
  _cheatsheet.md             One-page dense reference
  _flashcards.md             Active-recall Q→A
  _comparison.md             When-to-use-which decision tables
```

The `_` prefix sorts modality files below content files.

> **Note:** the current repo merges two modalities into `CHEATSHEET-COMPARISON.md`. Keep them separate going forward — a cheatsheet is for cramming, a comparison table is for deciding. They're used at different moments.

---

## Per-File Content Format

Because this repo is for interviews, every topic file follows this order. The first and last sections are non-negotiable.

```markdown
---
module: <section>
topic: <topic>
tags: [...]
prerequisites: [linear-algebra, gradient-descent]
last-reviewed: YYYY-MM-DD
---
# Topic

## The Problem It Solves      ← always first; motivation before mechanism
## Intuition                  ← the analogy that makes it stick
## The Mechanics              ← math + minimal code
## Worked Example             ← hand-computable, 3–5 data points
## When It Breaks             ← failure modes, edge cases
## Production Notes           ← latency, cost, scale, what breaks first
## Interview Angles           ← questions, cross-questions, traps
## Connections                ← links to related topics
```

### Two additions to the existing house style

**`## Worked Example`** — the highest-value addition. Computing one boosting round by hand on five rows converts notes you can *review* into material you actually *understand*. Interviewers probe exactly this boundary.

**`prerequisites:` frontmatter** — lets the dependency DAG be generated mechanically rather than maintained by hand.

### The `## Interview Angles` format

Questions live **inline with the concept they test**, not in a distant section-level bank. Each carries a cross-question ladder — the follow-ups that separate a round-2 answer from a round-3 answer.

```markdown
## Interview Angles

### Q: Why does L1 regularization induce sparsity? [Medium]

The L1 penalty λΣ|w| has a non-differentiable corner at zero. Geometrically,
the diamond-shaped constraint region has corners on the coordinate axes, so
the loss contour is most likely to first touch the constraint at a corner —
where one coordinate is exactly zero.

**Cross-questions to expect:**
- *"So why not always use L1?"* → When most features carry weak signal, L2
  outperforms; L1 arbitrarily picks one of a correlated group. ElasticNet
  when both matter.
- *"What's the Bayesian reading?"* → L1 = Laplace prior, L2 = Gaussian prior.
  Regularization is a prior belief about the weights.
- *"Show me geometrically."* → Draw the diamond vs. the circle; the corner
  is the whole argument.

**Trap:** Saying "L1 shrinks weights" without distinguishing *shrink to
exactly zero* from *shrink toward zero*. That distinction is the question.
```

**Difficulty tiers** (retained from the current convention, which is good):
- **Easy** — Round 1: definitions, intuition. Expect these cold.
- **Medium** — Round 2: connecting concepts, debugging, trade-offs.
- **Hard** — Round 3+: open-ended design, defending choices under pressure.

---

## What Not To Do

- **No parallel `flashcards/` tree.** Modality files live beside their content or they go stale.
- **No `99-misc/`.** Anything unplaceable reveals a missing category — add the category.
- **No topic folder under 3 files.** Fold it into its parent. The current `methods/` folder — 3 files with 5 numbering gaps — is the symptom.
- **No content duplicated across modalities.** Cheatsheets link to the deep-dive.
- **No link to a file that doesn't exist yet.** Use a visible `**TODO (planned):**` marker instead. Dangling links are how the repo accumulated 58 broken references.

---

## Migration Phases

Sequenced so each phase is verifiable before the next begins.

### Phase 0 — Make the repo verifiable ✅ COMPLETE (2026-07-19)

- [x] Add `00-meta/check-links.py`
- [x] Fix 58 broken links → **0**
- [x] Fix 39 dead `SUMMARY.md` entries → **0**
- [x] Add 81 unpublished files to `SUMMARY.md` → **0** (sections 13 and 14, 53 files of EA interview material, had zero nav presence)
- [x] Record removals in `00-meta/REMOVED-CONTENT.md`

### Phase 1 — Add missing content (net-new, no migration cost)

**This list was rewritten on 2026-07-20 after a keyword-density survey of the existing repo.** The original six-item list was written from the *target structure*, not from what the repo actually contains. Five of the six already have substantive homes — writing them fresh would have created a second source of truth for each, violating design principle #5. What was missing was folder-level *organization*, which is Phase 3 work, not new content.

- [x] `04-reinforcement-learning/` — foundations, bandits, value-based, policy gradient + RLHF **(complete, 2026-07-20)**

Placed at slot `04` rather than the target's `09` because `04` was vacant between `03-deep-learning/` and `05-llms/`, so it required no renumbering. It moves to `09-` in Phase 3 along with everything else.

#### Survey findings — why the other five were dropped

| Originally planned | Where the content actually lives | Verdict |
| :--- | :--- | :--- |
| `08-generative/` — VAE, GAN, diffusion | `07-interview-prep/dl/03-dl-architectures.md` §4–6 — ~4k words incl. WGAN-GP, ELBO + reparameterization, DDPM/DDIM, latent diffusion, consistency models. Already carries interviewer-framing and traps sections. | **Misfiled, not missing.** Phase 3 move. |
| `12-systems-and-scale/` — distributed training | `05-llms/06-fine-tuning-at-scale.md`, `05-llms/interview-notes/12-ai-infrastructure-and-scalability.md` | **Exists.** Phase 3 consolidation. |
| `07-domains/recsys` | Scattered across system-design case studies | **Partial.** Genuine gap for a *standalone* treatment (matrix factorization, ALS, two-tower). Re-evaluate after Phase 2. |
| `16-anomaly-detection.md` | Referenced by `unsupervised-learning.md` as a TODO marker | **Genuine gap**, but small — a single file, not a phase. Fold into Phase 2. |
| `14-responsible-ai/` | `07-interview-prep/ml/20-privacy-and-fairness.md`, `05-llms/interview-notes/10-ai-safety-...md` | **Exists**, thinly and split across two sections. Phase 3 consolidation. |

**Process note worth keeping:** the survey was nearly derailed twice — once by an `-E` regex whose escaped alternation matched nothing (reads as "topic absent" rather than "pattern broken"), and once by ranking hits by file size, which surfaced only the repo's largest files for every query. Rank by keyword density (`grep -rioE | cut -d: -f1 | sort | uniq -c | sort -rn`) and treat a zero result as suspect until the pattern is proven on a known match.

### Phase 2 — Split overloaded files (partially complete, 2026-07-20)

**Three of the five planned splits were wrong and were not executed.** As in Phase 1, the list was written from the target structure rather than from the files. Surveying first prevented two duplications and one destructive merge. What the survey found:

| Planned | Finding | Action |
| :--- | :--- | :--- |
| Split `01-supervised-learning.md` per-algorithm | `03-algorithms.md` **already covers the same eight algorithms** — but as *interview framing* ("what the interviewer is testing", traps, formulas cold) vs. the deep-dive's derivations. Not duplicates: different modalities. | **Not split.** Splitting would have created a third copy. Added a routing table to the folder README making the modality split explicit. |
| Extract gradient boosting to its own file | Already in three places: `01-supervised-learning.md` §XGBoost/LightGBM/CatBoost (732 words), `03-algorithms.md` §4, `04-ensemble-methods.md` §Boosting. | **Not extracted.** A fourth home makes the duplication worse. Consolidation is a separate decision. |
| Move `_interview-questions.md` content inline | Content is substantive and already tiered Easy/Medium/Hard, but has **no cross-questions and no traps**. Distributing it means authoring Interview Angles fresh per topic, and loses the end-to-end drillable ladder. | **Not moved.** The tiered ladder is a study asset; scattering it to satisfy a format spec is a net loss. |
| Split `_cheatsheet-comparison.md` → `_cheatsheet.md` + `_comparison.md` | All 7 files are a **single uniform comparison format** (What it is / Pros / Cons / Pick over alternatives when / Key hyperparameters; 578 table rows). There is no separable cram-sheet half — splitting means bisecting entries that read as a unit, or duplicating "What it is" into both. | **Renamed** `_cheatsheet-comparison.md` → `_comparison.md` (7 files). They were misnamed, not merged. |
| Write anomaly detection file | Genuine gap. (`16-interview-prep/ea/.../16-anomaly-detection.md` exists but is an EA streaming *case study*, not an algorithms reference.) | **Written** as `03-classical-ml/10-anomaly-detection.md`, resolving the TODO in `02-unsupervised-learning.md`. |

Also fixed: **20 files outside `16-interview-prep/` carried stale `module: Interview Prep` frontmatter** from the migration — which is *why* `03-algorithms.md` read as a duplicate rather than as a distinct modality. Corrected to their actual module, with an `interview-framing` tag added to keep them discoverable.

#### Still open

## Phase 4 — generative split + responsible-AI routing (COMPLETE)

Both items were planned as *moves*. Survey changed both: one was a partial move, the other was
not a move at all.

| Planned | Found | Action |
| :--- | :--- | :--- |
| Move `04-dl-architectures.md` §4–6 → `08-generative/` | §4 (GAN) and §6 (Diffusion) had no counterpart — clean move. §5 (VAE) **collided** with an existing deep-dive in `01-autoencoders.md` and is a different modality of the same subject. | Moved §4 → `02-gans.md`, §6 → `03-diffusion.md`, each with new Interview Angles. §5 kept in place with a reciprocal cross-reference. |
| Consolidate LLM safety → `14-responsible-ai/` | The three candidate files are LLM-*behavioural* (jailbreaks, prompt injection, RLHF, sycophancy, reward hacking), sit in a numbered 1–18 arc, and one has a paired `-snappy.md` modality file. Genuine overlap with the fairness math is ~3 questions of ~50. | **Not moved.** Added a routing table to `14-responsible-ai/README.md` splitting model-agnostic math (here) from LLM behaviour (there), plus cross-links. Corrected the README's own stale "should be consolidated here" claim. |

**Lesson, fourth occurrence.** The README in `08-generative/` asserted §4–6 was "misfiled, not
missing" — written from the plan, not from the files. It was two-thirds right. A folder that
*looks* thin may be thin for a reason worth preserving. Both READMEs now state their real gaps:
normalizing flows and autoregressive models in `08-generative/`; interpretability mechanics and
adversarial robustness in `14-responsible-ai/`.

- [ ] Decide whether standalone recsys coverage is in scope (see Phase 1 survey table)
- [ ] Decide whether gradient boosting should be consolidated to one home (currently 3) — a merge decision, not a split
- [ ] If Interview Angles format is wanted repo-wide, author cross-questions/traps per topic rather than relocating existing Q&A

#### The recurring lesson

Three phases running, plans written from the target structure have not survived contact with the files. **Survey before executing, and treat "this looks like a duplicate" as a hypothesis to test** — twice now the apparent duplicate was a distinct study modality whose frontmatter had gone stale.

### Phase 3 — Move folders ✅ COMPLETE (2026-07-20)

- [x] Renumber to close gaps — the repo now has exactly the 19 folders listed above, no holes
- [x] Move folders to target structure — 160 `git mv` operations, rename history preserved
- [x] Regenerate `SUMMARY.md` mechanically — generated from the folder tree, headings match folder numbering
- [x] Publish a redirect table (below)
- [x] Re-run `check-links.py` — reports **CLEAN** (0 broken links, 0 dead nav entries, 0 unpublished files)

#### Redirect table — old folder → new folder

| Old | New | Files |
| :--- | :--- | ---: |
| `12-projects/` | `17-projects/` | 66 |
| `05-llms/` | `10-llms/` | 51 |
| `13-interview-prep/` | `16-interview-prep/` | 31 |
| `14-ea-ai-system-design-playbook/` | `16-interview-prep/` | 24 |
| `09-study-plans/` | `16-interview-prep/` | 23 |
| `07-interview-prep/` | `16-interview-prep/` | 18 |
| `06-production-ml/` | `15-system-design/` | 17 |
| `03-deep-learning/` | `05-deep-learning-core/` | 15 |
| `06-production-ml/` | `13-production-ml/` | 14 |
| `05-llms/` | `11-llm-applications/` | 13 |
| `02-classical-ml/` | `03-classical-ml/` | 11 |
| `07-interview-prep/` | `01-math-foundations/` | 7 |
| `04-reinforcement-learning/` | `09-reinforcement-learning/` | 5 |
| `01-foundations/` | `01-math-foundations/` | 4 |
| `02-classical-ml/` | `04-evaluation/` | 4 |
| `02-classical-ml/` | `02-data/` | 3 |
| `03-deep-learning/` | `06-architectures/` | 3 |
| `03-deep-learning/` | `07-domains/` | 3 |
| `07-interview-prep/` | `07-domains/` | 3 |
| `03-deep-learning/` | `12-systems-and-scale/` | 3 |
| `05-llms/` | `18-resources/` | 3 |
| `01-foundations/` | `02-data/` | 2 |
| `07-interview-prep/` | `03-classical-ml/` | 2 |
| `06-production-ml/` | `12-systems-and-scale/` | 2 |
| `01-foundations/` | `00-meta/` | 1 |
| `07-interview-prep/` | `02-data/` | 1 |
| `07-interview-prep/` | `04-evaluation/` | 1 |
| `06-production-ml/` | `04-evaluation/` | 1 |
| `07-interview-prep/` | `05-deep-learning-core/` | 1 |
| `07-interview-prep/` | `06-architectures/` | 1 |
| `03-deep-learning/` | `08-generative/` | 1 |
| `07-interview-prep/` | `10-llms/` | 1 |
| `07-interview-prep/` | `13-production-ml/` | 1 |
| `07-interview-prep/` | `14-responsible-ai/` | 1 |
| `07-interview-prep/` | `15-system-design/` | 1 |
| `01-foundations/` | `16-interview-prep/` | 1 |
| `06-production-ml/` | `18-resources/` | 1 |
| `01-foundations/` | `18-resources/` | 1 |

Renames were executed with `git mv`, so `git log --follow <file>` traverses the move. To locate any file by its old path:

```bash
git log --all --diff-filter=A --name-only --pretty=format: -- "**/<old-basename>.md"
```

#### Notes from execution

- **Link repair was the bulk of the work, not the moves.** The moves produced 718 broken links. Path-remapping fixed 381; the remainder needed *basename* resolution (286) and then git's own rename detection (51), because many files were both moved *and* renamed (`flashcards.md` → `_flashcards.md`, `24-search-ranking-system.md` → `cases/09-search-ranking.md`). Git rename detection is the authoritative source here — prefer it over hand-built mappings.
- **`11-llm-applications/` initially nested as `11-llm-applications/applications/`.** Flattened. Subtree moves need the destination checked, not just the source.
- **4 folder READMEs were deleted rather than moved** (`03-deep-learning/components/`, `07-interview-prep/{dl,llm,ml}/`) — they indexed folders that no longer exist.

---

## Maintenance

Run before every commit:

```bash
python3 00-meta/check-links.py
```

Exit code 0 means clean. Wire it into a pre-commit hook or CI to prevent regression — the 58 broken links accumulated because nothing checked.

`SUMMARY.md` is GitBook's navigation source of truth (see `.gitbook.yaml`). **Any file move must update `SUMMARY.md` in the same commit**, or pages silently vanish from the published site.
