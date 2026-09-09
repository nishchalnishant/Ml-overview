---
module: Practice
topic: Active Recall
subtopic: Problem Index
status: unread
tags: [practice, index, navigation]
---
# Problem Index

Every problem in this section on one page, so you can pick one in ten seconds instead of
opening four files to decide. **Solutions are deliberately not linked from this page** —
looking up an answer should cost one extra deliberate step.

Difficulty is calibrated to a real screen, not to LeetCode: **E** = you should finish
inside the time box, **M** = expect to need a hint on one part, **H** = a strong candidate
often does not fully finish.

---

## Fastest path in

If you have never opened this section, do these six, in this order. They cover the widest
span of failure modes in about three hours.

| Order | Problem | Why this one |
|---|---|---|
| 1 | ML 1.1 stable softmax | Reveals whether you know *why* the max is subtracted |
| 2 | Debug D2 missing `zero_grad` | The single most-asked debug snippet |
| 3 | DSA 2.3 min window substring | The hardest window pattern; everything else is easier |
| 4 | ML 3.2 attention from scratch | Asked in nearly every LLM-adjacent loop |
| 5 | Mock 1 feature store | Point-in-time correctness is the classic disqualifier |
| 6 | Go ex. 6 cancellation | The idiom that matters most for inference roles |

---

## ML coding — [01-ml-problems.md](coding/01-ml-problems.md)

### Tier 1 — Warm-ups (15 min)

| # | Problem | Diff | The thing being tested |
|---|---|---|---|
| 1.1 | Stable softmax | E | Overflow, `keepdims`, the max-shift argument |
| 1.2 | Train/test split | E | Index permutation, X–y correspondence, seeding |
| 1.3 | StandardScaler | E | fit/transform discipline, zero-variance guard |
| 1.4 | Confusion matrix + metrics | E | Zero-denominator policy, defending it |
| 1.5 | One-hot with unseen categories | E | Stating a policy rather than crashing |
| 1.6 | Batch generator | E | Partial final batch, `drop_last` |

### Tier 2 — Core algorithms (25 min)

| # | Problem | Diff | The thing being tested |
|---|---|---|---|
| 2.1 | Linear regression, two ways | E | Normal equation vs GD; when each is right |
| 2.2 | Logistic regression with L2 | M | The branched sigmoid for stability |
| 2.3 | k-means with k-means++ init | M | Empty clusters, convergence check |
| 2.4 | k-nearest neighbours | M | Stable-sort tie-breaking |
| 2.5 | Decision tree — a single split | M | *Weighted* child impurity, duplicate values |
| 2.6 | Naive Bayes for text | M | Log space, `alpha * V` denominator |
| 2.7 | PCA from scratch | M | `eigh`, descending reorder, centering |
| 2.8 | ROC-AUC without sklearn | H | Average-rank ties, Mann–Whitney framing |
| 2.9 | Stratified k-fold | M | Round-robin assignment |
| 2.10 | Gradient descent variants | M | Adam bias correction — omitting it is the classic error |

### Tier 3 — Deep learning (30 min)

| # | Problem | Diff | The thing being tested |
|---|---|---|---|
| 3.1 | Backprop, two-layer network | H | Softmax+CE collapsing to `(P − Y)`; ReLU gated on `Z1 > 0` |
| 3.2 | Scaled dot-product attention | M | The `sqrt(d_k)` variance argument, additive `-inf` mask |
| 3.3 | Multi-head attention | H | Transpose to `(B, h, T, d)` and its exact inverse |
| 3.4 | LayerNorm, forward and backward | H | Both correction terms in the backward pass |
| 3.5 | A complete PyTorch training loop | M | `train()`/`eval()`, `zero_grad`, `no_grad` |
| 3.6 | Dropout at train and inference | E | Inverted scaling at *train* time |
| 3.7 | Positional encoding | M | Sinusoidal vs RoPE |
| 3.8 | Beam search decoding | H | Length normalization, finished-beam carryover |

### Tier 4 — Applied and messy (35 min)

| # | Problem | Diff | The thing being tested |
|---|---|---|---|
| 4.1 | Leak-free preprocessing pipeline | M | `ColumnTransformer` *inside* the CV loop |
| 4.2 | Time-series split | H | Walk-forward with a `gap` for label lag |
| 4.3 | Imbalanced 1:200, end to end | M | PR-AUC, threshold from a business constraint |
| 4.4 | Streaming mean and variance | M | Welford, using the *updated* mean |
| 4.5 | Top-k retrieval by cosine | M | `argpartition` over full sort; N = 10M |
| 4.6 | Sliding-window aggregation | H | `closed="left"` — no future leaks into a row |

---

## DSA — [02-dsa-problems.md](coding/02-dsa-problems.md)

### Tier 1 — Hashing and arrays

| # | Problem | Diff | Key idea |
|---|---|---|---|
| 1.1 | Two-sum, then all pairs | E | Complement map; dedupe for the all-pairs variant |
| 1.2 | Longest substring w/o repeats | M | `last[c] >= start` staleness guard |
| 1.3 | Group anagrams | E | Sorted-tuple or count-tuple key |
| 1.4 | Longest consecutive sequence | M | `x-1 not in s` gives the amortization |
| 1.5 | Majority element, O(1) space | M | Boyer–Moore pairing argument |
| 1.6 | Product except self | M | Prefix × suffix, no division |

### Tier 2 — Two pointers and windows

| # | Problem | Diff | Key idea |
|---|---|---|---|
| 2.1 | Dedupe sorted array in place | E | Write pointer |
| 2.2 | Container with most water | M | Why moving the taller pointer never helps |
| 2.3 | Minimum window substring | H | `missing` counter collapses the dict compare |
| 2.4 | Substring with ≤ k distinct | M | Delete zero-count keys or the size lies |
| 2.5 | Min-size subarray sum | M | Window **requires** positivity |
| 2.6 | Three-sum | M | Dedupe at both outer and inner pointers |

### Tier 3 — Heaps and intervals

| # | Problem | Diff | Key idea |
|---|---|---|---|
| 3.1 | k largest (+ streaming) | M | **Min**-heap to keep the largest k |
| 3.2 | Merge k sorted lists | M | List-index tiebreaker avoids `TypeError` |
| 3.3 | Merge intervals | E | `max(out[-1][1], e)` for nested |
| 3.4 | Insert interval | M | Three phases |
| 3.5 | Meeting rooms II | M | Decoupled start/end sweep |
| 3.6 | Median of a stream | H | `heappush(lo, -heappushpop(hi, x))` |
| 3.7 | Top-k frequent, two ways | M | Heap vs bucket sort |

### Tier 4 — Graphs, trees, DP

| # | Problem | Diff | Key idea |
|---|---|---|---|
| 4.1 | Number of islands | M | Iterative DFS, mark seen *at push* |
| 4.2 | Course schedule | M | Kahn's: ordering + cycle detection together |
| 4.3 | Binary tree level order | E | Snapshot `len(q)` per level |
| 4.4 | Lowest common ancestor | M | The split-point invariant |
| 4.5 | Coin change | M | Counterexample `[1,3,4]`, amount 6 |
| 4.6 | Longest common subsequence | M | The two-case recurrence |
| 4.7 | BFS shortest path | M | Why DFS is not optimal here |
| 4.8 | Validate BST | M | Bounds threading; `5 → (1 → (right 6))` |

---

## Debug drills — [03-debug-drills.md](coding/03-debug-drills.md)

All twelve run without raising. That is the point. **15 min each, 30 min for the full set
if you are warm.**

| # | Snippet | Diff | Bug class |
|---|---|---|---|
| D1 | Scale before split | E | Leakage |
| D2 | Missing `zero_grad` | E | Training loop |
| D3 | `correct / len(loader)` | E | Metric denominator |
| D4 | `rolling(7)` includes current row | M | Leakage |
| D5 | Softmax before `CrossEntropyLoss` | E | Double activation |
| D6 | Shuffled KFold on time series | M | Validation design |
| D7 | `losses.append(loss)` | M | Retained graph → OOM |
| D8 | `argsort(sims)[:10]` | E | Ascending sort |
| D9 | SMOTE before CV | H | Leakage |
| D10 | Batch-dependent padding | H | Masking |
| D11 | Tuning on the test set | M | Methodology |
| D12 | Summing an averaged loss | M | Metric weighting |

---

## System design mocks — [system-design/](system-design/README.md)

45 min each. Follow-ups open at 0:15 / 0:25 / 0:35.

| # | Mock | File | The trap |
|---|---|---|---|
| 1 | Real-time feature store | [01](system-design/01-ml-platform-mocks.md) | Point-in-time joins as a *type error* |
| 2 | Training pipeline, 100-person org | [01](system-design/01-ml-platform-mocks.md) | Full reproducibility set; gang-scheduling deadlock |
| 3 | Recommendation system | [01](system-design/01-ml-platform-mocks.md) | Feedback loop and popularity bias |
| 4 | Fraud detection | [01](system-design/01-ml-platform-mocks.md) | Delayed labels; never observing declined transactions |
| 5 | Multi-tenant inference platform | [02](system-design/02-llm-serving-mocks.md) | Noisy neighbors; prefix cache as a timing side channel |
| 6 | RAG at scale | [02](system-design/02-llm-serving-mocks.md) | Index memory arithmetic; staleness |
| 7 | KV cache tier | [02](system-design/02-llm-serving-mocks.md) | 0.3 MB/token → a 100k request is 30 GB |
| 8 | Agent platform | [02](system-design/02-llm-serving-mocks.md) | Cancellation; unbounded tool loops |

Score every attempt with [03-rubrics.md](system-design/03-rubrics.md). The 2→3 band —
prompted versus volunteered — is the one that decides outcomes.

---

## Go exercises — [go-inference/](go-inference/README.md)

| # | Exercise | Time | Prereq |
|---|---|---|---|
| 1 | Token counter service | 60 min | — |
| 2 | Concurrent batch scorer | 90 min | 1 |
| 3 | Queue with backpressure | 90 min | 2 |
| 4 | Streaming gRPC token API | 2 h | 1 |
| 5 | Prefix-affinity router | 90 min | 2 |
| 6 | Cancellation all the way down | 75 min | 3, 4 |
| 7 | Metrics and health | 75 min | any |
| 8 | Capstone gateway | 4 h+ | all |

**Three days only:** 2, 4, 6.

---

## Related

- [PROGRESS.md](PROGRESS.md) — where to record what you got wrong
- [DAY-ONE.md](DAY-ONE.md) — what to do in the first session
