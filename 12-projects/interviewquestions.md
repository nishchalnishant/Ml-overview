---
module: Projects
topic: Hands-On Projects Interview Q&A
subtopic: ""
status: unread
tags: [projects, interview, qa, tabular, rag, lora, mcp, a2a, hands-on]
---
# Hands-On Projects — Interview Q&A

Detailed interview questions and answers grounded in the actual code in this folder
(`12-projects/`), not generic textbook prose. Use this to rehearse "walk me through
this project" and "why did you do X" conversations for each of the six projects.

Organized by difficulty (Easy → Medium → Hard) rather than strictly by project, so
you can calibrate rehearsal to the interview stage you're prepping for. Each
difficulty section keeps per-project subsections for navigability.

---

## Easy

Definitional, single-concept, "what does this do" style questions — the baseline
you should be able to answer instantly and without hesitation for every project.

### Tabular ML Pipeline (Project 01)

#### Q: Walk me through what `01-tabular-ml-pipeline` does end to end.
`generate_data.py` synthesizes a 5,000-row customer-churn dataset with a fixed
`np.random.default_rng(42)` seed: numeric features (`age`, `tenure_months`,
`monthly_charge`, `total_charge`, `num_support_calls`) and categoricals
(`contract_type`, `payment_method`, `internet_service`, `tech_support`), with churn
generated from a logistic latent-propensity function thresholded at the 85th
percentile to hit a ~15% positive rate, plus injected missingness. `train.py` fits a
leakage-safe `sklearn.Pipeline` (impute/scale/one-hot inside a `ColumnTransformer`,
fit only on the train split) and compares `LogisticRegression` against
`XGBClassifier` via 5-fold `StratifiedKFold` cross-validated ROC-AUC, saving the
winner as `model.joblib`. `evaluate.py` reports held-out precision/recall/F1/ROC-AUC
and a confusion matrix. `train_dl.py` trains a PyTorch `TabularMLP` (categorical
embeddings + numeric features) with an 8-trial random hyperparameter search and
early stopping, then reruns the winning config across 5-fold CV for a confidence
interval. `optimize.py` profiles data-loading vs. compute time and compares fp32 vs.
AMP training. `serve.py` exposes the saved sklearn pipeline behind a FastAPI
`/score` endpoint plus a chunked batch-scoring CLI path.

#### Q: What exactly is `TabularMLP`'s architecture in `train_dl.py`?
It's an `nn.Module` that builds one `nn.Embedding(card + 1, embed_dim)` per
categorical column (the `+1` reserves an out-of-vocabulary bucket for unseen
categories at inference), concatenates all embeddings with the scaled numeric
features, then runs that through `Linear(in_dim, hidden) -> ReLU -> BatchNorm1d ->
Dropout -> Linear(hidden, hidden//2) -> ReLU -> Dropout -> Linear(hidden//2, 1)`,
returning a single logit (squeezed). Loss is `BCEWithLogitsLoss` with a `pos_weight`
computed from the training batch's class ratio to compensate for the ~15% positive
rate — the DL analogue of `class_weight="balanced"` used in the sklearn baseline's
`LogisticRegression`.

#### Q: What does `optimize.py` actually measure, and what's the catch with its result?
It imports `TabularMLP` and the preprocessing functions from `train_dl.py`, builds
one `DataLoader`, and for `use_amp in (False, True)` times, per epoch, cumulative
data-loading time (`time_epoch` sums the gap between "batch ready" and "batch
loaded") versus compute time (forward+backward+optimizer step), using
`torch.autocast` and `torch.amp.GradScaler` for the AMP path. The catch, stated
explicitly in the script's own printed note: AMP only accelerates CUDA GPU matmuls,
so on a CPU-only laptop (the default dev environment for this project) fp32 and AMP
timings will be nearly identical — the script prints that caveat rather than
claiming a fake speedup, modeling the interview behavior of stating profiling
limitations honestly rather than fabricating a result.

### RAG Pipeline (Project 02)

#### Q: Describe the full retrieval pipeline in `02-rag-pipeline`.
`build_index.py` globs `docs/*.md`, passes each document's raw text to
`chunk_text()` in `chunking.py`, embeds every resulting chunk with
`embeddings.py`, and persists the vectors as a plain `numpy` `.npy` file plus a
`metadata.json` (text, source filename, chunk_id) — no vector database required.
`query.py` is the CLI entry point: it calls `retrieve()` in `retrieval.py`, which
does dense cosine-similarity search over the top 20 candidates and then a lexical
rerank down to the top 5, then passes those chunks to `generate()` in
`generate.py`, which builds a citation-forcing prompt and dispatches to one of
three backends (`extractive`, `openai`, `anthropic`) selected via `--llm`.

#### Q: What embedding model and vector index does this project use?
`embeddings.py` wraps `sentence-transformers`' `all-MiniLM-L6-v2` model (loaded
lazily into a module-level `_MODEL` singleton) and calls
`.encode(texts, normalize_embeddings=True)`, so embeddings are pre-normalized unit
vectors. There is no dedicated vector database — `build_index.py`'s README notes
FAISS is used if installed, otherwise it falls back to a brute-force numpy matrix;
in the actual code path shown, `retrieval.py`'s `dense_search()` loads
`vectors.npy` and computes `vectors @ query_vec` directly (a dot product, which
equals cosine similarity because both sides are L2-normalized), then
`np.argsort(-scores)[:top_k]` for the top-k. This is a deliberate simplicity
choice: brute-force cosine search over a small markdown corpus needs no ANN index.

### LLM Fine-Tuning with LoRA (Project 03)

#### Q: Walk me through `03-llm-finetuning` end to end.
`prepare_data.py` writes a small, deterministic, hand-written 12-example
instruction/response dataset to `data/instructions.jsonl`, formatting each example
with the template `"### Instruction:\n{instruction}\n\n### Response:\n{response}"`.
`train_lora.py` loads a base causal LM (`sshleifer/tiny-gpt2` by default) and its
tokenizer via `transformers`, wraps the model with a `peft.LoraConfig`
(`r=8`, `lora_alpha=16`, `lora_dropout=0.05`, `bias="none"`, `task_type="CAUSAL_LM"`,
`target_modules=["c_attn"]` for GPT-2 architectures) via `get_peft_model()`,
tokenizes the JSONL dataset (`max_length=128`, padded), and trains with
`transformers.Trainer` using a `DataCollatorForLanguageModeling(mlm=False)`,
saving only the adapter weights to `./adapter`. `compare.py` then loads the base
model and a `PeftModel.from_pretrained(base, "adapter")`-wrapped fine-tuned model,
generates greedily (`do_sample=False`) on three held-out prompts, and prints both
outputs side by side.

#### Q: Why does `train_lora.py` default to `sshleifer/tiny-gpt2` instead of a real instruction-following model?
The stated reason (README and code comments) is that `tiny-gpt2` (~2M parameters)
lets the entire training loop — data loading, LoRA injection, `Trainer.train()`,
adapter save — complete as a correctness smoke test on CPU in well under a minute.
This project's point is to exercise the fine-tuning *mechanics* correctly, not
produce a good model. The `--model` CLI flag (e.g. `--model
Qwen/Qwen2.5-0.5B-Instruct --epochs 3`) swaps in a real base model for an actually
useful fine-tune, at the cost of needing a GPU and materially more time — the code
path is identical either way, only the model name and epoch count change.

#### Q: Why is only the adapter saved, not the full fine-tuned model?
`model.save_pretrained(args.output_dir)` on a `peft`-wrapped model saves only the
LoRA adapter's A/B matrices — a few MB — not a full copy of the (unchanged) base
model's weights. This is the standard PEFT deployment pattern: at inference time
(`compare.py`) you load the same base model fresh and then apply
`PeftModel.from_pretrained(base_model, adapter_dir)` to attach the adapter on top,
rather than shipping and storing a full duplicate model checkpoint per fine-tune.
This matters at scale — you can host one frozen base model and swap cheap adapters
per task/customer instead of maintaining N full model copies.

### Multi-Agent A2A/MCP System (Project 04)

#### Q: What does `run_pipeline.py` actually orchestrate?
It's the driver process. It spawns three independent OS subprocesses
(`research_agent.py`, `analysis_agent.py`, `report_agent.py` on ports 8001/8002/8003
respectively) via `subprocess.Popen`, polls each one's
`/.well-known/agent-card.json` endpoint with `wait_for()` until all three respond
`200`, then constructs an A2A `Task` (with a `goal` string and an `A2AMessage`
containing one text `MessagePart`) and calls `send_task("http://127.0.0.1:8001",
initial_task)` to hand it to the Research Agent. It awaits the final completed
`Task` returned up the chain, prints its text parts, and in a `finally` block
terminates all three subprocesses. The actual multi-step reasoning (Research →
Analysis → Report) happens entirely inside the three agent processes, not in the
driver.

### Live-Coding Interview Template (Project 05)

#### Q: What is `05-interview-template` for, and how is it different from Project 01?
Project 01 (`01-tabular-ml-pipeline`) is a finished, runnable reference
implementation against a fixed synthetic churn dataset. `05-interview-template` is
explicitly the opposite: a fill-in-the-blanks skeleton meant to be retyped live
during an actual interview, with `# TODO(interview)` markers and inline "say this
out loud" narration comments, generic enough to point at whatever dataset an
interviewer hands you. It covers the same six-phase shape (feature engineering →
DL model → HPO → CV → performance optimization → production scaling) but the point
is rehearsal and narration fluency, not a polished artifact.

#### Q: How is the template's directory structured and how would you use it in a real interview?
`template/` has one file per phase run in numeric order:
`00_problem_framing.md` (no code — clarify target, label window, metric before
typing), `01_feature_engineering.py`, `02_baseline_model.py`,
`03_deep_learning_model.py`, `04_hyperparameter_search.py`,
`05_cross_validation.py`, `06_performance_and_scaling.py`. Each file also runs
standalone against a bundled sklearn toy dataset (breast-cancer for classification)
so you can sanity-check the scaffold before real interview data shows up. There's
also `template-regression/run_regression_drill.py` — the same six phases collapsed
into one script against California-housing, for drilling the regression variant
(MSE/Huber loss, MAE/RMSE metric, plain `KFold`, no sigmoid/calibration step) once
the classification version is automatic. The README's 60-minute time-boxed script
says: run `01` and `03` live in full, fit `02` fast without tuning, *narrate only*
(don't execute) `04`/`05`, and close with `06`'s checklist statement.

### MCP Agent System (Project 09)

#### Q: What MCP tools does `server_weather.py` expose, and what's notable about how it's implemented?
Two tools via `@mcp.tool()` decorators on a `FastMCP("Weather & Search Server")`
instance: `get_weather(latitude: float, longitude: float) -> str`, which calls the
real, free Open-Meteo API (`https://api.open-meteo.com/v1/forecast`) via `httpx.get`
with a 5-second timeout and returns temperature/windspeed/time as JSON — this is
the one part of the project that hits a genuine external API, not a mock; and
`search_web(query: str) -> str`, which is explicitly a mocked/simulated web search
(hardcoded two-result JSON) with a docstring noting that a production system would
call Tavily, Brave, or Google's Search API instead. The server runs via
`mcp.run()`, defaulting to stdio transport when spawned as a subprocess.

#### Q: What MCP tools does `server_database.py` expose, and what security boundary does it enforce?
`execute_sql(query: str) -> str` and `list_tables() -> str`, backed by a local
SQLite database (`mock_database.db`) seeded on module load with a `users` table
(id, name, role, active, department) via `init_db()`. The security boundary — the
file's own docstring calls this out as the point of the example — is that
`execute_sql` uppercases and checks the query starts with `SELECT`, rejecting
anything else with `{"error": "Security policy violation: Only SELECT queries are
permitted."}`, and also rejects queries containing a `;` that isn't at the very
end (a crude multi-statement-injection guard). `orchestrator.py`'s
`run_simulated_flow()` deliberately demonstrates this by having a "Malicious Agent"
attempt `DROP TABLE users;` and showing the server reject it — a concrete,
runnable illustration of MCP servers enforcing their own security policy
independent of whatever the calling LLM decides to request.

### General Portfolio/Project Discussion

#### Q: Across all six projects, what's the throughline you'd tell an interviewer about why you built these?
Every study plan and interview-prep doc in this repo references milestones like
"build a tabular ML pipeline," "build a RAG system," "fine-tune an LLM," and "build
a multi-agent/MCP system" — but referencing them isn't the same as having built
one. These six projects are runnable, inspectable implementations of exactly those
milestones, each sized to run end-to-end on a laptop in minutes, so the answer to
"have you actually built X" is a specific file and a specific design decision, not
a description of what X generally involves.

#### Q: What is the difference between the six numbered projects and the reference implementations — which ones are "finished artifacts" vs. "templates"?
Projects 01, 02, 03, 04, and 09 are finished, runnable reference implementations
against fixed datasets/scenarios. Project 05 is the one explicit template —
deliberately unfinished, with `# TODO(interview)` markers, meant to be retyped live
rather than run as-is. Knowing this distinction matters because an interviewer
asking "show me your RAG project" expects Project 02's finished behavior, while
"let's pair on a tabular problem" is exactly what Project 05 is rehearsal for.

#### Q: What does "leakage-safe" mean in the context of these projects, in one sentence?
It means every preprocessing step that learns statistics from data (imputation
values, scaling mean/variance, one-hot category vocabularies) is fit exclusively on
the training split and only ever applied — never refit — to validation/test splits,
so no information from held-out rows influences how the training data itself gets
transformed.

---

## Medium

Questions that require connecting two or more concepts, explaining a design
tradeoff actually made in the code, or applying a concept to a scenario.

### Tabular ML Pipeline (Project 01)

#### Q: What hyperparameter optimization method does `train_dl.py` actually use, and why not Optuna or grid search?
`random_search()` in `train_dl.py` is a hand-rolled random search: for each of
`n_trials=8` trials it samples `lr` and `weight_decay` log-uniformly, `hidden`
from `{32, 64, 128}`, `embed_dim` from `{4, 8, 16}`, `dropout` uniformly in
`[0.1, 0.5]`, and `batch_size` from `{64, 128, 256}` using a seeded
`np.random.default_rng`, training a full model per trial and picking the config
with best validation ROC-AUC. The code's own comment cites Bergstra & Bengio:
random search dominates grid search once you have more than 2-3 hyperparameters,
because not every dimension matters equally and grid search wastes trials on
combinations of unimportant parameters. Optuna/TPE is explicitly reserved for the
*template* project (05) as the "next step up" — this reference implementation
keeps the dependency surface minimal.

#### Q: Why does cross-validation look different between `train.py` and `train_dl.py`?
`train.py` uses plain `StratifiedKFold(n_splits=5)` cross-validation directly for
model selection because sklearn pipelines are cheap to refit. `train_dl.py`
deliberately does *not* run K full neural-network training loops per HPO trial —
that would be `n_trials × k_folds` full trainings, too expensive — so HPO uses a
single held-out validation split, and K-fold CV is only run once, afterward, with
the already-chosen winning config, purely to report a confidence interval on the
final score rather than a single lucky/unlucky split. This split of concerns
(cheap search on one split, expensive confirmation via K-fold) is called out
explicitly in an inline comment in `train_dl.py`.

#### Q: How does `serve.py` structure the "scaling in production" phase?
It's a FastAPI app (`uvicorn serve:app --workers 4`) with a module-level `_model =
None` that's lazily loaded once per worker process inside `get_model()` — not
reloaded per request — so it's horizontally scalable behind a load balancer with no
per-request state. `POST /score` takes a Pydantic `ScoreRequest` matching the raw
feature schema, wraps it in a one-row DataFrame, and calls `model.predict_proba`
using the *same* saved pipeline artifact used offline (no re-implemented
preprocessing, avoiding training-serving skew). `GET /health` is a liveness probe.
Separately, `batch_score()` reads the input CSV in `chunksize=10_000` pieces via
`pd.read_csv(..., chunksize=...)` and appends scored chunks to the output file, so
memory stays bounded regardless of input file size — this is the nightly
batch-scoring path, same model artifact as the online path.

#### Q: Why is the churn dataset generation itself part of "feature engineering," not just a data-loading detail?
`generate_data.py` deliberately builds in realistic quirks that force feature-engineering
decisions downstream: `total_charge` and `tech_support` get MAR-style missingness
(more likely missing for customers with `tenure_months < 6`), `monthly_charge` is a
skewed/clipped normal, and categorical cardinality varies (3-4 categories each). This
means `train.py`'s `ColumnTransformer` isn't decorative — the median imputation on
numeric columns and most-frequent imputation + one-hot on categoricals are answering
real missing-data and skew problems the generator introduced on purpose, mirroring
what a live-coding interviewer's dataset would actually look like.

#### Q: What's a limitation of this pipeline as built?
The DL hyperparameter search space is small (8 trials, 6 dimensions) and would
under-explore a real search space; there's no persistent experiment tracking (no
MLflow/W&B, just printed trial logs); `model_dl.pt` saves architecture + config but
the comment in `train_dl.py` notes the final weights should be retrained before
serving (the code doesn't currently ship the final fit's actual trained weights
into that checkpoint in a production-safe way); and `serve.py`'s FastAPI app has no
auth, rate limiting, request logging, or drift monitoring — those are called out as
things to "talk through verbally" rather than implemented, since the file is
explicitly scoped to be the serving code itself, not the surrounding system.

#### Q: Why use `StratifiedKFold` rather than plain `KFold` here?
Because churn is imbalanced by construction (~15% positive rate, enforced via the
`np.quantile(prob, 0.85)` threshold in `generate_data.py`). Plain `KFold` can
produce folds with wildly different positive rates by chance, making CV scores
noisy and occasionally producing a fold with almost no positive examples;
`StratifiedKFold` (used in both `train.py` and `train_dl.py`) preserves the ~15%
class ratio in every fold, which is also why ROC-AUC and PR-AUC are the reported
metrics instead of accuracy — accuracy on a 15%-positive dataset is trivially
gameable by a majority-class classifier.

### RAG Pipeline (Project 02)

#### Q: What exact chunking strategy does `chunking.py` implement, and why not naive fixed-size splitting?
It's sentence-boundary-snapped fixed-size chunking with overlap: `split_sentences()`
splits on a regex `(?<=[.!?])\s+`, then `chunk_text(text, source, chunk_size=500,
overlap=100)` greedily packs sentences into a running buffer until adding the next
sentence would exceed 500 characters, at which point it closes the current chunk
and starts the next one by carrying backward as many trailing sentences as fit
within the 100-character overlap budget. The stated reason for snapping to sentence
boundaries rather than slicing raw characters: naive fixed-size chunking can cut a
sentence in half, degrading both the embedding (no longer one coherent idea) and
the readability of the eventual citation. Overlap exists so an answer spanning a
chunk boundary in the source document isn't silently dropped from both neighboring
chunks' context.

#### Q: How does the "reranking" stage actually work — is it a real cross-encoder?
No — `lexical_rerank()` in `retrieval.py` is a cheap stand-in, not a trained
cross-encoder model. It extracts a `_keywords()` set from the query (regex
word-tokenize, drop stopwords and words ≤2 chars), does the same for each candidate
chunk's text, computes `overlap = |query_kw ∩ chunk_kw| / |query_kw|`, and blends it
with the existing dense score as `rerank_score = 0.7 * dense_score + 0.3 *
overlap`, then sorts and truncates to the top 5. The docstring is explicit that this
corrects cases where dense cosine similarity surfaces a topically-close but
keyword-irrelevant chunk over a more lexically precise match, without needing an
extra cross-encoder model's weights or inference cost.

#### Q: How are hallucinations prevented, especially in the default (no API key) mode?
Two mechanisms. First, `build_prompt()` in `generate.py` constructs a prompt that
explicitly instructs the LLM to "Answer the question using ONLY the context below,"
requires a `[source: file.md#chunk_id]` citation tag on every claim, and instructs
the model to say explicitly if the context doesn't contain the answer — this is a
prompting-level mitigation, not a guarantee, for the `openai`/`anthropic` backends.
Second, and structurally stronger: the default `extractive` backend
(`_extractive_answer()`) never calls an LLM at all — it tokenizes the query into a
keyword set, splits every retrieved chunk into sentences, and returns the single
existing sentence with the highest keyword overlap with the query, verbatim, with
its source tag appended. Because the returned text is always a substring that
existed in a retrieved chunk, hallucination in this mode is structurally
impossible, which the README calls out as useful for evaluating retrieval quality
in isolation from generation quality.

#### Q: Why default to an extractive fallback instead of always calling an LLM?
So the whole pipeline runs offline with zero setup and zero API keys — anyone
cloning the repo can run `build_index.py` then `query.py "..."` immediately.
`generate.py`'s `generate(query, chunks, backend="extractive")` is the default in
`query.py`'s argparse; `--llm openai` or `--llm anthropic` are opt-in and raise a
`RuntimeError` if the corresponding `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` isn't set,
so the failure mode when someone forgets to set a key is an explicit error, not a
silent fallback.

#### Q: Why is retrieval two-stage (dense top-20, then rerank to top-5) instead of retrieving top-5 directly?
Dense cosine similarity alone at `top_k=5` risks missing a lexically precise match
that scores slightly lower on pure embedding similarity than a topically related
but less exact chunk. By first casting a wider net (`dense_search(..., top_k=max(20,
top_k*4))` in `retrieve()`) and then reranking that larger candidate pool with the
lexical-overlap signal before truncating to the requested `top_k`, the pipeline
gets a chance to promote a keyword-exact chunk that dense search alone ranked, say,
12th, into the final top 5 — the classic "retrieve broad, rerank precise" pattern
approximated here without a cross-encoder's cost.

#### Q: What does `query.py` print besides the answer, and why does that matter for debugging?
It prints each retrieved chunk's `source#chunk_id` and its `rerank_score` before
printing the generated answer. This makes retrieval quality inspectable
independently of generation quality — if the final answer is wrong, you can see
immediately whether the failure was retrieval (wrong chunks retrieved) or
generation (right chunks, but the answer paraphrases/misreads them), which is
exactly the diagnostic split the extractive-fallback design also supports.

### LLM Fine-Tuning with LoRA (Project 03)

#### Q: What exact LoRA configuration is used, and why those specific values?
`LoraConfig(r=8, lora_alpha=r*2=16, lora_dropout=0.05, bias="none",
task_type="CAUSAL_LM", target_modules=["c_attn"])`. Rank 8 is a common small-model
default that keeps trainable parameters to a tiny fraction of the base model while
still giving the adapter enough capacity to shift behavior; `lora_alpha=2r` is a
standard scaling convention (the effective update magnitude is `alpha/r` times the
low-rank product, so `alpha=2r` gives a scaling factor of 2, a common choice when
rank is small). `target_modules=["c_attn"]` targets GPT-2's fused
query/key/value attention projection matrix specifically — GPT-2's architecture
uses a single fused `c_attn` layer rather than separate `q_proj`/`v_proj` matrices
like Llama-style models, which is why the code conditionally sets
`target_modules=["c_attn"] if "gpt2" in args.model.lower() else None` — passing
`None` lets `peft` auto-detect target modules for other architectures like Qwen.
`bias="none"` means bias terms stay frozen (only the LoRA A/B matrices train).

#### Q: How does `compare.py` evaluate whether fine-tuning "worked"?
It's a qualitative, not quantitative, comparison — there's no held-out loss or
accuracy metric computed. `generate()` runs greedy decoding (`do_sample=False`,
so deterministic output) on the same three fixed prompts (capital of France, a
different arithmetic question than train, name a primary color) through both
`base_model` and `tuned_model`, decoding only the newly generated tokens
(`output[0][inputs["input_ids"].shape[1]:]`) and printing base vs. tuned outputs
side by side per prompt. This is explicitly the deliverable, per the README:
"exists specifically to make the effect of fine-tuning visible, not just to prove
training ran" — with a smoke-test-sized tiny-gpt2 and only 12 training examples,
a numeric eval metric would be noisy and less convincing than seeing the raw text
change.

#### Q: What's a limitation of this fine-tuning setup?
The training set is only 12 hand-written examples — nowhere near enough to
generalize, and with `tiny-gpt2`'s ~2M parameters the model has very limited
capacity to begin with, so `compare.py`'s side-by-side outputs mostly demonstrate
memorization/overfitting on the exact training prompts rather than genuine
instruction-following generalization. There's also no validation split, no
early stopping, no eval-loss tracking during `Trainer.train()` (only
`logging_steps=1` for loss printing), and no quantitative before/after metric
(e.g. perplexity on held-out prompts) — only qualitative eyeballing via
`compare.py`. Scaling to `Qwen2.5-0.5B-Instruct` with a real dataset would need a
proper train/val split and a quantitative eval harness.

#### Q: Why LoRA instead of full fine-tuning here?
Only the low-rank adapter matrices (`r=8`) are trainable — the base model's weights
are loaded and kept entirely frozen inside `get_peft_model()`. `model.
print_trainable_parameters()` (called right after wrapping) would show a trainable
parameter count that's a small fraction of the base model's total, drastically
cutting both the memory footprint (no need for full-model optimizer states/gradients)
and the artifact size at save time. This is exactly why the project can plausibly
scale from a 2M-parameter smoke-test model up to `Qwen2.5-0.5B-Instruct` with the
same code path and only a GPU as the added requirement, rather than needing
multi-GPU infrastructure for full fine-tuning.

### Multi-Agent A2A/MCP System (Project 04)

#### Q: What's the difference between MCP and A2A in this project, concretely?
MCP is the agent-to-*tool* protocol: each agent owns its own restricted MCP
client(s) — Research Agent's client only talks to `notes-server`, Analysis Agent's
only to `stats-server`, Report Agent's to both `filesystem-server` and
`notes-server` — spawned as `stdio` subprocesses via `agents/mcp_client.py`, doing
the MCP handshake and calling `list_tools()`/`call_tool()` for fine-grained,
stateless actions like `search_notes(keyword)`. A2A is the agent-to-*agent*
protocol: coarse-grained, cross-process handoffs of an entire unit of work,
implemented in `agents/a2a_protocol.py` as `AgentCard` (published at
`/.well-known/agent-card.json`), `Task` (lifecycle `submitted → working →
completed/failed`), and `Message`/`Part` (typed content, `text` or `data`), sent
over real HTTP `POST /tasks` calls via `send_task()`. Each agent is a separate OS
process running both a FastAPI server (the A2A side) and its own MCP client(s) (the
tool side) — the project deliberately keeps these two protocols and layers
distinct rather than collapsing everything into one client with universal tool
access.

#### Q: Why does each agent get its own restricted MCP client instead of one shared client with access to every server?
This is called the "Hierarchical Multi-Client" pattern in the README. Research
Agent's MCP client is wired only to `notes-server`, Analysis Agent's only to
`stats-server`, and Report Agent's to both `filesystem-server` (the only server
with write access) and `notes-server`. If every agent shared one client with
access to all three servers, the Research Agent could accidentally (or if
compromised, deliberately) call `write_report` and overwrite output files it has
no business touching. Restricting each agent's MCP client to exactly the servers
its job requires bounds the blast radius of a bug or a prompt-injection-style
failure to that agent's own narrow tool surface.

#### Q: What does the A2A `Task` lifecycle and Agent Card actually look like in code?
`agents/a2a_protocol.py` defines `AgentCard(name, description, url, skills:
list[AgentSkill])` served via `GET /.well-known/agent-card.json`, and `Task(id,
status: Literal["submitted","working","completed","failed"], message: A2AMessage,
goal: str)`, where `A2AMessage` wraps a `list[MessagePart]` and each `MessagePart`
has a `kind: Literal["text","data"]` plus arbitrary `content` — so a single message
can carry both natural-language prose and machine-readable structured data (e.g.
the Analysis Agent's computed trend numbers) in the same envelope. `make_agent_app()`
builds the FastAPI app every agent runs: `POST /tasks` sets `task.status = "working"`,
calls the agent's injected `handle_task(task) -> Task` function, and returns the
result. `send_task()` is the client-side call used to hand a task to the next agent.

#### Q: Why implement a trimmed version of the real A2A spec instead of using an existing library?
The docstring in `a2a_protocol.py` states this directly: the real A2A spec (Google,
2025) uses JSON-RPC/SSE transport and a fuller feature set; this project implements
the same three core concepts (Agent Cards, Tasks, Messages/Parts) over plain
JSON-over-HTTP, "trimmed to what's needed for one linear handoff chain." The goal
is pedagogical — to make the wire shape of A2A concrete and inspectable (you can
literally `curl` an Agent Card while the pipeline runs) — not to be a
spec-complete, production-grade A2A implementation.

#### Q: Why does the Report Agent need two MCP clients (filesystem-server and notes-server) instead of one?
Its job requires both write access (to persist the final report via
`write_report` on `filesystem-server`) and read access back into the raw notes
corpus (to re-check an exact quote while writing the report, via `notes-server`'s
tools) — two genuinely different capabilities. Rather than merging both into a
single do-everything server, the project keeps `filesystem-server` as "the only
server with write access" and gives the Report Agent a second, separate MCP client
pointed at `notes-server` alongside its `filesystem-server` client. This
demonstrates the flip side of the Hierarchical Multi-Client pattern: not just "one
client, one server, one agent" restriction, but also "one agent, multiple clients,
each scoped to a different server" when a single agent's job genuinely spans two
tool domains.

### Live-Coding Interview Template (Project 05)

#### Q: What HPO approach does `04_hyperparameter_search.py` demonstrate, and how does it differ from Project 01's approach?
It uses `sklearn.model_selection.RandomizedSearchCV` with `scipy.stats.loguniform`
and `randint` distributions over a `GradientBoostingClassifier`'s
`n_estimators`, `learning_rate`, `max_depth`, and a discrete `subsample` list,
`n_iter=N_TRIALS=20`, `cv=3`, scoring `roc_auc`. This is a different (and more
standard/library-driven) random search implementation than Project 01's hand-rolled
`random_search()` loop in `train_dl.py` — the template explicitly narrates that for
the GBT model, `RandomizedSearchCV` is the right tool, while for the deep-learning
model the narration comment says to *mention* Optuna/TPE with pruning as the
Bayesian alternative once trials get expensive enough that random search's
sample-inefficiency starts to matter, rather than actually wiring Optuna in code.

#### Q: What does `06_performance_and_scaling.py` cover that the other project files don't?
It explicitly splits "performance optimization" into two distinct things the
interviewer might mean, and addresses both: model-performance optimization via
`calibrate_model()`, which fits a `CalibratedClassifierCV(model, method="isotonic",
cv=3)` on top of a `GradientBoostingClassifier` and compares Brier score
before/after — relevant when predictions feed a decision threshold, not just when
AUC is the only metric that matters — and compute-performance optimization via
`profile_training_step()` (times a single forward+backward+optimizer step,
synchronizing CUDA if available before timing) and `train_step_with_amp()` (mixed
precision via a `GradScaler`). It closes by printing a production
serving/monitoring checklist as text output rather than standing up real infra,
since there's no deployment target inside a live-coding session.

#### Q: What's the "one preprocessing contract" every file in this template shares, and why does it matter?
Every phase enforces: split before fitting any transform; all
imputation/scaling/encoding lives inside one `sklearn.Pipeline`/`ColumnTransformer`
fit only on the training fold; and any feature using information from after the
label's observation window gets called out and dropped, unprompted, even if the
interviewer didn't ask about leakage. This is the same leakage-safety discipline as
Project 01, made explicit as a reusable contract here because the whole point of a
template is that it has to hold regardless of which dataset gets dropped in live —
you can't rely on a fixed, pre-vetted dataset (like Project 01's synthetic churn
CSV) to have already ruled out leakage risks for you.

#### Q: Why does the regression drill exist as a separate, consolidated single-file script instead of six files like the classification template?
The README frames it as a second drill specifically so practice isn't narrow to
one task type — once the six-phase shape feels automatic for binary
classification, `template-regression/run_regression_drill.py` forces the same
shape against a different loss (MSE/Huber instead of BCEWithLogitsLoss), metric
(MAE/RMSE instead of ROC-AUC), CV splitter (plain `KFold` instead of
`StratifiedKFold`, since regression targets aren't a class ratio to stratify by),
and no calibration/sigmoid step. Consolidating it into one script rather than six
mirrors how much faster a real second pass through the same shape should be once
you're fluent, and the deltas are called out inline in its own docstring rather
than requiring you to diff six separate files.

### MCP Agent System (Project 09)

#### Q: How does `09-mcp-agent-system` differ from Project 04, and why does the README call it a "design doc/spec"?
Both projects tour MCP, but Project 04 pairs MCP with a real A2A implementation
across three separate OS processes exchanging HTTP tasks. Project 09 has no A2A
layer at all — `orchestrator.py` is a single script acting as one central MCP
*client* that spawns two MCP servers (`server_weather.py`, `server_database.py`)
as `stdio` subprocesses and runs one linear tool-calling loop itself, either via a
real OpenAI function-calling loop (if `OPENAI_API_KEY` is set) or a hardcoded
`run_simulated_flow()` if not. The top-level README explicitly frames Project 09 as
"a broader MCP surface-area tour... as a design doc/spec — see project 04 for the
runnable A2A-protocol counterpart," and its own README's "Variations" section
(HITL approval servers, remote SSE servers, MCP Resources) is written as
prose/patterns to try, not code that's actually implemented in this folder.

#### Q: How does `orchestrator.py` decide whether to run a real LLM loop or a simulated one, and what does each path do?
It checks `os.environ.get("OPENAI_API_KEY")` at the end of
`run_multi_agent_system()`. If present, `run_llm_flow()` builds an OpenAI
function-calling request: it hand-writes JSON-schema `tools` definitions for
`get_weather` and `execute_sql` (the code comments note that in a production system
this schema-to-tool mapping would be generated dynamically from the MCP servers'
own `list_tools()` JSON schemas, not hand-duplicated), sends a user message asking
about Berlin's weather and active engineering-department users, and for each
`tool_call` the model returns, routes it to the matching MCP `ClientSession`
(`weather_session` or `db_session`), appends the tool's JSON result back into the
`messages` list with `role: "tool"`, then makes a second OpenAI call for the
model's final natural-language answer. If no API key is present,
`run_simulated_flow()` hardcodes the same two tool calls (plus the SQL-injection
demonstration) without ever calling an LLM, so the whole MCP wiring is exercisable
and demoable with zero API cost.

#### Q: What is `AsyncExitStack` doing in `orchestrator.py`, and why does connecting to two MCP servers need it?
`async with AsyncExitStack() as stack:` lets the orchestrator open an arbitrary,
dynamically-determined number of async context managers (here: two `stdio_client()`
connections and two `ClientSession()`s, one pair per server) and guarantees they
all get torn down correctly in reverse order on exit, even if an exception occurs
partway through setup. Each `stack.enter_async_context(...)` call both enters the
context manager and registers its cleanup with the stack, which is the standard
pattern for managing a variable number of resources (here exactly two servers, but
the pattern scales to N servers) that all need coordinated async cleanup.

#### Q: Why does `search_web` in `server_weather.py` return mocked data while `get_weather` calls a real API — isn't that inconsistent?
It's a deliberate cost/complexity tradeoff, not an oversight: Open-Meteo's forecast
API is free and keyless, so `get_weather` can hit it directly with zero setup,
demonstrating a genuine "wrap an external REST API as an MCP tool" pattern with no
friction for anyone running the project. A real search API (Tavily, Brave, Google)
requires an API key and a billing relationship, which would break the project's
zero-setup-required design goal, so `search_web` mocks two plausible-looking
results instead — its docstring is explicit about this being a stand-in, so a
reader isn't misled into thinking real search results are being returned.

### General Portfolio/Project Discussion

#### Q: Which of these projects would you pick as your "one end-to-end story you know cold," and why?
Project 01 (`01-tabular-ml-pipeline`) is the strongest candidate for a live-coding
ML interview specifically because it's a complete reference implementation of the
exact six-phase shape (feature engineering → DL model → HPO → CV → performance
optimization → production scaling) that recruiters at companies like EA
(per `07-interview-prep/EA-ml-deep-learning-interview.md`) explicitly test for, and
every phase maps to a real file with a specific technique (leakage-safe
`ColumnTransformer`, `TabularMLP` with categorical embeddings, an 8-trial random
search with early stopping, `StratifiedKFold` CV, AMP profiling, a FastAPI serving
stub) rather than a hand-wave.

#### Q: Every project claims to run "offline" or "with zero API keys" — how true is that, and where are the exceptions?
Mostly true, with two notable exceptions. Project 02's RAG pipeline defaults to
local `sentence-transformers` embeddings and an extractive (non-LLM) generation
fallback, but genuinely calls out to OpenAI or Anthropic if `--llm openai`/
`--llm anthropic` is passed with the corresponding key set. Project 09's
`server_weather.py`'s `get_weather` tool makes a real, keyless HTTP call to the
public Open-Meteo API — so "offline" there really means "no API key required," not
"no network access." Project 04's multi-agent A2A/MCP system is the one that's
genuinely fully offline and keyless end to end: all three MCP servers
(`notes-server`, `stats-server`, `filesystem-server`) operate on local files with
no external network calls at all.

#### Q: Which two projects share the most design DNA, and what's the common pattern?
Projects 01 and 05 share the most: Project 05 (`05-interview-template`) is
explicitly built as the retype-able, generic skeleton behind Project 01's
finished, fixed-dataset reference implementation — same six-phase shape
(feature engineering → DL model → HPO → CV → performance/scaling), same
leakage-safety contract (split before fit, all preprocessing inside one
`Pipeline`/`ColumnTransformer` fit only on train), same categorical-embeddings
MLP pattern for the DL model. The difference is purely packaging: Project 01 is
"here's the finished artifact," Project 05 is "here's how to build that artifact
live, from memory, against whatever dataset you're handed."

---

## Reference: Question Count by Difficulty

- Easy: 13 questions (definitional/walkthrough, one per project plus general framing)
- Medium: 24 questions (design tradeoffs, "why X not Y," scenario application)
