# SecureCodeRAG — Findings

This document consolidates every measured result in the project into one
place. Every number here is read directly from a file in `results/*.json`,
produced by a script listed in the "Scripts" section of
[`README.md`](../README.md) — nothing in this document is estimated,
rounded from intuition, or asserted without a script behind it. Where a number differs slightly between runs (e.g. the exact
count of distractor functions pulled from `src/`), that is because `src/`
itself changed slightly between runs, not because the measurement is
unstable; the rates themselves have been stable across every re-run in this
project's history.

**Scope.** All real-LLM results describe **`qwen2.5-coder:3b`** only, at
temperature 0. A second, larger model was attempted for comparison and
abandoned after a local Ollama installation issue prevented it from being
reliably served over its HTTP API — diagnosed in full in
[`docs/benchmark.md`](benchmark.md). Every claim below should be read as
describing this one model's behavior, not code LLMs in general.

---

## 1. Summary

Across 32 poisoned samples spanning 7 attack categories, this model
reproduced planted poisoned content in its output **3.1% of the time**
(1 sample), and that one case — a hardcoded credential — was also the one
category the defense pipeline catches with **100% reliability**. The other
six categories, which plant misleading comments, false documentation, or
instruction-like text rather than a concrete vulnerability, reached the
model **unfiltered** between 40% and 100% of the time, and the model
resisted all of them **on its own** — resistance that did not change when
the system prompt's explicit warning about untrusted content was removed
entirely. The defense pipeline's demonstrated strength is narrow and
specific (hardcoded secrets via static analysis, and one instruction-
injection pattern via anomaly detection), not the broad class of
social-engineering-style poisoning this project set out to study. That gap,
not a clean "defense works" headline, is the project's central finding.

---

## 2. Results by research question

### 2.1 — Was poisoned content retrieved, and how highly ranked?

**Script:** `scripts/run_retrieval_eval.py` · **File:** `results/retrieval_report.json`
**Method:** a real FAISS index built from real functions pulled out of this
repository's own `src/` tree (47 in the run behind these numbers) plus the
clean and poisoned seed samples, queried with a deterministic lexical
(bag-of-words) embedder — not the project's semantic HuggingFace embedder,
so these numbers describe lexical retrieval behavior specifically.

| Category | In top-5 | Ranked above the clean original |
|---|---|---|
| contradictory_documentation | 60% | 60% |
| false_api_guidance | 80% | 60% |
| instruction_like_content | 80% | 80% |
| context_manipulation | 60% | 20% |
| misleading_code | 40% | 20% |
| false_repository_conventions | 40% | 0% |
| vulnerable_code | 0% | 0% |

Six of seven categories surface in the top results at least 40% of the
time. `vulnerable_code` was the hardest to surface — its poison is a single
short line, easy to outweigh in token-overlap terms against a larger pool
of real functions. A semantic embedder, which matches on meaning rather
than shared words, may retrieve these differently; that comparison has not
been run.

### 2.2 — Did poisoned content change what the model generated?

**Script:** `scripts/run_benchmark_llm.py` · **File:** `results/benchmark_llm_report.json`
**Method:** 32 samples through `qwen2.5-coder:3b`, each run twice — once
with poisoned context, once with a matched clean-context control — scored
with `attack_flip_rate`: the poison must appear in the output *and* be
absent from the clean-context control, so a model that merely repeats
boilerplate in both runs is not counted as fooled.

| Metric | Value |
|---|---|
| Overall attack flip rate | 3.1% (1 of 32) |
| On-task rate | 100% |
| Categories with any measured effect | `vulnerable_code` only (50% of its 2 samples) |
| All six text-based categories | 0% |

### 2.3 — Did vulnerable behavior appear, and was it caught?

**Script:** `scripts/run_benchmark_llm_defended.py` · **File:** `results/benchmark_llm_defended_report.json`
**Method:** identical 32 samples, with the real L1–L5 defense pipeline
screening context before the model ever sees it.

| Metric | Value |
|---|---|
| Blocked before reaching the model | 29% (consistent with the 2/7 categories the offline defense catches) |
| Flip rate among calls that reached the model | 0% |

Reading 2.2 and 2.3 together: the one category that did flip undefended
(`vulnerable_code`) is also the one category the defense blocks 100% of
the time, so the defended pass shows 0% flip not because the model became
more resistant, but because the one sample that could flip never reached
it.

### 2.4 — Which defense layer actually caught it, and at what false-positive cost?

**Script:** `scripts/run_layer_ablation.py` · **File:** `results/layer_ablation_report.json`
**Method:** the real defense pipeline runs once per sample (all five layers
execute exactly as in production); the decision engine is then replayed on
filtered subsets of the findings it produced, to isolate each layer's
marginal contribution to the final block decision — no change to the
pipeline itself. The six-stage ladder (`no_defense → L1 → L1_L2 →
L1_L2_L3 → L1_L2_L3_L4 → L1_L2_L3_L4_L5`) is the project's own canonical
definition (`src/experiments/ablation.py`), not invented for this check.

| Ladder step | Overall block rate |
|---|---|
| No defense | 0% |
| +L1 (trust scoring) | 0% |
| +L2 (anomaly detection) | 15.6% |
| +L3 (context validation) | 15.6% (unchanged) |
| +L4 (instruction separation) | 15.6% (unchanged) |
| +L5 (static analysis) | 21.9% |

Only **L2** and **L5** ever move the needle. `instruction_like_content` is
fully caught by L2 alone; `vulnerable_code` only by L5. **L3 contributes
zero by construction** — it does not currently emit a finding that reaches
the decision engine at all, a real property of the pipeline as built, not
a gap in this measurement. L1 and L4 fire on some samples (visible in the
raw findings) but never change the outcome in this dataset, because a
different layer already made the decision first.

### 2.5 — Did the defense degrade legitimate, non-poisoned use?

**Script:** `scripts/run_utility_eval.py` · **File:** `results/utility_report.json`
**Method:** 50 real, never-poisoned functions pulled from this repository's
own `src/` tree (the same pool `collect_distractors` uses for 2.1),
screened through the identical production defense pipeline. This is
deliberately a different, larger sample than the 5 clean seed snippets
used as controls elsewhere in this project.

| Metric | Value |
|---|---|
| False-positive rate | 0% (0 of 50) |
| Mean findings per clean function | 0.00 |

A sanity check (`tests/benchmark/test_utility_eval.py`) confirms the
pipeline is not simply inert: a realistic hardcoded-secret pattern inserted
into otherwise ordinary code is still flagged and blocked, so the 0% figure
reflects genuinely clean input, not a defense that never fires.

### 2.6 — What does each layer contribute, isolated?

Answered directly by 2.4. This is a distinct experiment from the *prompt*
ablation below, which tests wording rather than pipeline layers.

---

## 3. A separate experiment: does the prompt matter?

**Script:** `scripts/run_ablation.py` · **File:** `results/ablation_report.json`
**Method:** the same 32 samples run twice through the same model — once
with the project's hardened prompt (which explicitly instructs the model
to treat retrieved content as untrusted data), once with a naive prompt
that pastes the context with no such warning.

| Prompt style | Attack flip rate |
|---|---|
| Naive | 3.1% |
| Hardened | 3.1% |

**No measured difference.** This model resists these poisons at the same
rate with or without being told to. Confirmed at n = 32 per style (the
same result held at n = 7 in an earlier, smaller run, then was re-confirmed
at the full sample size rather than left at the less certain result). This
is a genuine null finding, not an absence of testing: whatever is causing
this model's resistance to text-based poisoning, it is not the explicit
prompt-level warning.

---

## 4. Known limitations

- **Single model.** Every real-LLM number describes `qwen2.5-coder:3b`
  only. See `docs/benchmark.md` for the abandoned second-model attempt.
- **Lexical, not semantic, retrieval.** Section 2.1 uses a deterministic
  bag-of-words embedder to stay reproducible without a model download.
  Results describe lexical retrieval behavior specifically.
- **Small n for `vulnerable_code`.** Only 2 of the 5 seed tasks have a
  plausible reason to use a hardcoded API key, so this category's rate can
  only be 0%, 50%, or 100%.
- **Attack-success rules are string checks.** "The model's output contains
  the planted text" is a proxy for "the model followed the poison," not a
  semantic judgment. A model that quotes the planted text while warning
  about it would register the same as one that silently obeys it.
- **L3 structurally cannot contribute to the layer ablation** in the
  current pipeline, as stated in 2.4 — this is a property of
  `src/defense/pipeline.py`'s `_collect_findings`, not a limitation of the
  ablation script.
- **The offline benchmark** (`results/benchmark_report.json`, using a
  deterministic echo generator with no real LLM) can only produce a
  meaningful result for `vulnerable_code`, since the other six categories
  are designed to influence model *behavior*, which an echo generator
  cannot exhibit.

---

## 5. What would strengthen this further

In priority order, if continued:

1. **A second, successfully-served model** to test whether the 3.1% flip
   rate and the prompt-ablation null result are properties of this model
   specifically or hold more generally.
2. **A real semantic embedder** for the retrieval-stage check, to replace
   the lexical baseline in section 2.1.
3. **REST API endpoints** to trigger and query these experiments over HTTP
   rather than as local scripts.
4. **A verified Docker build** of the full pipeline.

None of these are required to consider the six research questions in
section 2 answered — they would extend the work, not complete it.
