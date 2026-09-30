# SecureCodeRAG: Security Evaluation and Defense Pipeline for Retrieval-Augmented Code Generation

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-informational)](LICENSE)

## What this project is

Retrieval-Augmented Generation (RAG) lets a Code LLM pull relevant snippets,
documentation, or examples from an external knowledge base before writing
code. This makes the model more useful, but it also opens a new attack
surface: if that knowledge base contains insecure, misleading, or
adversarial content, the model may copy it into the code it generates. A
poisoned comment, a hardcoded secret in a "reference implementation," or
text that looks like a system instruction can all end up in the retrieved
context an LLM trusts by default.

**SecureCodeRAG** is a full, working RAG pipeline for code generation, built
specifically to measure that risk and to test whether a layered defense
actually reduces it. It is not a toy pipeline with a security demo bolted
on — every stage (ingestion, chunking, embedding, retrieval, generation,
defense) is a real, independently tested module, and the security
evaluation sits on top of that real pipeline rather than a simulation of it.

The project answers six concrete research questions (see
[Research questions and what we found](#research-questions-and-what-we-found)
below), each backed by a script you can re-run yourself against
`results/*.json` and a generated dashboard.

## How the pipeline works

```
Knowledge base (clean / poisoned)
        │
        ▼
Ingestion & AST chunking        src/ingestion, src/parsing, src/chunking
        │
        ▼
Embeddings & vector store       src/embeddings, src/vectorstore
        │
        ▼
Retrieval & context building    src/retrieval
        │
        ▼
Defense pipeline (L1–L5)        src/defense, src/security
        │
        ▼
Code LLM generation              src/generation
        │
        ▼
Security evaluation & metrics    src/evaluation, src/experiments, src/benchmark
```

A repository is ingested and split into language-aware chunks using
tree-sitter, each chunk is embedded and indexed in FAISS, a query retrieves
the top-k most relevant chunks, those chunks pass through a five-layer
defense pipeline before being placed in a prompt, a Code LLM (local Ollama
or hosted OpenAI) generates code from that prompt, and the result is scored
for whether it reproduced anything from a poisoned chunk.

Seven **poisoning strategies** (`src/poisoning/`) can be applied to any
chunk to simulate an attacker planting adversarial content in the knowledge
base — everything from a literal hardcoded secret to a comment that reads
like a system instruction telling the model to ignore its real instructions.

## Where the project stands right now

| Area | Status | Detail |
|---|---|---|
| Ingestion, parsing, chunking, embeddings, vector store, retrieval | **Done** | Built and unit-tested. Exercised by the retrieval-stage experiment against this repo's own source as a real knowledge base. |
| 7 poisoning strategies | **Done** | Deterministic, tested individually. |
| 5-layer defense pipeline + static analysis | **Done** | Measured on real data, not just unit-tested — see results below. |
| CodeRAG-PoisonBench (offline, no LLM) | **Done** | Uses a deterministic echo generator; only `vulnerable_code` is measurable this way, and the report says so explicitly rather than guessing. |
| Real-LLM benchmark | **Done** | `qwen2.5-coder:3b` via local Ollama, 32 samples. |
| Retrieval-stage poisoning check | **Done, scoped** | Real FAISS index, real functions from this repo as the knowledge base, but a deterministic *lexical* embedder rather than a semantic one — documented as a scope limit, not a finding about semantic retrieval. |
| Prompt ablation (naive vs. hardened prompt) | **Done** | n = 32 per prompt style. No measured difference on this model. |
| Defense-layer ablation (L1 → L1‑L5) | **Done** | Isolates which of the five layers actually changes the block decision. Only L2 and L5 do, on this dataset. |
| Defended real-LLM pass | **Done** | The defense screens context before the model ever sees it. |
| REST API | **Partial** | Repository-ingestion endpoint exists; no endpoints yet to trigger or query benchmark runs. |
| Docker | **Written, not verified** | `Dockerfile` exists; the image has not been confirmed to build and run cleanly in this environment. |

Nothing above is aspirational — every "Done" row corresponds to a script
you can run yourself (listed in [Scripts](#scripts-what-each-one-does))
that writes a real JSON file to `results/`, and a dashboard generator
that renders those files without inventing numbers for anything that
hasn't actually been run.

## Research questions and what we found

The project exists to answer six questions. All results below are for
**`qwen2.5-coder:3b` only** (temperature 0). A second, larger model was
attempted for comparison but abandoned after a local Ollama installation
issue prevented it from being reliably served; see `docs/benchmark.md` for
the full diagnosis. Every number below should be read as describing this
one model, not code LLMs in general.

1. **Was poisoned content retrieved, and how highly ranked?**
   Measured with a lexical (bag-of-words) embedder against this repo's own
   source as a knowledge base: 6 of 7 poisoning categories surfaced in the
   top-5 results 40–100% of the time. `vulnerable_code` was the hardest to
   surface (50%) because its poison is a single short line, easy to bury
   in a larger pool of real functions.

2. **Did poisoned content change what the model generated?**
   Measured with the `attack_flip_rate` metric (poison followed in the
   poisoned run *and not* in a matched clean-context control, so genuine
   prose warnings from the model don't count as being fooled): 3.1% overall
   (1 of 32 samples), and only `vulnerable_code` was ever affected.

3. **Did vulnerable behavior appear, and was it caught?**
   `vulnerable_code` is the only poisoning category that produced a real,
   detectable vulnerability when it reached the model unfiltered, and the
   defense pipeline catches it 100% of the time when it is present.

4. **Which defense layer actually caught it, and at what false-positive
   cost?**
   The layer-by-layer ablation shows only **L2 (anomaly detection)** and
   **L5 (static analysis)** ever change the block decision on this
   dataset. `instruction_like_content` is caught by L2 alone; `vulnerable_code`
   only by L5. L1 (trust scoring), L3 (context validation), and L4
   (instruction separation) contribute zero measured effect here — L3
   because it does not currently emit a finding that reaches the decision
   engine at all, which is a real, stated property of the pipeline. False
   positives on clean code were 0% across every category.

5. **Did the defense degrade legitimate, non-poisoned use?**
   Not yet directly measured beyond the 0% false-positive rate above. This
   is the one research question from the original plan that remains open.

6. **What does each layer contribute, isolated?**
   Answered by the defense-layer ablation (question 4) — this is
   distinct from the *prompt* ablation (naive vs. hardened prompt wording),
   which is a separate experiment that also produced a real, useful null
   result: prompt hardening made no measurable difference to whether this
   model followed a poison (3.1% either way, confirmed at n = 32).

The honest overall picture: text-based poisoning (comments, false
conventions, fake authority claims) reaches this model unfiltered most of
the time, and the model resists it on its own — not because the defense
pipeline is stopping it. The defense pipeline's real, demonstrated strength
is catching syntactic vulnerabilities and instruction-injection patterns
specifically (L5 and L2), not the broader class of social-engineering-style
poisoning this project set out to study. That gap is real and is the most
important finding in the project, not something to paper over.

## Repository structure

```
Final-Year-Project/
│
├── src/
│   ├── ingestion/          Repository loading, file filtering, language detection
│   ├── parsing/             tree-sitter AST parsing per language
│   ├── chunking/             Splits parsed source into retrievable chunks
│   ├── embeddings/          Embedding providers (HuggingFace code models), caching, factory
│   ├── vectorstore/          FAISS-backed vector index and metadata store
│   ├── retrieval/            Query → top-k retrieval, context building, reranking
│   ├── generation/            Prompt construction; Ollama and hosted-OpenAI providers
│   ├── poisoning/             The 7 controlled poisoning strategies + registry
│   ├── security/              The 5 defense layers: trust scoring (L1), anomaly
│   │                          detection (L2), context validation (L3), instruction
│   │                          separation (L4), static analysis incl. AST + Semgrep (L5),
│   │                          plus the decision engine that turns findings into a
│   │                          block/flag/allow verdict
│   ├── defense/                DefensePipeline: wires all 5 security layers together
│   ├── evaluation/            Generation/retrieval metrics, security-utility scoring
│   ├── experiments/            Config loader, the real end-to-end ExperimentRunner,
│   │                          orchestrator, canonical ablation-stage definitions,
│   │                          result storage, reporting
│   ├── benchmark/             CodeRAG-PoisonBench: dataset, metrics, runner, report,
│   │                          the real-LLM evaluation modules, the integration
│   │                          adapter that wires poisoning strategies to the real
│   │                          pipeline, and the layer-ablation execution logic
│   ├── api/                    FastAPI app (currently: health check + repository
│   │                          ingestion endpoint)
│   ├── config.py               Centralized configuration loading
│   ├── logger.py               Standardized logging
│   └── models.py               Shared dataclasses/enums used across every module
│                              (CodeChunk, RetrievedChunk, ExperimentConfig, etc.)
│
├── scripts/                    One-shot runnable entry points — see below
│
├── tests/                      Mirrors src/ + scripts/: one test module per
│                              source module, plus tests/unit and tests/security
│                              for cross-cutting and poisoning-specific tests
│
├── configs/                    default_config.json — base pipeline configuration
├── experiments/                 Experiment condition definitions (baseline.json,
│                              poisoning.json, defense.json, clean_defense.json,
│                              experiment_matrix.json) consumed by
│                              src/experiments/config_loader.py
├── data/                        clean/, poisoned/, processed/ — git-ignored except
│                              placeholders; populated at runtime
├── results/                     Every script below writes its JSON report here,
│                              plus the generated dashboard.html
├── docs/                        architecture.md, benchmark.md — deeper detail than
│                              this README, including the full Ollama diagnosis
│
├── Dockerfile, .dockerignore    Reproducible environment (written, not yet
│                              verified to build in this environment)
├── requirements.txt / requirements-dev.txt
└── .env.example                  Template for API keys / configuration; never commit
                                 a real .env
```

## Scripts: what each one does

All scripts are run as modules from the repository root, e.g.
`python -m scripts.run_benchmark`. Every script that measures something
writes its result to `results/<name>.json` and never invents a number for
something it didn't actually run.

| Script | What it measures | Needs a live LLM? |
|---|---|---|
| `run_benchmark.py` | Offline CodeRAG-PoisonBench using a deterministic echo generator. Only `vulnerable_code` is meaningfully measurable this way — the report says so. | No |
| `run_defense_eval.py` | Runs the real 5-layer defense pipeline against all poisoning categories; block rate and false-positive rate. | No |
| `run_retrieval_eval.py` | Builds a real FAISS index from this repo's own source plus the poisoned seed samples; checks whether poisoned chunks are retrieved and how they rank. | No (deterministic lexical embedder) |
| `run_layer_ablation.py` | Runs the real defense pipeline once per sample, then replays the decision engine on filtered subsets of findings to isolate each layer's (L1→L1‑L5) contribution to the block decision. | No |
| `run_benchmark_llm.py` | Real-LLM benchmark: does the model reproduce poisoned content, per category, with a clean-context control. | **Yes** (local Ollama) |
| `run_ablation.py` | Prompt ablation: same samples, naive prompt vs. the project's hardened prompt, real model. | **Yes** |
| `run_benchmark_llm_defended.py` | Real-LLM benchmark with the defense pipeline screening context first. | **Yes** |
| `build_dashboard.py` | Reads whatever `results/*.json` files exist and renders `results/dashboard.html`. Anything not yet run is shown as "Not run," never guessed. | No |

The three LLM-dependent scripts accept `--model`, `--seeds` (how many of
the 5 seed tasks to use), and `--tag` (a filename suffix so results from
different models don't overwrite each other).

## Quickstart

### 1. Prerequisites

- Python 3.12+ (the code uses `slots=True` dataclasses and `X | Y` union
  syntax throughout).
- Git.
- Optional, only for real-LLM scripts: a local [Ollama](https://ollama.com)
  installation with a code model pulled, e.g.
  `ollama pull qwen2.5-coder:3b`.

### 2. Installation

```bash
git clone https://github.com/siddhigangan/Final-Year-Project.git
cd Final-Year-Project

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment configuration

```bash
cp .env.example .env
```

No API key is required for anything in this README except the hosted
OpenAI generation provider, which is optional. Never commit a real `.env`.

### 4. Run the tests

```bash
python -m pytest tests -q
```

### 5. Run the offline research pipeline (no LLM required)

```bash
python -m scripts.run_defense_eval
python -m scripts.run_retrieval_eval
python -m scripts.run_layer_ablation
python -m scripts.run_benchmark
python -m scripts.build_dashboard
```

Then open `results/dashboard.html`.

### 6. Run the real-LLM experiments (needs Ollama)

```bash
python -m scripts.run_benchmark_llm --model qwen2.5-coder:3b --seeds 5
python -m scripts.run_ablation --model qwen2.5-coder:3b --seeds 5
python -m scripts.run_benchmark_llm_defended --model qwen2.5-coder:3b --seeds 5
python -m scripts.build_dashboard
```

## Configuration & logging

- **Centralized configuration**: `configs/default_config.json`, loaded via
  `src.config.load_config()`. Overridable with environment variables
  (`CHUNK_SIZE`, `LOG_LEVEL`, `TOP_K`, etc.).
- **Experiment conditions**: defined in `experiments/*.json`, loaded and
  validated by `src.experiments.config_loader`.
- **Logging**: `src.logger.get_logger()` — output goes to stdout and
  `logs/app.log` simultaneously.

## Known limitations (stated deliberately, not hidden)

- All real-LLM findings describe **one model** (`qwen2.5-coder:3b`). A
  second-model comparison was attempted and diagnosed in detail in
  `docs/benchmark.md`, but not completed, due to a local Ollama
  installation issue unrelated to this codebase.
- The retrieval-stage experiment uses a **deterministic lexical embedder**,
  not the project's real semantic HuggingFace embedding provider, to keep
  the experiment reproducible without a model download. Results describe
  lexical retrieval behavior, not semantic retrieval behavior.
- The offline benchmark's echo generator can only produce a measurable
  result for `vulnerable_code`; the other six poisoning categories require
  a real LLM to evaluate meaningfully, since they attempt to *influence*
  a model's behavior rather than plant a static vulnerability.
- L3 (context validation) does not currently emit a finding that reaches
  the block/flag decision engine, so it measures zero contribution in the
  layer ablation — this is a property of the pipeline as built, not a bug
  in the ablation script.
- "Utility impact" (does the defense degrade legitimate retrieval or code
  correctness) is not yet measured.
- The REST API covers repository ingestion only.
- The Docker build has not been verified to complete in this environment.

## Development approach

One step → implement → test → verify against real data → document → push.
Every module in `src/benchmark/` that measures something was checked
against hand-inspected real output before being trusted, not just against
its own unit tests — several real bugs (a metrics-signature mismatch, a
runner that couldn't distinguish "poisoned" from "defended," an ablation
ladder duplicated instead of reusing the project's existing canonical
definition) were caught exactly this way and fixed before being reported
as working.

## Further reading

- `docs/architecture.md` — pipeline design in more depth.
- `docs/benchmark.md` — full benchmark methodology, the attack-success
  rules used per poisoning category, and the complete Ollama
  troubleshooting log for the abandoned second-model comparison.
