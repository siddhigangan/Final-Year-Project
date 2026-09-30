# SecureCodeRAG — Complete Implementation Blueprint

## Security Evaluation and Defense Pipeline for Retrieval-Augmented Code Generation

> **Purpose of this document:** This is the master engineering blueprint for taking the existing `Final-Year-Project` repository from its current scaffold/foundation state to the complete SecureCodeRAG final-year project.
>
> It is intended to be used as the project-level reference while implementing the system phase by phase.
>
> It combines the supplied implementation audit, the project architecture already established, and the free/local AI requirement:
>
> **No paid AI APIs.**
>
> Primary model stack:
>
> - **Hugging Face local model** for code embeddings
> - **Ollama local model** for code generation
> - **FAISS** for vector search
> - **Tree-sitter** for code parsing / AST-aware chunking
> - **FastAPI** for the API
> - **Semgrep + custom AST/security analysis** for security validation
> - **Docker** for reproducible application deployment

---

# 1. Project Overview

## 1.1 What is SecureCodeRAG?

SecureCodeRAG is a controlled research system for studying how **knowledge-base poisoning can affect Retrieval-Augmented Code Generation (Code-RAG)** and for evaluating whether multiple defense layers can reduce that risk without unnecessarily damaging useful retrieval and generated-code quality.

The project is not simply a coding chatbot.

The research flow is:

```text
Authorized / Synthetic Code
        ↓
Repository Ingestion
        ↓
AST Parsing / Code-Aware Chunking
        ↓
Local Code Embeddings
        ↓
FAISS Vector Store
        ↓
Retrieval
        ↓
Retrieved Context
        ↓
Code LLM
        ↓
Generated Code
        ↓
Security Analysis
        ↓
Evaluation
```

The security research flow introduces controlled poisoning:

```text
Clean Knowledge Base
        ↓
Controlled Poisoning Engine
        ↓
Poisoned Knowledge Base
        ↓
Retrieval
        ↓
Poisoned Context
        ↓
Defense Layers
        ↓
Code LLM
        ↓
Generated Code
        ↓
Security Analysis
        ↓
PRR / ASR / VIR
```

The final research contribution is intended to be a reproducible benchmark and evaluation pipeline, including:

- clean Code-RAG baseline,
- controlled poisoning,
- seven poisoning categories,
- multi-layer defense,
- retrieval/generation/security metrics,
- ablation studies,
- security–utility analysis,
- CodeRAG-PoisonBench,
- reproducible experiment results.

---

# 2. Project Scope

The supplied audit identifies the synopsis as the scope authority and the GitHub repository as the implementation-state authority.

The synopsis requires, at a minimum:

1. Working Code-RAG baseline
2. Authorized/synthetic code/document ingestion
3. Code-aware chunking
4. Code embeddings
5. Vector database
6. Code LLM
7. Software-engineering tasks
8. Controlled poisoning
9. Seven poisoning categories
10. Poison Retrieval Rate (PRR)
11. Attack Success Rate (ASR)
12. Vulnerability Introduction Rate (VIR)
13. Multi-layer defense
14. Static/security analysis
15. Security–utility analysis
16. Multiple retrievers/models/task types where practical
17. Ablation studies
18. CodeRAG-PoisonBench benchmark
19. Security evaluation report

The audit also establishes that the current repository contains the foundation/scaffold but does not yet contain the actual end-to-end RAG/security research pipeline.

---

# 3. Current Repository State

## 3.1 Current tree

```text
Final-Year-Project/
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
│
├── configs/
│   └── default_config.json
│
├── data/
│   ├── clean/
│   │   └── .gitkeep
│   ├── poisoned/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
│
├── docs/
│   └── .gitkeep
│
├── experiments/
│   └── .gitkeep
│
├── notebooks/
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   │
│   ├── chunking/
│   │   └── __init__.py
│   ├── defense/
│   │   └── __init__.py
│   ├── embeddings/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   ├── generation/
│   │   └── __init__.py
│   ├── ingestion/
│   │   └── __init__.py
│   ├── poisoning/
│   │   └── __init__.py
│   ├── retrieval/
│   │   └── __init__.py
│   └── vectorstore/
│       └── __init__.py
│
└── tests/
    ├── __init__.py
    └── test_foundation.py
```

## 3.2 Current status

### Implemented foundation

- Configuration
- JSON configuration loading
- Environment-variable overrides
- Logging
- Package structure
- Data directories
- Foundation tests

### Scaffolded but not implemented

- Ingestion
- AST/code parsing
- Code-aware chunking
- Embeddings
- Vector store
- Retrieval
- Generation
- Poisoning
- Defense
- Evaluation

### Missing research functionality

- End-to-end Code-RAG
- Seven poisoning categories
- PRR
- ASR
- VIR
- Defense layers
- Output security analysis
- Experiment runner
- Benchmark
- Ablation
- Security–utility analysis
- Research reports
- API
- Docker
- Optional dashboard

---

# 4. What We Are Building

The project will evolve through these major layers:

```text
PHASE 0
Environment + Foundation
        ↓
PHASE 1
Repository Ingestion
        ↓
PHASE 2
AST Parsing
        ↓
PHASE 3
AST-Aware Chunking
        ↓
PHASE 4
Local Hugging Face Embeddings
        ↓
PHASE 5
FAISS Vector Store
        ↓
PHASE 6
Retrieval
        ↓
PHASE 7
Local Ollama Code Generation
        ↓
PHASE 8
Clean Code-RAG Baseline
        ↓
PHASE 9
Baseline Evaluation + Security Analysis
        ↓
PHASE 10
Controlled Poisoning
        ↓
PHASE 11
Poisoning Metrics
        ↓
PHASE 12
Defense Layers
        ↓
PHASE 13
Generated-Code Security Validation
        ↓
PHASE 14
Experiment Runner
        ↓
PHASE 15
CodeRAG-PoisonBench
        ↓
PHASE 16
Ablation Studies
        ↓
PHASE 17
Security–Utility Analysis
        ↓
PHASE 18
FastAPI
        ↓
PHASE 19
Optional Research Dashboard
        ↓
PHASE 20
Docker / Reproducible Deployment
        ↓
PHASE 21
Documentation + Final Validation
```

---

# 5. Core Design Principle

The system is intentionally built in dependency order.

The central rule is:

```text
Do not build a later layer on an unverified earlier layer.
```

For example:

```text
Do not build poisoning
before clean retrieval works.

Do not build defenses
before poisoning can be measured.

Do not build the dashboard
before the research core works.

Do not claim experimental results
before experiments actually run.
```

---

# 6. Final Architecture

```text
                         ┌─────────────────────────────┐
                         │ Authorized / Synthetic Code │
                         │ Repositories + Documents   │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │ Repository Loader │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ File Filtering   │
                              │ Language Detect. │
                              │ Metadata         │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ AST Parser       │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ AST Chunker      │
                              └────────┬─────────┘
                                       │
                                       ▼
                         ┌───────────────────────────┐
                         │ Local HF Code Embeddings │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ FAISS Vector     │
                              │ Store            │
                              └────────┬─────────┘
                                       │
                                       ▼
                                  Retriever
                                       │
                         ┌─────────────┴─────────────┐
                         │                           │
                         ▼                           ▼
                   Clean Corpus                Poisoning Engine
                                                     │
                                                     ▼
                                               Poisoned Corpus
                                                     │
                                                     ▼
                                             Poisoned Index
                         │                           │
                         └─────────────┬─────────────┘
                                       ▼
                              Retrieved Context
                                       │
                                       ▼
                              Defense Pipeline
                                       │
              ┌────────────────────────┼─────────────────────┐
              │                        │                     │
              ▼                        ▼                     ▼
        Trust Scoring          Anomaly Detection      Context Validation
              │                        │                     │
              └────────────────────────┼─────────────────────┘
                                       │
                                       ▼
                         Instruction/Data Separation
                                       │
                                       ▼
                              Safe/Filtered Context
                                       │
                                       ▼
                                Ollama Local LLM
                                       │
                                       ▼
                                 Generated Code
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
                    Code Quality              Security Analysis
                         │                           │
                         └─────────────┬─────────────┘
                                       ▼
                                 Metrics Engine
                                       │
                                       ▼
                              Experiment Runner
                                       │
                                       ▼
                              Result Store / Reports
                                       │
                                       ▼
                             CodeRAG-PoisonBench
```

---

# 7. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.11.x | Main implementation |
| Configuration | python-dotenv + JSON | Environment/configuration |
| Models | PyTorch | Local model execution |
| Model hub | Hugging Face Transformers | Local code embeddings |
| AST | Tree-sitter | Structural code parsing |
| Vector search | FAISS | Local vector similarity search |
| Generation | Ollama | Local Code LLM runtime |
| API | FastAPI | Backend API |
| Security | AST rules + Semgrep | Static/security analysis |
| Testing | pytest | Test automation |
| Lint/format | Ruff | Code quality |
| Reporting | pandas + matplotlib | Experiment processing/reporting |
| Containerization | Docker | Reproducible runtime |
| Version control | Git/GitHub | Source control |

## Important cost rule

The core project must work with:

```text
Hugging Face local model
+
Ollama local model
+
FAISS
```

No paid AI API is required.

---

# 8. Environment Architecture

## 8.1 Python

Use:

```text
Python 3.11.x
```

## 8.2 Virtual environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

## 8.3 Ollama

Install Ollama locally.

Verify:

```powershell
ollama --version
```

Check installed models:

```powershell
ollama list
```

Pull the selected model:

```powershell
ollama pull <MODEL_NAME>
```

Run a direct test:

```powershell
ollama run <MODEL_NAME>
```

The application should use:

```text
OLLAMA_BASE_URL=http://localhost:11434
```

## 8.4 Hugging Face

The selected embedding model should be public and runnable locally.

First run can download the model.

Subsequent runs should reuse the local cache.

Where possible, support offline operation after the model has been downloaded.

---

# 9. Environment File

The canonical file is:

```text
.env
```

and the committed template is:

```text
.env.example
```

Recommended structure:

```env
# =====================================================
# APPLICATION
# =====================================================

APP_ENV=development

# =====================================================
# LOGGING
# =====================================================

LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# =====================================================
# EMBEDDINGS
# =====================================================

EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=<PUBLIC_HUGGINGFACE_MODEL>

HF_HOME=.cache/huggingface
HF_TOKEN=

# =====================================================
# LOCAL LLM
# =====================================================

LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=<OLLAMA_MODEL>

# =====================================================
# VECTOR STORE
# =====================================================

VECTORSTORE_PROVIDER=faiss
VECTORSTORE_PATH=data/processed/faiss_index

# =====================================================
# CHUNKING / RETRIEVAL
# =====================================================

CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=5

# =====================================================
# DATA
# =====================================================

CLEAN_DATA_DIR=data/clean
POISONED_DATA_DIR=data/poisoned
PROCESSED_DATA_DIR=data/processed
BENCHMARK_DATA_DIR=data/benchmarks

# =====================================================
# SECURITY
# =====================================================

DEFENSE_ENABLED=true
SECURITY_SCAN_ENABLED=true

# =====================================================
# EXPERIMENTS
# =====================================================

RANDOM_SEED=42

# =====================================================
# API
# =====================================================

API_HOST=0.0.0.0
API_PORT=8000
```

## Environment rules

- Never commit `.env`
- Never hard-code credentials
- Public Hugging Face models should be preferred
- `HF_TOKEN` should remain empty unless genuinely needed
- No paid provider key is required
- All paths must be relative/configurable
- Every environment variable must have one canonical name

---

# 10. Final Folder Structure

The final repository should evolve toward:

```text
Final-Year-Project/
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
│
├── configs/
│   ├── default_config.json
│   ├── baseline.json
│   ├── poisoning.json
│   ├── defense.json
│   └── experiment_matrix.json
│
├── data/
│   ├── clean/
│   │   └── repositories/
│   ├── poisoned/
│   │   └── repositories/
│   ├── processed/
│   └── benchmarks/
│
├── docs/
│   ├── architecture.md
│   ├── threat-model.md
│   ├── dataset.md
│   ├── experiments.md
│   ├── metrics.md
│   ├── reproducibility.md
│   ├── api.md
│   └── deployment.md
│
├── experiments/
│   ├── configs/
│   └── runs/
│
├── notebooks/
│
├── results/
│   ├── raw/
│   ├── metrics/
│   ├── plots/
│   └── reports/
│
├── scripts/
│   ├── health_check.py
│   ├── prepare_dataset.py
│   ├── ingest.py
│   ├── build_index.py
│   ├── run_pipeline.py
│   ├── run_experiment.py
│   └── generate_report.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── models.py
│   ├── pipeline.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── repository_loader.py
│   │   ├── file_filter.py
│   │   ├── language_detector.py
│   │   └── metadata.py
│   │
│   ├── parsing/
│   │   ├── __init__.py
│   │   ├── language_registry.py
│   │   └── ast_parser.py
│   │
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── chunk_models.py
│   │   └── ast_chunker.py
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── hf_code_embeddings.py
│   │   ├── cache.py
│   │   └── factory.py
│   │
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── faiss_store.py
│   │   └── metadata_store.py
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retriever.py
│   │   ├── context_builder.py
│   │   └── reranker.py
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── prompt_builder.py
│   │   ├── local_ollama.py
│   │   └── factory.py
│   │
│   ├── poisoning/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── misleading_code.py
│   │   ├── vulnerable_code.py
│   │   ├── false_api_guidance.py
│   │   ├── contradictory_docs.py
│   │   ├── instruction_like.py
│   │   ├── false_conventions.py
│   │   └── context_manipulation.py
│   │
│   ├── defense/
│   │   ├── __init__.py
│   │   ├── trust_scoring.py
│   │   ├── anomaly_detection.py
│   │   ├── context_validation.py
│   │   ├── instruction_separator.py
│   │   ├── static_analysis.py
│   │   └── risk_engine.py
│   │
│   ├── security/
│   │   ├── __init__.py
│   │   ├── findings.py
│   │   ├── ast_analyzer.py
│   │   ├── semgrep_runner.py
│   │   └── decision.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── retrieval_metrics.py
│   │   ├── generation_metrics.py
│   │   ├── security_metrics.py
│   │   ├── benchmark.py
│   │   ├── experiment_runner.py
│   │   ├── result_store.py
│   │   └── reporting.py
│   │
│   └── api/
│       ├── __init__.py
│       ├── schemas.py
│       └── main.py
│
└── tests/
    ├── __init__.py
    ├── test_foundation.py
    │
    ├── unit/
    │   ├── test_models.py
    │   ├── test_ingestion.py
    │   ├── test_ast_parser.py
    │   ├── test_chunking.py
    │   ├── test_embeddings.py
    │   ├── test_vectorstore.py
    │   ├── test_retrieval.py
    │   ├── test_generation.py
    │   └── test_metrics.py
    │
    ├── integration/
    │   ├── test_clean_rag.py
    │   ├── test_baseline.py
    │   └── test_experiments.py
    │
    ├── security/
    │   ├── test_poisoning.py
    │   ├── test_defense.py
    │   └── test_output_validation.py
    │
    ├── regression/
    │
    └── e2e/
        └── test_api.py
```

---

# 11. File Responsibility Map

## Root files

### `.env.example`
Documents the environment contract.

### `.gitignore`
Prevents secrets, caches, large generated files, logs, indexes, and results from being committed accidentally.

### `README.md`
Complete installation, architecture, experiment, testing, and deployment guide.

### `requirements.txt`
Runtime dependencies.

### `requirements-dev.txt`
Development/test/lint dependencies.

### `Dockerfile`
Builds the application image.

### `docker-compose.yml`
Runs the application and any explicitly required supporting services.

### `.dockerignore`
Prevents unnecessary files entering the image build context.

---

# 12. Configs

## `configs/default_config.json`

Central application defaults.

Should eventually cover:

```text
application
logging
ingestion
parsing
chunking
embeddings
vectorstore
retrieval
generation
poisoning
defense
security
evaluation
experiments
api
```

## `baseline.json`

Parameters for clean baseline experiments.

## `poisoning.json`

Poisoning experiment settings.

## `defense.json`

Defense configuration and thresholds.

## `experiment_matrix.json`

Defines the planned experimental combinations.

---

# 13. Data Folders

## `data/clean/`

Only clean, authorized, non-poisoned benchmark material.

## `data/poisoned/`

Separate controlled poisoned versions.

## `data/processed/`

Derived artifacts:

```text
chunks
metadata
embedding cache
FAISS index
index metadata
```

## `data/benchmarks/`

Versioned CodeRAG-PoisonBench data.

---

# 14. Source Modules

# `src/models.py`

Shared data contracts.

Expected models:

```text
SourceFile
CodeChunk
RetrievedChunk
GenerationRequest
GenerationResult
SecurityFinding
ExperimentConfig
ExperimentResult
```

Why it exists:

It prevents different modules from inventing different dictionary formats.

---

# `src/config.py`

Loads:

```text
.env
+
default_config.json
```

and exposes strongly structured configuration.

---

# `src/logger.py`

Central logging.

Should support:

```text
console
+
file
```

Never log secrets.

---

# `src/pipeline.py`

Orchestrates the overall clean/poisoned RAG flow.

It should not contain every implementation detail.

Instead it coordinates:

```text
ingestion
→ parsing
→ chunking
→ embedding
→ indexing
→ retrieval
→ defense
→ generation
→ security analysis
→ evaluation
```

---

# 15. Ingestion Layer

## `repository_loader.py`

Input:

```text
local repository path
```

Output:

```text
SourceFile[]
```

Responsibilities:

- recursive file discovery
- repository metadata
- safe path handling

---

## `file_filter.py`

Exclude:

```text
.git
node_modules
.venv
venv
__pycache__
dist
build
binary files
generated files
large files
.env
credential files
private keys
```

The filter should be configurable.

---

## `language_detector.py`

Determines source language from:

- extension
- optional content checks

---

## `metadata.py`

Extracts:

```text
repository
file_path
language
file_size
hash
commit
```

and later:

```text
class_name
function_name
line ranges
```

---

# 16. Parsing Layer

## `language_registry.py`

Central mapping of supported languages to Tree-sitter grammars/parsers.

Start small.

Recommended initial languages:

```text
Python
JavaScript
Java
```

Add more only after core functionality is stable.

---

## `ast_parser.py`

Responsibilities:

```text
source
↓
language parser
↓
AST
```

Must gracefully handle syntax errors.

A single malformed file should not necessarily terminate ingestion.

---

# 17. Chunking Layer

## `chunk_models.py`

Defines chunk structures.

## `ast_chunker.py`

Prefer:

```text
module
class
function
method
```

boundaries.

Fallback to safe structural/text chunking when required.

Every chunk should preserve provenance.

Example metadata:

```text
chunk_id
repository
file_path
language
class_name
function_name
line_start
line_end
commit_hash
source_type
poison_status
security_flags
content_hash
```

---

# 18. Embedding Layer

## `base.py`

Interface:

```text
embed(text)
embed_batch(texts)
dimension()
model_name()
```

## `hf_code_embeddings.py`

Loads a public Hugging Face model locally.

Responsibilities:

```text
load model
tokenize
batch
infer
normalize if configured
return vectors
```

No paid service.

## `cache.py`

Uses content/model hashes to avoid recomputation.

Important cache key:

```text
content hash
+
embedding model
+
model version/config
```

## `factory.py`

Selects embedding provider from configuration.

---

# 19. Vector Store Layer

## `base.py`

Abstract interface:

```text
build
add
search
save
load
```

## `faiss_store.py`

Primary local implementation.

Responsibilities:

- create index
- add embeddings
- nearest-neighbor search
- persistence

## `metadata_store.py`

Stores metadata associated with vector IDs.

Important:

```text
vector_id → CodeChunk metadata
```

Validate embedding dimension before querying/loading.

---

# 20. Retrieval Layer

## `retriever.py`

Pipeline:

```text
query
↓
query embedding
↓
FAISS
↓
top-k
↓
RetrievedChunk[]
```

Each result must contain:

```text
chunk_id
score
content
metadata
```

## `context_builder.py`

Builds LLM context.

Retrieved information must be explicitly labeled as untrusted reference data.

## `reranker.py`

Optional abstraction.

Do not make expensive reranking mandatory unless required.

---

# 21. Generation Layer

## `base.py`

Interface for the local LLM.

Expected behavior:

```text
generate(request)
```

## `prompt_builder.py`

Creates prompts with clear separation:

```text
application instructions
```

and:

```text
retrieved reference data
```

Retrieved content must never automatically gain privileged instruction authority.

## `local_ollama.py`

Communicates directly with Ollama's local API.

Must handle:

- connection failure
- missing model
- timeout
- malformed response
- empty output

## `factory.py`

Creates the configured LLM provider.

There is no paid OpenAI adapter.

---

# 22. Clean Baseline

The first major milestone is:

```text
Repository
↓
Ingestion
↓
AST
↓
Chunking
↓
HF Embeddings
↓
FAISS
↓
Retrieval
↓
Context
↓
Ollama
↓
Generated Code
```

Acceptance criteria:

```text
[ ] ingest works
[ ] chunks are produced
[ ] embeddings are generated
[ ] index is persisted
[ ] retrieval returns expected chunks
[ ] Ollama responds
[ ] generated code is returned
[ ] complete pipeline runs
```

Only after this milestone is verified should poisoning become a required pipeline stage.

---

# 23. Security Analysis Layer

## `findings.py`

Standardized finding object:

```text
finding_id
severity
category
rule
message
file
line
evidence
confidence
source
recommended_action
```

## `ast_analyzer.py`

Detects structurally relevant security patterns where practical.

## `semgrep_runner.py`

Runs configured Semgrep rules.

Semgrep is one signal, not the complete security system.

## `decision.py`

Converts findings into a standardized decision:

```text
PASS
FLAG
REJECT
```

with rationale.

---

# 24. Poisoning Layer

Seven required research categories:

```text
1. Misleading code
2. Vulnerable code
3. False API guidance
4. Contradictory documentation
5. Instruction-like content
6. False repository conventions
7. Context manipulation
```

All attacks must operate on:

```text
authorized
+
synthetic
+
controlled
```

research material.

Never modify unauthorized external systems.

Every poisoned artifact must retain:

```text
attack_id
attack_type
source
metadata
seed
poison_status
```

---

# 25. Defense Layer

## L1 — Trust Scoring

Inputs can include:

```text
source provenance
repository information
metadata consistency
content type
suspicious indicators
```

Output:

```text
trust_score
risk_level
reason
```

---

## L2 — Anomaly Detection

Detect suspicious:

```text
instruction-like content
unexpected structures
metadata anomalies
unusual patterns
```

---

## L3 — Context Validation

Before retrieved context reaches the LLM:

```text
retrieve
↓
validate
↓
allow/filter
```

---

## L4 — Instruction/Data Separation

Ensure retrieved material is treated as data, not privileged instructions.

---

## L5 — Static Analysis

Use:

```text
AST analysis
+
Semgrep
+
custom security checks
```

---

## `risk_engine.py`

Combines defense signals.

Output:

```text
decision
risk_score
triggered_layers
reasons
evidence
```

---

# 26. Evaluation Layer

## Retrieval Metrics

Potential metrics:

```text
Recall@K
Precision@K
MRR
similarity distributions
```

## Generation Metrics

Potential metrics:

```text
syntax/compile success
unit-test success
functional correctness
Pass@K where applicable
```

## Security Metrics

Required:

```text
PRR
ASR
VIR
```

Additional:

```text
false-positive rate
false-negative rate
defense detection rate
```

Only use metrics that are supported by the experiment design.

---

# 27. Causal Tracking

This is a major academic requirement.

For every experiment distinguish:

```text
Poison exists?
      ↓
Poison retrieved?
      ↓
Poison included in context?
      ↓
Generation changed?
      ↓
Vulnerability introduced?
```

Do not collapse these into one field such as:

```text
attack_success=true
```

without retaining the intermediate evidence.

---

# 28. Experiment Conditions

At minimum:

| Experiment | Knowledge Base | Defense | Purpose |
|---|---|---|---|
| A | Clean | No | Baseline |
| B | Poisoned | No | Poisoning impact |
| C | Poisoned | Yes | Defense effectiveness |
| D | Clean | Yes | Utility / false-positive assessment |

---

# 29. Security–Utility Analysis

A defense should not simply reject everything.

Analyze:

```text
Security improvement
        vs
Retrieval quality
        vs
Code correctness
```

Track:

```text
TP
TN
FP
FN
```

where applicable.

---

# 30. CodeRAG-PoisonBench

The benchmark should contain structured records such as:

```text
experiment_id
dataset_version
task_id
repository
language
attack_type
poison_ratio
retriever
embedding_model
llm
defense
seed
retrieved_chunks
generated_code
expected_behavior
security_findings
metrics
```

The benchmark must be generated and versioned through code rather than manually assembled whenever practical.

---

# 31. Experiment Runner

Flow:

```text
ExperimentConfig
        ↓
Dataset
        ↓
Retriever
        ↓
Defense
        ↓
Ollama
        ↓
Evaluator
        ↓
Metrics
        ↓
Result Store
        ↓
Report
```

Experiment configuration should live in JSON/config files rather than being hard-coded into business logic.

---

# 32. Ablation Studies

At minimum:

```text
No Defense
     ↓
L1
     ↓
L1 + L2
     ↓
L1 + L2 + L3
     ↓
L1 + L2 + L3 + L4
     ↓
L1 + L2 + L3 + L4 + L5
```

Purpose:

Determine what each defense layer contributes.

---

# 33. API

Only after the core research pipeline works.

Candidate endpoints:

```text
GET  /health
POST /ingest
POST /index
POST /retrieve
POST /generate
POST /analyze
POST /poison
POST /evaluate
POST /run-experiment
GET  /results/{experiment_id}
GET  /metrics
```

Each endpoint requires:

```text
request schema
validation
response schema
error handling
tests
documentation
```

---

# 34. Optional Dashboard

The dashboard is secondary.

It should eventually visualize:

```text
Query
↓
Retrieved chunks
↓
Defense decisions
↓
Generated code
↓
Security findings
↓
Experiment configuration
↓
Metrics
```

The dashboard must not delay the core research pipeline.

---

# 35. Scripts

## `health_check.py`

Checks:

```text
configuration
Ollama
selected model
embedding model availability
paths
vector store state
```

## `prepare_dataset.py`

Prepares clean/benchmark material.

## `ingest.py`

Runs ingestion.

Example:

```powershell
python scripts/ingest.py --source data/clean/repositories/example
```

## `build_index.py`

Builds FAISS.

Example:

```powershell
python scripts/build_index.py --dataset clean
```

## `run_pipeline.py`

Runs end-to-end query flow.

Example:

```powershell
python scripts/run_pipeline.py --query "Create a REST endpoint for user registration"
```

## `run_experiment.py`

Runs configured experiment.

Example:

```powershell
python scripts/run_experiment.py --config experiments/configs/baseline.json
```

## `generate_report.py`

Converts result artifacts into tables/plots/reports.

---

# 36. Ruff — What It Is and How We Use It

## 36.1 What is Ruff?

Ruff is the project's fast Python **linter and formatter**.

It helps us identify:

- unused imports
- undefined names
- import problems
- style violations
- error-prone code patterns
- formatting inconsistencies

Ruff should be part of the engineering workflow, not just an optional command.

---

# 37. Ruff's Role in the Project

Our code-quality pipeline becomes:

```text
Write Code
   ↓
Ruff Check
   ↓
Fix Lint Problems
   ↓
Ruff Format
   ↓
Run Tests
   ↓
Integrate
```

Ruff does NOT replace tests.

```text
Ruff = static code quality
pytest = behavioral correctness
```

We need both.

---

# 38. Ruff Commands

Install:

```powershell
pip install ruff
```

Check the entire project:

```powershell
ruff check .
```

Automatically fix safe lint issues:

```powershell
ruff check . --fix
```

Check formatting:

```powershell
ruff format --check .
```

Format:

```powershell
ruff format .
```

Run both:

```powershell
ruff check . && ruff format --check .
```

On PowerShell, if command chaining is inconvenient:

```powershell
ruff check .
ruff format --check .
```

---

# 39. How to Show What Ruff Is Doing

Use verbose output:

```powershell
ruff check . --verbose
```

To inspect a specific file:

```powershell
ruff check src/models.py
```

To see formatting differences:

```powershell
ruff format src/models.py --diff
```

This is useful during development because you can show:

```text
Before
↓
Ruff detects issue
↓
Fix
↓
Ruff passes
```

For example:

```text
F401 unused import
E501 line too long
F821 undefined name
```

The exact codes depend on the selected Ruff rule set.

---

# 40. Ruff Configuration

Recommended configuration in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = [
    "E",
    "F",
    "I",
    "B",
    "UP",
]

ignore = [
    "E501",
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
```

This should be adjusted only when the actual project's codebase requires it.

---

# 41. Quality Gate

A phase should not be considered complete when:

```text
ruff check .
```

fails on newly introduced issues.

Ideal phase gate:

```text
Ruff passes
+
Unit tests pass
+
Integration test passes where applicable
+
Manual verification complete
```

---

# 42. Complete Development Workflow

The entire engineering workflow is:

```text
1. Read the phase objective
        ↓
2. Inspect existing files
        ↓
3. Identify dependencies
        ↓
4. Add/update configuration
        ↓
5. Implement one logical file
        ↓
6. Run Ruff
        ↓
7. Fix Ruff findings
        ↓
8. Run unit tests
        ↓
9. Run integration test
        ↓
10. Run a real command
        ↓
11. Inspect generated artifacts
        ↓
12. Update documentation
        ↓
13. Git commit
        ↓
14. Move to next file
```

---

# 43. Phase-by-Phase Build Plan

# PHASE 0 — Foundation and Environment

## Goal

Convert the current foundation into a stable development base.

## Files

```text
src/models.py
src/config.py
configs/default_config.json
.env.example
requirements.txt
requirements-dev.txt
tests/test_foundation.py
tests/unit/test_models.py
```

## Tasks

- define domain models
- normalize environment names
- extend configuration
- add Ruff
- add pytest
- establish test directories
- verify Python environment

## Acceptance

```text
python --version
pip install -r requirements.txt
ruff check .
ruff format --check .
pytest
```

---

# PHASE 1 — Ingestion

## Files

```text
src/ingestion/repository_loader.py
src/ingestion/file_filter.py
src/ingestion/language_detector.py
src/ingestion/metadata.py
tests/unit/test_ingestion.py
```

## Flow

```text
Repository
↓
Discover Files
↓
Filter
↓
Detect Language
↓
Metadata
↓
SourceFile[]
```

## Acceptance

A small authorized test repository can be ingested successfully.

---

# PHASE 2 — Parsing

## Files

```text
src/parsing/__init__.py
src/parsing/language_registry.py
src/parsing/ast_parser.py
tests/unit/test_ast_parser.py
```

## Flow

```text
SourceFile
↓
Language Registry
↓
Tree-sitter
↓
AST
```

## Acceptance

Supported test files parse correctly.

Malformed files are handled predictably.

---

# PHASE 3 — AST Chunking

## Files

```text
src/chunking/chunk_models.py
src/chunking/ast_chunker.py
tests/unit/test_chunking.py
```

## Flow

```text
AST
↓
Class / Function / Method
↓
CodeChunk[]
```

## Acceptance

Chunks preserve provenance and line metadata.

---

# PHASE 4 — Local Hugging Face Embeddings

## Files

```text
src/embeddings/base.py
src/embeddings/hf_code_embeddings.py
src/embeddings/cache.py
src/embeddings/factory.py
tests/unit/test_embeddings.py
```

## Flow

```text
CodeChunk
↓
Local HF Model
↓
Embedding Vector
↓
Cache
```

## Acceptance

- model loads locally
- embeddings are generated
- dimension is consistent
- repeated identical content can use cache

---

# PHASE 5 — FAISS

## Files

```text
src/vectorstore/base.py
src/vectorstore/metadata_store.py
src/vectorstore/faiss_store.py
tests/unit/test_vectorstore.py
```

## Flow

```text
Embeddings
↓
FAISS
↓
Index
↓
Persist
```

## Acceptance

Index can:

```text
build
save
load
search
```

---

# PHASE 6 — Retrieval

## Files

```text
src/retrieval/retriever.py
src/retrieval/context_builder.py
src/retrieval/reranker.py
tests/unit/test_retrieval.py
```

## Flow

```text
Query
↓
Embedding
↓
FAISS
↓
Top-K
↓
Metadata
↓
Context
```

## Acceptance

Relevant test chunks are returned.

---

# PHASE 7 — Ollama Generation

## Files

```text
src/generation/base.py
src/generation/prompt_builder.py
src/generation/local_ollama.py
src/generation/factory.py
tests/unit/test_generation.py
src/pipeline.py
tests/integration/test_clean_rag.py
```

## Flow

```text
Query
+
Retrieved Context
↓
Prompt
↓
Ollama
↓
Generated Code
```

## Acceptance

A real local Ollama model can answer a controlled test query.

---

# PHASE 8 — Clean Code-RAG

## Goal

Combine everything.

```text
Repository
↓
Ingestion
↓
AST
↓
Chunking
↓
HF Embeddings
↓
FAISS
↓
Retrieval
↓
Context
↓
Ollama
↓
Generated Code
```

## Acceptance

The end-to-end clean baseline works without any paid service.

---

# PHASE 9 — Baseline Evaluation

## Files

```text
src/evaluation/retrieval_metrics.py
src/evaluation/generation_metrics.py
src/security/findings.py
src/security/ast_analyzer.py
src/security/decision.py
tests/unit/test_metrics.py
tests/integration/test_baseline.py
```

## Goal

Measure the clean baseline.

---

# PHASE 10 — Controlled Poisoning

## Files

```text
src/poisoning/base.py
src/poisoning/misleading_code.py
src/poisoning/vulnerable_code.py
src/poisoning/false_api_guidance.py
src/poisoning/contradictory_docs.py
src/poisoning/instruction_like.py
src/poisoning/false_conventions.py
src/poisoning/context_manipulation.py
tests/security/test_poisoning.py
```

## Goal

Create controlled research-only poisoned corpora.

---

# PHASE 11 — Poisoning Evaluation

## Files

```text
src/evaluation/security_metrics.py
```

## Metrics

```text
PRR
ASR
VIR
```

## Required evidence

```text
poison exists
↓
poison retrieved
↓
poison included
↓
generation affected
↓
vulnerability introduced
```

---

# PHASE 12 — Defense

## Files

```text
src/defense/trust_scoring.py
src/defense/anomaly_detection.py
src/defense/context_validation.py
src/defense/instruction_separator.py
src/defense/static_analysis.py
src/defense/risk_engine.py
tests/security/test_defense.py
```

## Flow

```text
Retrieved Chunk
↓
L1 Trust
↓
L2 Anomaly
↓
L3 Context Validation
↓
L4 Instruction/Data Separation
↓
L5 Static Analysis
↓
Risk Engine
↓
Safe Context
```

---

# PHASE 13 — Output Security Validation

## Files

```text
src/security/semgrep_runner.py
tests/security/test_output_validation.py
```

## Flow

```text
Generated Code
↓
Syntax
↓
AST
↓
Semgrep
↓
Security Rules
↓
Findings
↓
PASS / FLAG / REJECT
```

---

# PHASE 14 — Experiment Framework

## Files

```text
src/evaluation/experiment_runner.py
src/evaluation/result_store.py
src/evaluation/reporting.py
experiments/configs/baseline.json
experiments/configs/poisoning.json
experiments/configs/defense.json
experiments/configs/experiment_matrix.json
tests/integration/test_experiments.py
```

---

# PHASE 15 — CodeRAG-PoisonBench

## Goal

Make the benchmark a first-class artifact.

## Required characteristics

- versioned
- reproducible
- controlled
- machine-readable
- traceable
- linked to experiments

---

# PHASE 16 — Ablation

## Goal

Measure the contribution of each defense layer.

---

# PHASE 17 — Security–Utility Analysis

## Goal

Measure:

```text
security
vs
retrieval
vs
generation quality
```

---

# PHASE 18 — API

## Files

```text
src/api/__init__.py
src/api/schemas.py
src/api/main.py
tests/e2e/test_api.py
```

---

# PHASE 19 — Optional Dashboard

Only now.

---

# PHASE 20 — Docker

## Files

```text
Dockerfile
docker-compose.yml
.dockerignore
```

Document:

```text
what runs inside the container
what runs outside
how the application reaches Ollama
where data lives
where results live
```

---

# PHASE 21 — Final Documentation

Files:

```text
docs/architecture.md
docs/threat-model.md
docs/dataset.md
docs/experiments.md
docs/metrics.md
docs/reproducibility.md
docs/api.md
docs/deployment.md
README.md
```

---

# 44. Exact File-by-File Order

## Phase 0

```text
1.  src/models.py
2.  src/config.py
3.  configs/default_config.json
4.  .env.example
5.  requirements.txt
6.  requirements-dev.txt
7.  tests/unit/test_models.py
8.  tests/test_foundation.py
```

## Phase 1

```text
9.  src/ingestion/repository_loader.py
10. src/ingestion/file_filter.py
11. src/ingestion/language_detector.py
12. src/ingestion/metadata.py
13. tests/unit/test_ingestion.py
```

## Phase 2

```text
14. src/parsing/__init__.py
15. src/parsing/language_registry.py
16. src/parsing/ast_parser.py
17. tests/unit/test_ast_parser.py
```

## Phase 3

```text
18. src/chunking/chunk_models.py
19. src/chunking/ast_chunker.py
20. tests/unit/test_chunking.py
```

## Phase 4

```text
21. src/embeddings/base.py
22. src/embeddings/hf_code_embeddings.py
23. src/embeddings/cache.py
24. src/embeddings/factory.py
25. tests/unit/test_embeddings.py
```

## Phase 5

```text
26. src/vectorstore/base.py
27. src/vectorstore/metadata_store.py
28. src/vectorstore/faiss_store.py
29. tests/unit/test_vectorstore.py
```

## Phase 6

```text
30. src/retrieval/retriever.py
31. src/retrieval/context_builder.py
32. src/retrieval/reranker.py
33. tests/unit/test_retrieval.py
```

## Phase 7

```text
34. src/generation/base.py
35. src/generation/prompt_builder.py
36. src/generation/local_ollama.py
37. src/generation/factory.py
38. tests/unit/test_generation.py
39. src/pipeline.py
40. tests/integration/test_clean_rag.py
```

## Phase 9

```text
41. src/evaluation/retrieval_metrics.py
42. src/evaluation/generation_metrics.py
43. src/security/findings.py
44. src/security/ast_analyzer.py
45. src/security/decision.py
46. tests/unit/test_metrics.py
47. tests/integration/test_baseline.py
```

## Phase 10–11

```text
48. src/poisoning/base.py
49. src/poisoning/misleading_code.py
50. src/poisoning/vulnerable_code.py
51. src/poisoning/false_api_guidance.py
52. src/poisoning/contradictory_docs.py
53. src/poisoning/instruction_like.py
54. src/poisoning/false_conventions.py
55. src/poisoning/context_manipulation.py
56. src/evaluation/security_metrics.py
57. tests/security/test_poisoning.py
```

## Phase 12–13

```text
58. src/defense/trust_scoring.py
59. src/defense/anomaly_detection.py
60. src/defense/context_validation.py
61. src/defense/instruction_separator.py
62. src/defense/static_analysis.py
63. src/defense/risk_engine.py
64. src/security/semgrep_runner.py
65. tests/security/test_defense.py
66. tests/security/test_output_validation.py
```

## Phase 14–17

```text
67. src/evaluation/experiment_runner.py
68. src/evaluation/benchmark.py
69. src/evaluation/result_store.py
70. src/evaluation/reporting.py
71. experiments/configs/baseline.json
72. experiments/configs/poisoning.json
73. experiments/configs/defense.json
74. experiments/configs/experiment_matrix.json
75. tests/integration/test_experiments.py
```

## Scripts

```text
76. scripts/health_check.py
77. scripts/prepare_dataset.py
78. scripts/ingest.py
79. scripts/build_index.py
80. scripts/run_pipeline.py
81. scripts/run_experiment.py
82. scripts/generate_report.py
```

## API

```text
83. src/api/__init__.py
84. src/api/schemas.py
85. src/api/main.py
86. tests/e2e/test_api.py
```

## Deployment

```text
87. Dockerfile
88. docker-compose.yml
89. .dockerignore
```

## Documentation

```text
90. docs/architecture.md
91. docs/threat-model.md
92. docs/dataset.md
93. docs/experiments.md
94. docs/metrics.md
95. docs/reproducibility.md
96. docs/api.md
97. docs/deployment.md
98. README.md
```

---

# 45. Dependency Graph

```text
config
  ↓
models
  ↓
ingestion
  ↓
parsing
  ↓
chunking
  ↓
embeddings
  ↓
vectorstore
  ↓
retrieval
  ↓
generation
  ↓
clean baseline
  ↓
baseline evaluation
  ↓
poisoning
  ↓
security metrics
  ↓
defense
  ↓
output validation
  ↓
experiment runner
  ↓
benchmark
  ↓
ablation
  ↓
security-utility analysis
  ↓
API
  ↓
dashboard
  ↓
Docker
```

Supporting dependencies:

```text
Ruff ───────→ all source code
pytest ─────→ all implementation phases
configuration ─→ all configurable components
logging ───────→ all major pipeline components
```

---

# 46. Runtime Data Flow

## Clean mode

```text
Repository
↓
SourceFile[]
↓
AST
↓
CodeChunk[]
↓
Embedding vectors
↓
FAISS
↓
Query vector
↓
RetrievedChunk[]
↓
Context
↓
Ollama
↓
Generated code
↓
Security analysis
↓
Result
```

## Poisoned mode

```text
Clean corpus
↓
Poisoning engine
↓
Poisoned corpus
↓
Poisoned index
↓
Query
↓
Retrieved chunks
↓
Poison trace
↓
Defense
↓
Context
↓
Ollama
↓
Generated code
↓
Security analysis
↓
PRR / ASR / VIR
```

---

# 47. Configuration Flow

```text
.env
   +
configs/default_config.json
   ↓
src/config.py
   ↓
AppConfig
   ↓
Modules
```

The rule is:

```text
No hidden magic constants.
```

Configuration belongs in configuration.

---

# 48. Logging Flow

```text
Module
  ↓
Central Logger
  ├── Console
  └── logs/app.log
```

Important events:

```text
repository loaded
files discovered
file filtered
AST parsed
chunks produced
embeddings created
index built
query received
chunks retrieved
defense triggered
LLM call started
generation complete
security scan complete
experiment started
experiment completed
```

Never log:

```text
API keys
tokens
passwords
private keys
full secret contents
```

---

# 49. Testing Pyramid

```text
                 E2E Tests
                    ▲
                    │
             Integration Tests
                    ▲
                    │
                Unit Tests
                    ▲
                    │
           Static Analysis / Ruff
```

A feature is strongest when:

```text
Ruff passes
+
unit tests pass
+
integration test passes
+
real command succeeds
```

---

# 50. Example Development Checkpoint

For a new file:

```text
src/ingestion/repository_loader.py
```

the workflow is:

```text
1. Understand requirement
2. Inspect models/config
3. Implement loader
4. Run Ruff
5. Fix lint
6. Create test
7. Run pytest
8. Test against tiny real repository
9. Inspect output
10. Update documentation
11. Commit
```

---

# 51. Git Commit Strategy

Use meaningful commits.

Examples:

```text
feat: add shared SecureCodeRAG domain models
feat: implement repository ingestion
feat: add Tree-sitter AST parsing
feat: implement AST-aware chunking
feat: add local Hugging Face embeddings
feat: add FAISS vector store
feat: implement retrieval pipeline
feat: add Ollama generation adapter
feat: establish clean Code-RAG baseline
feat: add poisoning benchmark
feat: add security metrics
feat: implement defense layers
feat: add experiment runner
feat: add CodeRAG-PoisonBench
test: add retrieval integration tests
docs: update reproducibility guide
chore: configure Ruff
```

Avoid:

```text
update
changes
final
done
stuff
```

---

# 52. Phase Completion Gate

At the end of every phase record:

```text
PHASE:
OBJECTIVE:

FILES CREATED:
FILES MODIFIED:

DEPENDENCIES ADDED:

ENVIRONMENT VARIABLES ADDED:

RUFF:
PASSED / FAILED

UNIT TESTS:
PASSED / FAILED

INTEGRATION:
PASSED / FAILED / NOT APPLICABLE

REAL COMMAND:
PASSED / FAILED / NOT RUN

ARTIFACT VERIFICATION:
PASSED / FAILED / NOT RUN

DOCUMENTATION:
UPDATED / NOT UPDATED

SYNOPSIS REQUIREMENTS SATISFIED:

KNOWN ISSUES:

NEXT PHASE:
NEXT FILE:
```

---

# 53. Verification States

Never confuse these states:

## IMPLEMENTED

Code exists.

## VERIFIED

Code was actually executed and observed to work.

## PARTIALLY VERIFIED

Some behavior was verified, but complete functionality was not.

## NOT VERIFIED

Execution could not be performed.

## NOT IMPLEMENTED

Functionality does not exist yet.

Never write:

> tests pass

when tests were not actually executed.

---

# 54. Clean-Machine Installation Flow

A new Windows user should eventually be able to follow:

```text
1. Install Git
2. Install Python 3.11
3. Install Ollama
4. Pull the selected Ollama model
5. Clone repository
6. Create .venv
7. Activate .venv
8. Install requirements
9. Create .env
10. Download/cache HF embedding model
11. Run health_check.py
12. Prepare clean dataset
13. Run ingestion
14. Build index
15. Run clean baseline
16. Run security analysis
17. Run poisoning experiments
18. Run defenses
19. Run evaluation
20. Generate reports
21. Start API
22. Run Docker when applicable
```

---

# 55. Health Check

The project should eventually provide a command such as:

```powershell
python scripts/health_check.py
```

which checks:

```text
Python version
Configuration
Required directories
Ollama reachable
Ollama model installed
Embedding model available
FAISS environment
Security tooling
API configuration
```

A useful output style is:

```text
[OK] Python
[OK] Configuration
[OK] Ollama
[OK] LLM Model
[OK] Embedding Model
[OK] FAISS
[OK] Data Paths
[OK] Security Tools

SecureCodeRAG environment is ready.
```

---

# 56. Common Failure Handling

## Ollama unavailable

Explain:

```text
Ollama is not reachable at configured URL.
```

Then:

```powershell
ollama serve
```

or start Ollama normally.

## Model missing

```powershell
ollama list
ollama pull <MODEL_NAME>
```

## Hugging Face model failure

Check:

- model name
- local cache
- disk space
- Python dependency compatibility
- network during first download

## FAISS failure

Check:

- Python version
- package version
- architecture
- installation compatibility

## Tree-sitter failure

Check:

- Python version
- language package
- parser registration

## Semgrep missing

Install the configured dependency or provide the documented fallback path.

## Port conflict

Identify the process using the port and either terminate it or change:

```text
API_PORT
```

---

# 57. Performance Strategy

Use:

```text
embedding batching
embedding cache
persistent FAISS
incremental ingestion
configurable top-k
```

Avoid premature complexity.

Do not add:

```text
Kubernetes
Kafka
Redis
microservices
cloud databases
```

unless a real requirement appears.

---

# 58. Security-by-Design Rules

The project must:

- isolate clean and poisoned data
- preserve provenance
- never trust retrieved text as privileged instructions
- log defense decisions
- preserve security evidence
- avoid secret ingestion
- avoid unauthorized repositories
- avoid real-world exploitation
- keep poisoning experiments controlled and synthetic/authorized

---

# 59. Research Integrity Rules

Never fabricate:

```text
accuracy
recall
PRR
ASR
VIR
detection rate
security improvement
```

If not measured:

```text
NOT MEASURED
```

If code exists but was not run:

```text
IMPLEMENTED — NOT VERIFIED
```

---

# 60. Final Research Questions the System Must Answer

The finished project should enable questions such as:

### Retrieval

- Was poisoned content retrieved?
- How often?
- How highly ranked?

### Influence

- Was poisoned content inserted into the LLM context?
- Did the model output change?

### Security

- Did vulnerable behavior appear?
- Was it caught?

### Defense

- Which defense layer caught it?
- What was the false-positive cost?

### Utility

- Did defense degrade useful retrieval?
- Did code correctness change?

### Ablation

- What does each layer contribute?

---

# 61. Final Project Lifecycle

```text
                PROJECT START
                     │
                     ▼
             FOUNDATION AUDIT
                     │
                     ▼
             ENVIRONMENT SETUP
                     │
                     ▼
             INGESTION WORKS
                     │
                     ▼
               AST WORKS
                     │
                     ▼
             CHUNKING WORKS
                     │
                     ▼
          HF EMBEDDINGS WORK
                     │
                     ▼
             FAISS WORKS
                     │
                     ▼
             RETRIEVAL WORKS
                     │
                     ▼
           OLLAMA GENERATION
                     │
                     ▼
          CLEAN RAG BASELINE
                     │
                     ▼
          BASELINE EVALUATION
                     │
                     ▼
          CONTROLLED POISONING
                     │
                     ▼
            POISON METRICS
                     │
                     ▼
             DEFENSE LAYERS
                     │
                     ▼
          OUTPUT VALIDATION
                     │
                     ▼
          EXPERIMENT RUNNER
                     │
                     ▼
        CODERAG-POISONBENCH
                     │
                     ▼
           ABLATION STUDIES
                     │
                     ▼
        SECURITY–UTILITY STUDY
                     │
                     ▼
                FASTAPI
                     │
                     ▼
          OPTIONAL DASHBOARD
                     │
                     ▼
                 DOCKER
                     │
                     ▼
            FINAL DOCUMENTATION
                     │
                     ▼
                VIVA READY
```

---

# 62. What the Project Looks Like After All Phases

The final repository should look approximately like:

```text
Final-Year-Project/
│
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
│
├── configs/
│   ├── default_config.json
│   ├── baseline.json
│   ├── poisoning.json
│   ├── defense.json
│   └── experiment_matrix.json
│
├── data/
│   ├── clean/
│   │   └── repositories/
│   ├── poisoned/
│   │   └── repositories/
│   ├── processed/
│   └── benchmarks/
│
├── docs/
│   ├── architecture.md
│   ├── threat-model.md
│   ├── dataset.md
│   ├── experiments.md
│   ├── metrics.md
│   ├── reproducibility.md
│   ├── api.md
│   └── deployment.md
│
├── experiments/
│   ├── configs/
│   └── runs/
│
├── notebooks/
│
├── results/
│   ├── raw/
│   ├── metrics/
│   ├── plots/
│   └── reports/
│
├── scripts/
│   ├── health_check.py
│   ├── prepare_dataset.py
│   ├── ingest.py
│   ├── build_index.py
│   ├── run_pipeline.py
│   ├── run_experiment.py
│   └── generate_report.py
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── models.py
│   ├── pipeline.py
│   ├── ingestion/
│   ├── parsing/
│   ├── chunking/
│   ├── embeddings/
│   ├── vectorstore/
│   ├── retrieval/
│   ├── generation/
│   ├── poisoning/
│   ├── defense/
│   ├── security/
│   ├── evaluation/
│   └── api/
│
└── tests/
    ├── __init__.py
    ├── test_foundation.py
    ├── unit/
    ├── integration/
    ├── security/
    ├── regression/
    └── e2e/
```

---

# 63. Final User Workflow

A developer using the completed project should be able to do:

```powershell
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Check environment
python scripts/health_check.py

# 3. Run quality checks
ruff check .
ruff format --check .

# 4. Run tests
pytest

# 5. Prepare dataset
python scripts/prepare_dataset.py

# 6. Ingest
python scripts/ingest.py --source data/clean/repositories/example

# 7. Build index
python scripts/build_index.py --dataset clean

# 8. Run baseline
python scripts/run_pipeline.py --query "Create a secure REST API endpoint"

# 9. Run experiment
python scripts/run_experiment.py --config experiments/configs/baseline.json

# 10. Generate report
python scripts/generate_report.py

# 11. Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The exact commands may evolve as implementation progresses, but the final README must contain the actual verified commands.

---

# 64. What We Should NOT Build

Do not add complexity simply to make the project look advanced.

Unless a documented requirement appears, do not add:

```text
Kubernetes
Kafka
Redis
Celery
microservices
cloud database
paid model API
complex orchestration frameworks
large distributed infrastructure
```

The focus is:

```text
research correctness
+
reproducibility
+
security evaluation
+
working RAG
```

---

# 65. What Makes This a Complete Final-Year Project

The project is complete only when three levels are satisfied.

## Engineering

```text
code works
tests work
deployment works
```

## Research

```text
poisoning works
defense works
metrics work
experiments work
benchmark works
```

## Academic

```text
architecture can be explained
methodology can be defended
results are reproducible
limitations are documented
viva questions can be answered
```

---

# 66. Final Completion Criteria

```text
[ ] Environment documented
[ ] No paid AI API dependency
[ ] Hugging Face local embeddings work
[ ] Ollama local LLM works
[ ] Ingestion works
[ ] AST parsing works
[ ] AST chunking works
[ ] Embeddings work
[ ] FAISS works
[ ] Retrieval works
[ ] Clean RAG works
[ ] Baseline evaluation works
[ ] Seven poisoning categories implemented
[ ] PRR works
[ ] ASR works
[ ] VIR works
[ ] L1 trust scoring works
[ ] L2 anomaly detection works
[ ] L3 context validation works
[ ] L4 instruction/data separation works
[ ] L5 static analysis works
[ ] Generated-code validation works
[ ] Experiment runner works
[ ] Benchmark works
[ ] Ablation works
[ ] Security–utility analysis works
[ ] Results are reproducible
[ ] API works
[ ] Docker works
[ ] Ruff passes
[ ] Unit tests pass
[ ] Integration tests pass
[ ] Security tests pass
[ ] E2E tests pass
[ ] README complete
[ ] Technical documentation complete
[ ] Final report artifacts generated
[ ] Viva preparation complete
```

---

# 67. Master Rule for Future Implementation Sessions

When implementing this project in an AI coding environment:

## NEVER

```text
Generate all files blindly
```

## ALWAYS

```text
Inspect
→ plan
→ implement
→ lint
→ test
→ run
→ verify
→ document
→ commit
→ next
```

The AI must preserve the implementation ledger:

```text
COMPLETED
CURRENT
NEXT
REMAINING
```

---

# 68. How the AI Should Respond to "NEXT"

When the developer says:

```text
NEXT
```

respond with:

```text
CURRENT PHASE:
CURRENT FILE:

WHY THIS FILE:

DEPENDENCIES:

FILES IT DEPENDS ON:

FILES THAT DEPEND ON IT:

COMPLETE CODE:

CONFIGURATION:

INSTALLATION:

TEST:

RUN COMMAND:

EXPECTED OUTPUT:

VERIFICATION:

RUFF RESULT:

NEXT FILE:
```

Do not restart from the beginning.

Do not repeat already completed files.

---

# 69. How the AI Should Respond to an Error

When the developer provides an error:

```text
ERROR
↓
ROOT CAUSE
↓
AFFECTED FILE
↓
MINIMAL FIX
↓
COMPLETE UPDATED FILE
↓
TEST COMMAND
↓
EXPECTED RESULT
```

Do not rewrite unrelated modules.

---

# 70. Final Mental Model

The whole project can be remembered as:

```text
                UNDERSTAND
                    ↓
                 INGEST
                    ↓
                  PARSE
                    ↓
                 CHUNK
                    ↓
                EMBED
                    ↓
                 INDEX
                    ↓
                RETRIEVE
                    ↓
                GENERATE
                    ↓
                 ATTACK
                    ↓
                DEFEND
                    ↓
                ANALYZE
                    ↓
                MEASURE
                    ↓
                COMPARE
                    ↓
               DOCUMENT
                    ↓
                DEPLOY
```

And the central research chain is:

```text
POISON
  ↓
RETRIEVAL
  ↓
CONTEXT
  ↓
GENERATION
  ↓
VULNERABILITY
  ↓
DEFENSE
  ↓
MEASUREMENT
```

---

# 71. Final Principle

The objective is not:

> "Create many files."

The objective is:

> **Build a real, local-first, reproducible SecureCodeRAG research system in which every important component works, every experiment is measurable, every result is traceable, every security decision is explainable, and the final project can be demonstrated and defended academically.**

