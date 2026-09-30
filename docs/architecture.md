# Architecture

SecureCodeRAG evaluates how poisoned retrieval context affects code
generation, and how much a layered defense recovers.

## Pipeline

```
Knowledge base (clean / poisoned)
  -> ingestion & AST chunking      src/ingestion, src/parsing, src/chunking
  -> embeddings & vector store     src/embeddings, src/vectorstore
  -> retrieval & context building  src/retrieval
  -> defense (L1-L5)               src/defense, src/security
  -> code generation               src/generation
  -> evaluation & metrics          src/evaluation, src/experiments
```

## Defense layers

| Layer | Purpose |
|-------|---------|
| L1 | Source trust scoring |
| L2 | Anomaly detection |
| L3 | Retrieved-context validation |
| L4 | Instruction / data separation |
| L5 | Static analysis (AST, optional Semgrep) |

## Experiment conditions

Defined in `experiments/*.json` and loaded by
`src/experiments/config_loader.py`: clean baseline, poisoned without
defense, poisoned with defense, clean with defense.

## Key design rules

- External services (LLM providers, embedding APIs) are injected, never
  hardcoded, so everything runs offline in tests.
- No API keys in the repository. Use `.env` (see `.env.example`).
- Runs are deterministic given a seed.

