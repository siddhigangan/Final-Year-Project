# CodeRAG-PoisonBench

## What it measures

For each sample the benchmark runs three passes:

| Pass | Meaning |
|------|---------|
| clean | Clean code in context, no defense |
| poisoned | Poisoned code in context, no defense |
| defended | Poisoned code in context, defense pipeline on |

Metrics: `clean_success_rate`, `poisoned_success_rate`,
`defended_success_rate`, `attack_impact`
(clean minus poisoned) and `detection_rate`.

## Running it

```powershell
python -m scripts.run_benchmark
```

Writes `results/benchmark_report.json`.

## Important limitation

The default offline run uses `EchoGenerationProvider`, which returns the
retrieved context as the "generated" code. This is deterministic and
needs no API key, but it does **not** simulate an LLM.

Only `vulnerable_code` poisoning is measurable this way, because it
plants a syntactic vulnerability that static analysis can see. The other
six categories plant comments or text that mislead an LLM. Static
analysis cannot see them, so the report marks them `"measured": false`
instead of inventing a number.

To measure all seven categories, inject a real `GenerationProvider`
(local Ollama or hosted OpenAI) into `build_benchmark_runner`.

The aggregate block averages measured and unmeasured categories
together. Read the `per_category` block for honest results.

## Extending

- New poisoning strategy: add a class in `src/poisoning/`, add it to
  `ALL_POISONING_STRATEGIES` in `src/benchmark/integration.py`.
- New seed samples: edit `SEEDS` in `scripts/run_benchmark.py`.

