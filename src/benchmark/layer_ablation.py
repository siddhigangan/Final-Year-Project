"""Defense layer ablation: what does each L1-L5 layer contribute?

Blueprint section 32 specifies the ladder no_defense -> L1 -> L1_L2 ->
L1_L2_L3 -> L1_L2_L3_L4 -> L1_L2_L3_L4_L5, to isolate each layer's
contribution to the final block/flag decision. The stage names and
order come from src.experiments.ablation.ABLATION_STAGES, the
project's existing canonical (tested, validated) definition of this
ladder -- this module adds execution against the real defense
pipeline on top of that definition, rather than redefining it.

``DefensePipelineConfig`` has no per-layer enable flags: all five
layers always run together (see src/defense/pipeline.py). Rather than
modify that tested module, this runs the full pipeline once per
sample (all five layers execute, exactly as in production), then
re-evaluates the block decision using only the subset of findings that
belong to each ladder step. The decision engine
(``SecurityDecisionEngine.evaluate``) is a pure function over a
findings list, so replaying it on a filtered subset reproduces exactly
what running only that subset of layers would have decided, without
touching pipeline.py.

Important, measured caveat: L3 (context validation) does not currently
emit any SecurityFinding at all (see pipeline.py's _collect_findings,
which has no L3 case) -- it only sanitizes the built context, and
therefore never changes the block decision. Any ladder step that
"adds" L3 will show identical results to the step before it. This is
not a bug in this module; it reflects the real pipeline as built, and
is reported explicitly rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.benchmark.dataset import BenchmarkDataset, BenchmarkSample
from src.defense.pipeline import DefensePipeline, DefensePipelineConfig
from src.experiments.ablation import ABLATION_STAGES, DefenseLayer
from src.models import CodeChunk, ProgrammingLanguage, RetrievedChunk
from src.retrieval.context_builder import ContextBuilder
from src.security.decision import SecurityDecisionEngine
from src.security.findings import SecurityFinding

# Rule-id prefixes each canonical layer's findings use. L3 has none: it
# never produces a SecurityFinding in the current pipeline (see module
# docstring), so it contributes no prefix and its step matches the one
# before it exactly.
_LAYER_RULE_PREFIXES: dict[DefenseLayer, tuple[str, ...]] = {
    DefenseLayer.L1: ("DEF-L1",),
    DefenseLayer.L2: ("DEF-L2",),
    DefenseLayer.L3: (),
    DefenseLayer.L4: ("DEF-L4",),
    # L5 (static analysis) findings come straight from the analyzer,
    # not a hand-built DEF-L5-* id; see pipeline.py's _collect_findings.
    DefenseLayer.L5: ("PY-", "JS-", "GENERIC-"),
}

# Ladder step name -> rule_id prefixes included at that step (cumulative),
# derived from the project's single canonical stage list in
# src.experiments.ablation so this module cannot silently diverge from it.
LAYER_PREFIXES: dict[str, tuple[str, ...]] = {
    stage.name: tuple(
        prefix
        for layer in stage.layers
        for prefix in _LAYER_RULE_PREFIXES[layer]
    )
    for stage in ABLATION_STAGES
}
LADDER: tuple[str, ...] = tuple(LAYER_PREFIXES)


def _matches(finding: SecurityFinding, prefixes: tuple[str, ...]) -> bool:
    if not prefixes:
        return False
    return any(finding.rule_id.startswith(p) for p in prefixes)


@dataclass(frozen=True)
class LadderStepOutcome:
    """Whether one ladder step would have blocked one sample."""

    step: str
    blocked: bool
    finding_count: int


@dataclass(frozen=True)
class AblationSampleResult:
    """Every ladder step's outcome for one sample."""

    sample_id: str
    category: str
    all_findings: tuple[SecurityFinding, ...]
    steps: tuple[LadderStepOutcome, ...]

    def step(self, name: str) -> LadderStepOutcome:
        return next(s for s in self.steps if s.step == name)


def _screen_full(sample: BenchmarkSample, code: str, pipeline: DefensePipeline):
    chunk = CodeChunk(
        chunk_id=f"{sample.sample_id}-ablation",
        source_file_id=f"{sample.sample_id}-file",
        repository_id="benchmark-ablation-repository",
        content=code,
        language=ProgrammingLanguage(sample.language),
        start_line=1,
        end_line=len(code.splitlines()) or 1,
    )
    retrieved = RetrievedChunk(chunk=chunk, retrieval_score=1.0, rank=1)
    context = ContextBuilder().build([retrieved], query=sample.query)
    return pipeline.analyze_context(
        query=sample.query, context=context, retrieved_chunks=[retrieved]
    )


def evaluate_ablation(
    sample: BenchmarkSample,
    *,
    pipeline: DefensePipeline,
    engine: SecurityDecisionEngine | None = None,
) -> AblationSampleResult:
    """Run the layer ablation ladder for one sample's poisoned code.

    The full pipeline runs once (all five layers execute for real);
    each ladder step re-evaluates the block decision on a filtered
    subset of the resulting findings.
    """
    decision_engine = engine or SecurityDecisionEngine()
    result = _screen_full(sample, sample.poisoned_code, pipeline)

    all_findings = (
        result.decision_report.findings
        if result.decision_report is not None
        else ()
    )

    steps = []
    for step_name, prefixes in LAYER_PREFIXES.items():
        subset = tuple(f for f in all_findings if _matches(f, prefixes))
        report = decision_engine.evaluate(subset)
        steps.append(
            LadderStepOutcome(
                step=step_name,
                blocked=report.decision.value == "reject",
                finding_count=len(subset),
            )
        )

    return AblationSampleResult(
        sample_id=sample.sample_id,
        category=sample.poisoning_category,
        all_findings=all_findings,
        steps=tuple(steps),
    )


def evaluate_layer_ablation(
    dataset: BenchmarkDataset,
    pipeline: DefensePipeline | None = None,
) -> dict[str, dict[str, float | int]]:
    """Per-ladder-step block rate across every sample in the dataset."""
    pipe = pipeline or DefensePipeline(config=DefensePipelineConfig(enabled=True))
    results = [evaluate_ablation(s, pipeline=pipe) for s in dataset.samples]

    summary: dict[str, dict[str, float | int]] = {}
    for step_name in LADDER:
        blocked = [r.step(step_name).blocked for r in results]
        summary[step_name] = {
            "samples": len(blocked),
            "block_rate": sum(blocked) / len(blocked) if blocked else 0.0,
        }
    return summary


def evaluate_layer_ablation_by_category(
    dataset: BenchmarkDataset,
    pipeline: DefensePipeline | None = None,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Per-category, per-ladder-step block rate.

    This is the view that actually answers "which layer caught it":
    a category whose block_rate jumps from 0.0 to 1.0 between two
    adjacent steps was caught by the layer added at that step.
    """
    pipe = pipeline or DefensePipeline(config=DefensePipelineConfig(enabled=True))
    by_category: dict[str, list[AblationSampleResult]] = {}
    for sample in dataset.samples:
        by_category.setdefault(sample.poisoning_category, []).append(
            evaluate_ablation(sample, pipeline=pipe)
        )

    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for category, results in sorted(by_category.items()):
        out[category] = {}
        for step_name in LADDER:
            blocked = [r.step(step_name).blocked for r in results]
            out[category][step_name] = {
                "samples": len(blocked),
                "block_rate": sum(blocked) / len(blocked) if blocked else 0.0,
            }
    return out
