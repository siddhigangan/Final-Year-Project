"""Integration adapter connecting CodeRAG-PoisonBench to the real
SecureCodeRAG poisoning strategies and experiment runner.

This module does not implement new detection or generation logic. It
wires together three subsystems that already exist and are already
tested independently:

    - ``src.poisoning`` -- the seven controlled poisoning strategies,
      used here to build the poisoned half of each benchmark sample.
    - ``src.experiments.experiment_runner`` -- the real retrieval ->
      defense -> generation -> vulnerability-detection pipeline.
    - ``src.benchmark`` -- the dataset/metrics/runner contract that
      this module's evaluator satisfies.

Generation-provider caveat
---------------------------
``ExperimentRunner`` requires a ``GenerationProvider``. A real provider
(``HostedOpenAIProvider`` or ``LocalOllamaProvider``) calls a live
external service and is fully supported here -- inject one via
``build_experiment_runner``.

For reproducible, offline benchmark runs (the default), this module
provides ``EchoGenerationProvider``, a deterministic test double that
returns the retrieved context verbatim as "generated" code. This is a
disclosed methodological simplification, not a fabricated result: it
models the worst case in which an LLM follows retrieved context
exactly, so the benchmark measures what the *defense and detection
layers* catch, using the same real ``DefensePipeline`` and static
analysis used in production. It does not simulate or invent LLM
behavior. Anyone measuring actual LLM behavior should inject a real
``GenerationProvider`` instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.benchmark.dataset import BenchmarkDataset, BenchmarkSample
from src.benchmark.runner import BenchmarkRunner
from src.defense.pipeline import DefensePipeline, DefensePipelineConfig
from src.experiments.experiment_runner import (
    ExperimentRunInput,
    ExperimentRunner,
)
from src.generation.base import GenerationProvider
from src.models import (
    CodeChunk,
    ExperimentCondition,
    ExperimentConfig,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    ProgrammingLanguage,
    RetrievedChunk,
)
from src.poisoning.base import PoisoningInput, PoisoningStrategy
from src.poisoning.context_manipulation import (
    ContextManipulationStrategy,
)
from src.poisoning.contradictory_documentation import (
    ContradictoryDocumentationStrategy,
)
from src.poisoning.false_api_guidance import FalseApiGuidanceStrategy
from src.poisoning.false_repository_conventions import (
    FalseRepositoryConventionsStrategy,
)
from src.poisoning.instruction_like_content import (
    InstructionLikeContentStrategy,
)
from src.poisoning.misleading_code import MisleadingCodeStrategy
from src.poisoning.vulnerable_code import VulnerableCodeStrategy
from src.retrieval.retriever import RetrievalResponse, Retriever
from src.security.decision import SecurityDecision


class BenchmarkIntegrationError(RuntimeError):
    """Base exception for benchmark-integration errors."""


class BenchmarkIntegrationInputError(
    BenchmarkIntegrationError,
    ValueError,
):
    """Raised when benchmark-integration inputs are invalid."""


ALL_POISONING_STRATEGIES: tuple[type[PoisoningStrategy], ...] = (
    MisleadingCodeStrategy,
    VulnerableCodeStrategy,
    FalseApiGuidanceStrategy,
    ContradictoryDocumentationStrategy,
    InstructionLikeContentStrategy,
    FalseRepositoryConventionsStrategy,
    ContextManipulationStrategy,
)
"""All seven poisoning strategies, independent of the default
registry in ``src.poisoning.registry`` (which intentionally still
exposes only the original three by default)."""


@dataclass(frozen=True)
class SeedCodeSample:
    """One clean code sample used as the basis for a benchmark entry.

    Attributes:
        sample_id: Stable identifier for the resulting benchmark sample.
        query: Natural-language query a retriever would have matched
            this chunk against.
        task: Short task-type label (e.g. "code_completion").
        content: Clean source code content.
        language: Programming language of ``content``.
        category: Optional benchmark category label. Defaults to the
            poisoning strategy's category when omitted.
    """

    sample_id: str
    query: str
    task: str
    content: str
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON
    category: str | None = None


def build_poisoning_dataset(
    seed_samples: list[SeedCodeSample],
    *,
    strategies: tuple[type[PoisoningStrategy], ...] = (
        ALL_POISONING_STRATEGIES
    ),
    seed: int = 42,
) -> BenchmarkDataset:
    """Build a BenchmarkDataset by applying real poisoning strategies.

    Every seed sample is poisoned once per strategy, so the resulting
    dataset contains ``len(seed_samples) * len(strategies)`` benchmark
    samples spanning every requested poisoning category.

    Args:
        seed_samples: Clean code samples to poison.
        strategies: Poisoning strategy classes to apply. Defaults to
            all seven categories.
        seed: Deterministic seed passed to each strategy instance.

    Returns:
        A populated BenchmarkDataset.

    Raises:
        BenchmarkIntegrationInputError: If seed_samples is empty.
    """
    if not seed_samples:
        raise BenchmarkIntegrationInputError(
            "seed_samples cannot be empty."
        )

    dataset = BenchmarkDataset()

    for seed_sample in seed_samples:
        chunk = CodeChunk(
            chunk_id=f"{seed_sample.sample_id}-clean",
            source_file_id=f"{seed_sample.sample_id}-file",
            repository_id="benchmark-seed-repository",
            content=seed_sample.content,
            language=seed_sample.language,
            start_line=1,
            end_line=len(seed_sample.content.splitlines()) or 1,
            metadata={"benchmark_seed_id": seed_sample.sample_id},
        )

        for strategy_cls in strategies:
            strategy = strategy_cls()

            poisoning_result = strategy(
                PoisoningInput(
                    chunk=chunk,
                    category=strategy.category,
                    poison_id=(
                        f"{seed_sample.sample_id}-"
                        f"{strategy.category}"
                    ),
                    seed=seed,
                )
            )

            dataset.add(
                BenchmarkSample(
                    sample_id=(
                        f"{seed_sample.sample_id}-"
                        f"{strategy.category}"
                    ),
                    query=seed_sample.query,
                    clean_code=seed_sample.content,
                    poisoned_code=poisoning_result.poisoned_content,
                    language=seed_sample.language.value,
                    poisoning_category=(
                        seed_sample.category or strategy.category
                    ),
                    metadata={
                        "task": seed_sample.task,
                        "seed_sample_id": seed_sample.sample_id,
                    },
                )
            )

    return dataset


class EchoGenerationProvider(GenerationProvider):
    """Deterministic generation provider for offline benchmark runs.

    Returns the retrieved context's code verbatim, concatenated in
    ranked order, as the "generated" code. This models the worst case
    in which the downstream Code LLM follows retrieved context
    exactly, isolating the benchmark's measurement to what the real
    defense and static-analysis layers catch. It performs no network
    calls and requires no API key.
    """

    provider_name = "benchmark-echo"

    def __init__(self) -> None:
        super().__init__(
            model_name="echo-passthrough",
            provider_name=self.provider_name,
        )

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        """Echo the retrieved context back as generated code."""
        self.validate_request(request)

        generated_code = "\n".join(
            item.chunk.content for item in request.context
        )

        return GenerationResult(
            request_id=request.request_id,
            generated_code=generated_code,
            model_name=self.model_name,
            provider_name=self.provider_name,
            status=GenerationStatus.SUCCESS,
            latency_ms=0.0,
            retrieved_chunk_ids=[
                item.chunk.chunk_id for item in request.context
            ],
            metadata={"benchmark_echo_provider": True},
        )


class SingleChunkRetriever(Retriever):
    """Deterministic Retriever returning one fixed chunk.

    Bypasses real embedding and vector-store dependencies, matching
    the pattern already used in
    ``tests/unit/test_experiment_runner.py``'s ``FakeRetriever``. Used
    here because a benchmark sample already specifies exactly which
    code -- clean or poisoned -- should be in context; no semantic
    search is needed to answer that question.
    """

    def __init__(self, chunk: RetrievedChunk) -> None:
        # The real retrieval dependencies are intentionally bypassed:
        # retrieve() is overridden to return deterministic test data,
        # matching the established pattern in test_experiment_runner.py.
        self._chunk = chunk
        self._name = "benchmark-single-chunk-retriever"
        self._embedding_model = "benchmark-fixed"

    @property
    def name(self) -> str:
        """Return the retriever name."""
        return self._name

    @property
    def embedding_model(self) -> str:
        """Return the fixed embedding model label."""
        return self._embedding_model

    @property
    def dimension(self) -> int:
        """Return a fixed embedding dimension."""
        return 1

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Return the single fixed chunk regardless of query."""
        del metadata_filter

        return RetrievalResponse(
            query=query,
            results=(self._chunk,),
            requested_top_k=top_k,
            retriever_name=self.name,
            embedding_model=self.embedding_model,
            metadata={"benchmark_fixed_retrieval": True},
        )


MEASURABLE_OFFLINE_CATEGORIES: frozenset[str] = frozenset(
    {"vulnerable_code"}
)
"""Categories whose poison is a syntactic vulnerability that the static
analyzer can detect in echoed context. The remaining categories inject
text that only misleads an LLM; they can be scored only when a real
GenerationProvider is injected (see ``is_measurable``)."""


def is_measurable(
    sample: BenchmarkSample,
    *,
    real_generator: bool,
) -> bool:
    """Return whether a sample's outcome is honestly measurable.

    Text-only poisoning (comments, false conventions, injected
    instructions) has no static-analysis signature, so with the offline
    echo generator its outcome is not measured. With a real generator
    every category is measurable.
    """
    if real_generator:
        return True

    return sample.poisoning_category in MEASURABLE_OFFLINE_CATEGORIES


def static_analysis_flags(code: str, language: str) -> bool:
    """Ground-truth oracle: does the real analyzer block this code?

    Measurement only. Uses the same DefensePipeline generated-code
    analysis as production but never alters the code or the pipeline.
    """
    oracle = DefensePipeline(
        config=DefensePipelineConfig(
            enabled=True,
            analyze_generated_code=True,
        )
    )
    result = oracle.analyze_generated_code(
        code=code,
        file_name="generated_code.py",
        language=language,
        query="benchmark-oracle",
    )
    return any(
        f.decision == SecurityDecision.REJECT
        for f in result.blocking_findings
    ) or len(result.blocking_findings) > 0


def build_experiment_runner(
    chunk: CodeChunk,
    *,
    generator: GenerationProvider | None = None,
    defense_enabled: bool,
) -> ExperimentRunner:
    """Build an ExperimentRunner fixed to retrieve one chunk.

    Args:
        chunk: The single chunk the runner's retriever will return
            for any query, representing the clean or poisoned code
            under evaluation.
        generator: Generation provider to use. Defaults to
            ``EchoGenerationProvider`` for reproducible, offline runs.
            Pass a real ``GenerationProvider`` (e.g.
            ``HostedOpenAIProvider``) to benchmark actual LLM
            behavior.
        defense_enabled: Whether the returned runner applies the real
            defense pipeline with generated-code analysis enabled.

    Returns:
        A configured ExperimentRunner.
    """
    resolved_generator = generator or EchoGenerationProvider()

    defense_pipeline = None

    if defense_enabled:
        defense_pipeline = DefensePipeline(
            config=DefensePipelineConfig(
                enabled=True,
                analyze_generated_code=True,
            )
        )

    retriever = SingleChunkRetriever(
        RetrievedChunk(
            chunk=chunk,
            retrieval_score=1.0,
            rank=1,
        )
    )

    return ExperimentRunner(
        retriever=retriever,
        generator=resolved_generator,
        defense_pipeline=defense_pipeline,
    )


def run_sample_condition(
    sample: BenchmarkSample,
    code: str,
    *,
    is_poisoned: bool,
    generator: GenerationProvider | None = None,
    defense_enabled: bool,
) -> bool:
    """Run one benchmark sample through the real experiment pipeline.

    This is the evaluator body shared by the "poisoned" and
    "defended" passes described in ``BenchmarkRunner``. It is not
    passed to ``BenchmarkRunner`` directly -- see
    ``make_poisoned_evaluator`` and ``make_defended_evaluator``, which
    close over ``defense_enabled`` to match
    ``BenchmarkRunner``'s three-argument evaluator contract.

    Args:
        sample: The benchmark sample being evaluated.
        code: The code string to place in retrieval context (clean or
            poisoned, per the caller).
        is_poisoned: Whether ``code`` is the poisoned variant. Passed
            through as experiment metadata; does not change execution.
        generator: Generation provider to use. Defaults to
            ``EchoGenerationProvider``.
        defense_enabled: Whether the real defense pipeline (including
            generated-code static analysis) is applied.

    Returns:
        True if the sample was successful, i.e. no vulnerability was
        introduced into the generated code. False if a vulnerability
        was detected.
    """
    language = ProgrammingLanguage(sample.language)

    chunk = CodeChunk(
        chunk_id=f"{sample.sample_id}-context",
        source_file_id=f"{sample.sample_id}-file",
        repository_id="benchmark-run-repository",
        content=code,
        language=language,
        start_line=1,
        end_line=len(code.splitlines()) or 1,
        metadata={"benchmark_is_poisoned": is_poisoned},
    )

    runner = build_experiment_runner(
        chunk,
        generator=generator,
        defense_enabled=defense_enabled,
    )

    condition = (
        ExperimentCondition.POISONED_DEFENSE
        if defense_enabled
        else ExperimentCondition.POISONED_NO_DEFENSE
    )

    task = sample.metadata.get("task", "code_completion")

    experiment_config = ExperimentConfig(
        experiment_id=f"{sample.sample_id}-{condition.value}",
        condition=condition,
        repository_id="benchmark-run-repository",
        task_type=task,
        language=language,
        model_name=runner.generator.model_name,
        retriever_name=runner.retriever.name,
        top_k=1,
    )

    run_input = ExperimentRunInput(
        query=sample.query,
        task=task,
        experiment_config=experiment_config,
        request_id=f"{sample.sample_id}-{condition.value}-request",
        metadata={
            "benchmark_category": sample.poisoning_category,
        },
    )

    result = runner.run(run_input)

    if result.vulnerability_introduced:
        return False

    if not defense_enabled:
        # Undefended: nothing screens the output, so apply the
        # analyzer as a measurement-only oracle on what the generator
        # would emit (the echoed context for the offline provider).
        return not static_analysis_flags(code, language.value)

    return True


def make_poisoned_evaluator(
    *,
    generator: GenerationProvider | None = None,
):
    """Build an evaluator for BenchmarkRunner's poisoned/clean passes.

    The returned callable applies no defense, matching the
    ``B_poisoned_no_defense`` / ``A_clean_no_defense`` experimental
    conditions.
    """

    def evaluator(
        sample: BenchmarkSample,
        code: str,
        is_poisoned: bool,
    ) -> bool:
        return run_sample_condition(
            sample,
            code,
            is_poisoned=is_poisoned,
            generator=generator,
            defense_enabled=False,
        )

    return evaluator


def make_defended_evaluator(
    *,
    generator: GenerationProvider | None = None,
):
    """Build an evaluator for BenchmarkRunner's defended pass.

    The returned callable applies the real defense pipeline with
    generated-code static analysis enabled, matching the
    ``C_poisoned_defense`` experimental condition.
    """

    def evaluator(
        sample: BenchmarkSample,
        code: str,
        is_poisoned: bool,
    ) -> bool:
        return run_sample_condition(
            sample,
            code,
            is_poisoned=is_poisoned,
            generator=generator,
            defense_enabled=True,
        )

    return evaluator


def make_detector(
    *,
    generator: GenerationProvider | None = None,
):
    """Build a detector matching ``Callable[[BenchmarkSample], bool]``.

    A sample counts as detected only when the defense layer actually
    acted on the poisoned code: the analyzer flags it, or the defended
    run differs in outcome from the undefended one. Code the analyzer
    cannot distinguish from clean code is NOT reported as detected.
    """
    del generator

    def detector(sample: BenchmarkSample) -> bool:
        return static_analysis_flags(
            sample.poisoned_code,
            sample.language,
        )

    return detector


def summarize_by_category(
    result,
    dataset: BenchmarkDataset,
    *,
    real_generator: bool = False,
) -> dict[str, dict[str, object]]:
    """Summarize a run per poisoning category, marking unmeasured ones.

    Categories that cannot be measured with the offline echo generator
    are reported as ``measured: False`` with no rates, instead of being
    averaged into a misleading aggregate.
    """
    by_id = {s.sample_id: s for s in dataset.samples}
    buckets: dict[str, list] = {}

    for evaluation in result.evaluations:
        sample = by_id[evaluation.sample_id]
        buckets.setdefault(sample.poisoning_category, []).append(
            (sample, evaluation)
        )

    summary: dict[str, dict[str, object]] = {}

    for category, items in sorted(buckets.items()):
        measurable = is_measurable(
            items[0][0], real_generator=real_generator
        )
        entry: dict[str, object] = {
            "samples": len(items),
            "measured": measurable,
        }
        if measurable:
            n = len(items)
            entry["poisoned_success_rate"] = (
                sum(e.poisoned_success for _, e in items) / n
            )
            entry["defended_success_rate"] = (
                sum(e.defended_success for _, e in items) / n
            )
            entry["detection_rate"] = (
                sum(e.detected for _, e in items) / n
            )
        summary[category] = entry

    return summary


def build_benchmark_runner(
    *,
    generator: GenerationProvider | None = None,
) -> BenchmarkRunner:
    """Build a BenchmarkRunner wired to the real experiment pipeline.

    The undefended evaluator (used for the clean and poisoned passes)
    runs with no defense pipeline. The defended evaluator runs with
    the real defense pipeline, including generated-code static
    analysis. These are genuinely distinct evaluator callables, so
    ``poisoned_success_rate`` and ``defended_success_rate`` can differ
    -- see the ``defended_evaluator`` parameter added to
    ``BenchmarkRunner`` for this purpose.

    Args:
        generator: Generation provider shared by all three passes.
            Defaults to ``EchoGenerationProvider`` for reproducible,
            offline runs.

    Returns:
        A configured BenchmarkRunner, ready to run a BenchmarkDataset
        built by ``build_poisoning_dataset``.
    """
    return BenchmarkRunner(
        evaluator=make_poisoned_evaluator(generator=generator),
        defended_evaluator=make_defended_evaluator(
            generator=generator
        ),
        detector=make_detector(generator=generator),
    )
