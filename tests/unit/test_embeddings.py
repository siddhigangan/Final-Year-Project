"""Unit tests for the SecureCodeRAG embedding subsystem."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import AppConfig
from src.embeddings.base import (
    EmbeddingConfigurationError,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingProvider,
    normalize_embeddings,
    normalize_vector,
)
from src.embeddings.cache import EmbeddingCache, EmbeddingCacheError
from src.embeddings.factory import EmbeddingFactory, EmbeddingFactoryError
from src.embeddings.hf_code_embeddings import HFCodeEmbeddingProvider
from src.models import (
    CodeChunk,
    ProgrammingLanguage,
    SourceTrust,
)


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic provider used to test the embedding interface."""

    def __init__(
        self,
        *,
        model_name: str = "fake-code-model",
        dimension: int = 4,
        normalize: bool = False,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        self._model_name = model_name
        self._dimension = dimension
        self._normalize = normalize

    @property
    def model_name(self) -> str:
        """Return the fake model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the fake embedding dimension."""
        return self._dimension

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate deterministic vectors from text content."""
        if texts is None:
            raise EmbeddingInputError("texts cannot be None.")

        if not isinstance(texts, list):
            raise EmbeddingInputError("texts must be a list.")

        validated_texts: list[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise EmbeddingInputError(
                    f"texts[{index}] must be a string."
                )

            if not text.strip():
                raise EmbeddingInputError(
                    f"texts[{index}] cannot be empty."
                )

            validated_texts.append(text)

        if not validated_texts:
            return np.empty(
                (0, self._dimension),
                dtype=np.float32,
            )

        vectors: list[np.ndarray] = []

        for text in validated_texts:
            value = float(sum(ord(char) for char in text) % 1000)

            vector = np.full(
                self._dimension,
                value,
                dtype=np.float32,
            )

            vectors.append(vector)

        embeddings = np.vstack(vectors)

        if self._normalize:
            return normalize_embeddings(embeddings)

        return embeddings


def build_config(**overrides: object) -> AppConfig:
    """Build a minimal AppConfig for factory tests."""
    values: dict[str, object] = {
        "project_name": "SecureCodeRAG",
        "version": "0.1.0",
        "log_level": "INFO",
        "log_file": "logs/app.log",
        "chunk_size": 512,
        "chunk_overlap": 64,
        "top_k": 5,
        "reranking_enabled": False,
        "embedding_provider": "huggingface",
        "embedding_model": "microsoft/unixcoder-base",
        "vectorstore_provider": "faiss",
        "vectorstore_path": "data/processed/faiss_index",
        "code_llm_provider": "ollama",
        "code_llm": "deepseek-coder:6.7b",
        "code_llm_base_url": "http://localhost:11434",
        "hosted_llm_provider": "openai",
        "hosted_llm_model": "gpt-4o-mini",
        "data_clean_dir": "data/clean",
        "data_poisoned_dir": "data/poisoned",
        "data_processed_dir": "data/processed",
        "benchmark_dir": "data/benchmarks",
        "results_dir": "results",
        "experiments_dir": "experiments",
        "seed": 42,
        "poisoning_enabled": False,
        "poisoning_rate": 0.1,
        "defense_enabled": True,
        "defense_trust_scoring": True,
        "defense_anomaly_detection": True,
        "defense_context_validation": True,
        "defense_instruction_separation": True,
        "defense_static_analysis": True,
        "static_analysis_enabled": True,
        "semgrep_enabled": False,
        "experiment_repetitions": 1,
        "save_intermediate_results": True,
        "extra": {},
    }

    values.update(overrides)

    return AppConfig(**values)


class TestNormalization:
    """Tests for embedding normalization helpers."""

    def test_normalize_embeddings(self) -> None:
        embeddings = np.array(
            [
                [3.0, 4.0],
                [5.0, 12.0],
            ],
            dtype=np.float32,
        )

        normalized = normalize_embeddings(embeddings)

        assert normalized.dtype == np.float32
        assert np.allclose(
            np.linalg.norm(normalized, axis=1),
            1.0,
        )

    def test_normalize_vector(self) -> None:
        vector = np.array([3.0, 4.0], dtype=np.float32)

        normalized = normalize_vector(vector)

        assert normalized.dtype == np.float32
        assert np.isclose(np.linalg.norm(normalized), 1.0)
        assert np.allclose(normalized, [0.6, 0.8])

    def test_normalize_embeddings_rejects_one_dimensional_input(self) -> None:
        with pytest.raises(EmbeddingError):
            normalize_embeddings(
                np.array([1.0, 2.0], dtype=np.float32)
            )

    def test_normalize_vector_rejects_two_dimensional_input(self) -> None:
        with pytest.raises(EmbeddingError):
            normalize_vector(
                np.array([[1.0, 2.0]], dtype=np.float32)
            )

    def test_normalize_embeddings_rejects_zero_vector(self) -> None:
        embeddings = np.array(
            [
                [0.0, 0.0],
                [1.0, 2.0],
            ],
            dtype=np.float32,
        )

        with pytest.raises(EmbeddingError):
            normalize_embeddings(embeddings)

    def test_normalize_vector_rejects_zero_vector(self) -> None:
        with pytest.raises(EmbeddingError):
            normalize_vector(
                np.zeros(4, dtype=np.float32)
            )

    def test_normalize_embeddings_rejects_non_finite_values(self) -> None:
        embeddings = np.array(
            [[1.0, np.nan]],
            dtype=np.float32,
        )

        with pytest.raises(EmbeddingError):
            normalize_embeddings(embeddings)


class TestEmbeddingProvider:
    """Tests for the embedding provider behavior."""

    def test_fake_provider_metadata(self) -> None:
        provider = FakeEmbeddingProvider()

        metadata = provider.metadata()

        assert metadata["provider"] == "FakeEmbeddingProvider"
        assert metadata["model_name"] == "fake-code-model"
        assert metadata["dimension"] == 4

    def test_fake_provider_repr(self) -> None:
        provider = FakeEmbeddingProvider()

        representation = repr(provider)

        assert "FakeEmbeddingProvider" in representation
        assert "fake-code-model" in representation

    def test_embed_text(self) -> None:
        provider = FakeEmbeddingProvider()

        vector = provider.embed_text("hello")

        assert vector.shape == (4,)
        assert vector.dtype == np.float32

    def test_embed_texts(self) -> None:
        provider = FakeEmbeddingProvider()

        embeddings = provider.embed_texts(
            [
                "hello",
                "world",
            ]
        )

        assert embeddings.shape == (2, 4)
        assert embeddings.dtype == np.float32

    def test_embed_chunks(self) -> None:
        provider = FakeEmbeddingProvider()

        chunks = [
            CodeChunk(
                chunk_id="chunk-1",
                source_file_id="file-1",
                repository_id="repo-1",
                content="def add(a, b): return a + b",
                language=ProgrammingLanguage.PYTHON,
                start_line=1,
                end_line=1,
                symbol_name="add",
                symbol_type="function",
                ast_node_type="function_definition",
                parent_symbol=None,
                is_documentation=False,
                trust=SourceTrust.TRUSTED,
                metadata={},
            )
        ]

        embeddings = provider.embed_chunks(chunks)

        assert embeddings.shape == (1, 4)

    def test_embed_text_rejects_empty_text(self) -> None:
        provider = FakeEmbeddingProvider()

        with pytest.raises(EmbeddingInputError):
            provider.embed_text("")

    def test_embed_texts_rejects_none(self) -> None:
        provider = FakeEmbeddingProvider()

        with pytest.raises(EmbeddingInputError):
            provider.embed_texts(None)  # type: ignore[arg-type]

    def test_embed_texts_rejects_non_string(self) -> None:
        provider = FakeEmbeddingProvider()

        with pytest.raises(EmbeddingInputError):
            provider.embed_texts([123])  # type: ignore[list-item]

    def test_embed_chunks_rejects_invalid_chunk(self) -> None:
        provider = FakeEmbeddingProvider()

        with pytest.raises(EmbeddingInputError):
            provider.embed_chunks([object()])  # type: ignore[list-item]

    def test_validate_embeddings_rejects_wrong_dimension(self) -> None:
        provider = FakeEmbeddingProvider(dimension=4)

        embeddings = np.ones(
            (2, 3),
            dtype=np.float32,
        )

        with pytest.raises(EmbeddingError):
            provider.validate_embeddings(embeddings)

    def test_validate_embeddings_rejects_wrong_count(self) -> None:
        provider = FakeEmbeddingProvider(dimension=4)

        embeddings = np.ones(
            (2, 4),
            dtype=np.float32,
        )

        with pytest.raises(EmbeddingError):
            provider.validate_embeddings(
                embeddings,
                expected_count=3,
            )

    def test_validate_embeddings_rejects_non_finite_values(self) -> None:
        provider = FakeEmbeddingProvider(dimension=4)

        embeddings = np.array(
            [[1.0, 2.0, np.inf, 4.0]],
            dtype=np.float32,
        )

        with pytest.raises(EmbeddingError):
            provider.validate_embeddings(embeddings)

    def test_validate_vector(self) -> None:
        provider = FakeEmbeddingProvider(dimension=4)

        vector = np.ones(
            4,
            dtype=np.float32,
        )

        validated = provider.validate_vector(vector)

        assert validated.shape == (4,)
        assert validated.dtype == np.float32


class TestHFCodeEmbeddingProvider:
    """Tests for provider configuration without model inference."""

    def test_default_model_name(self) -> None:
        provider = HFCodeEmbeddingProvider()

        assert provider.model_name == "microsoft/unixcoder-base"

    def test_custom_configuration(self) -> None:
        provider = HFCodeEmbeddingProvider(
            model_name="custom-model",
            device="cpu",
            batch_size=8,
            normalize=False,
            max_seq_length=256,
        )

        assert provider.model_name == "custom-model"
        assert provider.device == "cpu"
        assert provider.batch_size == 8
        assert provider.normalize is False

    def test_invalid_model_name(self) -> None:
        with pytest.raises(EmbeddingConfigurationError):
            HFCodeEmbeddingProvider(model_name="")

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(EmbeddingConfigurationError):
            HFCodeEmbeddingProvider(batch_size=0)

    def test_invalid_sequence_length(self) -> None:
        with pytest.raises(EmbeddingConfigurationError):
            HFCodeEmbeddingProvider(max_seq_length=0)

    def test_unload_before_loading(self) -> None:
        provider = HFCodeEmbeddingProvider()

        provider.unload()

        assert provider._model is None


class TestEmbeddingCache:
    """Tests for persistent embedding caching."""

    def test_cache_key_is_deterministic(self, tmp_path: Path) -> None:
        cache = EmbeddingCache(
            tmp_path,
            namespace="test",
        )

        key1 = cache.build_key(
            "hello",
            model_name="model-a",
        )
        key2 = cache.build_key(
            "hello",
            model_name="model-a",
        )

        assert key1 == key2

    def test_cache_key_changes_for_different_text(
        self,
        tmp_path: Path,
    ) -> None:
        cache = EmbeddingCache(
            tmp_path,
            namespace="test",
        )

        key1 = cache.build_key(
            "hello",
            model_name="model-a",
        )
        key2 = cache.build_key(
            "world",
            model_name="model-a",
        )

        assert key1 != key2

    def test_cache_key_changes_for_different_models(
        self,
        tmp_path: Path,
    ) -> None:
        cache = EmbeddingCache(
            tmp_path,
            namespace="test",
        )

        key1 = cache.build_key(
            "hello",
            model_name="model-a",
        )
        key2 = cache.build_key(
            "hello",
            model_name="model-b",
        )

        assert key1 != key2

    def test_put_and_get(
        self,
        tmp_path: Path,
    ) -> None:
        cache = EmbeddingCache(
            tmp_path,
            namespace="test",
        )

        key = cache.build_key(
            "hello",
            model_name="model-a",
        )

        vector = np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        )

        cache.put(key, vector)

        result = cache.get(key)

        assert result is not None
        assert np.array_equal(result, vector)

    def test_cache_persists_to_disk(
        self,
        tmp_path: Path,
    ) -> None:
        cache = EmbeddingCache(
            tmp_path,
            namespace="test",
        )

        key = cache.build_key(
            "hello",
            model_name="model-a",
        )

        vector = np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        )

        cache.put(key, vector)

        new_cache = EmbeddingCache(
            tmp_path,
            namespace="test",
        )

        result = new_cache.get(key)

        assert result is not None
        assert np.array_equal(result, vector)

    def test_missing_key_returns_none(
        self,
        tmp_path: Path,
    ) -> None:
        cache = EmbeddingCache(
            tmp_path,
            namespace="test",
        )

        key = cache.build_key(
            "missing",
            model_name="model-a",
        )

        assert cache.get(key) is None

    def test_contains(
        self,
        tmp_path: Path,
    ) -> None:
        cache = EmbeddingCache(
            tmp_path,
            namespace="test",
        )

        key = cache.build_key(
            "hello",
            model_name="model-a",
        )

        vector = np.ones(
            3,
            dtype=np.float32,
        )

        assert cache.contains(key) is False

        cache.put(key, vector)

        assert cache.contains(key) is True

    def test_delete(
        self,
        tmp_path: Path,
    ) -> None:
        cache = EmbeddingCache(
            tmp_path,
            namespace="test",
        )

        key = cache.build_key(
            "hello",
            model_name="model-a",
        )

        cache.put(
            key,
            np.ones(3, dtype=np.float32),
        )

        assert cache.delete(key) is True
        assert cache.get(key) is None
        assert cache.delete(key) is False

    def test_clear(
        self,
        tmp_path: Path,
    ) -> None:
        cache = EmbeddingCache(
            tmp_path,
            namespace="test",
        )

        for text in ("one", "two", "three"):
            key = cache.build_key(
                text,
                model_name="model-a",
            )
            cache.put(
                key,
                np.ones(3, dtype=np.float32),
            )

        assert cache.size == 3
        assert cache.clear() == 3
        assert cache.size == 0

    def test_cache_rejects_invalid_key(
        self,
        tmp_path: Path,
    ) -> None:
        cache = EmbeddingCache(
            tmp_path,
            namespace="test",
        )

        with pytest.raises(EmbeddingCacheError):
            cache.get("../unsafe")

    def test_cache_rejects_empty_vector(
        self,
        tmp_path: Path,
    ) -> None:
        cache = EmbeddingCache(
            tmp_path,
            namespace="test",
        )

        key = cache.build_key(
            "hello",
            model_name="model-a",
        )

        with pytest.raises(EmbeddingCacheError):
            cache.put(
                key,
                np.array([], dtype=np.float32),
            )


class TestEmbeddingFactory:
    """Tests for embedding provider construction."""

    def test_create_from_config(self) -> None:
        config = build_config()

        provider = EmbeddingFactory.create(config)

        assert isinstance(
            provider,
            HFCodeEmbeddingProvider,
        )

        assert provider.model_name == "microsoft/unixcoder-base"

    def test_provider_override(self) -> None:
        config = build_config()

        provider = EmbeddingFactory.create(
            config,
            provider_name="hf",
        )

        assert isinstance(
            provider,
            HFCodeEmbeddingProvider,
        )

    def test_model_override(self) -> None:
        config = build_config()

        provider = EmbeddingFactory.create(
            config,
            model_name="custom-code-model",
        )

        assert provider.model_name == "custom-code-model"

    def test_explicit_values(self) -> None:
        provider = EmbeddingFactory.create_from_values(
            provider_name="huggingface",
            model_name="test-model",
            device="cpu",
            batch_size=8,
            normalize=False,
        )

        assert isinstance(
            provider,
            HFCodeEmbeddingProvider,
        )
        assert provider.model_name == "test-model"
        assert provider.device == "cpu"
        assert provider.batch_size == 8
        assert provider.normalize is False

    def test_invalid_provider(self) -> None:
        config = build_config(
            embedding_provider="unsupported",
        )

        with pytest.raises(EmbeddingFactoryError):
            EmbeddingFactory.create(config)

    def test_invalid_batch_size(self) -> None:
        config = build_config()

        with pytest.raises(EmbeddingFactoryError):
            EmbeddingFactory.create(
                config,
                batch_size=0,
            )

    def test_invalid_model_name(self) -> None:
        config = build_config()

        with pytest.raises(EmbeddingFactoryError):
            EmbeddingFactory.create(
                config,
                model_name="",
            )

    def test_provider_metadata(self) -> None:
        provider = FakeEmbeddingProvider()

        metadata = EmbeddingFactory.provider_metadata(provider)

        assert metadata["model_name"] == "fake-code-model"
        assert metadata["dimension"] == 4