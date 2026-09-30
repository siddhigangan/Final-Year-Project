"""Unit tests for the SecureCodeRAG vector-store layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import (
    CodeChunk,
    ProgrammingLanguage,
    RetrievedChunk,
    SourceTrust,
)
from src.vectorstore.base import (
    VectorSearchResult,
    VectorStoreInputError,
    VectorStoreNotFoundError,
)
from src.vectorstore.faiss_store import (
    FAISSVectorStore,
    FAISSVectorStoreError,
)
from src.vectorstore.metadata_store import (
    MetadataStore,
    MetadataStoreCorruptionError,
)

DIMENSION = 4


def make_chunk(
    chunk_id: str,
    *,
    trust: SourceTrust = SourceTrust.TRUSTED,
    content: str | None = None,
) -> CodeChunk:
    """Create a valid CodeChunk for tests."""
    return CodeChunk(
        chunk_id=chunk_id,
        source_file_id=f"file-{chunk_id}",
        repository_id="repo-test",
        content=content or f"def {chunk_id}():\n    return 1",
        language=ProgrammingLanguage.PYTHON,
        start_line=1,
        end_line=2,
        symbol_name=chunk_id,
        symbol_type="function",
        ast_node_type="function_definition",
        parent_symbol=None,
        is_documentation=False,
        trust=trust,
        metadata={
            "relative_path": f"src/{chunk_id}.py",
        },
    )


def make_store(
    tmp_path: Path,
) -> FAISSVectorStore:
    """Create a temporary FAISS store."""
    return FAISSVectorStore(
        dimension=DIMENSION,
        path=tmp_path / "faiss_index",
    )


def make_embeddings() -> np.ndarray:
    """Return deterministic test embeddings."""
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


class TestVectorSearchResult:
    """Tests for VectorSearchResult."""

    def test_valid_result(self) -> None:
        result = VectorSearchResult(
            chunk_id="chunk-1",
            score=0.95,
            rank=1,
            metadata={"language": "python"},
        )

        assert result.chunk_id == "chunk-1"
        assert result.score == pytest.approx(0.95)
        assert result.rank == 1
        assert result.metadata["language"] == "python"

    def test_to_dict(self) -> None:
        result = VectorSearchResult(
            chunk_id="chunk-1",
            score=0.5,
            rank=2,
        )

        data = result.to_dict()

        assert data["chunk_id"] == "chunk-1"
        assert data["score"] == 0.5
        assert data["rank"] == 2

    def test_rejects_empty_chunk_id(self) -> None:
        with pytest.raises(VectorStoreInputError):
            VectorSearchResult(
                chunk_id="",
                score=0.5,
                rank=1,
            )

    def test_rejects_non_finite_score(self) -> None:
        with pytest.raises(VectorStoreInputError):
            VectorSearchResult(
                chunk_id="chunk-1",
                score=float("nan"),
                rank=1,
            )

    def test_rejects_invalid_rank(self) -> None:
        with pytest.raises(VectorStoreInputError):
            VectorSearchResult(
                chunk_id="chunk-1",
                score=0.5,
                rank=0,
            )


class TestMetadataStore:
    """Tests for MetadataStore."""

    def test_empty_store(self) -> None:
        store = MetadataStore()

        assert store.size == 0
        assert store.items() == []
        assert store.chunks() == []

    def test_add_and_get(self) -> None:
        store = MetadataStore()
        chunk = make_chunk("chunk-1")

        store.add([0], [chunk])

        assert store.size == 1
        assert store.get(0) == chunk

    def test_get_by_chunk_id(self) -> None:
        store = MetadataStore()
        chunk = make_chunk("chunk-1")

        store.add([7], [chunk])

        assert store.get_by_chunk_id("chunk-1") == chunk
        assert store.get_vector_id("chunk-1") == 7

    def test_contains_methods(self) -> None:
        store = MetadataStore()
        chunk = make_chunk("chunk-1")

        store.add([3], [chunk])

        assert store.contains_vector_id(3)
        assert store.contains_chunk_id("chunk-1")
        assert not store.contains_vector_id(4)
        assert not store.contains_chunk_id("chunk-2")

    def test_add_rejects_length_mismatch(self) -> None:
        store = MetadataStore()

        with pytest.raises(VectorStoreInputError):
            store.add(
                [0, 1],
                [make_chunk("chunk-1")],
            )

    def test_add_rejects_duplicate_vector_id(self) -> None:
        store = MetadataStore()

        store.add(
            [0],
            [make_chunk("chunk-1")],
        )

        with pytest.raises(VectorStoreInputError):
            store.add(
                [0],
                [make_chunk("chunk-2")],
            )

    def test_add_rejects_duplicate_chunk_id(self) -> None:
        store = MetadataStore()

        store.add(
            [0],
            [make_chunk("chunk-1")],
        )

        with pytest.raises(VectorStoreInputError):
            store.add(
                [1],
                [make_chunk("chunk-1")],
            )

    def test_get_missing_vector_raises(self) -> None:
        store = MetadataStore()

        with pytest.raises(VectorStoreNotFoundError):
            store.get(100)

    def test_get_missing_chunk_raises(self) -> None:
        store = MetadataStore()

        with pytest.raises(VectorStoreNotFoundError):
            store.get_by_chunk_id("missing")

    def test_remove(self) -> None:
        store = MetadataStore()
        chunk = make_chunk("chunk-1")

        store.add([0], [chunk])

        removed = store.remove([0])

        assert removed == 1
        assert store.size == 0
        assert not store.contains_chunk_id("chunk-1")

    def test_clear(self) -> None:
        store = MetadataStore()

        store.add(
            [0, 1],
            [
                make_chunk("chunk-1"),
                make_chunk("chunk-2"),
            ],
        )

        store.clear()

        assert store.size == 0

    def test_save_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "metadata.json"

        original = MetadataStore()

        chunk = make_chunk(
            "chunk-1",
            trust=SourceTrust.SUSPICIOUS,
        )

        original.add([4], [chunk])
        original.save(path)

        loaded = MetadataStore()
        loaded.load(path)

        assert loaded.size == 1
        assert loaded.get(4) == chunk
        assert loaded.get(4).trust == SourceTrust.SUSPICIOUS

    def test_save_creates_parent_directory(
        self,
        tmp_path: Path,
    ) -> None:
        path = (
            tmp_path
            / "nested"
            / "directory"
            / "metadata.json"
        )

        store = MetadataStore()
        store.add([0], [make_chunk("chunk-1")])

        saved_path = store.save(path)

        assert saved_path.is_file()

    def test_load_missing_file_raises(
        self,
        tmp_path: Path,
    ) -> None:
        store = MetadataStore()

        with pytest.raises(VectorStoreNotFoundError):
            store.load(tmp_path / "missing.json")

    def test_load_corrupt_json_raises(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "metadata.json"
        path.write_text(
            "{invalid json",
            encoding="utf-8",
        )

        store = MetadataStore()

        with pytest.raises(MetadataStoreCorruptionError):
            store.load(path)

    def test_load_wrong_format_version_raises(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "metadata.json"

        path.write_text(
            (
                '{"format_version":999,'
                '"records":[]}'
            ),
            encoding="utf-8",
        )

        store = MetadataStore()

        with pytest.raises(MetadataStoreCorruptionError):
            store.load(path)


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore."""

    def test_empty_store(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)

        assert store.size == 0
        assert store.dimension == DIMENSION
        assert store.provider == "faiss"

    def test_add_vectors(self, tmp_path: Path) -> None:
        store = make_store(tmp_path)

        chunks = [
            make_chunk("chunk-1"),
            make_chunk("chunk-2"),
            make_chunk("chunk-3"),
        ]

        embeddings = make_embeddings()

        ids = store.add(
            embeddings,
            chunks,
        )

        assert ids == [
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ]
        assert store.size == 3
        assert store.metadata_store.size == 3

    def test_add_and_search(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        chunks = [
            make_chunk("chunk-1"),
            make_chunk("chunk-2"),
            make_chunk("chunk-3"),
        ]

        store.add(
            make_embeddings(),
            chunks,
        )

        query = np.asarray(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        results = store.search(
            query,
            top_k=2,
        )

        assert len(results) == 2
        assert all(
            isinstance(result, RetrievedChunk)
            for result in results
        )
        assert results[0].chunk.chunk_id == "chunk-1"
        assert results[0].rank == 1
        assert results[0].retrieval_score == pytest.approx(1.0)

    def test_search_empty_store(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        query = np.ones(
            DIMENSION,
            dtype=np.float32,
        )

        assert store.search(query, top_k=5) == []

    def test_top_k_larger_than_store(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        store.add(
            make_embeddings()[:2],
            [
                make_chunk("chunk-1"),
                make_chunk("chunk-2"),
            ],
        )

        query = np.asarray(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        results = store.search(
            query,
            top_k=10,
        )

        assert len(results) == 2

    def test_add_rejects_dimension_mismatch(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        embeddings = np.ones(
            (1, DIMENSION + 1),
            dtype=np.float32,
        )

        with pytest.raises(VectorStoreInputError):
            store.add(
                embeddings,
                [make_chunk("chunk-1")],
            )

    def test_add_rejects_nan(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        embeddings = np.asarray(
            [[1.0, np.nan, 0.0, 0.0]],
            dtype=np.float32,
        )

        with pytest.raises(VectorStoreInputError):
            store.add(
                embeddings,
                [make_chunk("chunk-1")],
            )

    def test_add_rejects_mismatched_lengths(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        with pytest.raises(VectorStoreInputError):
            store.add(
                np.ones(
                    (2, DIMENSION),
                    dtype=np.float32,
                ),
                [make_chunk("chunk-1")],
            )

    def test_add_rejects_duplicate_chunk_ids(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        embeddings = np.ones(
            (2, DIMENSION),
            dtype=np.float32,
        )

        chunks = [
            make_chunk("duplicate"),
            make_chunk("duplicate"),
        ]

        with pytest.raises(VectorStoreInputError):
            store.add(
                embeddings,
                chunks,
            )

    def test_search_rejects_invalid_top_k(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        query = np.ones(
            DIMENSION,
            dtype=np.float32,
        )

        with pytest.raises(VectorStoreInputError):
            store.search(query, top_k=0)

        with pytest.raises(VectorStoreInputError):
            store.search(query, top_k=-1)

    def test_search_rejects_wrong_dimension(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        query = np.ones(
            DIMENSION + 1,
            dtype=np.float32,
        )

        with pytest.raises(VectorStoreInputError):
            store.search(query)

    def test_search_rejects_multiple_query_vectors(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        query = np.ones(
            (2, DIMENSION),
            dtype=np.float32,
        )

        with pytest.raises(VectorStoreInputError):
            store.search(query)

    def test_add_chunks_helper(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        ids = store.add_chunks(
            make_embeddings()[:1],
            [make_chunk("chunk-1")],
        )

        assert ids == ["chunk-1"]
        assert store.size == 1

    def test_search_vector_helper(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        store.add(
            make_embeddings()[:1],
            [make_chunk("chunk-1")],
        )

        query = np.asarray(
            [1.0, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

        results = store.search_vector(
            query,
            top_k=1,
        )

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "chunk-1"

    def test_trust_metadata_is_preserved(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        poisoned_chunk = make_chunk(
            "poisoned-chunk",
            trust=SourceTrust.POISONED,
        )

        store.add(
            np.asarray(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            [poisoned_chunk],
        )

        results = store.search(
            np.asarray(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            top_k=1,
        )

        assert results[0].chunk.trust == SourceTrust.POISONED

    def test_save_and_load(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "persistent_index"

        original = FAISSVectorStore(
            DIMENSION,
            path,
        )

        chunks = [
            make_chunk("chunk-1"),
            make_chunk("chunk-2"),
        ]

        original.add(
            make_embeddings()[:2],
            chunks,
        )

        original.save()

        assert path.with_suffix(".index").is_file()
        assert path.with_suffix(".json").is_file()

        loaded = FAISSVectorStore(
            DIMENSION,
            path,
        )

        loaded.load()

        assert loaded.size == 2

        results = loaded.search(
            np.asarray(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            top_k=2,
        )

        assert len(results) == 2
        assert results[0].chunk.chunk_id == "chunk-1"

    def test_save_with_explicit_path(
        self,
        tmp_path: Path,
    ) -> None:
        store = FAISSVectorStore(DIMENSION)

        store.add(
            make_embeddings()[:1],
            [make_chunk("chunk-1")],
        )

        path = tmp_path / "explicit_index"

        store.save(path)

        assert path.with_suffix(".index").is_file()
        assert path.with_suffix(".json").is_file()

    def test_load_rejects_dimension_mismatch(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "index"

        original = FAISSVectorStore(
            DIMENSION,
            path,
        )

        original.add(
            make_embeddings()[:1],
            [make_chunk("chunk-1")],
        )

        original.save()

        loaded = FAISSVectorStore(
            DIMENSION + 1,
            path,
        )

        with pytest.raises(FAISSVectorStoreError):
            loaded.load()

    def test_load_rejects_missing_index(
        self,
        tmp_path: Path,
    ) -> None:
        store = FAISSVectorStore(
            DIMENSION,
            tmp_path / "missing",
        )

        with pytest.raises(VectorStoreNotFoundError):
            store.load()

    def test_load_rejects_missing_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "index"

        store = FAISSVectorStore(
            DIMENSION,
            path,
        )

        store.add(
            make_embeddings()[:1],
            [make_chunk("chunk-1")],
        )

        store.save()

        path.with_suffix(".json").unlink()

        loaded = FAISSVectorStore(
            DIMENSION,
            path,
        )

        with pytest.raises(VectorStoreNotFoundError):
            loaded.load()

    def test_load_rejects_mismatched_record_counts(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "index"

        store = FAISSVectorStore(
            DIMENSION,
            path,
        )

        store.add(
            make_embeddings()[:2],
            [
                make_chunk("chunk-1"),
                make_chunk("chunk-2"),
            ],
        )

        store.save()

        metadata_path = path.with_suffix(".json")

        metadata_path.write_text(
            (
                '{"format_version":1,'
                '"records":[]}'
            ),
            encoding="utf-8",
        )

        loaded = FAISSVectorStore(
            DIMENSION,
            path,
        )

        with pytest.raises(FAISSVectorStoreError):
            loaded.load()

    def test_delete(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        store.add(
            make_embeddings(),
            [
                make_chunk("chunk-1"),
                make_chunk("chunk-2"),
                make_chunk("chunk-3"),
            ],
        )

        removed = store.delete(["chunk-2"])

        assert removed == 1
        assert store.size == 2
        assert not store.metadata_store.contains_chunk_id(
            "chunk-2"
        )

        results = store.search(
            np.asarray(
                [0.0, 1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            top_k=2,
        )

        assert all(
            result.chunk.chunk_id != "chunk-2"
            for result in results
        )

    def test_delete_missing_chunk(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        store.add(
            make_embeddings()[:1],
            [make_chunk("chunk-1")],
        )

        assert store.delete(["missing"]) == 0
        assert store.size == 1

    def test_delete_multiple_chunks(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        store.add(
            make_embeddings(),
            [
                make_chunk("chunk-1"),
                make_chunk("chunk-2"),
                make_chunk("chunk-3"),
            ],
        )

        removed = store.delete(
            [
                "chunk-1",
                "chunk-3",
            ]
        )

        assert removed == 2
        assert store.size == 1
        assert store.metadata_store.contains_chunk_id(
            "chunk-2"
        )

    def test_delete_empty_list(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        assert store.delete([]) == 0

    def test_clear(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        store.add(
            make_embeddings()[:2],
            [
                make_chunk("chunk-1"),
                make_chunk("chunk-2"),
            ],
        )

        store.clear()

        assert store.size == 0
        assert store.metadata_store.size == 0
        assert store.search(
            np.ones(
                DIMENSION,
                dtype=np.float32,
            )
        ) == []

    def test_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        store.add(
            make_embeddings()[:1],
            [make_chunk("chunk-1")],
        )

        metadata = store.metadata()

        assert metadata["provider"] == "faiss"
        assert metadata["dimension"] == DIMENSION
        assert metadata["size"] == 1
        assert metadata["metadata_size"] == 1
        assert metadata["index_type"] == "IndexFlatIP"

    def test_repr(
        self,
        tmp_path: Path,
    ) -> None:
        store = make_store(tmp_path)

        representation = repr(store)

        assert "FAISSVectorStore" in representation
        assert "provider='faiss'" in representation
        assert f"dimension={DIMENSION}" in representation
        assert "size=0" in representation

    def test_windows_style_path(
        self,
        tmp_path: Path,
    ) -> None:
        nested_path = (
            tmp_path
            / "data"
            / "processed"
            / "faiss_index"
        )

        store = FAISSVectorStore(
            DIMENSION,
            nested_path,
        )

        store.add(
            make_embeddings()[:1],
            [make_chunk("chunk-1")],
        )

        store.save()

        assert nested_path.with_suffix(".index").exists()
        assert nested_path.with_suffix(".json").exists()

    def test_round_trip_preserves_all_chunk_metadata(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "round_trip"

        chunk = CodeChunk(
            chunk_id="chunk-special",
            source_file_id="file-special",
            repository_id="repo-special",
            content="def secure_function():\n    return True",
            language=ProgrammingLanguage.PYTHON,
            start_line=10,
            end_line=11,
            symbol_name="secure_function",
            symbol_type="function",
            ast_node_type="function_definition",
            parent_symbol="ExampleClass",
            is_documentation=False,
            trust=SourceTrust.SUSPICIOUS,
            metadata={
                "relative_path": "src/example.py",
                "custom": "value",
            },
        )

        original = FAISSVectorStore(
            DIMENSION,
            path,
        )

        original.add(
            np.asarray(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            [chunk],
        )

        original.save()

        loaded = FAISSVectorStore(
            DIMENSION,
            path,
        )

        loaded.load()

        result = loaded.search(
            np.asarray(
                [1.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            top_k=1,
        )[0]

        loaded_chunk = result.chunk

        assert loaded_chunk.chunk_id == chunk.chunk_id
        assert loaded_chunk.source_file_id == chunk.source_file_id
        assert loaded_chunk.repository_id == chunk.repository_id
        assert loaded_chunk.content == chunk.content
        assert loaded_chunk.language == chunk.language
        assert loaded_chunk.start_line == chunk.start_line
        assert loaded_chunk.end_line == chunk.end_line
        assert loaded_chunk.symbol_name == chunk.symbol_name
        assert loaded_chunk.symbol_type == chunk.symbol_type
        assert loaded_chunk.ast_node_type == chunk.ast_node_type
        assert loaded_chunk.parent_symbol == chunk.parent_symbol
        assert loaded_chunk.is_documentation == chunk.is_documentation
        assert loaded_chunk.trust == SourceTrust.SUSPICIOUS
        assert loaded_chunk.metadata == chunk.metadata
