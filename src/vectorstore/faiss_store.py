"""FAISS-backed vector store for SecureCodeRAG."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.models import CodeChunk, RetrievedChunk
from src.vectorstore.base import (
    VectorStore,
    VectorStoreError,
    VectorStoreInputError,
    VectorStoreNotFoundError,
)
from src.vectorstore.metadata_store import MetadataStore


class FAISSVectorStoreError(VectorStoreError):
    """Raised when a FAISS vector-store operation fails."""


class FAISSVectorStore(VectorStore):
    """Persistent FAISS vector store with external chunk metadata."""

    provider_name = "faiss"

    INDEX_SUFFIX = ".index"
    METADATA_SUFFIX = ".json"

    def __init__(
        self,
        dimension: int,
        path: str | Path | None = None,
    ) -> None:
        super().__init__(dimension)

        self._index = faiss.IndexFlatIP(self.dimension)
        self._metadata = MetadataStore()
        self._path: Path | None = None

        if path is not None:
            self._path = self._normalize_base_path(path)

    @property
    def size(self) -> int:
        """Return the number of vectors currently indexed."""
        return self._index.ntotal

    @property
    def index(self) -> faiss.Index:
        """Return the underlying FAISS index."""
        return self._index

    @property
    def metadata_store(self) -> MetadataStore:
        """Return the metadata store."""
        return self._metadata

    @property
    def path(self) -> Path | None:
        """Return the configured persistence base path."""
        return self._path

    def add(
        self,
        embeddings: np.ndarray,
        chunks: list[CodeChunk],
    ) -> list[str]:
        """Add embeddings and their corresponding chunks."""
        validated_embeddings = self.validate_embeddings(embeddings)

        if not chunks:
            raise VectorStoreInputError(
                "chunks cannot be empty."
            )

        if len(validated_embeddings) != len(chunks):
            raise VectorStoreInputError(
                "The number of embeddings must match the number of chunks."
            )

        self._validate_chunk_ids(chunks)

        start_vector_id = self.size

        vector_ids = list(
            range(
                start_vector_id,
                start_vector_id + len(chunks),
            )
        )

        self._metadata.add(
            vector_ids=vector_ids,
            chunks=chunks,
        )

        try:
            self._index.add(validated_embeddings)
        except Exception as exc:
            self._metadata.remove(vector_ids)

            raise FAISSVectorStoreError(
                f"Failed to add embeddings to FAISS: {exc}"
            ) from exc

        return [chunk.chunk_id for chunk in chunks]

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """Search the FAISS index and return retrieved chunks."""
        query = self.validate_query_embedding(query_embedding)

        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise VectorStoreInputError(
                "top_k must be an integer."
            )

        if top_k <= 0:
            raise VectorStoreInputError(
                "top_k must be greater than zero."
            )

        if self.size == 0:
            return []

        actual_k = min(top_k, self.size)

        query_matrix = np.ascontiguousarray(
            query.reshape(1, -1),
            dtype=np.float32,
        )

        try:
            scores, vector_ids = self._index.search(
                query_matrix,
                actual_k,
            )
        except Exception as exc:
            raise FAISSVectorStoreError(
                f"FAISS search failed: {exc}"
            ) from exc

        results: list[RetrievedChunk] = []

        for rank, (score, vector_id) in enumerate(
            zip(scores[0], vector_ids[0], strict=True),
            start=1,
        ):
            if vector_id < 0:
                continue

            try:
                chunk = self._metadata.get(int(vector_id))
            except VectorStoreNotFoundError as exc:
                raise FAISSVectorStoreError(
                    "FAISS returned a vector ID without metadata: "
                    f"{vector_id}"
                ) from exc

            results.append(
                self._to_retrieved_chunk(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                )
            )

        return results

    def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks from the store by rebuilding the FAISS index."""
        if not chunk_ids:
            return 0

        normalized_ids = self._validate_chunk_id_list(chunk_ids)

        vector_ids: list[int] = []

        for chunk_id in normalized_ids:
            if not self._metadata.contains_chunk_id(chunk_id):
                continue

            vector_ids.append(
                self._metadata.get_vector_id(chunk_id)
            )

        if not vector_ids:
            return 0

        remove_set = set(vector_ids)

        try:
            remaining_chunks = [
                chunk
                for vector_id, chunk in self._metadata.items()
                if vector_id not in remove_set
            ]

            remaining_vectors = self._reconstruct_remaining_vectors(
                remove_set
            )

            new_index = faiss.IndexFlatIP(self.dimension)

            if remaining_vectors:
                new_index.add(
                    np.ascontiguousarray(
                        np.vstack(remaining_vectors),
                        dtype=np.float32,
                    )
                )

            new_metadata = MetadataStore()

            new_vector_ids = list(
                range(len(remaining_chunks))
            )

            if remaining_chunks:
                new_metadata.add(
                    vector_ids=new_vector_ids,
                    chunks=remaining_chunks,
                )

            self._index = new_index
            self._metadata = new_metadata

        except Exception as exc:
            raise FAISSVectorStoreError(
                f"Failed to delete chunks: {exc}"
            ) from exc

        return len(vector_ids)

    def clear(self) -> None:
        """Remove every vector and metadata record."""
        self._index = faiss.IndexFlatIP(self.dimension)
        self._metadata.clear()

    def save(
        self,
        path: str | Path | None = None,
    ) -> None:
        """Persist FAISS index and metadata atomically."""
        base_path = self._resolve_path(path)

        base_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_path = self._index_path(base_path)
        metadata_path = self._metadata_path(base_path)

        temporary_index: Path | None = None
        temporary_metadata: Path | None = None

        try:
            temporary_index = self._temporary_path(
                index_path,
                ".index.tmp",
            )

            temporary_metadata = self._temporary_path(
                metadata_path,
                ".json.tmp",
            )

            faiss.write_index(
                self._index,
                str(temporary_index),
            )

            self._write_metadata_file(
                temporary_metadata,
            )

            os.replace(
                temporary_index,
                index_path,
            )

            temporary_index = None

            os.replace(
                temporary_metadata,
                metadata_path,
            )

            temporary_metadata = None

            self._path = base_path

        except Exception as exc:
            if temporary_index is not None:
                temporary_index.unlink(missing_ok=True)

            if temporary_metadata is not None:
                temporary_metadata.unlink(missing_ok=True)

            raise FAISSVectorStoreError(
                f"Failed to save FAISS vector store: {exc}"
            ) from exc

    def load(
        self,
        path: str | Path | None = None,
    ) -> None:
        """Load FAISS index and metadata from disk."""
        base_path = self._resolve_path(path)

        index_path = self._index_path(base_path)
        metadata_path = self._metadata_path(base_path)

        if not index_path.is_file():
            raise VectorStoreNotFoundError(
                f"FAISS index does not exist: {index_path}"
            )

        if not metadata_path.is_file():
            raise VectorStoreNotFoundError(
                f"Metadata file does not exist: {metadata_path}"
            )

        try:
            loaded_index = faiss.read_index(
                str(index_path)
            )

            if loaded_index.d != self.dimension:
                raise FAISSVectorStoreError(
                    "Loaded FAISS index dimension does not match the "
                    f"configured dimension: {loaded_index.d} != "
                    f"{self.dimension}"
                )

            loaded_metadata = MetadataStore()
            loaded_metadata.load(metadata_path)

            if loaded_index.ntotal != loaded_metadata.size:
                raise FAISSVectorStoreError(
                    "FAISS index and metadata store contain different "
                    f"numbers of records: {loaded_index.ntotal} != "
                    f"{loaded_metadata.size}"
                )

        except FAISSVectorStoreError:
            raise
        except Exception as exc:
            raise FAISSVectorStoreError(
                f"Failed to load FAISS vector store: {exc}"
            ) from exc

        self._index = loaded_index
        self._metadata = loaded_metadata
        self._path = base_path

    def metadata(self) -> dict[str, Any]:
        """Return vector-store metadata."""
        return {
            "provider": self.provider,
            "dimension": self.dimension,
            "size": self.size,
            "index_type": type(self._index).__name__,
            "path": str(self._path) if self._path else None,
            "metadata_size": self._metadata.size,
        }

    def _write_metadata_file(
        self,
        path: Path,
    ) -> None:
        """Write metadata directly to an exact temporary path."""
        payload = {
            "format_version": MetadataStore.FORMAT_VERSION,
            "records": [
                {
                    "vector_id": vector_id,
                    "chunk": chunk.to_dict(),
                }
                for vector_id, chunk in self._metadata.items()
            ],
        }

        try:
            with path.open(
                "w",
                encoding="utf-8",
            ) as metadata_file:
                import json

                json.dump(
                    payload,
                    metadata_file,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                metadata_file.write("\n")
        except OSError as exc:
            raise FAISSVectorStoreError(
                f"Failed to write metadata file {path}: {exc}"
            ) from exc

    def _reconstruct_remaining_vectors(
        self,
        remove_set: set[int],
    ) -> list[np.ndarray]:
        """Reconstruct vectors that should remain after deletion."""
        vectors: list[np.ndarray] = []

        for vector_id in range(self.size):
            if vector_id in remove_set:
                continue

            try:
                vector = self._index.reconstruct(vector_id)
            except Exception as exc:
                raise FAISSVectorStoreError(
                    f"Unable to reconstruct vector {vector_id}: {exc}"
                ) from exc

            vectors.append(
                np.asarray(
                    vector,
                    dtype=np.float32,
                )
            )

        return vectors

    @staticmethod
    def _to_retrieved_chunk(
        chunk: CodeChunk,
        score: float,
        rank: int,
    ) -> RetrievedChunk:
        """Convert a CodeChunk and similarity score to RetrievedChunk."""
        return RetrievedChunk(
            chunk=chunk,
            retrieval_score=score,
            rank=rank,
            retriever_name="faiss",
            retrieval_metadata={
                "vector_store": "faiss",
            },
        )

    @staticmethod
    def _validate_chunk_id_list(
        chunk_ids: list[str],
    ) -> list[str]:
        """Validate a list of chunk IDs."""
        normalized_ids: list[str] = []

        for chunk_id in chunk_ids:
            if not isinstance(chunk_id, str):
                raise VectorStoreInputError(
                    "chunk_ids must contain only strings."
                )

            normalized_id = chunk_id.strip()

            if not normalized_id:
                raise VectorStoreInputError(
                    "chunk_ids cannot contain empty strings."
                )

            normalized_ids.append(normalized_id)

        return list(dict.fromkeys(normalized_ids))

    def _resolve_path(
        self,
        path: str | Path | None,
    ) -> Path:
        """Resolve a supplied or configured persistence path."""
        if path is not None:
            return self._normalize_base_path(path)

        if self._path is None:
            raise VectorStoreInputError(
                "A persistence path must be supplied or configured."
            )

        return self._path

    @classmethod
    def _normalize_base_path(
        cls,
        path: str | Path,
    ) -> Path:
        """Normalize a persistence base path."""
        normalized = Path(path)

        if normalized.suffix.lower() in {
            cls.INDEX_SUFFIX,
            cls.METADATA_SUFFIX,
        }:
            normalized = normalized.with_suffix("")

        return normalized

    @classmethod
    def _index_path(
        cls,
        base_path: Path,
    ) -> Path:
        """Return the FAISS index file path."""
        return base_path.with_suffix(
            cls.INDEX_SUFFIX
        )

    @classmethod
    def _metadata_path(
        cls,
        base_path: Path,
    ) -> Path:
        """Return the metadata JSON path."""
        return base_path.with_suffix(
            cls.METADATA_SUFFIX
        )

    @staticmethod
    def _temporary_path(
        destination: Path,
        suffix: str,
    ) -> Path:
        """Create a unique temporary path beside a destination."""
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.stem}.",
            suffix=suffix,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

        temporary_path.unlink(
            missing_ok=True,
        )

        return temporary_path