"""
Search Backend Abstraction for Face Recognition.

Provides an interface for embedding search so the current NumPy
implementation can later be replaced by FAISS or pgvector without
changing the rest of the recognition pipeline.

Default: NumpySearchBackend (exact cosine similarity via np.dot).
Future:  FAISSSearchBackend, PgvectorSearchBackend (not implemented).
"""

import numpy as np
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


class SearchBackend(ABC):
    """
    Abstract interface for embedding search.

    Implementations must support:
    - Building/rebuilding a search index from all known encodings
    - Incremental add/remove operations
    - Top-K similarity search
    """

    @abstractmethod
    def build_index(
        self,
        encodings: np.ndarray,
        person_ids: List[str],
        encoding_ids: List[str],
        qualities: List[float],
    ):
        """
        Build/rebuild the search index from all known encodings.

        Args:
            encodings: (N, D) matrix of normalized embeddings
            person_ids: person_id for each encoding
            encoding_ids: unique ID for each encoding row
            qualities: quality score for each encoding
        """
        ...

    @abstractmethod
    def add(
        self,
        encoding: np.ndarray,
        person_id: str,
        encoding_id: str,
        quality: float = 1.0,
    ):
        """Incrementally add a single encoding to the index."""
        ...

    @abstractmethod
    def remove_person(self, person_id: str):
        """Remove all encodings for a person from the index."""
        ...

    @abstractmethod
    def search(
        self, query: np.ndarray, top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Return top-k matches by cosine similarity.

        Args:
            query: normalized query embedding (D,)
            top_k: number of results

        Returns:
            List of (index, similarity) tuples, sorted descending by similarity.
            Index corresponds to position in the encodings matrix.
        """
        ...

    @abstractmethod
    def search_batch(
        self, queries: np.ndarray, top_k: int = 5
    ) -> List[List[Tuple[int, float]]]:
        """
        Batch search: return top-k matches for multiple queries.

        Args:
            queries: (N, D) matrix of normalized query embeddings
            top_k: number of results per query

        Returns:
            List of N result lists, each containing (index, similarity) tuples.
        """
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of encodings in the index."""
        ...

    def get_person_id(self, index: int) -> Optional[str]:
        """Get the person_id for an encoding at the given index."""
        ...

    def get_encoding_quality(self, index: int) -> float:
        """Get the quality score for an encoding at the given index."""
        ...


class NumpySearchBackend(SearchBackend):
    """
    Exact cosine similarity search using NumPy vectorized operations.

    Performance: handles hundreds of faces in <1ms. Use FAISS when
    the encoding count exceeds ~1,000–2,000.
    """

    def __init__(self):
        self._encodings: Optional[np.ndarray] = None  # (N, D) matrix
        self._person_ids: List[str] = []
        self._encoding_ids: List[str] = []
        self._qualities: List[float] = []
        self._size = 0

    def build_index(
        self,
        encodings: np.ndarray,
        person_ids: List[str],
        encoding_ids: List[str],
        qualities: List[float],
    ):
        """Build the index from a pre-computed encoding matrix."""
        if encodings is not None and len(encodings) > 0:
            self._encodings = encodings
            self._person_ids = list(person_ids)
            self._encoding_ids = list(encoding_ids)
            self._qualities = list(qualities)
            self._size = len(encodings)
        else:
            self._encodings = None
            self._person_ids = []
            self._encoding_ids = []
            self._qualities = []
            self._size = 0

        logger.debug(f"NumpySearchBackend: built index with {self._size} encodings")

    def add(
        self,
        encoding: np.ndarray,
        person_id: str,
        encoding_id: str,
        quality: float = 1.0,
    ):
        """Incrementally add a single encoding."""
        enc_row = encoding.reshape(1, -1)
        if self._encodings is not None:
            self._encodings = np.vstack([self._encodings, enc_row])
        else:
            self._encodings = enc_row

        self._person_ids.append(person_id)
        self._encoding_ids.append(encoding_id)
        self._qualities.append(quality)
        self._size += 1

    def remove_person(self, person_id: str):
        """Remove all encodings for a person."""
        indices_to_keep = [
            i for i, pid in enumerate(self._person_ids) if pid != person_id
        ]
        if not indices_to_keep:
            self._encodings = None
            self._person_ids = []
            self._encoding_ids = []
            self._qualities = []
            self._size = 0
            return

        if self._encodings is not None:
            self._encodings = self._encodings[indices_to_keep]
        self._person_ids = [self._person_ids[i] for i in indices_to_keep]
        self._encoding_ids = [self._encoding_ids[i] for i in indices_to_keep]
        self._qualities = [self._qualities[i] for i in indices_to_keep]
        self._size = len(indices_to_keep)

    def search(
        self, query: np.ndarray, top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Exact cosine similarity search via np.dot."""
        if self._encodings is None or self._size == 0:
            return []

        # Cosine similarity: dot product of normalized vectors
        similarities = np.dot(self._encodings, query)

        # Get top-k indices (descending similarity)
        k = min(top_k, self._size)
        if k >= self._size:
            top_indices = np.argsort(similarities)[::-1]
        else:
            # Use argpartition for efficiency when k << N
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def search_batch(
        self, queries: np.ndarray, top_k: int = 5
    ) -> List[List[Tuple[int, float]]]:
        """Batch search: single matrix multiply for all queries."""
        if self._encodings is None or self._size == 0:
            return [[] for _ in range(len(queries))]

        # (N_queries, D) @ (D, N_known) -> (N_queries, N_known)
        all_similarities = np.dot(queries, self._encodings.T)

        results = []
        k = min(top_k, self._size)
        for i in range(len(queries)):
            sims = all_similarities[i]
            if k >= self._size:
                top_indices = np.argsort(sims)[::-1]
            else:
                top_indices = np.argpartition(sims, -k)[-k:]
                top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]
            results.append([(int(idx), float(sims[idx])) for idx in top_indices])

        return results

    @property
    def size(self) -> int:
        return self._size

    def get_person_id(self, index: int) -> Optional[str]:
        if 0 <= index < len(self._person_ids):
            return self._person_ids[index]
        return None

    def get_encoding_quality(self, index: int) -> float:
        if 0 <= index < len(self._qualities):
            return self._qualities[index]
        return 1.0


def create_search_backend(backend_name: str = None) -> SearchBackend:
    """
    Factory: create the appropriate search backend.

    Args:
        backend_name: "numpy" (default) or "faiss" (future)

    Returns:
        SearchBackend instance
    """
    name = (backend_name or settings.SEARCH_BACKEND).lower()

    if name == "numpy":
        return NumpySearchBackend()
    elif name == "faiss":
        # Future: return FAISSSearchBackend()
        logger.warning("FAISS backend not yet implemented, falling back to NumPy")
        return NumpySearchBackend()
    else:
        logger.warning(f"Unknown search backend '{name}', falling back to NumPy")
        return NumpySearchBackend()


# Avoid circular import — import settings at module level
from app.config import settings
