from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(name=self._collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None
            self._client = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        embedding = self._embedding_fn(doc.content)
        metadata = dict(doc.metadata)
        metadata["doc_id"] = doc.id
        return {
            "id": f"{doc.id}:{self._next_index}",
            "doc_id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": embedding,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records:
            return []

        query_embedding = self._embedding_fn(query)
        scored = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append({**record, "score": float(score)})

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(0, top_k)]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        if self._use_chroma and self._collection is not None:
            ids = []
            contents = []
            embeddings = []
            metadatas = []
            for doc in docs:
                metadata = dict(doc.metadata)
                metadata["doc_id"] = doc.id
                ids.append(doc.id)
                contents.append(doc.content)
                embeddings.append(self._embedding_fn(doc.content))
                metadatas.append(metadata)
            self._collection.add(ids=ids, documents=contents, embeddings=embeddings, metadatas=metadatas)
            return

        for doc in docs:
            record = self._make_record(doc)
            self._store.append(record)
            self._next_index += 1

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if not self._store and not self._use_chroma:
            return []

        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k, include=["documents", "metadatas", "distances"])
            output = []
            for index, content in enumerate(results.get("documents", [[]])[0]):
                metadata = results.get("metadatas", [[]])[0][index]
                score = 1.0 / (1.0 + results.get("distances", [[]])[0][index])
                output.append({
                    "id": results.get("ids", [[]])[0][index],
                    "content": content,
                    "metadata": metadata,
                    "score": float(score),
                })
            return output

        ranked = self._search_records(query, self._store, top_k)
        return [
            {
                "id": item["id"],
                "content": item["content"],
                "metadata": item["metadata"],
                "score": float(item["score"]),
            }
            for item in ranked
        ]

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if metadata_filter is None:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            filtered = self._collection.get(where=metadata_filter, include=["documents", "metadatas", "embeddings"])
            if not filtered or not filtered.get("ids"):
                return []
            query_embedding = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, len(filtered["ids"])),
                where=metadata_filter,
                include=["documents", "metadatas", "distances"],
            )
            output = []
            for index, content in enumerate(results.get("documents", [[]])[0]):
                metadata = results.get("metadatas", [[]])[0][index]
                score = 1.0 / (1.0 + results.get("distances", [[]])[0][index])
                output.append({"id": results.get("ids", [[]])[0][index], "content": content, "metadata": metadata, "score": float(score)})
            return output

        candidates = [
            item for item in self._store
            if all(item.get("metadata", {}).get(key) == value for key, value in metadata_filter.items())
        ]
        ranked = self._search_records(query, candidates, top_k)
        return [
            {
                "id": item["id"],
                "content": item["content"],
                "metadata": item["metadata"],
                "score": float(item["score"]),
            }
            for item in ranked
        ]

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        if self._use_chroma and self._collection is not None:
            matching_ids = [item["id"] for item in self._collection.get(where={"doc_id": doc_id}, include=["ids"]).get("ids", [])]
            if matching_ids:
                self._collection.delete(ids=matching_ids)
                return True
            return False

        before = len(self._store)
        self._store = [
            record for record in self._store
            if record.get("doc_id") != doc_id
        ]
        return len(self._store) < before
