"""ChromaDB client for xaihi memory system."""
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

try:
    from .config import config
except ImportError:
    from config import config


class ChromaDBClient:
    """ChromaDB client for memory operations."""

    _instance = None
    _client = None
    _collection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connect()
        return cls._instance

    def _connect(self) -> None:
        chroma_cfg = config.get_chroma()
        persist_dir = os.path.expanduser(chroma_cfg.get("persist_dir", "~/.claude/memory/chroma_db"))
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        return self._collection

    def add_memory(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Add a memory document."""
        # Filter out empty lists and None values (ChromaDB doesn't support them)
        clean_metadata = {
            k: v for k, v in metadata.items()
            if v is not None and v != "" and (not isinstance(v, list) or len(v) > 0)
        }
        self._collection.add(
            ids=[memory_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[clean_metadata],
        )

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        min_importance: float = 0.3,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Search memories by vector similarity."""
        # Build where filter for importance
        query_where = where or {}
        if min_importance > 0:
            # ChromaDB uses a different filter syntax
            query_where = {"importance": {"$gte": min_importance}}

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=query_where if query_where else None,
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                memories.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                })
        return memories

    def count(self) -> int:
        """Count total memories."""
        return self._collection.count()

    def delete(self, memory_id: str) -> None:
        """Delete a memory by ID."""
        self._collection.delete(ids=[memory_id])

    def clear(self) -> None:
        """Clear all memories."""
        self._collection.delete(where={})

    def get_all(self) -> list[dict[str, Any]]:
        """Get all memories (for debugging)."""
        all_data = self._collection.get(include=["documents", "metadatas"])
        results = []
        for i, doc_id in enumerate(all_data["ids"]):
            results.append({
                "id": doc_id,
                "content": all_data["documents"][i],
                "metadata": all_data["metadatas"][i],
            })
        return results

    # ── v2: metadata updates ──────────────────────────────

    def update_metadata(self, memory_id: str, metadata: dict[str, Any]) -> None:
        """Update metadata fields for a single memory."""
        # Clean None/empty values before upserting
        clean_metadata = {
            k: v for k, v in metadata.items()
            if v is not None and v != "" and (not isinstance(v, list) or len(v) > 0)
        }
        self._collection.update(
            ids=[memory_id],
            metadatas=[clean_metadata],
        )

    def upsert_memory(
        self,
        memory_id: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any],
    ) -> None:
        """Insert or update a memory document."""
        clean_metadata = {
            k: v for k, v in metadata.items()
            if v is not None and v != "" and (not isinstance(v, list) or len(v) > 0)
        }
        self._collection.upsert(
            ids=[memory_id],
            documents=[content],
            embeddings=[embedding],
            metadatas=[clean_metadata],
        )

    # ── v2: decay helpers ─────────────────────────────────

    def get_where(self, where: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Get all memories matching a where filter."""
        try:
            data = self._collection.get(
                where=where,
                include=["documents", "metadatas"],
            )
        except Exception:
            return []
        results = []
        for i, doc_id in enumerate(data["ids"]):
            results.append({
                "id": doc_id,
                "content": data["documents"][i],
                "metadata": data["metadatas"][i] or {},
            })
        return results

    # ── v2: tier / date queries ───────────────────────────

    def find_by_tier_and_date(
        self, tier: str, date_key: str
    ) -> list[dict[str, Any]]:
        """Find memories of a given tier whose created_at starts with date_key."""
        all_mems = self.get_all()
        return [
            m for m in all_mems
            if m["metadata"].get("tier") == tier
            and m["metadata"].get("created_at", "").startswith(date_key)
        ]


chroma_client = ChromaDBClient()
