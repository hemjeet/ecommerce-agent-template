"""
Semantic Cache — PostgreSQL-backed cache using pgvector cosine similarity.

Instead of exact-match caching, this embeds queries into vectors and finds
cache hits by cosine similarity (threshold ≥ 0.92 by default). This means
"what is your return policy?" and "how do I return a product?" resolve to
the same cached response.

Storage: PostgreSQL table `semantic_cache` with a `vector(1536)` column
indexed via IVFFlat for fast approximate nearest-neighbour search.
"""

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)


class CacheResult:
    """Wrapper for a semantic cache hit."""

    def __init__(self, response: str, similarity: float, query_text: str):
        self.response = response
        self.similarity = similarity
        self.original_query = query_text

    def __repr__(self):
        return (
            f"CacheResult(similarity={self.similarity:.4f}, "
            f"original_query={self.original_query!r})"
        )


class SemanticCache:
    """
    PostgreSQL + pgvector backed semantic cache.

    Parameters
    ----------
    embeddings : langchain Embeddings instance (e.g. OpenAIEmbeddings)
    session_factory : SQLAlchemy sessionmaker bound to the Postgres engine
    threshold : float
        Minimum cosine similarity to count as a cache hit (0.0–1.0).
    ttl : int
        Time-to-live in seconds for cached entries.
    """

    # ── SQL templates ────────────────────────────────────────────────

    _CREATE_TABLE_SQL = text("""
        CREATE TABLE IF NOT EXISTS semantic_cache (
            id          SERIAL PRIMARY KEY,
            query_text  TEXT NOT NULL,
            embedding   vector(1536) NOT NULL,
            response    TEXT NOT NULL,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            expires_at  TIMESTAMPTZ NOT NULL
        );
    """)

    _CREATE_INDEX_SQL = text("""
        CREATE INDEX IF NOT EXISTS idx_semantic_cache_embedding
        ON semantic_cache
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10);
    """)

    _SEARCH_SQL = text("""
        SELECT
            id,
            query_text,
            response,
            1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
        FROM semantic_cache
        WHERE expires_at > NOW()
        ORDER BY embedding <=> CAST(:query_embedding AS vector)
        LIMIT 1;
    """)

    _INSERT_SQL = text("""
        INSERT INTO semantic_cache (query_text, embedding, response, expires_at)
        VALUES (:query_text, CAST(:embedding AS vector), :response, :expires_at);
    """)

    _DELETE_EXPIRED_SQL = text("""
        DELETE FROM semantic_cache WHERE expires_at <= NOW();
    """)

    _STATS_SQL = text("""
        SELECT
            COUNT(*) AS total_entries,
            COUNT(*) FILTER (WHERE expires_at > NOW()) AS active_entries,
            COUNT(*) FILTER (WHERE expires_at <= NOW()) AS expired_entries
        FROM semantic_cache;
    """)

    _TRUNCATE_SQL = text("TRUNCATE TABLE semantic_cache;")

    def __init__(
        self,
        embeddings,
        session_factory,
        threshold: float = 0.92,
        ttl: int = 600,
    ):
        self.embeddings = embeddings
        self.session_factory = session_factory
        self.threshold = threshold
        self.ttl = ttl

        # Stats counters (thread-safe via lock)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ── Public API ───────────────────────────────────────────────────

    def setup(self) -> None:
        """Create the semantic_cache table and index if they don't exist."""
        db = self.session_factory()
        try:
            db.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            db.execute(self._CREATE_TABLE_SQL)
            db.commit()

            # IVFFlat index needs at least some rows to build; CREATE INDEX
            # IF NOT EXISTS is safe even on an empty table with lists=10.
            try:
                db.execute(self._CREATE_INDEX_SQL)
                db.commit()
            except Exception:
                db.rollback()
                logger.info(
                    "IVFFlat index creation deferred "
                    "(needs rows or already exists)"
                )

            logger.info("Semantic cache table ready")
        except Exception as e:
            db.rollback()
            logger.error("Failed to set up semantic cache table: %s", e)
            raise
        finally:
            db.close()

    def get(self, query: str, embedding: Optional[list[float]] = None) -> Optional[CacheResult]:
        """
        Look up a semantically similar cached response.

        Parameters
        ----------
        query : str
            The search query text.
        embedding : list[float], optional
            Pre-computed embedding vector. When supplied the method skips
            the internal ``_embed()`` call, saving a round-trip to the
            embeddings API.

        Returns a CacheResult on hit, or None on miss.
        """
        query_embedding = embedding or self._embed(query)
        if query_embedding is None:
            return None

        embedding_str = self._to_pgvector_literal(query_embedding)

        db = self.session_factory()
        try:
            row = db.execute(
                self._SEARCH_SQL,
                {"query_embedding": embedding_str},
            ).mappings().first()

            if row and row["similarity"] >= self.threshold:
                with self._lock:
                    self._hits += 1
                result = CacheResult(
                    response=row["response"],
                    similarity=row["similarity"],
                    query_text=row["query_text"],
                )
                logger.info(
                    "Semantic cache HIT (sim=%.4f, original=%r, query=%r)",
                    result.similarity,
                    result.original_query,
                    query,
                )
                return result

            with self._lock:
                self._misses += 1
            logger.info("Semantic cache MISS for query=%r", query)
            return None
        except Exception as e:
            logger.error("Semantic cache lookup failed: %s", e)
            with self._lock:
                self._misses += 1
            return None
        finally:
            db.close()

    def put(self, query: str, response: str, embedding: Optional[list[float]] = None) -> None:
        """Store a query-response pair in the cache.

        Parameters
        ----------
        embedding : list[float], optional
            Pre-computed embedding vector. When supplied the method skips
            the internal ``_embed()`` call, saving a round-trip to the
            embeddings API.
        """
        query_embedding = embedding or self._embed(query)
        if query_embedding is None:
            return

        embedding_str = self._to_pgvector_literal(query_embedding)
        expires_at = datetime.now(timezone.utc).timestamp() + self.ttl
        expires_dt = datetime.fromtimestamp(expires_at, tz=timezone.utc)

        db = self.session_factory()
        try:
            db.execute(
                self._INSERT_SQL,
                {
                    "query_text": query,
                    "embedding": embedding_str,
                    "response": response,
                    "expires_at": expires_dt,
                },
            )
            db.commit()
            logger.info("Semantic cache PUT for query=%r (ttl=%ds)", query, self.ttl)
        except Exception as e:
            db.rollback()
            logger.error("Semantic cache store failed: %s", e)
        finally:
            db.close()

    def cleanup_expired(self) -> int:
        """Delete expired cache entries. Returns count deleted."""
        db = self.session_factory()
        try:
            result = db.execute(self._DELETE_EXPIRED_SQL)
            db.commit()
            deleted = result.rowcount
            if deleted:
                logger.info("Semantic cache: cleaned up %d expired entries", deleted)
            return deleted
        except Exception as e:
            db.rollback()
            logger.error("Semantic cache cleanup failed: %s", e)
            return 0
        finally:
            db.close()

    def clear(self) -> None:
        """Truncate the entire cache."""
        db = self.session_factory()
        try:
            db.execute(self._TRUNCATE_SQL)
            db.commit()
            with self._lock:
                self._hits = 0
                self._misses = 0
            logger.info("Semantic cache cleared")
        except Exception as e:
            db.rollback()
            logger.error("Semantic cache clear failed: %s", e)
        finally:
            db.close()

    def stats(self) -> dict:
        """Return cache statistics."""
        db = self.session_factory()
        try:
            row = db.execute(self._STATS_SQL).mappings().first()
            with self._lock:
                hits = self._hits
                misses = self._misses
            total_lookups = hits + misses
            return {
                "hits": hits,
                "misses": misses,
                "hit_rate": f"{(hits / total_lookups * 100):.1f}%" if total_lookups else "N/A",
                "total_lookups": total_lookups,
                "active_entries": row["active_entries"] if row else 0,
                "expired_entries": row["expired_entries"] if row else 0,
                "total_entries": row["total_entries"] if row else 0,
                "threshold": self.threshold,
                "ttl_seconds": self.ttl,
            }
        except Exception as e:
            logger.error("Semantic cache stats failed: %s", e)
            return {"error": str(e)}
        finally:
            db.close()

    # ── Internal helpers ─────────────────────────────────────────────

    def _embed(self, text: str) -> Optional[list[float]]:
        """Generate an embedding vector for the given text."""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error("Embedding generation failed: %s", e)
            return None

    @staticmethod
    def _to_pgvector_literal(embedding: list[float]) -> str:
        """Convert a list of floats to a pgvector literal string '[0.1,0.2,...]'."""
        return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"
