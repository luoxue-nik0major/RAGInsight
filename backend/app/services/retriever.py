from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from starlette.concurrency import run_in_threadpool
from app.core.config import get_settings
from app.services.cache import embedding_cache
from app.utils.text_utils import is_chinese_text, tokenize_for_bm25
import os
import json

settings = get_settings()

# Embedding model cache directory: configurable via env; defaults to HF cache.
# Set RAGINSIGHT_EMBEDDING_CACHE_DIR to reuse an existing model directory.
EMBEDDING_CACHE_DIR = os.environ.get("RAGINSIGHT_EMBEDDING_CACHE_DIR") or None
if EMBEDDING_CACHE_DIR:
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

# Offline mode: set RAGINSIGHT_EMBEDDING_LOCAL_ONLY=true to forbid model
# downloads (requires the model to be present in the cache directory).
_EMBEDDING_LOCAL_ONLY = os.environ.get(
    "RAGINSIGHT_EMBEDDING_LOCAL_ONLY", "false"
).lower() == "true"

# Use BAAI/bge-small-zh-v1.5 for Chinese semantic understanding
# Vector dim: 512 (vs all-MiniLM-L6-v2's 384)
_chinese_embedding_fn = None


class _CachingEmbeddingFunction:
    """Wrap a ChromaDB embedding function with a per-text TTL cache.

    Avoids repeated embedding inference for identical texts (query cache
    misses, repeated documents across init runs, etc.).
    """

    def __init__(self, inner):
        self._inner = inner

    def __call__(self, input):
        texts = list(input)
        results: List[Any] = [None] * len(texts)
        missing_idx: List[int] = []
        missing_texts: List[str] = []
        for i, text in enumerate(texts):
            hit = embedding_cache.get(text)
            if hit is None:
                missing_idx.append(i)
                missing_texts.append(text)
            else:
                results[i] = hit
        if missing_texts:
            computed = self._inner(missing_texts)
            for idx, text, vec in zip(missing_idx, missing_texts, computed):
                embedding_cache.set(text, vec)
                results[idx] = vec
        return results

    def __getattr__(self, name):
        # Delegate attributes (e.g. ChromaDB's name()/is_legacy()) to inner
        return getattr(self._inner, name)


def _get_embedding_fn():
    global _chinese_embedding_fn
    if _chinese_embedding_fn is None:
        _chinese_embedding_fn = _CachingEmbeddingFunction(
            SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-small-zh-v1.5",
                cache_folder=EMBEDDING_CACHE_DIR,
                normalize_embeddings=True,
                local_files_only=_EMBEDDING_LOCAL_ONLY,
            )
        )
    return _chinese_embedding_fn


def _extract_entities(text: str) -> List[str]:
    """Extract entities from text using jieba POS tagging."""
    import jieba.posseg as pseg
    entities = []
    for word, flag in pseg.cut(text):
        if flag in ("nr", "ns", "nt", "nz"):
            if word not in entities:
                entities.append(word)
    return entities


def list_available_collections() -> list:
    """List all available ChromaDB collection names."""
    client = chromadb.PersistentClient(
        path=settings.chroma_db_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        return [c.name for c in client.list_collections()]
    except Exception:
        return ["nq_documents", "chinese_poetry"]


class RetrieverAdapter(ABC):
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5, collection_name: str = None) -> Dict[str, Any]:
        """Return standardized retrieval result. If collection_name is given, use it instead of auto-detect."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def _resolve_collection(self, query: str, collection_name: str = None):
        """Resolve which collection to use. Explicit collection_name overrides language auto-detect."""
        if collection_name:
            return self.client.get_or_create_collection(
                collection_name,
                embedding_function=_get_embedding_fn(),
            )
        return self._collection(query)


class VectorRetrieverAdapter(RetrieverAdapter):
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_en = self.client.get_or_create_collection(
            "nq_documents",
            embedding_function=_get_embedding_fn(),
        )
        self._collection_zh = self.client.get_or_create_collection(
            "chinese_poetry",
            embedding_function=_get_embedding_fn(),
        )

    def _collection(self, query: str):
        return self._collection_zh if is_chinese_text(query) else self._collection_en

    @property
    def name(self) -> str:
        return "vector"

    async def retrieve(self, query: str, top_k: int = 5, collection_name: str = None) -> Dict[str, Any]:
        collection = self._resolve_collection(query, collection_name)
        # ChromaDB query + embedding inference are blocking; offload to thread pool
        results = await run_in_threadpool(
            collection.query,
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]
            # ChromaDB HNSW returns squared L2 distance.
            # With normalize_embeddings=True, squared_L2 = 2 - 2*cos_sim,
            # so cos_sim = 1 - squared_L2 / 2. Clamp to [0, 1].
            similarity = max(0.0, 1.0 - distance / 2.0)
            chunks.append({
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", ""),
                "relevance_score": float(similarity),
                "chunk_index": i,
            })
        return {
            "chunks": chunks,
            "total_found": len(chunks),
        }

    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
        """Batch add documents to ChromaDB (English collection by default)."""
        self._collection_en.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )


class HybridRetrieverAdapter(RetrieverAdapter):
    """Hybrid retrieval: dense vector + sparse BM25 with RRF fusion."""

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_en = self.client.get_or_create_collection(
            "nq_documents",
            embedding_function=_get_embedding_fn(),
        )
        self._collection_zh = self.client.get_or_create_collection(
            "chinese_poetry",
            embedding_function=_get_embedding_fn(),
        )
        self._bm25_en: Optional[Any] = None
        self._bm25_zh: Optional[Any] = None
        self._doc_ids_en: List[str] = []
        self._documents_en: List[str] = []
        self._metadatas_en: List[Dict[str, Any]] = []
        self._doc_ids_zh: List[str] = []
        self._documents_zh: List[str] = []
        self._metadatas_zh: List[Dict[str, Any]] = []
        self._bm25_collections: Dict[str, Any] = {}
        self._bm25_collection_ids: Dict[str, List[str]] = {}
        self._bm25_collection_docs: Dict[str, List[str]] = {}
        self._bm25_collection_meta: Dict[str, List[Dict[str, Any]]] = {}

    def _collection(self, query: str):
        return self._collection_zh if is_chinese_text(query) else self._collection_en

    @property
    def name(self) -> str:
        return "hybrid"

    def _build_bm25_index(self, is_chinese: bool):
        """Lazy-load BM25 index from ChromaDB collection."""
        if is_chinese and self._bm25_zh is not None:
            return
        if not is_chinese and self._bm25_en is not None:
            return

        from rank_bm25 import BM25Okapi

        collection = self._collection_zh if is_chinese else self._collection_en
        all_data = collection.get(include=["documents", "metadatas"])
        doc_ids = all_data["ids"]
        documents = all_data["documents"]
        metadatas = all_data["metadatas"]

        tokenized_corpus = [tokenize_for_bm25(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)

        if is_chinese:
            self._doc_ids_zh = doc_ids
            self._documents_zh = documents
            self._metadatas_zh = metadatas
            self._bm25_zh = bm25
        else:
            self._doc_ids_en = doc_ids
            self._documents_en = documents
            self._metadatas_en = metadatas
            self._bm25_en = bm25

    def _build_bm25_index_for_collection(self, collection_name: str):
        """Lazy-load BM25 index for an explicitly named collection."""
        if collection_name in self._bm25_collections and self._bm25_collections[collection_name] is not None:
            return

        from rank_bm25 import BM25Okapi

        collection = self.client.get_or_create_collection(
            collection_name,
            embedding_function=_get_embedding_fn(),
        )
        all_data = collection.get(include=["documents", "metadatas"])
        if not all_data["ids"]:
            return

        documents = all_data["documents"]
        tokenized_corpus = [tokenize_for_bm25(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)

        self._bm25_collections[collection_name] = bm25
        self._bm25_collection_ids[collection_name] = all_data["ids"]
        self._bm25_collection_docs[collection_name] = documents
        self._bm25_collection_meta[collection_name] = all_data["metadatas"]

    async def retrieve(self, query: str, top_k: int = 5, collection_name: str = None) -> Dict[str, Any]:
        is_chinese = is_chinese_text(query)
        # When using explicit collection, still need to pick the right BM25 index
        # For simplicity, use the language-appropriate index unless explicitly overridden
        if collection_name:
            # Build BM25 for the explicit collection if not already cached
            await run_in_threadpool(self._build_bm25_index_for_collection, collection_name)
            is_chinese = False  # Use the collection-specific BM25
            collection = self._resolve_collection(query, collection_name)
            doc_ids = self._bm25_collection_ids.get(collection_name, [])
            documents = self._bm25_collection_docs.get(collection_name, [])
            metadatas = self._bm25_collection_meta.get(collection_name, [])
            bm25 = self._bm25_collections.get(collection_name)
        else:
            await run_in_threadpool(self._build_bm25_index, is_chinese)
            collection = self._collection(query)
            doc_ids = self._doc_ids_zh if is_chinese else self._doc_ids_en
            documents = self._documents_zh if is_chinese else self._documents_en
            metadatas = self._metadatas_zh if is_chinese else self._metadatas_en
            bm25 = self._bm25_zh if is_chinese else self._bm25_en

        # 1. Dense retrieval (vector) — blocking, offload to thread pool
        dense_results = await run_in_threadpool(
            collection.query,
            query_texts=[query],
            n_results=min(top_k * 4, len(documents)),
            include=["documents", "metadatas", "distances"],
        )

        dense_ranked: Dict[str, Dict[str, Any]] = {}
        for i in range(len(dense_results["documents"][0])):
            doc_id = dense_results["ids"][0][i]
            distance = dense_results["distances"][0][i]
            similarity = max(0.0, 1.0 - distance / 2.0)
            dense_ranked[doc_id] = {
                "content": dense_results["documents"][0][i],
                "source": dense_results["metadatas"][0][i].get("source", ""),
                "relevance_score": float(similarity),
                "rank": i + 1,
            }

        # 2. Sparse retrieval (BM25) — tokenization + scoring are blocking
        query_tokens = await run_in_threadpool(tokenize_for_bm25, query)
        bm25_scores = await run_in_threadpool(bm25.get_scores, query_tokens)
        # Get top indices by BM25 score
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:top_k * 4]

        sparse_ranked: Dict[str, Dict[str, Any]] = {}
        for rank, idx in enumerate(top_indices, start=1):
            doc_id = doc_ids[idx]
            sparse_ranked[doc_id] = {
                "content": documents[idx],
                "source": metadatas[idx].get("source", ""),
                "relevance_score": float(bm25_scores[idx]),
                "rank": rank,
            }

        # 3. RRF fusion
        K = 60  # RRF constant
        all_ids = set(dense_ranked.keys()) | set(sparse_ranked.keys())
        rrf_scores: Dict[str, float] = {}
        for doc_id in all_ids:
            score = 0.0
            if doc_id in dense_ranked:
                score += 1.0 / (K + dense_ranked[doc_id]["rank"])
            if doc_id in sparse_ranked:
                score += 1.0 / (K + sparse_ranked[doc_id]["rank"])
            rrf_scores[doc_id] = score

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        chunks = []
        for i, doc_id in enumerate(sorted_ids):
            # Prefer dense metadata for consistency
            if doc_id in dense_ranked:
                item = dense_ranked[doc_id]
            else:
                item = sparse_ranked[doc_id]
            chunks.append({
                "content": item["content"],
                "source": item["source"],
                "relevance_score": rrf_scores[doc_id],  # use RRF score as combined relevance
                "chunk_index": i,
            })

        return {
            "chunks": chunks,
            "total_found": len(chunks),
        }


class GraphRetrieverAdapter(RetrieverAdapter):
    """Graph-based retrieval: entity co-occurrence graph with multi-hop BFS."""

    GRAPH_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "knowledge_graph.json")

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection_en = self.client.get_or_create_collection(
            "nq_documents",
            embedding_function=_get_embedding_fn(),
        )
        self._collection_zh = self.client.get_or_create_collection(
            "chinese_poetry",
            embedding_function=_get_embedding_fn(),
        )
        self._graph: Optional[Any] = None
        self._chunk_map: Dict[str, Dict[str, Any]] = {}  # doc_id -> {content, source}
        self._load_graph()

    def _collection(self, query: str):
        return self._collection_zh if is_chinese_text(query) else self._collection_en

    def _load_graph(self):
        """Lazy-load knowledge graph from JSON."""
        if not os.path.exists(self.GRAPH_PATH):
            return
        try:
            with open(self.GRAPH_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            import networkx as nx
            self._graph = nx.Graph()
            for node in data.get("nodes", []):
                self._graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
            for edge in data.get("edges", []):
                self._graph.add_edge(edge["source"], edge["target"], **{k: v for k, v in edge.items() if k not in ("source", "target")})
        except Exception:
            self._graph = None

    def _build_chunk_map(self, query: str = "", collection_name: str = None):
        """Build map of chunk id -> content/source from ChromaDB."""
        if self._chunk_map:
            return
        try:
            if collection_name:
                collection = self.client.get_or_create_collection(
                    collection_name,
                    embedding_function=_get_embedding_fn(),
                )
            else:
                collection = self._collection(query) if query else self._collection_en
            all_data = collection.get(include=["documents", "metadatas"])
            for doc_id, content, metadata in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
                self._chunk_map[doc_id] = {
                    "content": content,
                    "source": metadata.get("source", ""),
                }
        except Exception:
            pass

    @property
    def name(self) -> str:
        return "graph"

    async def retrieve(self, query: str, top_k: int = 5, collection_name: str = None) -> Dict[str, Any]:
        await run_in_threadpool(self._build_chunk_map, query, collection_name)

        # If no graph available, fall back to vector search
        if self._graph is None:
            return await self._vector_fallback(query, top_k, collection_name)

        query_entities = _extract_entities(query)
        if not query_entities:
            return await self._vector_fallback(query, top_k)

        import networkx as nx

        # Collect chunks reachable from query entities within 2 hops
        chunk_scores: Dict[str, float] = {}
        chunk_paths: Dict[str, List[str]] = {}

        for ent in query_entities:
            ent_node = f"ent:{ent}"
            if not self._graph.has_node(ent_node):
                continue

            # BFS up to 2 hops
            for target, path in nx.single_source_shortest_path(self._graph, ent_node, cutoff=2).items():
                if not target.startswith("chunk:"):
                    continue
                doc_id = target.replace("chunk:", "")
                if doc_id not in self._chunk_map:
                    continue

                hop = len(path) - 1
                score = 1.0 if hop == 1 else 0.5
                if doc_id in chunk_scores:
                    chunk_scores[doc_id] = max(chunk_scores[doc_id], score)
                else:
                    chunk_scores[doc_id] = score
                    # Build human-readable path
                    path_names = []
                    for node in path:
                        if node.startswith("ent:"):
                            path_names.append(node.replace("ent:", ""))
                        elif node.startswith("chunk:"):
                            src = self._chunk_map.get(doc_id, {}).get("source", "")
                            path_names.append(src or doc_id)
                    chunk_paths[doc_id] = path_names

        if not chunk_scores:
            return await self._vector_fallback(query, top_k)

        # Sort by score descending
        sorted_ids = sorted(chunk_scores.keys(), key=lambda x: chunk_scores[x], reverse=True)[:top_k]

        chunks = []
        for i, doc_id in enumerate(sorted_ids):
            info = self._chunk_map[doc_id]
            chunks.append({
                "content": info["content"],
                "source": info["source"],
                "relevance_score": chunk_scores[doc_id],
                "chunk_index": i,
                "reasoning_path": chunk_paths.get(doc_id, []),
            })

        return {
            "chunks": chunks,
            "total_found": len(chunks),
        }

    async def _vector_fallback(self, query: str, top_k: int = 5, collection_name: str = None) -> Dict[str, Any]:
        """Fallback to vector search when graph is unavailable."""
        collection = self._resolve_collection(query, collection_name)
        results = await run_in_threadpool(
            collection.query,
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for i in range(len(results["documents"][0])):
            distance = results["distances"][0][i]
            similarity = max(0.0, 1.0 - distance / 2.0)
            chunks.append({
                "content": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", ""),
                "relevance_score": float(similarity),
                "chunk_index": i,
                "reasoning_path": [],
            })
        return {
            "chunks": chunks,
            "total_found": len(chunks),
        }


class RetrieverRegistry:
    def __init__(self):
        self._retrievers: Dict[str, RetrieverAdapter] = {}
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            self.register(VectorRetrieverAdapter())
            self.register(HybridRetrieverAdapter())
            self.register(GraphRetrieverAdapter())
            self._initialized = True

    def register(self, adapter: RetrieverAdapter):
        self._retrievers[adapter.name] = adapter

    def get(self, name: str) -> RetrieverAdapter:
        self._ensure_initialized()
        adapter = self._retrievers.get(name)
        if adapter is None:
            raise ValueError(f"Unknown retriever strategy: {name}. Available: {self.list()}")
        return adapter

    def list(self) -> List[str]:
        self._ensure_initialized()
        return list(self._retrievers.keys())


retriever_registry = RetrieverRegistry()
