import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from llama_index.core import (
    Settings,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from src.config import config
from src.engines.model_factory import (
    _SYNONYM_CANONICAL_MAP,
    initialize_models,
    is_gemini_quota_exhausted,
)
from src.vault.doc_loader import (
    _compute_dir_signature,
    evict_vault_document_cache,
    load_single_vault_document,
    load_vault_documents,
)

# In-memory cache of loaded indices: {storage_dir_str: (dir_sig, VectorStoreIndex, SummaryIndex)}
_INDEX_MEM_CACHE: Dict[str, Tuple[Tuple[Tuple[str, int, int], ...], VectorStoreIndex, SummaryIndex]] = {}

_TOKEN_CLEAN_RE = re.compile(r"[^a-z0-9#]+")
_STOP_WORDS = frozenset({
    "what", "is", "the", "for", "a", "an", "to", "in", "of", "and", "or",
    "on", "can", "how", "do", "does", "are", "with", "from", "by", "that",
    "this", "they", "their", "my", "she", "he", "it", "be", "if", "have",
    "has", "had", "was", "were", "according", "compared", "about", "when",
})


class _ResilientVaultRetriever(BaseRetriever):
    """Retriever that delegates to UnstructuredEngineManager.retrieve_nodes for resilient semantic + lexical retrieval."""

    def __init__(self, manager: "UnstructuredEngineManager", similarity_top_k: int = 3):
        super().__init__()
        self._manager = manager
        self._similarity_top_k = similarity_top_k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        return self._manager.retrieve_nodes(query_bundle.query_str, similarity_top_k=self._similarity_top_k)


class UnstructuredEngineManager:
    """
    Manages vector indexing, document chunking, semantic retrieval,
    and holistic summarization over unstructured knowledge vault documents.
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        force_reindex: bool = False,
    ):
        # Only initialize default models if global Settings have not been configured yet
        if getattr(Settings, "_llm", None) is None or getattr(Settings, "_embed_model", None) is None:
            initialize_models()
        self.storage_dir = ((storage_dir or config.storage_dir) / "unstructured_vault").resolve()
        self.vector_index: Optional[VectorStoreIndex] = None
        self.summary_index: Optional[SummaryIndex] = None
        self._initialize_indices(force_reindex=force_reindex)

    def _initialize_indices(self, force_reindex: bool = False) -> None:
        """Loads index from in-memory cache, disk cache, or constructs from source documents."""
        storage_key = str(self.storage_dir)
        current_sig = _compute_dir_signature(config.unstructured_dir)

        if not force_reindex and storage_key in _INDEX_MEM_CACHE:
            cached_sig, cached_vec, cached_sum = _INDEX_MEM_CACHE[storage_key]
            if cached_sig == current_sig:
                self.vector_index = cached_vec
                self.summary_index = cached_sum
                return

        index_exists = (
            self.storage_dir.exists()
            and (self.storage_dir / "docstore.json").exists()
        )

        if index_exists and not force_reindex:
            try:
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self.storage_dir)
                )
                self.vector_index = load_index_from_storage(
                    storage_context,
                    embed_model=Settings.embed_model,
                )
                # Build in-memory summary index from already-parsed docstore nodes (zero disk re-parsing)
                existing_nodes = list(storage_context.docstore.docs.values())
                if existing_nodes:
                    self.summary_index = SummaryIndex(existing_nodes)
                else:
                    docs = load_vault_documents()
                    self.summary_index = SummaryIndex.from_documents(docs) if docs else SummaryIndex([])
                _INDEX_MEM_CACHE[storage_key] = (current_sig, self.vector_index, self.summary_index)
                return
            except Exception as e:
                print(f"[Warning] Index loading failed ({e}). Rebuilding indices...")

        # Build fresh indices
        docs = load_vault_documents(force_reload=force_reindex)
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        nodes = splitter.get_nodes_from_documents(docs) if docs else []

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        if nodes:
            self.vector_index = VectorStoreIndex(nodes, embed_model=Settings.embed_model)
            self.vector_index.storage_context.persist(persist_dir=str(self.storage_dir))
            self.summary_index = SummaryIndex(nodes)
        else:
            self.vector_index = VectorStoreIndex([], embed_model=Settings.embed_model)
            self.summary_index = SummaryIndex([])

        _INDEX_MEM_CACHE[storage_key] = (_compute_dir_signature(config.unstructured_dir), self.vector_index, self.summary_index)

    def reindex(self, force: bool = True) -> None:
        """Forces a clean re-indexing of all documents currently in the vault."""
        self._initialize_indices(force_reindex=force)

    def delete_document(self, file_name: str) -> bool:
        """Deletes a document from the vault directory and incrementally removes its nodes from the indices."""
        target_path = config.unstructured_dir / file_name
        if not target_path.exists():
            return False

        target_path.unlink()
        evict_vault_document_cache(file_name)

        # Incrementally remove matching nodes to avoid re-embedding remaining documents
        if self.vector_index is not None and self.summary_index is not None:
            try:
                node_ids_to_delete = [
                    node_id
                    for node_id, node in self.vector_index.docstore.docs.items()
                    if node.metadata.get("file_name") == file_name
                ]
                if node_ids_to_delete:
                    self.vector_index.delete_nodes(node_ids_to_delete, delete_from_docstore=True)
                    try:
                        self.summary_index.delete_nodes(node_ids_to_delete, delete_from_docstore=True)
                    except Exception:
                        remaining_nodes = list(self.vector_index.docstore.docs.values())
                        self.summary_index = SummaryIndex(remaining_nodes)
                    self.vector_index.storage_context.persist(persist_dir=str(self.storage_dir))
                _INDEX_MEM_CACHE[str(self.storage_dir)] = (
                    _compute_dir_signature(config.unstructured_dir),
                    self.vector_index,
                    self.summary_index,
                )
                return True
            except Exception:
                pass

        self.reindex(force=True)
        return True

    def add_document(self, file_name: str, content: bytes) -> Path:
        """Saves a new document into the vault and incrementally indexes its chunks."""
        config.unstructured_dir.mkdir(parents=True, exist_ok=True)
        target_path = config.unstructured_dir / file_name
        is_overwrite = target_path.exists()
        target_path.write_bytes(content)

        # Parse only the newly added/updated file rather than re-reading the entire vault
        new_docs = load_single_vault_document(target_path)

        if self.vector_index is not None and self.summary_index is not None and new_docs:
            try:
                if is_overwrite:
                    old_node_ids = [
                        node_id
                        for node_id, node in self.vector_index.docstore.docs.items()
                        if node.metadata.get("file_name") == file_name
                    ]
                    if old_node_ids:
                        self.vector_index.delete_nodes(old_node_ids, delete_from_docstore=True)

                splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
                new_nodes = splitter.get_nodes_from_documents(new_docs)
                if new_nodes:
                    self.vector_index.insert_nodes(new_nodes)
                    all_nodes = list(self.vector_index.docstore.docs.values())
                    self.summary_index = SummaryIndex(all_nodes)
                    self.vector_index.storage_context.persist(persist_dir=str(self.storage_dir))
                    _INDEX_MEM_CACHE[str(self.storage_dir)] = (
                        _compute_dir_signature(config.unstructured_dir),
                        self.vector_index,
                        self.summary_index,
                    )
                    return target_path
            except Exception:
                pass

        self.reindex(force=True)
        return target_path

    def get_vault_file_list(self) -> List[dict]:
        """Returns metadata for all physical files currently stored in the unstructured vault."""
        files = []
        for ext in ("*.md", "*.txt", "*.pdf"):
            for p in config.unstructured_dir.glob(ext):
                try:
                    stat = p.stat()
                    files.append({
                        "name": p.name,
                        "extension": p.suffix.lower(),
                        "size_bytes": stat.st_size,
                        "size_kb": round(stat.st_size / 1024, 2),
                        "modified_time": stat.st_mtime,
                        "path": p,
                    })
                except OSError:
                    continue
        return sorted(files, key=lambda x: x["name"])

    def get_vector_query_engine(self, similarity_top_k: int = 3, llm=None):
        """Returns a query engine for semantic retrieval over policy and spec documents."""
        if self.vector_index is None:
            self._initialize_indices(force_reindex=True)
        retriever = _ResilientVaultRetriever(self, similarity_top_k=similarity_top_k)
        return RetrieverQueryEngine.from_args(
            retriever=retriever,
            response_mode="compact",
            llm=llm or Settings.llm,
        )

    def get_summary_query_engine(self, llm=None):
        """Returns a query engine for holistic summarization across all documents."""
        if self.summary_index is None:
            self._initialize_indices(force_reindex=True)
        return self.summary_index.as_query_engine(
            response_mode="tree_summarize",
            llm=llm or Settings.llm,
        )

    def get_hyde_query_engine(self, llm=None):
        """
        Constructs a HyDE (Hypothetical Document Embeddings) Query Engine.
        Generates a hypothetical answer before embedding lookup to eliminate
        paraphrase penalties and vocabulary mismatch between queries and documents.
        """
        from llama_index.core.indices.query.query_transform import HyDEQueryTransform
        from llama_index.core.query_engine import TransformQueryEngine

        base_engine = self.get_vector_query_engine()
        hyde = HyDEQueryTransform(llm=llm or Settings.llm, include_original=True)
        return TransformQueryEngine(base_engine, hyde)

    def _retrieve_nodes_hybrid_fallback(self, query_str: str, similarity_top_k: int = 3) -> List[NodeWithScore]:
        """
        Scores docstore nodes using keyword + synonym-expanded concept overlap when cloud embeddings
        are unavailable or when persisted vector embeddings belong to a rate-limited cloud model.
        """
        if self.vector_index is None:
            return []
        all_nodes = list(self.vector_index.docstore.docs.values())
        if not all_nodes:
            return []

        clean_q = _TOKEN_CLEAN_RE.sub(" ", query_str.lower()).strip()
        q_tokens = [t for t in clean_q.split() if len(t) > 1 and t not in _STOP_WORDS]
        expanded_q: List[str] = list(q_tokens)
        for t in q_tokens:
            if t.endswith("s") and len(t) > 3:
                expanded_q.append(t[:-1])
            if t in _SYNONYM_CANONICAL_MAP:
                expanded_q.extend(_SYNONYM_CANONICAL_MAP[t].split())

        scored: List[NodeWithScore] = []
        denom = max(len(expanded_q), 1)
        for node in all_nodes:
            text_lower = node.text.lower()
            title_lower = str(node.metadata.get("title", "")).lower()
            fn_lower = str(node.metadata.get("file_name", "")).lower()
            combined = f"{title_lower} {fn_lower} {text_lower}"

            direct_hits = sum(1 for t in q_tokens if t in combined)
            expanded_hits = sum(1 for t in expanded_q if t in combined)
            raw_ratio = (direct_hits * 1.3 + expanded_hits * 0.7) / (denom * 1.5)
            calibrated_score = (
                min(0.94, round(0.62 + 0.32 * min(raw_ratio, 1.0), 4))
                if (direct_hits + expanded_hits) > 0
                else 0.45
            )
            scored.append(NodeWithScore(node=node, score=calibrated_score))

        scored.sort(key=lambda x: float(x.score or 0.0), reverse=True)
        return scored[:similarity_top_k]

    def retrieve_nodes(self, query_str: str, similarity_top_k: int = 3) -> List[NodeWithScore]:
        """Retrieves raw semantic nodes for auditing and UI inspection, with automatic hybrid fallback."""
        if self.vector_index is None:
            self._initialize_indices(force_reindex=True)

        embed_model = Settings.embed_model
        embed_is_fallback = getattr(embed_model, "is_fallback", False) or is_gemini_quota_exhausted()
        if not embed_is_fallback:
            try:
                retriever = self.vector_index.as_retriever(similarity_top_k=similarity_top_k)
                nodes = retriever.retrieve(query_str)
                # Check if embed_model tripped fallback during retrieve()
                if not getattr(embed_model, "is_fallback", False) and not is_gemini_quota_exhausted() and nodes:
                    return nodes
            except Exception:
                pass

        return self._retrieve_nodes_hybrid_fallback(query_str, similarity_top_k=similarity_top_k)

    def evaluate_query_retrieval(
        self,
        query_str: str,
        target_doc_keyword: str,
        similarity_top_k: int = 3,
    ) -> dict:
        """
        Directly evaluates retriever performance for a given query:
        computes Hit Rate@k, Reciprocal Rank (RR), and top similarity score.
        """
        nodes = self.retrieve_nodes(query_str, similarity_top_k=similarity_top_k)
        hit = False
        rank = None
        max_score = 0.0
        kw_lower = target_doc_keyword.lower()

        for idx, node in enumerate(nodes):
            file_name = node.node.metadata.get("file_name", "")
            title = node.node.metadata.get("title", "")
            score = float(node.score or 0.0)
            if score > max_score:
                max_score = score

            if (
                kw_lower in file_name.lower()
                or kw_lower in title.lower()
                or kw_lower in node.node.text.lower()
            ):
                if not hit:
                    hit = True
                    rank = idx + 1

        reciprocal_rank = (1.0 / rank) if rank else 0.0
        return {
            "hit": hit,
            "rank": rank,
            "reciprocal_rank": reciprocal_rank,
            "max_score": round(max_score, 4),
            "retrieved_nodes_count": len(nodes),
            "top_node_snippet": nodes[0].node.text[:150] if nodes else "",
        }
