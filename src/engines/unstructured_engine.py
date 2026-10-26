from pathlib import Path
from typing import Optional, List
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore
from src.config import config
from src.vault.doc_loader import load_vault_documents
from src.engines.model_factory import initialize_models


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
        initialize_models()  # Ensure global Settings.embed_model and Settings.llm are configured
        self.storage_dir = (storage_dir or config.storage_dir) / "unstructured_vault"
        self.vector_index: Optional[VectorStoreIndex] = None
        self.summary_index: Optional[SummaryIndex] = None
        self._initialize_indices(force_reindex=force_reindex)

    def _initialize_indices(self, force_reindex: bool = False) -> None:
        """Loads index from disk cache or constructs from source documents."""
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
                # Build in-memory summary index over documents
                docs = load_vault_documents()
                self.summary_index = SummaryIndex.from_documents(docs)
                return
            except Exception as e:
                print(f"[Warning] Index loading failed ({e}). Rebuilding indices...")

        # Build fresh indices
        docs = load_vault_documents()
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

    def reindex(self, force: bool = True) -> None:
        """Forces a clean re-indexing of all documents currently in the vault."""
        self._initialize_indices(force_reindex=force)

    def delete_document(self, file_name: str) -> bool:
        """Deletes a document from the vault directory and refreshes indices."""
        target_path = config.unstructured_dir / file_name
        if target_path.exists():
            target_path.unlink()
            self.reindex(force=True)
            return True
        return False

    def add_document(self, file_name: str, content: bytes) -> Path:
        """Saves a new document into the vault and triggers index rebuild."""
        config.unstructured_dir.mkdir(parents=True, exist_ok=True)
        target_path = config.unstructured_dir / file_name
        target_path.write_bytes(content)
        self.reindex(force=True)
        return target_path

    def get_vault_file_list(self) -> List[dict]:
        """Returns metadata for all physical files currently stored in the unstructured vault."""
        files = []
        for ext in ("*.md", "*.txt", "*.pdf"):
            for p in config.unstructured_dir.glob(ext):
                stat = p.stat()
                files.append({
                    "name": p.name,
                    "extension": p.suffix.lower(),
                    "size_bytes": stat.st_size,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "modified_time": stat.st_mtime,
                    "path": p,
                })
        return sorted(files, key=lambda x: x["name"])

    def get_vector_query_engine(self, similarity_top_k: int = 3):
        """Returns a query engine for semantic retrieval over policy and spec documents."""
        if self.vector_index is None:
            self._initialize_indices(force_reindex=True)
        return self.vector_index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode="compact",
        )

    def get_summary_query_engine(self):
        """Returns a query engine for holistic summarization across all documents."""
        if self.summary_index is None:
            self._initialize_indices(force_reindex=True)
        return self.summary_index.as_query_engine(
            response_mode="tree_summarize",
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

    def retrieve_nodes(self, query_str: str, similarity_top_k: int = 3) -> List[NodeWithScore]:
        """Retrieves raw semantic nodes for auditing and UI inspection."""
        if self.vector_index is None:
            self._initialize_indices(force_reindex=True)
        retriever = self.vector_index.as_retriever(similarity_top_k=similarity_top_k)
        return retriever.retrieve(query_str)

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

        for idx, node in enumerate(nodes):
            file_name = node.node.metadata.get("file_name", "")
            title = node.node.metadata.get("title", "")
            score = float(node.score or 0.0)
            if score > max_score:
                max_score = score

            if (target_doc_keyword.lower() in file_name.lower() or 
                target_doc_keyword.lower() in title.lower() or
                target_doc_keyword.lower() in node.node.text.lower()):
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
