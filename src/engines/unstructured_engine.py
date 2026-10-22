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
                self.vector_index = load_index_from_storage(storage_context)
                # Build in-memory summary index over documents
                docs = load_vault_documents()
                self.summary_index = SummaryIndex.from_documents(docs)
                return
            except Exception as e:
                print(f"[Warning] Index loading failed ({e}). Rebuilding indices...")

        # Build fresh indices
        docs = load_vault_documents()
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
        nodes = splitter.get_nodes_from_documents(docs)

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.vector_index = VectorStoreIndex(nodes)
        self.vector_index.storage_context.persist(persist_dir=str(self.storage_dir))

        self.summary_index = SummaryIndex(nodes)

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

    def retrieve_nodes(self, query_str: str, similarity_top_k: int = 3) -> List[NodeWithScore]:
        """Retrieves raw semantic nodes for auditing and UI inspection."""
        if self.vector_index is None:
            self._initialize_indices(force_reindex=True)
        retriever = self.vector_index.as_retriever(similarity_top_k=similarity_top_k)
        return retriever.retrieve(query_str)
