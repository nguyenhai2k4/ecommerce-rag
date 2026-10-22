from pathlib import Path
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader, Document
from src.config import config


def load_vault_documents(docs_dir: Optional[Path] = None) -> List[Document]:
    """
    Loads all markdown/text documents from the unstructured knowledge vault.
    Enriches each Document with structured metadata for filtering and citation tracking.
    """
    target_dir = docs_dir or config.unstructured_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    reader = SimpleDirectoryReader(
        input_dir=str(target_dir),
        required_exts=[".md", ".txt", ".pdf"],
        recursive=True,
    )
    documents = reader.load_data()

    # Enrich metadata for each document
    for doc in documents:
        file_name = doc.metadata.get("file_name", "")
        if "return" in file_name.lower():
            doc.metadata["doc_type"] = "policy"
            doc.metadata["domain"] = "returns_and_warranty"
            doc.metadata["title"] = "Return, Refund & Warranty Policy"
        elif "shipping" in file_name.lower():
            doc.metadata["doc_type"] = "sla"
            doc.metadata["domain"] = "logistics_and_shipping"
            doc.metadata["title"] = "Shipping & Fulfillment SLA"
        elif "specs" in file_name.lower():
            doc.metadata["doc_type"] = "technical_specification"
            doc.metadata["domain"] = "hardware_specs_troubleshooting"
            doc.metadata["title"] = "Product Specs & Troubleshooting Guide"
        elif "feedback" in file_name.lower():
            doc.metadata["doc_type"] = "customer_feedback"
            doc.metadata["domain"] = "sentiment_and_reviews"
            doc.metadata["title"] = "Customer Feedback & Sentiment Report Q3"
        else:
            doc.metadata["doc_type"] = "general_knowledge"
            doc.metadata["domain"] = "operations"
            doc.metadata["title"] = file_name

    return documents


if __name__ == "__main__":
    docs = load_vault_documents()
    print(f"Loaded {len(docs)} documents from knowledge vault.")
    for d in docs:
        print(f" - {d.metadata.get('title')} ({d.metadata.get('file_name')})")
