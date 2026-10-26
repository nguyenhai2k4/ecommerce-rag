from pathlib import Path
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader, Document
from src.config import config


def _infer_document_title(doc: Document, file_name: str) -> str:
    """Extracts a clean title from markdown header or cleans the filename."""
    if doc.text:
        for line in doc.text.splitlines()[:5]:
            line = line.strip()
            if line.startswith("# "):
                cleaned = line.lstrip("#").strip()
                if cleaned:
                    return cleaned
    name_stem = Path(file_name).stem
    return name_stem.replace("_", " ").replace("-", " ").title()


def _infer_doc_type_and_domain(file_name: str, title: str, text: str) -> tuple[str, str]:
    """Infers document type and domain from filename, title, and initial text snippet."""
    combined = f"{file_name} {title} {text[:300]}".lower()
    if "return" in combined or "refund" in combined or "warranty" in combined:
        return "policy", "returns_and_warranty"
    elif "shipping" in combined or "sla" in combined or "delivery" in combined or "fulfillment" in combined:
        return "sla", "logistics_and_shipping"
    elif "specs" in combined or "specification" in combined or "troubleshoot" in combined or "manual" in combined:
        return "technical_specification", "hardware_specs_troubleshooting"
    elif "feedback" in combined or "review" in combined or "sentiment" in combined or "csat" in combined:
        return "customer_feedback", "sentiment_and_reviews"
    elif "contract" in combined or "agreement" in combined:
        return "contract", "legal_and_compliance"
    elif "policy" in combined or "guideline" in combined:
        return "policy", "operations_and_policies"
    else:
        return "general_knowledge", "operations"


def load_vault_documents(docs_dir: Optional[Path] = None) -> List[Document]:
    """
    Loads all markdown/text/pdf documents from the unstructured knowledge vault.
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
        fn_lower = file_name.lower()
        
        # Preserve canonical titles for core suite
        if "return" in fn_lower and "policy" in fn_lower:
            doc.metadata["doc_type"] = "policy"
            doc.metadata["domain"] = "returns_and_warranty"
            doc.metadata["title"] = "Return, Refund & Warranty Policy"
        elif "shipping" in fn_lower:
            doc.metadata["doc_type"] = "sla"
            doc.metadata["domain"] = "logistics_and_shipping"
            doc.metadata["title"] = "Shipping & Fulfillment SLA"
        elif "specs" in fn_lower:
            doc.metadata["doc_type"] = "technical_specification"
            doc.metadata["domain"] = "hardware_specs_troubleshooting"
            doc.metadata["title"] = "Product Specs & Troubleshooting Guide"
        elif "feedback" in fn_lower:
            doc.metadata["doc_type"] = "customer_feedback"
            doc.metadata["domain"] = "sentiment_and_reviews"
            doc.metadata["title"] = "Customer Feedback & Sentiment Report Q3"
        else:
            inferred_title = _infer_document_title(doc, file_name)
            doc_type, domain = _infer_doc_type_and_domain(file_name, inferred_title, doc.text or "")
            doc.metadata["doc_type"] = doc_type
            doc.metadata["domain"] = domain
            doc.metadata["title"] = inferred_title

    return documents


if __name__ == "__main__":
    docs = load_vault_documents()
    print(f"Loaded {len(docs)} documents from knowledge vault.")
    for d in docs:
        print(f" - {d.metadata.get('title')} ({d.metadata.get('file_name')})")
