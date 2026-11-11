import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from llama_index.core import Document, SimpleDirectoryReader
from src.config import config

# In-memory cache keyed by directory path -> (dir_signature, List[Document])
_VAULT_DOCS_CACHE: Dict[str, Tuple[Tuple[Tuple[str, int, int], ...], List[Document]]] = {}

# Per-file in-memory cache keyed by resolved file path -> (mtime_ns, size_bytes, List[Document])
_FILE_DOCS_CACHE: Dict[str, Tuple[int, int, List[Document]]] = {}
_DISK_CACHE_LOADED: bool = False

# In-memory manifest cache keyed by directory path -> (dir_signature, List[Dict[str, str]])
_MANIFEST_CACHE: Dict[str, Tuple[Tuple[Tuple[str, int, int], ...], List[Dict[str, str]]]] = {}


def _get_disk_cache_path() -> Path:
    return config.storage_dir / ".vault_docs_cache.json"


def _load_disk_cache_once() -> None:
    """Loads persisted per-file parsed document cache from disk on first call."""
    global _DISK_CACHE_LOADED
    if _DISK_CACHE_LOADED:
        return
    _DISK_CACHE_LOADED = True
    cache_path = _get_disk_cache_path()
    if not cache_path.exists():
        return
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, dict):
            return
        for path_key, entry in raw_data.items():
            if not isinstance(entry, dict):
                continue
            mtime_ns = int(entry.get("mtime_ns", 0))
            size_b = int(entry.get("size", 0))
            raw_docs = entry.get("docs", [])
            docs = [
                Document(text=d.get("text", ""), metadata=dict(d.get("metadata", {})))
                for d in raw_docs
                if isinstance(d, dict)
            ]
            if docs:
                _FILE_DOCS_CACHE[path_key] = (mtime_ns, size_b, docs)
    except Exception:
        pass


def _save_disk_cache() -> None:
    """Persists per-file parsed documents so subsequent processes skip expensive PDF parsing."""
    cache_path = _get_disk_cache_path()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {}
        for path_key, (mtime_ns, size_b, docs) in _FILE_DOCS_CACHE.items():
            payload[path_key] = {
                "mtime_ns": mtime_ns,
                "size": size_b,
                "docs": [{"text": d.text or "", "metadata": dict(d.metadata or {})} for d in docs],
            }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass


def _compute_dir_signature(target_dir: Path) -> Tuple[Tuple[str, int, int], ...]:
    """Computes a lightweight fingerprint of all supported files in the directory."""
    entries = []
    if target_dir.exists():
        for ext in ("*.md", "*.txt", "*.pdf"):
            for p in target_dir.rglob(ext):
                try:
                    st = p.stat()
                    entries.append((str(p.resolve()), st.st_mtime_ns, st.st_size))
                except OSError:
                    continue
    entries.sort(key=lambda x: x[0])
    return tuple(entries)


def _infer_document_title_from_text(text: str, file_name: str) -> str:
    """Extracts a clean title from markdown header or cleans the filename."""
    if text:
        for line in text.splitlines()[:5]:
            line = line.strip()
            if line.startswith("# "):
                cleaned = line.lstrip("#").strip()
                if cleaned:
                    return cleaned
    name_stem = Path(file_name).stem
    return name_stem.replace("_", " ").replace("-", " ").title()


def _infer_document_title(doc: Document, file_name: str) -> str:
    """Extracts a clean title from a Document instance."""
    return _infer_document_title_from_text(doc.text or "", file_name)


def _infer_doc_type_and_domain(file_name: str, title: str, text: str) -> Tuple[str, str]:
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


def _resolve_canonical_metadata(file_name: str, raw_text: str) -> Dict[str, str]:
    """Computes enriched metadata (file_name, title, doc_type, domain) for a document."""
    fn_lower = file_name.lower()
    if "return" in fn_lower and "policy" in fn_lower:
        return {
            "file_name": file_name,
            "doc_type": "policy",
            "domain": "returns_and_warranty",
            "title": "Return, Refund & Warranty Policy",
        }
    elif "shipping" in fn_lower:
        return {
            "file_name": file_name,
            "doc_type": "sla",
            "domain": "logistics_and_shipping",
            "title": "Shipping & Fulfillment SLA",
        }
    elif "specs" in fn_lower:
        return {
            "file_name": file_name,
            "doc_type": "technical_specification",
            "domain": "hardware_specs_troubleshooting",
            "title": "Product Specs & Troubleshooting Guide",
        }
    elif "feedback" in fn_lower:
        return {
            "file_name": file_name,
            "doc_type": "customer_feedback",
            "domain": "sentiment_and_reviews",
            "title": "Customer Feedback & Sentiment Report Q3",
        }
    else:
        inferred_title = _infer_document_title_from_text(raw_text, file_name)
        doc_type, domain = _infer_doc_type_and_domain(file_name, inferred_title, raw_text)
        return {
            "file_name": file_name,
            "doc_type": doc_type,
            "domain": domain,
            "title": inferred_title,
        }


def _enrich_document_metadata(doc: Document) -> None:
    """Enriches a Document's metadata in-place."""
    file_name = doc.metadata.get("file_name", "")
    meta = _resolve_canonical_metadata(file_name, doc.text or "")
    doc.metadata["doc_type"] = meta["doc_type"]
    doc.metadata["domain"] = meta["domain"]
    doc.metadata["title"] = meta["title"]


def get_vault_manifest_entries(docs_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    Lightweight, sub-millisecond extraction of document metadata (file_name, title,
    doc_type, domain) across the unstructured vault without parsing full PDF bodies.
    """
    target_dir = Path(docs_dir or config.unstructured_dir).resolve()
    if not target_dir.exists():
        return []

    dir_key = str(target_dir)
    current_sig = _compute_dir_signature(target_dir)
    if not current_sig:
        _MANIFEST_CACHE[dir_key] = (current_sig, [])
        return []

    if dir_key in _MANIFEST_CACHE:
        cached_sig, cached_entries = _MANIFEST_CACHE[dir_key]
        if cached_sig == current_sig:
            return list(cached_entries)

    entries: List[Dict[str, str]] = []
    seen_names = set()
    for path_str, _, _ in current_sig:
        p = Path(path_str)
        fn = p.name
        if fn in seen_names:
            continue
        seen_names.add(fn)

        head_text = ""
        if p.suffix.lower() in (".md", ".txt"):
            try:
                with open(p, "r", encoding="utf-8", errors="replace") as f:
                    head_text = f.read(512)
            except OSError:
                head_text = ""

        entries.append(_resolve_canonical_metadata(fn, head_text))

    _MANIFEST_CACHE[dir_key] = (current_sig, entries)
    return list(entries)


def load_single_vault_document(file_path: Path) -> List[Document]:
    """Loads and enriches a single file from the vault, updating the per-file cache."""
    _load_disk_cache_once()
    resolved = file_path.resolve()
    if not resolved.exists():
        return []

    path_key = str(resolved)
    try:
        st = resolved.stat()
        mtime_ns, size_b = st.st_mtime_ns, st.st_size
    except OSError:
        return []

    cached = _FILE_DOCS_CACHE.get(path_key)
    if cached and cached[0] == mtime_ns and cached[1] == size_b:
        return list(cached[2])

    reader = SimpleDirectoryReader(input_files=[path_key])
    docs = reader.load_data()
    for doc in docs:
        doc.metadata.setdefault("file_name", resolved.name)
        _enrich_document_metadata(doc)

    _FILE_DOCS_CACHE[path_key] = (mtime_ns, size_b, docs)
    _save_disk_cache()
    return list(docs)


def evict_vault_document_cache(file_name: str, docs_dir: Optional[Path] = None) -> None:
    """Evicts a deleted file from the in-memory and disk caches without re-reading remaining files."""
    target_dir = Path(docs_dir or config.unstructured_dir).resolve()
    path_key = str((target_dir / file_name).resolve())
    _FILE_DOCS_CACHE.pop(path_key, None)
    _VAULT_DOCS_CACHE.pop(str(target_dir), None)
    _MANIFEST_CACHE.pop(str(target_dir), None)
    _save_disk_cache()


def load_vault_documents(docs_dir: Optional[Path] = None, force_reload: bool = False) -> List[Document]:
    """
    Loads all markdown/text/pdf documents from the unstructured knowledge vault.
    Enriches each Document with structured metadata for filtering and citation tracking.
    Uses per-file mtime+size caching so unchanged large PDFs are not re-parsed unnecessarily.
    """
    _load_disk_cache_once()
    target_dir = Path(docs_dir or config.unstructured_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    dir_key = str(target_dir)
    current_sig = _compute_dir_signature(target_dir)
    if not current_sig:
        _VAULT_DOCS_CACHE[dir_key] = (current_sig, [])
        return []

    if not force_reload and dir_key in _VAULT_DOCS_CACHE:
        cached_sig, cached_docs = _VAULT_DOCS_CACHE[dir_key]
        if cached_sig == current_sig:
            return list(cached_docs)

    documents: List[Document] = []
    cache_dirty = False
    active_path_keys = set()

    for path_str, mtime_ns, size_b in current_sig:
        active_path_keys.add(path_str)
        cached_file = _FILE_DOCS_CACHE.get(path_str)
        if cached_file and cached_file[0] == mtime_ns and cached_file[1] == size_b:
            documents.extend(cached_file[2])
        else:
            reader = SimpleDirectoryReader(input_files=[path_str])
            file_docs = reader.load_data()
            file_name = Path(path_str).name
            for doc in file_docs:
                doc.metadata.setdefault("file_name", file_name)
                _enrich_document_metadata(doc)
            _FILE_DOCS_CACHE[path_str] = (mtime_ns, size_b, file_docs)
            documents.extend(file_docs)
            cache_dirty = True

    # Prune deleted files from per-file cache if needed
    stale_keys = [
        k for k in list(_FILE_DOCS_CACHE.keys())
        if k.startswith(dir_key) and k not in active_path_keys
    ]
    for k in stale_keys:
        _FILE_DOCS_CACHE.pop(k, None)
        cache_dirty = True

    if cache_dirty:
        _save_disk_cache()

    _VAULT_DOCS_CACHE[dir_key] = (current_sig, documents)
    return list(documents)


if __name__ == "__main__":
    docs = load_vault_documents()
    print(f"Loaded {len(docs)} documents from knowledge vault.")
    for d in docs:
        print(f" - {d.metadata.get('title')} ({d.metadata.get('file_name')})")
