import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

from src.agent.document_agent import SpecializedDocumentAgent
from src.engines.structured_engine import StructuredEngineManager
from src.engines.unstructured_engine import UnstructuredEngineManager
from src.vault.doc_loader import load_vault_documents
from src.evaluation.eval_suite import (
    SemanticEvaluationSuite,
    SEMANTIC_BENCHMARK_CASES,
    run_agent_benchmark,
)
from src.config import config


def format_file_size(size_bytes: int) -> str:
    """Formats file size into human-readable B, KB, or MB."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def get_mime_type(ext: str) -> str:
    """Returns appropriate MIME type for downloads based on file extension."""
    ext_lower = ext.lower().strip()
    if ext_lower == ".md":
        return "text/markdown"
    elif ext_lower == ".txt":
        return "text/plain"
    elif ext_lower == ".pdf":
        return "application/pdf"
    return "application/octet-stream"


def extract_pdf_preview(pdf_path: Path) -> str:
    """Extracts text from a PDF file for UI preview."""
    try:
        from llama_index.core import SimpleDirectoryReader
        docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
        preview_parts = []
        for i, d in enumerate(docs, 1):
            preview_parts.append(f"--- Page {i} ---\n{d.text.strip()}")
        return "\n\n".join(preview_parts) if preview_parts else "(No text extracted from PDF)"
    except Exception as e:
        return f"Error extracting PDF preview: {e}"


# -----------------------------------------------------------------------------
# Page Configuration & Modern Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nexus Document Agent | AI Engineering Portfolio",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Modern Glassmorphic Dark Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        color: #f8fafc;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-right: 6px;
    }
    .source-box {
        background-color: #0f172a;
        border-left: 3px solid #38bdf8;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.88rem;
        margin-top: 10px;
        color: #cbd5e1;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am **NexusDoc-Agent**, an enterprise AI Document & Data Agent.\n\n"
                "I can query and synthesize information across:\n"
                "- 📊 **Structured SQLite Database**: Products, customer loyalty tiers, order history, inventory logs.\n"
                "- 📚 **Unstructured Knowledge Vault**: Return & warranty policies, shipping SLAs, technical specs, and customer reviews.\n\n"
                "Try asking a complex multi-hop question, or click one of the quick prompts in the sidebar!"
            ),
            "sources": [],
            "latency": 0.0,
        }
    ]

if "api_key" not in st.session_state:
    st.session_state.api_key = config.gemini_api_key

if "agent" not in st.session_state:
    st.session_state.agent = SpecializedDocumentAgent(
        api_key=st.session_state.api_key,
        verbose=False,
    )


# -----------------------------------------------------------------------------
# Sidebar: Configurations, Vault Stats & Quick Demos
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 Agent Control Center")

    # API Key Configuration
    user_key = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.api_key,
        type="password",
        help="Paste your Google AI Studio Gemini API Key here (or set in .env).",
    )
    if user_key != st.session_state.api_key:
        st.session_state.api_key = user_key
        st.session_state.agent = SpecializedDocumentAgent(api_key=user_key, verbose=False)
        st.success("API Key updated!")

    # Model Info
    st.markdown(
        f"""
        <div style="margin-bottom: 12px;">
            <span class="badge-pill">LLM: {config.llm_model}</span>
            <span class="badge-pill">Embed: {config.embedding_model}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Vault Metrics
    st.markdown("#### 📦 Knowledge Vault Health")
    db_mgr = StructuredEngineManager()
    counts = db_mgr.get_table_summary()
    docs = load_vault_documents()

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Products", counts.get("products", 0))
        st.metric("Orders", counts.get("orders", 0))
    with c2:
        st.metric("Customers", counts.get("customers", 0))
        st.metric("Doc Files", len(docs))

    st.divider()

    # Quick Demo Prompts
    st.markdown("#### ⚡ Quick Demo Scenarios")
    demo_queries = [
        ("🔍 Alice Chen Order Return", "Customer Alice Chen (Customer ID 101) wants to return order #5001. She received it on 2026-09-10. Is she eligible for a full refund today?"),
        ("📊 Low Stock & Reviews", "Which products currently have low stock (< 20 units) in the database, and what are customers complaining about in their reviews?"),
        ("📜 Policy Comparison", "Compare the return windows, restocking fees, and shipping costs for Standard tier vs VIP tier customers."),
        ("🎧 Audio Specs & Reset", "What are the Bluetooth specifications for AeroSound Pro headphones, and how do I perform a factory reset?"),
    ]

    for label, query_text in demo_queries:
        if st.button(label, use_container_width=True):
            st.session_state.preset_query = query_text

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent.reset_conversation()
        st.rerun()


# -----------------------------------------------------------------------------
# Main Application Tabs
# -----------------------------------------------------------------------------
tab_chat, tab_vault, tab_eval, tab_arch = st.tabs([
    "💬 Agent Chat & Synthesis",
    "🗄️ Knowledge Vault Explorer",
    "🧪 Evaluation & Benchmark",
    "🏛️ System Architecture",
])


# =============================================================================
# TAB 1: Agent Chat & Reasoning
# =============================================================================
with tab_chat:
    st.markdown(
        """
        ### 🧠 Specialized Document Agent
        *Autonomous multi-hop retrieval and synthesis across structured database tables and unstructured document vaults.*
        """
    )

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                sources_str = ", ".join([f"`{s}`" for s in msg["sources"]])
                st.markdown(f"<div class='source-box'>📚 <b>Grounded Sources:</b> {sources_str}</div>", unsafe_allow_html=True)
            if msg.get("latency", 0) > 0:
                st.caption(f"⏱️ Response latency: {msg['latency']}s")

    # Chat Input
    preset = st.session_state.pop("preset_query", None)
    user_prompt = st.chat_input("Ask a question across products, orders, return policies, or specs...")

    active_prompt = preset or user_prompt

    if active_prompt:
        # Add user query to chat
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        # Execute agent
        with st.chat_message("assistant"):
            with st.spinner("Agent analyzing query, running Text-to-SQL & semantic search..."):
                start_time = time.time()
                response = st.session_state.agent.chat(active_prompt)
                latency = round(time.time() - start_time, 2)

                st.markdown(response.response_text)

                if response.sources:
                    sources_str = ", ".join([f"`{s}`" for s in response.sources])
                    st.markdown(f"<div class='source-box'>📚 <b>Grounded Sources:</b> {sources_str}</div>", unsafe_allow_html=True)

                st.caption(f"⏱️ Response latency: {latency}s")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.response_text,
                    "sources": response.sources,
                    "latency": latency,
                })


# =============================================================================
# TAB 2: Knowledge Vault Explorer
# =============================================================================
# =============================================================================
# TAB 2: Knowledge Vault & Document Management Studio
# =============================================================================
with tab_vault:
    st.markdown("### 🗄️ Knowledge Vault & Document Management Studio")
    st.markdown(
        "Manage unstructured knowledge documents (upload, preview, download, delete), "
        "inspect live relational tables, and observe dynamic system prompts, schema, and reasoning relationships."
    )

    vault_sub1, vault_sub2 = st.tabs([
        "📄 Document Management (Upload, Preview, Download, Delete)",
        "📊 Structured Database (SQLite)",
    ])

    # -------------------------------------------------------------------------
    # SUBTAB 1: Document Manager
    # -------------------------------------------------------------------------
    with vault_sub1:
        st.markdown("#### 📄 Document Vault Management")
        st.caption("Upload company policies, SLAs, specifications, and customer sentiment reports (.md, .txt, .pdf).")

        # Top Vault Metrics & Sync Action
        vault_files = st.session_state.agent.unstructured_manager.get_vault_file_list()
        total_size_bytes = sum(f["size_bytes"] for f in vault_files)
        total_size_formatted = format_file_size(total_size_bytes)

        m_col1, m_col2, m_col3, m_col4 = st.columns([2, 2, 3, 3])
        with m_col1:
            st.metric("Total Documents", len(vault_files))
        with m_col2:
            st.metric("Vault Storage", total_size_formatted)
        with m_col3:
            st.caption(f"📂 **Vault Path:** `{config.unstructured_dir}`")
            st.caption("🏷️ **Supported Formats:** `.md`, `.txt`, `.pdf`")
        with m_col4:
            if st.button("🔄 Sync & Re-index Vault", use_container_width=True, help="Re-embed and rebuild all vector and summary indices"):
                with st.spinner("Re-indexing knowledge vault documents and updating dynamic agent prompt..."):
                    st.session_state.agent.reload_vault(force_reindex=True)
                    st.success("Knowledge Vault re-indexed successfully!")
                    time.sleep(0.5)
                    st.rerun()

        st.divider()

        # Document Upload Section
        st.markdown("##### 📤 Upload New Documents to Vault")
        uploaded_files = st.file_uploader(
            "Choose markdown, text, or PDF files to ingest",
            type=["md", "txt", "pdf"],
            accept_multiple_files=True,
            help="Files will be automatically saved, parsed, chunked, and vectorized into the knowledge vault.",
            key="vault_file_uploader",
        )

        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) ready for ingestion.")
            if st.button("🚀 Ingest & Index Uploaded Documents", type="primary"):
                with st.spinner("Ingesting files, extracting content, and re-indexing vector store..."):
                    for up_file in uploaded_files:
                        content_bytes = up_file.read()
                        st.session_state.agent.unstructured_manager.add_document(
                            file_name=up_file.name,
                            content=content_bytes,
                        )
                    st.session_state.agent.reload_vault(force_reindex=True)
                    st.success(f"✅ Successfully ingested {len(uploaded_files)} document(s) and refreshed agent prompt!")
                    time.sleep(0.7)
                    st.rerun()

        st.divider()

        # Document Inventory & Management Workspace
        st.markdown("##### 📚 Active Vault Documents")
        if not vault_files:
            st.warning("No documents currently found in the knowledge vault directory. Upload some documents above!")
        else:
            # Summary Table
            docs_summary_data = []
            for f in vault_files:
                mod_str = datetime.fromtimestamp(f["modified_time"]).strftime("%Y-%m-%d %H:%M:%S")
                docs_summary_data.append({
                    "Document Name": f["name"],
                    "Format": f["extension"].upper().replace(".", ""),
                    "Size": format_file_size(f["size_bytes"]),
                    "Last Modified": mod_str,
                })
            st.dataframe(pd.DataFrame(docs_summary_data), use_container_width=True)

            # Selected Document Actions (Preview, Download, Delete)
            doc_names = [f["name"] for f in vault_files]
            selected_doc_name = st.selectbox("Select a Document to Preview, Download, or Delete", doc_names)

            if selected_doc_name:
                selected_meta = next(f for f in vault_files if f["name"] == selected_doc_name)
                selected_path = selected_meta["path"]
                ext = selected_meta["extension"]

                # Action Bar
                act_col1, act_col2 = st.columns([3, 3])
                with act_col1:
                    with open(selected_path, "rb") as bf:
                        file_bytes = bf.read()
                    mime_type = get_mime_type(ext)
                    st.download_button(
                        label=f"📥 Download `{selected_doc_name}`",
                        data=file_bytes,
                        file_name=selected_doc_name,
                        mime=mime_type,
                        use_container_width=True,
                    )
                with act_col2:
                    with st.popover(f"🗑️ Delete `{selected_doc_name}`", use_container_width=True):
                        st.warning(f"Are you sure you want to delete `{selected_doc_name}`?")
                        st.caption("This will permanently remove the file from disk and re-index the knowledge vault vector store.")
                        if st.button("⚠️ Confirm Delete", key=f"confirm_delete_{selected_doc_name}", type="primary"):
                            st.session_state.agent.unstructured_manager.delete_document(selected_doc_name)
                            st.session_state.agent.reload_vault(force_reindex=True)
                            st.success(f"Deleted `{selected_doc_name}` and rebuilt knowledge vault!")
                            time.sleep(0.6)
                            st.rerun()

                # Document Content Preview
                st.markdown("###### 👁️ Document Content Preview")
                if ext == ".md":
                    view_mode = st.radio("View Format", ["Rendered Markdown", "Raw Source Code"], horizontal=True, key=f"view_{selected_doc_name}")
                    try:
                        with open(selected_path, "r", encoding="utf-8") as rf:
                            md_text = rf.read()
                        if view_mode == "Rendered Markdown":
                            st.markdown(md_text)
                        else:
                            st.code(md_text, language="markdown")
                    except Exception as e:
                        st.error(f"Error reading markdown file: {e}")

                elif ext == ".txt":
                    try:
                        with open(selected_path, "r", encoding="utf-8") as rf:
                            txt_content = rf.read()
                        st.code(txt_content, language="text")
                    except Exception as e:
                        st.error(f"Error reading text file: {e}")

                elif ext == ".pdf":
                    st.caption("📕 PDF Document Text Extraction Preview:")
                    pdf_text = extract_pdf_preview(selected_path)
                    st.text_area("Extracted PDF Content", pdf_text, height=400)

    # -------------------------------------------------------------------------
    # SUBTAB 2: Structured Tables (SQLite)
    # -------------------------------------------------------------------------
    with vault_sub2:
        st.markdown("#### 📊 Relational Database Tables (SQLite)")
        st.caption("Direct read-only inspection of structured retail database records.")

        table_names = db_mgr.get_table_names()
        tbl_choice = st.selectbox("Select SQLite Table to Inspect", table_names)

        if tbl_choice:
            rows = db_mgr.execute_raw_sql(f"SELECT * FROM {tbl_choice}")
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.caption(f"Displaying {len(df)} records from table `{tbl_choice}`.")




# =============================================================================
# TAB 3: Advanced Semantic Evaluation & Benchmarks
# =============================================================================
with tab_eval:
    st.markdown("### 🧪 Production-Grade Semantic Search Evaluation Suite")
    st.markdown(
        "Evaluate the agent beyond naive substring matching. Measures **Retriever Semantic Robustness** "
        "(Hit Rate@k, MRR, Paraphrase Penalties) and **Agent Synthesis Quality** "
        "(Embedding Cosine Similarity, Concept Coverage vs. Lexical False Negatives)."
    )

    eval_tab1, eval_tab2, eval_tab3 = st.tabs([
        "🚀 Live Dual-Stage Benchmark",
        "📋 Paraphrase Test Matrix",
        "💡 Retrieval Pitfalls & Mitigations",
    ])

    with eval_tab1:
        st.markdown("#### Execute Dual-Stage Semantic Evaluation")
        st.caption("Runs both Retriever-level robustness testing and Agent-level semantic synthesis scoring.")

        if st.button("▶️ Run Full Semantic Benchmark", type="primary"):
            with st.spinner("Running semantic retriever evaluations and agent synthesis across paraphrased queries..."):
                suite = SemanticEvaluationSuite(agent=st.session_state.agent)
                benchmark_data = suite.run_full_suite()

                retriever_res = benchmark_data["retriever_benchmark"]
                synthesis_res = benchmark_data["synthesis_benchmark"]

                # Metrics Overview
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    canon_hit = sum(1 for r in retriever_res if r["canonical_hit"]) / len(retriever_res)
                    st.metric("Canonical Hit Rate", f"{canon_hit*100:.1f}%")
                with c2:
                    para_hit = sum(1 for r in retriever_res if r["paraphrase_hit"]) / len(retriever_res)
                    st.metric("Paraphrase Hit Rate", f"{para_hit*100:.1f}%")
                with c3:
                    avg_penalty = sum(r["paraphrase_penalty"] for r in retriever_res) / len(retriever_res)
                    st.metric("Mean Paraphrase Penalty", f"{avg_penalty:+.4f}", help="Score drop when queries use slang/synonyms")
                with c4:
                    semantic_pass = sum(1 for r in synthesis_res if r["semantic_passed"]) / len(synthesis_res)
                    st.metric("Semantic Cosine Pass", f"{semantic_pass*100:.1f}%")

                st.divider()

                # Stage 1: Retriever Level Robustness
                st.markdown("#### 1️⃣ Stage 1: Dense Retriever Robustness (Hit Rate & Paraphrase Penalty)")
                retriever_df = pd.DataFrame([
                    {
                        "Case ID": r["id"],
                        "Target Concept": r["concept_name"],
                        "Canonical Hit": "✅ Yes (Rank " + str(r["canonical_rank"]) + ")" if r["canonical_hit"] else "❌ No",
                        "Paraphrase Hit": "✅ Yes (Rank " + str(r["paraphrase_rank"]) + ")" if r["paraphrase_hit"] else "❌ No",
                        "Canon Score": f"{r['canonical_score']:.4f}",
                        "Para Score": f"{r['paraphrase_score']:.4f}",
                        "Paraphrase Penalty (Δ)": f"{r['paraphrase_penalty']:+.4f}",
                        "Verdict": r["robustness"],
                    }
                    for r in retriever_res
                ])
                st.dataframe(retriever_df, use_container_width=True)

                st.divider()

                # Stage 2: Agent Synthesis Level
                st.markdown("#### 2️⃣ Stage 2: Agent Synthesis (Semantic Embedding Similarity vs. Naive Lexical)")
                synthesis_df = pd.DataFrame([
                    {
                        "Case ID": s["id"],
                        "Category": s["category"],
                        "Naive Lexical Hit": f"{s['naive_score']*100:.0f}% ({'PASS' if s['naive_passed'] else 'FAIL'})",
                        "Embedding Cosine Similarity": f"{s['cosine_score']:.3f} ({'PASS' if s['semantic_passed'] else 'FAIL'})",
                        "Concept Coverage": f"{s['concept_coverage']*100:.0f}%",
                        "Latency": f"{s['latency']}s",
                        "Outcome": s["outcome"],
                    }
                    for s in synthesis_res
                ])
                st.dataframe(synthesis_df, use_container_width=True)

                st.markdown("#### Detailed Case Inspection")
                for s in synthesis_res:
                    status_icon = "🟢" if s["semantic_passed"] else "🔴"
                    with st.expander(f"{status_icon} [{s['id']}] {s['category']} | Cosine Similarity: {s['cosine_score']:.3f}"):
                        st.markdown(f"**Query Evaluated (Paraphrased):** *\"{s['query']}\"*")
                        st.markdown(f"**Golden Reference Answer:**\n> {s['reference_answer']}")
                        st.markdown(f"**Agent Generated Response:**\n{s['response']}")
                        st.markdown(f"**Concepts Captured:** {len(s['concepts_hit'])} propositions verified")
        else:
            st.info("Click 'Run Full Semantic Benchmark' above to evaluate retriever paraphrase penalties and agent embedding similarity.")

    with eval_tab2:
        st.markdown("#### Canonical vs. Paraphrased Test Pairs Matrix")
        st.caption("Each test pair compares formal documentation phrasing against real-world colloquial slang and synonyms.")
        matrix_data = [
            {
                "ID": tc.id,
                "Concept": tc.concept_name,
                "Canonical Phrasing": tc.canonical_query,
                "Colloquial / Paraphrased Query": tc.paraphrased_query,
                "Target Document": tc.target_doc_keyword,
                "Key Semantic Concepts": "; ".join(tc.core_concepts),
            }
            for tc in SEMANTIC_BENCHMARK_CASES
        ]
        st.dataframe(pd.DataFrame(matrix_data), use_container_width=True)

    with eval_tab3:
        st.markdown("### 🧠 Semantic Retrieval Failure Modes & Production Mitigations")
        st.markdown(
            """
            #### 1. Why Naive Exact / Keyword Matching Fails in Semantic Evaluation
            - **Vocabulary Mismatch:** Users ask for *"premier loyalty patrons"* while documentation specifies *"VIP Tier Customers"*. A substring check for `"VIP"` produces a **False Negative** if the LLM adopts the user's synonym.
            - **Lexical Overfitting:** Benchmarks relying on exact words mask weak embeddings because any keyword search (BM25) would retrieve them anyway.
            
            #### 2. The Conceptual Retrieval Penalty (Semantic Dispersion)
            - Dense embeddings project queries into a shared vector space. When a query is colloquial, symptom-based, or long, its cosine similarity to the document chunk degrades (e.g., in our benchmark, **SEM-04 suffered a +0.1283 paraphrase penalty**).
            
            #### 3. Enterprise AI Engineering Solutions:
            1. **Embedding Cosine Evaluation (`SemanticSimilarityEvaluator`):**
               - Evaluates generated responses against a golden reference answer using vector cosine similarity. Invariance to rewordings, active vs passive voice, and synonyms.
            2. **HyDE (Hypothetical Document Embeddings):**
               - Uses an LLM to generate a hypothetical answer *first*, and embeds that hypothetical document. Aligns user query semantics with documentation format.
               - Available in this repository via `unstructured_manager.get_hyde_query_engine()`.
            3. **Hybrid Dense + Sparse Search (BM25 + Vector):**
               - Preserves exact model code / SKU precision while retaining semantic synonym reach.
            4. **Cross-Encoder Re-Ranking:**
               - Re-ranks candidate chunks using full self-attention across `(query, document)` pairs, removing false-positive semantic drift.
            """
        )


# =============================================================================
# TAB 4: System Architecture & AI Engineering Highlights
# =============================================================================
with tab_arch:
    st.markdown("### 🏛️ System Architecture & Engineering Design Decisions")

    st.markdown(
        """
        ```mermaid
        graph TD
            User([User Query]) --> Agent[Specialized Document Agent / AgentWorkflow]
            
            subgraph Agent Orchestration Layer
                Agent --> Router{Multi-Hop Reasoning Loop}
                Router --> ToolSQL[QueryEngineTool: retail_sql_analytics]
                Router --> ToolVector[QueryEngineTool: knowledge_vault_search]
                Router --> ToolSummary[QueryEngineTool: document_summary_tool]
                Router --> ToolDate[FunctionTool: date_calculator]
            end
            
            subgraph Data & Knowledge Vault
                ToolSQL --> SQLDB[(SQLite Retail Database)]
                ToolVector --> VectorIndex[VectorStoreIndex / Top-K Embeddings]
                ToolSummary --> SummaryIndex[SummaryIndex / Whole-Doc Tree]
            end
            
            subgraph Synthesis & Grounding
                SQLDB --> Facts[Structured Facts: Orders, Tiers, Dates]
                VectorIndex --> Policies[Unstructured Text: Return Rules, SLAs, Specs]
                Facts --> Synthesizer[Grounded Synthesizer & Citation Engine]
                Policies --> Synthesizer
                Synthesizer --> Response([Final Grounded Response])
            end
        ```
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ---
        ### 🔑 Key AI Engineering Highlights for Interviewers:
        1. **Hybrid Data Ingestion**:
           - Structured data stored in normalized SQLite tables queried via deterministic SQLAlchemy / `NLSQLTableQueryEngine`.
           - Unstructured documents chunked with `SentenceSplitter(chunk_size=512, chunk_overlap=64)` and indexed with Google Gemini `text-embedding-004`.
        2. **Multi-Hop Cross-Domain Reasoning**:
           - Resolves complex business logic that cannot be answered by SQL or RAG alone (e.g. looking up a customer's loyalty tier from SQL, looking up the tier's return window in policy docs, and calculating eligibility).
        3. **Deterministic Tool Augmentation**:
           - Custom `date_calculator` tool eliminates LLM arithmetic hallucinations when determining elapsed days.
        4. **Production Security & Grounding**:
           - SQL execution enforces read-only operations (disallowing `DROP`, `DELETE`, `UPDATE`).
           - Explicit citation enforcement in agent prompts prevents ungrounded hallucinations.
        """
    )
