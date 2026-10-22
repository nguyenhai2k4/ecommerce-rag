import time
import pandas as pd
import streamlit as st

from src.agent.document_agent import SpecializedDocumentAgent
from src.engines.structured_engine import StructuredEngineManager
from src.engines.unstructured_engine import UnstructuredEngineManager
from src.vault.doc_loader import load_vault_documents
from src.evaluation.eval_suite import EVAL_TEST_CASES, run_agent_benchmark
from src.config import config


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
with tab_vault:
    st.markdown("### 🗄️ External Data & Knowledge Vault Explorer")
    st.markdown("Inspect both the underlying relational SQLite tables and the unstructured vectorized Markdown documents.")

    vault_sub1, vault_sub2 = st.tabs(["📊 Structured Tables (SQLite)", "📄 Unstructured Documents (Markdown)"])

    with vault_sub1:
        tbl_choice = st.selectbox("Select SQLite Table to Inspect", ["products", "customers", "orders", "inventory_logs"])
        rows = db_mgr.execute_raw_sql(f"SELECT * FROM {tbl_choice}")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.caption(f"Displaying {len(df)} records from table `{tbl_choice}`.")

    with vault_sub2:
        doc_files = list(config.unstructured_dir.glob("*.md"))
        doc_names = [f.name for f in doc_files]
        chosen_file = st.selectbox("Select Document from Vault", doc_names)

        if chosen_file:
            file_path = config.unstructured_dir / chosen_file
            with open(file_path, "r", encoding="utf-8") as f:
                doc_content = f.read()
            st.markdown(doc_content)


# =============================================================================
# TAB 3: Evaluation & Benchmarks
# =============================================================================
with tab_eval:
    st.markdown("### 🧪 Production Evaluation & Benchmark Suite")
    st.markdown(
        "Evaluate the agent's performance across **Structured SQL**, **Unstructured RAG**, "
        "and **Multi-Hop Cross-Domain Synthesis**."
    )

    if st.button("🚀 Run Live Benchmark Suite", type="primary"):
        with st.spinner("Executing test cases..."):
            results = run_agent_benchmark(st.session_state.agent)

            total_cases = len(results)
            passed_cases = sum(1 for r in results if r["passed"])
            avg_latency = round(sum(r["latency"] for r in results) / total_cases, 2)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Pass Rate", f"{round(passed_cases/total_cases*100, 1)}%", f"{passed_cases}/{total_cases} Passed")
            with col_b:
                st.metric("Avg Latency", f"{avg_latency}s")
            with col_c:
                st.metric("Total Test Cases", total_cases)

            st.markdown("#### Test Case Breakdown")
            for res in results:
                status_icon = "✅ PASS" if res["passed"] else "❌ FAIL"
                with st.expander(f"{status_icon} [{res['id']}] {res['category']} - {res['query'][:60]}..."):
                    st.markdown(f"**Query:** {res['query']}")
                    st.markdown(f"**Latency:** {res['latency']}s")
                    st.markdown(f"**Keywords Hit:** {', '.join(res['matched_keywords'])}")
                    st.markdown(f"**Response:**\n{res['response']}")
    else:
        st.info("Click the button above to run the automated benchmark suite against the agent.")
        st.markdown("#### Standard Test Suite Matrix")
        suite_df = pd.DataFrame([
            {"ID": tc.id, "Category": tc.category, "Description": tc.description, "Key Targets": ", ".join(tc.expected_keywords)}
            for tc in EVAL_TEST_CASES
        ])
        st.table(suite_df)


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
