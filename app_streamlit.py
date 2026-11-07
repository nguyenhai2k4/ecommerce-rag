import os
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import streamlit as st

from src.engines.model_factory import (
    check_ollama_status,
    get_active_model_metadata,
    get_available_models_for_provider,
    initialize_models,
)
from src.agent.chat_history import ChatHistoryManager, auto_generate_title
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


def format_relative_time(timestamp_str: str) -> str:
    """Formats an ISO timestamp into a compact relative time tag like 2h, 14h, 3d, 1w."""
    if not timestamp_str:
        return ""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        now = datetime.now()
        diff = now - dt
        seconds = int(diff.total_seconds())
        if seconds < 0 or seconds < 60:
            return "now"
        elif seconds < 3600:
            return f"{seconds // 60}m"
        elif seconds < 86400:
            return f"{seconds // 3600}h"
        elif seconds < 86400 * 7:
            return f"{seconds // 86400}d"
        elif seconds < 86400 * 30:
            return f"{seconds // (86400 * 7)}w"
        else:
            return dt.strftime("%b %d")
    except Exception:
        return ""


# -----------------------------------------------------------------------------
# Page Configuration & Dark Antigravity Aesthetic
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Nexus Document Agent | AI Engineering Portfolio",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Global Background & Typography */
    .stApp {
        background-color: #080b11 !important;
        color: #f8fafc !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Left Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 14px !important;
    }

    /* Top Left Hamburger & Logo Control (When Sidebar is Collapsed) */
    [data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        align-items: center !important;
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        padding: 6px 12px !important;
        top: 14px !important;
        left: 16px !important;
        z-index: 999999 !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
        backdrop-filter: blur(10px) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebarCollapsedControl"]:hover {
        background: rgba(30, 41, 59, 0.95) !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
        transform: scale(1.02);
    }

    [data-testid="stSidebarCollapsedControl"]::before {
        content: "▲";
        font-size: 15px;
        font-weight: 800;
        color: #38bdf8;
        margin-right: 8px;
    }

    .top-logo-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 8px;
        background: linear-gradient(135deg, #38bdf8, #6366f1);
        color: #ffffff;
        font-weight: 800;
        font-size: 15px;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.3);
    }

    /* Hero Center Section */
    .hero-container {
        text-align: center;
        padding: 40px 20px 20px 20px;
        max-width: 820px;
        margin: 0 auto;
    }

    .hero-logo-box {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 72px;
        height: 72px;
        border-radius: 22px;
        background: radial-gradient(circle at 35% 35%, #38bdf8, #1e1b4b);
        box-shadow: 0 0 55px rgba(56, 189, 248, 0.35);
        color: #ffffff;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 20px;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 600;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.96rem;
        line-height: 1.5;
        max-width: 640px;
        margin: 0 auto 28px auto;
    }

    /* Badges & Source Chips */
    .badge-pill {
        display: inline-block;
        padding: 4px 11px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.25);
        margin-right: 6px;
    }

    .source-box {
        background-color: #0b1120;
        border-left: 3px solid #38bdf8;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.88rem;
        margin-top: 10px;
        color: #cbd5e1;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Scenario Card */
    .scenario-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 18px;
        transition: all 0.2s ease;
        margin-bottom: 10px;
    }
    .scenario-card:hover {
        background: rgba(30, 41, 59, 0.8);
        border-color: rgba(56, 189, 248, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Session State Initialization & Chat History Persistence
# -----------------------------------------------------------------------------
if "chat_manager" not in st.session_state:
    st.session_state.chat_manager = ChatHistoryManager()


def get_user_initialized_sessions() -> list:
    """Helper to retrieve saved conversations that have user messages, resilient to cached instances."""
    mgr = st.session_state.chat_manager
    try:
        return mgr.list_sessions(require_user_messages=True)
    except TypeError:
        all_s = mgr.list_sessions()
        return [s for s in all_s if s.get("has_user_messages") or s.get("message_count", 0) > 1]


if "active_nav" not in st.session_state:
    st.session_state.active_nav = "💬 Agent Chat & Synthesis"

if "current_session_id" not in st.session_state:
    existing_sessions = get_user_initialized_sessions()
    if existing_sessions:
        latest = st.session_state.chat_manager.get_session(existing_sessions[0]["id"])
        if latest:
            st.session_state.current_session_id = latest["id"]
            st.session_state.messages = latest.get("messages", [])
        else:
            new_s = st.session_state.chat_manager.create_session()
            st.session_state.current_session_id = new_s["id"]
            st.session_state.messages = new_s["messages"]
    else:
        new_s = st.session_state.chat_manager.create_session()
        st.session_state.current_session_id = new_s["id"]
        st.session_state.messages = new_s["messages"]

if "messages" not in st.session_state:
    curr = st.session_state.chat_manager.get_session(st.session_state.current_session_id)
    st.session_state.messages = curr.get("messages", []) if curr else []

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = getattr(config, "llm_provider", os.getenv("LLM_PROVIDER", "google"))

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = getattr(config, "gemini_api_key", os.getenv("GEMINI_API_KEY", ""))

if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = getattr(config, "llm_model", os.getenv("LLM_MODEL", "gemini-2.5-flash-lite"))

if "deepseek_api_key" not in st.session_state:
    st.session_state.deepseek_api_key = getattr(config, "deepseek_api_key", os.getenv("DEEPSEEK_API_KEY", ""))

if "deepseek_model" not in st.session_state:
    st.session_state.deepseek_model = getattr(config, "deepseek_model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))

if "deepseek_base_url" not in st.session_state:
    st.session_state.deepseek_base_url = getattr(config, "deepseek_base_url", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))

if "ollama_base_url" not in st.session_state:
    st.session_state.ollama_base_url = getattr(config, "ollama_base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

if "ollama_model" not in st.session_state:
    st.session_state.ollama_model = getattr(config, "ollama_model", os.getenv("OLLAMA_MODEL", "llama3.2:latest"))

if "enable_ollama_fallback" not in st.session_state:
    st.session_state.enable_ollama_fallback = getattr(config, "enable_ollama_fallback", True)


def build_agent_instance() -> SpecializedDocumentAgent:
    curr_p = st.session_state.llm_provider.lower()
    if "gemini" in curr_p or "google" in curr_p:
        p = "google"
        key = st.session_state.gemini_api_key
        mod = st.session_state.gemini_model
        burl = None
    elif "deepseek" in curr_p:
        p = "deepseek"
        key = st.session_state.deepseek_api_key
        mod = st.session_state.deepseek_model
        burl = st.session_state.deepseek_base_url
    else:
        p = "ollama"
        key = None
        mod = st.session_state.ollama_model
        burl = st.session_state.ollama_base_url

    existing_agent = st.session_state.get("agent")
    ag = SpecializedDocumentAgent(
        provider=p,
        api_key=key,
        llm_model=mod,
        base_url=burl,
        fallback_to_ollama=st.session_state.enable_ollama_fallback,
        structured_manager=getattr(existing_agent, "structured_manager", None),
        unstructured_manager=getattr(existing_agent, "unstructured_manager", None),
        verbose=False,
    )
    if st.session_state.get("messages"):
        ag.load_conversation(st.session_state.messages)
    return ag


if "agent" not in st.session_state:
    st.session_state.agent = build_agent_instance()


# -----------------------------------------------------------------------------
# LEFT SIDEBAR: Redesigned matching Antigravity / AI IDE
# -----------------------------------------------------------------------------
with st.sidebar:
    # 1. Top Logo & Header Bar
    st.markdown(
        """
        <div style="display: flex; align-items: center; padding-bottom: 12px; margin-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div class="top-logo-badge">▲</div>
                <div>
                    <div style="font-weight: 700; font-size: 15px; color: #f8fafc; letter-spacing: -0.02em;">Nexus Agent</div>
                    <div style="font-size: 11px; color: #94a3b8;">Multi-Hop RAG</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Action Button: + New Conversation
    has_user_messages = any(m.get("role") == "user" for m in st.session_state.get("messages", []))

    if st.button(
        "➕ New Conversation",
        type="primary",
        use_container_width=True,
        disabled=not has_user_messages,
        help="Start a fresh conversation" if has_user_messages else "A new chat can only be created once the current chat has been initialized with messages.",
        key="btn_sidebar_new_conversation",
    ):
        if has_user_messages:
            if hasattr(st.session_state.chat_manager, "cleanup_uninitialized_sessions"):
                st.session_state.chat_manager.cleanup_uninitialized_sessions(exclude_session_id=st.session_state.current_session_id)
            new_sess = st.session_state.chat_manager.create_session()
            st.session_state.current_session_id = new_sess["id"]
            st.session_state.messages = new_sess["messages"]
            st.session_state.agent.reset_conversation()
            st.session_state.active_nav = "💬 Agent Chat & Synthesis"
            st.rerun()
        else:
            st.warning("A new chat can only be created once the current chat has been initialized with messages.")

    if not has_user_messages:
        st.caption("🔒 *Send a message to initialize this chat and unlock 'New Conversation'.*")

    # 3. Sidebar Navigation: Tabs moved to the left side of the screen!
    NAV_ITEMS = [
        ("💬 Agent Chat & Synthesis", "💬 Agent Chat & Synthesis", "Main AI document chat and cross-domain reasoning"),
        ("🗄️ Knowledge Vault Explorer", "🗄️ Knowledge Vault Explorer", "Inspect SQL tables & manage documents"),
        ("🧪 Evaluation & Benchmark", "🧪 Evaluation & Benchmark", "Dense retriever & synthesis quality testing"),
        ("🏛️ System Architecture", "🏛️ System Architecture", "Architectural flowcharts & engineering highlights"),
    ]

    st.markdown("<div style='margin-top: 10px; margin-bottom: 8px;'>", unsafe_allow_html=True)
    for label, nav_id, help_tip in NAV_ITEMS:
        is_active = (st.session_state.active_nav == nav_id)
        if st.button(
            label,
            key=f"nav_btn_{nav_id}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
            help=help_tip,
        ):
            st.session_state.active_nav = nav_id
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # 4. Projects / Chat History Panel (matching Image 2)
    all_sessions = get_user_initialized_sessions()

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; padding: 0 4px;">
            <span style="font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8;">Projects / Chats</span>
            <span style="font-size: 0.72rem; color: #64748b;">({len(all_sessions)})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not all_sessions:
        st.caption("No previous chats saved. Start asking questions to build your history!")
    else:
        # Search filter if user has multiple sessions
        search_filter = ""
        if len(all_sessions) > 3:
            search_filter = st.text_input(
                "Filter chats",
                placeholder="🔍 Search history...",
                label_visibility="collapsed",
                key="chat_history_search",
            ).strip().lower()

        filtered_sessions = [
            s for s in all_sessions
            if not search_filter or search_filter in s["title"].lower() or search_filter in s.get("preview", "").lower()
        ]

        hist_box = st.container(height=260)
        with hist_box:
            for sess in filtered_sessions:
                is_active = (sess["id"] == st.session_state.current_session_id)
                time_tag = format_relative_time(sess.get("updated_at", ""))

                col_btn, col_opts = st.columns([5, 1])
                with col_btn:
                    icon = "👉 " if is_active else "💬 "
                    disp_title = sess["title"]
                    if len(disp_title) > 20:
                        disp_title = disp_title[:17] + "..."

                    # Button with title and relative time badge
                    btn_label = f"{icon}{disp_title}  ({time_tag})" if time_tag else f"{icon}{disp_title}"
                    if st.button(
                        btn_label,
                        key=f"hist_btn_{sess['id']}",
                        type="primary" if is_active else "secondary",
                        use_container_width=True,
                        help=f"{sess['title']}\n{sess.get('preview', '')}\nUpdated: {sess['updated_at'][:16].replace('T', ' ')}",
                    ):
                        if not is_active:
                            loaded = st.session_state.chat_manager.get_session(sess["id"])
                            if loaded:
                                st.session_state.current_session_id = loaded["id"]
                                st.session_state.messages = loaded.get("messages", [])
                                st.session_state.agent.load_conversation(st.session_state.messages)
                                st.session_state.active_nav = "💬 Agent Chat & Synthesis"
                                st.rerun()

                with col_opts:
                    with st.popover("⋮", help="Options for this conversation"):
                        st.caption(f"**Chat ID:** `{sess['id']}`")
                        st.caption(f"📅 **Last Active:** {sess['updated_at'][:16].replace('T', ' ')}")
                        st.caption(f"💬 **Messages:** {sess['message_count']}")

                        rename_input = st.text_input("Rename Title", value=sess["title"], key=f"rename_input_{sess['id']}")
                        if st.button("Save Name", key=f"save_rename_{sess['id']}", use_container_width=True):
                            st.session_state.chat_manager.rename_session(sess["id"], rename_input)
                            st.rerun()

                        if st.button("🗑️ Delete Chat", key=f"del_chat_{sess['id']}", type="primary", use_container_width=True):
                            st.session_state.chat_manager.delete_session(sess["id"])
                            if is_active:
                                remaining = get_user_initialized_sessions()
                                if remaining:
                                    nxt = st.session_state.chat_manager.get_session(remaining[0]["id"])
                                    st.session_state.current_session_id = nxt["id"]
                                    st.session_state.messages = nxt.get("messages", [])
                                    st.session_state.agent.load_conversation(st.session_state.messages)
                                else:
                                    fresh = st.session_state.chat_manager.create_session()
                                    st.session_state.current_session_id = fresh["id"]
                                    st.session_state.messages = fresh["messages"]
                                    st.session_state.agent.reset_conversation()
                            st.rerun()

    st.divider()

    # 5. Bottom Settings & Multi-Model Engine (Settings drawer)
    with st.expander("⚙️ Settings & Models", expanded=False):
        st.markdown("<div style='font-weight: 600; font-size: 13px; color: #f8fafc; margin-bottom: 6px;'>LLM Provider</div>", unsafe_allow_html=True)
        provider_options = ["Google Gemini", "DeepSeek", "Ollama (Local)"]
        default_idx = 0
        if "deepseek" in st.session_state.llm_provider.lower():
            default_idx = 1
        elif "ollama" in st.session_state.llm_provider.lower():
            default_idx = 2

        selected_provider_label = st.selectbox(
            "Select Provider",
            options=provider_options,
            index=default_idx,
            label_visibility="collapsed",
            key="ui_provider_select",
        )

        provider_changed = False
        new_provider_slug = "google" if "Google" in selected_provider_label else ("deepseek" if "DeepSeek" in selected_provider_label else "ollama")
        if new_provider_slug != st.session_state.llm_provider:
            st.session_state.llm_provider = new_provider_slug
            provider_changed = True

        # Check Ollama connection status
        ollama_alive, ollama_models = check_ollama_status(st.session_state.ollama_base_url)

        if new_provider_slug == "google":
            g_key = st.text_input(
                "Google Gemini API Key",
                value=st.session_state.gemini_api_key,
                type="password",
                help="Paste your Google AI Studio Gemini API Key here (or set in .env).",
                key="ui_gemini_key",
            )
            if g_key != st.session_state.gemini_api_key:
                st.session_state.gemini_api_key = g_key
                provider_changed = True

            g_models = get_available_models_for_provider("google")
            if st.session_state.gemini_model and st.session_state.gemini_model not in g_models:
                g_models.insert(0, st.session_state.gemini_model)
            curr_g_idx = g_models.index(st.session_state.gemini_model) if st.session_state.gemini_model in g_models else 0
            g_mod = st.selectbox("Gemini Model", options=g_models, index=curr_g_idx, key="ui_gemini_model")
            if g_mod != st.session_state.gemini_model:
                st.session_state.gemini_model = g_mod
                provider_changed = True

        elif new_provider_slug == "deepseek":
            d_key = st.text_input(
                "DeepSeek API Key",
                value=st.session_state.deepseek_api_key,
                type="password",
                placeholder="sk-...",
                help="Paste your DeepSeek API Key here (https://platform.deepseek.com/).",
                key="ui_deepseek_key",
            )
            if d_key != st.session_state.deepseek_api_key:
                st.session_state.deepseek_api_key = d_key
                provider_changed = True

            d_models = get_available_models_for_provider("deepseek")
            if st.session_state.deepseek_model and st.session_state.deepseek_model not in d_models:
                d_models.insert(0, st.session_state.deepseek_model)
            curr_d_idx = d_models.index(st.session_state.deepseek_model) if st.session_state.deepseek_model in d_models else 0
            d_mod = st.selectbox("DeepSeek Model", options=d_models, index=curr_d_idx, key="ui_deepseek_model")
            if d_mod != st.session_state.deepseek_model:
                st.session_state.deepseek_model = d_mod
                provider_changed = True

            d_burl = st.text_input(
                "DeepSeek Base URL",
                value=st.session_state.deepseek_base_url,
                help="OpenAI-compatible DeepSeek endpoint.",
                key="ui_deepseek_base_url",
            )
            if d_burl != st.session_state.deepseek_base_url:
                st.session_state.deepseek_base_url = d_burl
                provider_changed = True

        else:  # Ollama
            o_burl = st.text_input(
                "Ollama Base URL",
                value=st.session_state.ollama_base_url,
                help="Local Ollama daemon endpoint.",
                key="ui_ollama_base_url",
            )
            if o_burl != st.session_state.ollama_base_url:
                st.session_state.ollama_base_url = o_burl
                provider_changed = True

            if ollama_alive:
                st.caption(f"🟢 **Ollama Connected** ({len(ollama_models)} model(s) available)")
                avail_o = ollama_models if ollama_models else ["llama3.2:latest"]
                curr_o_idx = avail_o.index(st.session_state.ollama_model) if st.session_state.ollama_model in avail_o else 0
                o_mod = st.selectbox("Ollama Model", options=avail_o, index=curr_o_idx, key="ui_ollama_model")
                if o_mod != st.session_state.ollama_model:
                    st.session_state.ollama_model = o_mod
                    provider_changed = True
            else:
                st.caption("⚪ **Ollama Offline** (Run `ollama serve` to connect)")
                o_mod = st.text_input("Ollama Model Name", value=st.session_state.ollama_model, key="ui_ollama_model_txt")
                if o_mod != st.session_state.ollama_model:
                    st.session_state.ollama_model = o_mod
                    provider_changed = True

        # Fallback to Ollama Toggle
        fb_checked = st.checkbox(
            "🛡️ Enable Ollama Fallback",
            value=st.session_state.enable_ollama_fallback,
            help="If cloud API calls fail or credentials are missing, automatically fall back to local Ollama.",
            key="ui_ollama_fallback_check",
        )
        if fb_checked != st.session_state.enable_ollama_fallback:
            st.session_state.enable_ollama_fallback = fb_checked
            provider_changed = True

        if provider_changed:
            st.session_state.agent = build_agent_instance()
            st.success("Model configuration updated!")

        # Active Model Status Badges
        agent_meta = getattr(st.session_state.agent, "metadata", {})
        active_p = agent_meta.get("provider", st.session_state.llm_provider)
        active_m = agent_meta.get("model", "")
        is_fb = agent_meta.get("is_fallback", False)

        st.markdown(
            f"""
            <div style="margin-top: 10px; margin-bottom: 8px; display: flex; flex-wrap: wrap; gap: 4px;">
                <span class="badge-pill">Provider: {active_p.capitalize()}</span>
                <span class="badge-pill">Model: {active_m}</span>
                {'<span class="badge-pill" style="border-color: #22c55e; color: #4ade80;">🛡️ Fallback Active</span>' if is_fb else ('<span class="badge-pill">🛡️ Fallback Ready</span>' if ollama_alive and st.session_state.enable_ollama_fallback else '')}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if len(all_sessions) > 1:
            st.divider()
            with st.popover("🗑️ Clear All Chat History", use_container_width=True):
                st.warning("Permanently delete all saved conversations?")
                if st.button("Confirm Delete All", type="primary", key="confirm_clear_all_history", use_container_width=True):
                    st.session_state.chat_manager.clear_all_sessions()
                    fresh = st.session_state.chat_manager.create_session()
                    st.session_state.current_session_id = fresh["id"]
                    st.session_state.messages = fresh["messages"]
                    st.session_state.agent.reset_conversation()
                    st.rerun()


# -----------------------------------------------------------------------------
# MAIN VIEWPORT ROUTER (Switched dynamically by left sidebar navigation)
# -----------------------------------------------------------------------------


# =============================================================================
# VIEW 1: 💬 Agent Chat & Synthesis
# =============================================================================
if st.session_state.active_nav == "💬 Agent Chat & Synthesis":
    curr_sess = st.session_state.chat_manager.get_session(st.session_state.current_session_id)
    sess_title = curr_sess.get("title", "Active Conversation") if curr_sess else "Active Conversation"
    created_at_fmt = curr_sess.get("created_at", "")[:16].replace("T", " ") if curr_sess else ""
    user_msg_count = len([m for m in st.session_state.messages if m.get("role") == "user"])

    # If Fresh Conversation (0 user messages), render the Hero design matching Image 1
    if user_msg_count == 0:
        st.markdown(
            """
            <div class="hero-container">
                <div class="hero-logo-box">▲</div>
                <h1 class="hero-title">Hi Hải, what can we get done today?</h1>
                <p class="hero-subtitle">
                    Autonomous multi-hop retrieval and reasoning across SQLite transactional database tables
                    and unstructured knowledge policies.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Demo Prompts as Centered Scenario Cards
        sc_col1, sc_col2 = st.columns(2)
        with sc_col1:
            if st.button("🔍 Alice Chen Order Return", use_container_width=True, help="Check eligibility for refund on order #5001"):
                st.session_state.preset_query = "Customer Alice Chen (Customer ID 101) wants to return order #5001. She received it on 2026-09-10. Is she eligible for a full refund today?"
                st.rerun()
            if st.button("📊 Low Stock & Reviews", use_container_width=True, help="Analyze low stock items and customer complaints"):
                st.session_state.preset_query = "Which products currently have low stock (< 20 units) in the database, and what are customers complaining about in their reviews?"
                st.rerun()
        with sc_col2:
            if st.button("📜 Policy Comparison (Standard vs VIP)", use_container_width=True, help="Compare return window and shipping fees"):
                st.session_state.preset_query = "Compare the return windows, restocking fees, and shipping costs for Standard tier vs VIP tier customers."
                st.rerun()
            if st.button("🎧 Audio Specs & Factory Reset", use_container_width=True, help="AeroSound Pro Bluetooth specs and reset procedure"):
                st.session_state.preset_query = "What are the Bluetooth specifications for AeroSound Pro headphones, and how do I perform a factory reset?"
                st.rerun()

    else:
        # Active Chat Header Bar
        head_col1, head_col2 = st.columns([7, 3])
        with head_col1:
            active_m_label = getattr(st.session_state.agent, "metadata", {}).get("model", config.llm_model)
            st.markdown(f"### 💬 {sess_title}")
            st.caption(f"Session: `{st.session_state.current_session_id}` | Model: `{active_m_label}` | Started: {created_at_fmt} | {len(st.session_state.messages)} message(s)")
        with head_col2:
            export_md = st.session_state.chat_manager.export_session_markdown(st.session_state.current_session_id)
            st.download_button(
                label="📥 Export Chat (.md)",
                data=export_md,
                file_name=f"{st.session_state.current_session_id}.md",
                mime="text/markdown",
                use_container_width=True,
                help="Export full conversation transcript as Markdown document",
            )

        st.divider()

        # Render Chat History
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    sources_str = ", ".join([f"`{s}`" for s in msg["sources"]])
                    st.markdown(f"<div class='source-box'>📚 <b>Grounded Sources:</b> {sources_str}</div>", unsafe_allow_html=True)
                if msg.get("trace_steps"):
                    with st.expander(f"🔍 Reasoning & Tool Trace ({len(msg['trace_steps'])} step(s))"):
                        for step_idx, step in enumerate(msg["trace_steps"], 1):
                            t_name = step.get("tool_name") or step.get("step_type", "step")
                            t_in = step.get("input_data", "")
                            st.markdown(f"**{step_idx}. `{t_name}`**" + (f" — `{t_in}`" if t_in else ""))
                if msg.get("latency", 0) > 0:
                    st.caption(f"⏱️ Response latency: {msg['latency']}s")

    # Chat Input
    preset = st.session_state.pop("preset_query", None)
    user_prompt = st.chat_input("Ask anything across products, orders, return policies, or specs...")

    active_prompt = preset or user_prompt

    if active_prompt:
        st.session_state.messages.append({"role": "user", "content": active_prompt})

        st.session_state.chat_manager.save_session(
            session_id=st.session_state.current_session_id,
            title=sess_title,
            messages=st.session_state.messages,
        )

        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent analyzing query, running Text-to-SQL & semantic search..."):
                start_time = time.time()
                response = st.session_state.agent.chat(active_prompt)
                latency = round(time.time() - start_time, 2)

                st.markdown(response.response_text)

                if response.sources:
                    sources_str = ", ".join([f"`{s}`" for s in response.sources])
                    st.markdown(f"<div class='source-box'>📚 <b>Grounded Sources:</b> {sources_str}</div>", unsafe_allow_html=True)

                serialized_trace = [
                    {
                        "step_type": s.step_type,
                        "tool_name": s.tool_name,
                        "input_data": s.input_data,
                        "output_data": s.output_data,
                    }
                    for s in (response.trace_steps or [])
                ]
                if serialized_trace:
                    with st.expander(f"🔍 Reasoning & Tool Trace ({len(serialized_trace)} step(s))"):
                        for step_idx, step in enumerate(serialized_trace, 1):
                            t_name = step.get("tool_name") or step.get("step_type", "step")
                            t_in = step.get("input_data", "")
                            st.markdown(f"**{step_idx}. `{t_name}`**" + (f" — `{t_in}`" if t_in else ""))

                st.caption(f"⏱️ Response latency: {latency}s")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.response_text,
                    "sources": response.sources,
                    "trace_steps": serialized_trace,
                    "latency": latency,
                })

                st.session_state.chat_manager.save_session(
                    session_id=st.session_state.current_session_id,
                    title=sess_title,
                    messages=st.session_state.messages,
                )
                st.rerun()


# =============================================================================
# VIEW 2: 🗄️ Knowledge Vault Explorer
# =============================================================================
elif st.session_state.active_nav == "🗄️ Knowledge Vault Explorer":
    st.markdown("### 🗄️ Knowledge Vault & Document Management Studio")
    st.markdown(
        "Manage unstructured knowledge documents (upload, preview, download, delete), "
        "inspect live relational tables, and observe dynamic system prompts, schema, and reasoning relationships."
    )

    vault_sub1, vault_sub2 = st.tabs([
        "📄 Document Management (Upload, Preview, Download, Delete)",
        "📊 Structured Database (SQLite Records)",
    ])

    with vault_sub1:
        st.markdown("#### 📄 Document Vault Management")
        st.caption("Upload company policies, SLAs, specifications, and customer sentiment reports (.md, .txt, .pdf).")

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

        # Active Vault Documents List
        st.markdown("##### 📚 Active Vault Documents")
        if not vault_files:
            st.warning("No documents currently found in the knowledge vault directory. Upload some documents above!")
        else:
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

            doc_names = [f["name"] for f in vault_files]
            selected_doc_name = st.selectbox("Select a Document to Preview, Download, or Delete", doc_names)

            if selected_doc_name:
                selected_meta = next(f for f in vault_files if f["name"] == selected_doc_name)
                selected_path = selected_meta["path"]
                ext = selected_meta["extension"]

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

    with vault_sub2:
        st.markdown("#### 📊 Relational Database Tables (SQLite)")
        st.caption("Direct read-only inspection of structured retail database records.")

        db_mgr_view = st.session_state.agent.structured_manager
        table_names = db_mgr_view.get_table_names()
        tbl_choice = st.selectbox("Select SQLite Table to Inspect", table_names)

        if tbl_choice and tbl_choice in table_names:
            rows = db_mgr_view.execute_raw_sql(f'SELECT * FROM "{tbl_choice}"')
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.caption(f"Displaying {len(df)} records from table `{tbl_choice}`.")


# =============================================================================
# VIEW 3: 🧪 Evaluation & Benchmark
# =============================================================================
elif st.session_state.active_nav == "🧪 Evaluation & Benchmark":
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
            - Dense embeddings project queries into a shared vector space. When a query is colloquial, symptom-based, or long, its cosine similarity to the document chunk degrades.
            
            #### 3. Enterprise AI Engineering Solutions:
            1. **Embedding Cosine Evaluation (`SemanticSimilarityEvaluator`):**
               - Evaluates generated responses against a golden reference answer using vector cosine similarity.
            2. **HyDE (Hypothetical Document Embeddings):**
               - Generates a hypothetical answer first and embeds it to align query and document representations.
            3. **Hybrid Dense + Sparse Search (BM25 + Vector):**
               - Preserves SKU precision while maintaining semantic recall.
            4. **Cross-Encoder Re-Ranking:**
               - Re-ranks candidate chunks using full cross-attention.
            """
        )


# =============================================================================
# VIEW 4: 🏛️ System Architecture
# =============================================================================
elif st.session_state.active_nav == "🏛️ System Architecture":
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
        ### 🔑 Key AI Engineering Highlights:
        1. **Hybrid Data Ingestion**:
           - Structured data stored in normalized SQLite tables queried via deterministic SQLAlchemy / `NLSQLTableQueryEngine`.
           - Unstructured documents chunked with `SentenceSplitter(chunk_size=512, chunk_overlap=64)` and indexed with Google Gemini embeddings.
        2. **Multi-Hop Cross-Domain Reasoning**:
           - Resolves complex business logic across SQL database tables and vector document stores.
        3. **Deterministic Tool Augmentation**:
           - Custom `date_calculator` tool eliminates LLM arithmetic hallucinations when evaluating elapsed calendar days.
        4. **Production Security & Grounding**:
           - Read-only SQL queries prevent database mutations.
           - Explicit citation enforcement in agent prompts prevents hallucination.
        """
    )
