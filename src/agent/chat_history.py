"""
Chat History Management Module.
Handles session persistence, creation, retrieval, auto-titling,
and deletion of multi-session agent conversations.
"""
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.config import config

# In-memory cache of session summary metadata keyed by resolved file path -> (mtime_ns, summary_dict)
_SESSION_SUMMARY_CACHE: Dict[str, Tuple[int, Dict[str, Any]]] = {}


def generate_session_id() -> str:
    """Generates a timestamped unique ID for a chat conversation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_suffix = uuid.uuid4().hex[:6]
    return f"chat_{timestamp}_{unique_suffix}"


def auto_generate_title(text: str, max_words: int = 6) -> str:
    """
    Generates a clean and concise conversation title from initial user prompt.
    E.g. "Customer Alice Chen (Customer ID 101) wants to return..." -> "Alice Chen Return Order #5001"
    """
    if not text:
        return "New Conversation"
    # Remove excessive formatting symbols and clean whitespace
    cleaned = re.sub(r"[#*_`>\n\r]+", " ", text).strip()
    # Remove leading common conversational phrases
    cleaned = re.sub(r"^(can you please |could you please |can you |could you |please |i would like to |how do i |what is |which |tell me about )+", "", cleaned, flags=re.IGNORECASE).strip()
    words = cleaned.split()
    if not words:
        return "New Conversation"
    title = " ".join(words[:max_words])
    if len(words) > max_words:
        title += "..."
    # Cap total character length
    if len(title) > 42:
        title = title[:39] + "..."
    # Capitalize only first character to preserve casing of proper nouns (names, SKUs, tiers)
    if title:
        title = title[0].upper() + title[1:]
    return title


class ChatHistoryManager:
    """
    Manages persistent storage and retrieval of chat sessions on disk.
    Each session is saved as a structured JSON file inside storage/chat_history/.
    """

    DEFAULT_GREETING = (
        "👋 Hello! I am **NexusDoc-Agent**, an enterprise AI Document & Data Agent.\n\n"
        "I can query and synthesize information across:\n"
        "- 📊 **Structured SQLite Database**: Products, customer loyalty tiers, order history, inventory logs.\n"
        "- 📚 **Unstructured Knowledge Vault**: Return & warranty policies, shipping SLAs, technical specs, and customer reviews.\n\n"
        "Try asking a complex multi-hop question, or click one of the quick prompts in the sidebar!"
    )

    def __init__(self, storage_dir: Optional[Path] = None):
        self.storage_dir = storage_dir or config.chat_history_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        # Sanitize session_id to prevent directory traversal
        safe_id = "".join(c for c in session_id if c.isalnum() or c in ("_", "-"))
        return self.storage_dir / f"{safe_id}.json"

    def list_sessions(self, require_user_messages: bool = False, *args, **kwargs) -> List[Dict[str, Any]]:
        """
        Returns summary metadata for all saved conversation sessions,
        sorted chronologically by updated_at (most recent first).
        If require_user_messages is True, sessions without any user queries are excluded.
        Uses mtime-based in-memory caching to avoid redundant JSON parsing on every UI render.
        """
        filter_user_msgs = kwargs.get("require_user_messages", require_user_messages)
        sessions = []
        if not self.storage_dir.exists():
            return []

        for file_path in self.storage_dir.glob("*.json"):
            try:
                stat = file_path.stat()
                path_key = str(file_path.resolve())
                cached_entry = _SESSION_SUMMARY_CACHE.get(path_key)

                if cached_entry and cached_entry[0] == stat.st_mtime_ns:
                    summary = dict(cached_entry[1])
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    messages = data.get("messages", [])
                    has_user_msg = any(msg.get("role") == "user" for msg in messages)

                    # Extract preview from the first user query or last message
                    preview = ""
                    for msg in messages:
                        if msg.get("role") == "user":
                            preview = msg.get("content", "")[:70]
                            break
                    if not preview and messages:
                        preview = messages[-1].get("content", "")[:70]

                    updated_at = data.get(
                        "updated_at",
                        datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    )
                    created_at = data.get(
                        "created_at",
                        datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    )

                    summary = {
                        "id": data.get("id", file_path.stem),
                        "title": data.get("title", "Untitled Chat"),
                        "created_at": created_at,
                        "updated_at": updated_at,
                        "message_count": len(messages),
                        "preview": preview,
                        "has_user_messages": has_user_msg,
                    }
                    _SESSION_SUMMARY_CACHE[path_key] = (stat.st_mtime_ns, summary)

                if filter_user_msgs and not summary.get("has_user_messages"):
                    continue

                sessions.append(summary)
            except Exception:
                continue

        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def cleanup_uninitialized_sessions(self, exclude_session_id: Optional[str] = None) -> int:
        """
        Deletes orphaned conversation sessions that were never initialized with user messages.
        Optionally preserves the currently active session via exclude_session_id.
        """
        count = 0
        if not self.storage_dir.exists():
            return 0

        for file_path in self.storage_dir.glob("*.json"):
            if exclude_session_id and file_path.stem == exclude_session_id:
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                messages = data.get("messages", [])
                has_user_msg = any(m.get("role") == "user" for m in messages)
                if not has_user_msg:
                    path_key = str(file_path.resolve())
                    file_path.unlink()
                    _SESSION_SUMMARY_CACHE.pop(path_key, None)
                    count += 1
            except Exception:
                continue
        return count

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Loads and returns complete conversation session data."""
        session_path = self._get_session_path(session_id)
        if not session_path.exists():
            return None
        try:
            with open(session_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def create_session(
        self,
        title: Optional[str] = "New Conversation",
        initial_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Creates a brand new conversation session and saves it to disk."""
        session_id = generate_session_id()
        now_str = datetime.now().isoformat()

        if initial_messages is None:
            initial_messages = [
                {
                    "role": "assistant",
                    "content": self.DEFAULT_GREETING,
                    "sources": [],
                    "latency": 0.0,
                    "timestamp": now_str,
                }
            ]

        session_data = {
            "id": session_id,
            "title": title or "New Conversation",
            "created_at": now_str,
            "updated_at": now_str,
            "messages": initial_messages,
        }
        self.save_session(
            session_id=session_id,
            title=session_data["title"],
            messages=initial_messages,
            created_at=now_str,
        )
        return session_data

    def save_session(
        self,
        session_id: str,
        title: str,
        messages: List[Dict[str, Any]],
        created_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Saves or updates a conversation session on disk."""
        session_path = self._get_session_path(session_id)
        now_str = datetime.now().isoformat()

        if not created_at and session_path.exists():
            try:
                with open(session_path, "r", encoding="utf-8") as f:
                    old_data = json.load(f)
                    created_at = old_data.get("created_at", now_str)
            except Exception:
                created_at = now_str
        elif not created_at:
            created_at = now_str

        # Auto-update title if default and user messages exist
        clean_title = (title or "").strip()
        if clean_title in ("New Conversation", "New Chat", "Untitled Chat", ""):
            for msg in messages:
                if msg.get("role") == "user":
                    clean_title = auto_generate_title(msg.get("content", ""))
                    break
        if not clean_title:
            clean_title = "New Conversation"

        session_data = {
            "id": session_id,
            "title": clean_title,
            "created_at": created_at,
            "updated_at": now_str,
            "messages": messages,
        }

        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)

        return session_data

    def rename_session(self, session_id: str, new_title: str) -> bool:
        """Renames a saved conversation session."""
        session_data = self.get_session(session_id)
        if not session_data:
            return False
        clean_title = new_title.strip() or "Untitled Chat"
        self.save_session(
            session_id=session_id,
            title=clean_title,
            messages=session_data.get("messages", []),
            created_at=session_data.get("created_at"),
        )
        return True

    def delete_session(self, session_id: str) -> bool:
        """Deletes a conversation session by ID."""
        session_path = self._get_session_path(session_id)
        if session_path.exists():
            try:
                path_key = str(session_path.resolve())
                session_path.unlink()
                _SESSION_SUMMARY_CACHE.pop(path_key, None)
                return True
            except Exception:
                return False
        return False

    def clear_all_sessions(self) -> int:
        """Deletes all saved conversations."""
        count = 0
        for file_path in self.storage_dir.glob("*.json"):
            try:
                path_key = str(file_path.resolve())
                file_path.unlink()
                _SESSION_SUMMARY_CACHE.pop(path_key, None)
                count += 1
            except Exception:
                continue
        return count

    def export_session_markdown(self, session_id: str) -> str:
        """Generates a clean Markdown transcript of the conversation for export."""
        session = self.get_session(session_id)
        if not session:
            return ""

        md_lines = [
            f"# {session.get('title', 'Chat Transcript')}",
            f"- **Session ID:** `{session.get('id')}`",
            f"- **Created:** {session.get('created_at')}",
            f"- **Updated:** {session.get('updated_at')}",
            "\n---\n",
        ]

        for idx, msg in enumerate(session.get("messages", []), 1):
            role_label = "👤 User" if msg.get("role") == "user" else "🤖 Assistant (NexusDoc-Agent)"
            content = msg.get("content", "")
            sources = msg.get("sources", [])
            latency = msg.get("latency", 0.0)

            md_lines.append(f"### {idx}. {role_label}")
            md_lines.append(content)
            if sources:
                sources_str = ", ".join([f"`{s}`" for s in sources])
                md_lines.append(f"\n> 📚 **Grounded Sources:** {sources_str}")
            if latency:
                md_lines.append(f"\n*⏱️ Latency: {latency}s*")
            md_lines.append("\n---\n")

        return "\n".join(md_lines)
