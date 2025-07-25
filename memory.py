from __future__ import annotations
import json, sqlite3, time, re, os
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# Your existing MemoryStore (unchanged)
DB_PATH = "agent_memory.db"

def _ensure_tables():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            thread_id TEXT,
            key       TEXT,
            value     TEXT,
            updated   REAL,
            PRIMARY KEY (thread_id, key)
        )
        """)
_ensure_tables()

class MemoryStore:
    """Tiny key‑value store → {thread_id, key} → value (JSON)."""

    @staticmethod
    def set(thread_id: str, key: str, value: Any):
        print(f"[MEMORY_DEBUG] MemoryStore.set called: thread_id={thread_id}, key={key}, value={value}")
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                "REPLACE INTO memory VALUES (?,?,?,?)",
                (thread_id, key, json.dumps(value), time.time()),
            )
        print(f"[MEMORY_DEBUG] Successfully stored in database")

    @staticmethod
    def get(thread_id: str, key: str, default: Any = None) -> Any:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.execute(
                "SELECT value FROM memory WHERE thread_id=? AND key=?",
                (thread_id, key),
            ).fetchone()
        return json.loads(cur[0]) if cur else default

    @staticmethod
    def all(thread_id: str) -> Dict[str, Any]:
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT key, value FROM memory WHERE thread_id=?", (thread_id,)
            ).fetchall()
        return {k: json.loads(v) for k, v in rows}

# Your AgentState
class AgentState(BaseModel):
    thread_id:       str
    user_input:      str
    history:         str
    route:           Optional[str] = Field(default=None)
    assistant_reply: Optional[str] = None
    last_user:       Optional[str] = None