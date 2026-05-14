from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from text2sql.config import Settings, get_settings
from text2sql.graph import GraphState, build_text2sql_graph
from text2sql.query_executor import QueryExecutor


class Chatbot:
    """多轮对话客户端：维护 thread、schema 缓存，调用 LangGraph。"""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        thread_id: str = "default",
        query_executor: QueryExecutor | None = None,
    ):
        self._settings = settings or get_settings()
        self._qe = query_executor or QueryExecutor(self._settings)
        self._thread_id = thread_id
        self._schema_text: str | None = None
        workflow = build_text2sql_graph(self._settings, query_executor=self._qe)
        self._app = workflow.compile(checkpointer=MemorySaver())

    def refresh_schema(self) -> str:
        allow = self._settings.schema_allow_tables
        tables = [t.strip() for t in allow.split(",") if t.strip()] if allow else None
        self._schema_text = self._qe.fetch_schema_digest(table_filter=tables)
        return self._schema_text

    @property
    def schema_text(self) -> str:
        if self._schema_text is None:
            self.refresh_schema()
        assert self._schema_text is not None
        return self._schema_text

    def chat(self, user_text: str) -> dict[str, Any]:
        """处理一轮用户输入，返回完整状态片段（含 analysis、sql、error 等）。"""
        init: GraphState = {
            "messages": [HumanMessage(content=user_text)],
            "schema_text": self.schema_text,
            "sql": "",
            "columns": [],
            "rows": [],
            "error": None,
            "retry_count": 0,
            "analysis": "",
            "route": "",
            "rag_context": [],
            "rag_error": None,
        }
        cfg: dict[str, Any] = {"configurable": {"thread_id": self._thread_id}}
        # 合并 checkpoint：新输入用 add_messages 追加
        out = self._app.invoke(init, cfg)
        return out
