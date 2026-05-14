from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from text2sql.analyzer_agent import AnalyzerAgent
from text2sql.config import Settings, get_settings
from text2sql.query_executor import QueryExecutor
from text2sql.rag_engine import RagEngine
from text2sql.text2sql_agent import Text2sqlAgent, last_user_text


class GraphState(TypedDict):
    """LangGraph 状态：多轮对话 + SQL 执行 + 分析。"""

    messages: Annotated[list[BaseMessage], add_messages]
    schema_text: str
    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    error: str | None
    retry_count: int
    analysis: str
    route: str
    rag_context: list[str]
    rag_error: str | None


def build_text2sql_graph(
    settings: Settings | None = None,
    *,
    query_executor: QueryExecutor | None = None,
    text2sql: Text2sqlAgent | None = None,
    analyzer: AnalyzerAgent | None = None,
):
    s = settings or get_settings()
    qe = query_executor or QueryExecutor(s)
    gen = text2sql or Text2sqlAgent()
    ana = analyzer or AnalyzerAgent()
    rag: RagEngine | None = None
    rag_init_error: str | None = None
    if s.rag_enabled:
        try:
            rag = RagEngine(s)
        except Exception as e:
            rag = None
            rag_init_error = f"{type(e).__name__}: {e}"

    def route_query(state: GraphState) -> dict[str, Any]:
        if rag is None:
            return {"route": "sql", "rag_error": rag_init_error}
        uq = last_user_text(state["messages"])
        route = rag.route(uq)
        return {"route": route, "rag_error": None}

    def route_after_classifier(state: GraphState) -> Literal["rag", "sql"]:
        return "rag" if state.get("route") == "kb" else "sql"

    def generate_sql(state: GraphState) -> dict[str, Any]:
        sql = gen.generate(
            messages=state["messages"],
            schema_text=state.get("schema_text") or "(无 schema)",
            previous_sql=state.get("sql") or "",
            last_error=state.get("error"),
        )
        return {"sql": sql}

    def execute_sql(state: GraphState) -> dict[str, Any]:
        res = qe.execute(state["sql"])
        if res.error:
            return {
                "columns": [],
                "rows": [],
                "error": res.error,
            }
        return {
            "columns": res.columns,
            "rows": res.rows,
            "error": None,
        }

    def answer_with_rag(state: GraphState) -> dict[str, Any]:
        if rag is None:
            detail = state.get("rag_error") or "unknown initialization error"
            return {
                "analysis": (
                    "当前环境 RAG 不可用，已回退为 SQL 模式。\n"
                    f"具体原因：{detail}"
                ),
                "sql": "",
                "columns": [],
                "rows": [],
                "error": f"RAG unavailable: {detail}",
                "rag_context": [],
                "messages": [AIMessage(content="当前环境 RAG 不可用，已回退为 SQL 模式。")],
            }
        uq = last_user_text(state["messages"])
        result = rag.ask(uq)
        contexts = [
            f"{d.metadata.get('source', 'unknown')} | score={d.metadata.get('rerank_score', 'n/a')}"
            for d in result.contexts
        ]
        return {
            "analysis": result.answer,
            "sql": "",
            "columns": [],
            "rows": [],
            "error": None,
            "rag_context": contexts,
            "rag_error": None,
            "messages": [AIMessage(content=result.answer)],
        }

    def route_after_execute(state: GraphState) -> Literal["retry", "analyze", "fail"]:
        if not state.get("error"):
            return "analyze"
        if state.get("retry_count", 0) < s.max_sql_retries:
            return "retry"
        return "fail"

    def bump_retry(state: GraphState) -> dict[str, Any]:
        return {"retry_count": state.get("retry_count", 0) + 1}

    def run_analyze(state: GraphState) -> dict[str, Any]:
        uq = last_user_text(state["messages"])
        text = ana.analyze(
            user_question=uq,
            sql=state["sql"],
            columns=state["columns"],
            rows=state["rows"],
        )
        rag_error = state.get("rag_error")
        if rag_error:
            text = (
                "提示：当前会话未启用 RAG，已自动回退 SQL 路径。\n"
                f"RAG 初始化错误：{rag_error}\n\n{text}"
            )
        return {
            "analysis": text,
            "messages": [AIMessage(content=text)],
        }

    def run_fail(state: GraphState) -> dict[str, Any]:
        err = state.get("error") or "未知错误"
        n = state.get("retry_count", 0)
        text = (
            f"SQL 执行仍失败（已达最大重试次数 {s.max_sql_retries} 次）。\n\n"
            f"最后错误：{err}\n\n"
            f"已尝试的 SQL：\n```sql\n{state.get('sql', '')}\n```"
        )
        return {
            "analysis": text,
            "messages": [AIMessage(content=text)],
        }

    g = StateGraph(GraphState)
    g.add_node("route_query", route_query)
    g.add_node("answer_with_rag", answer_with_rag)
    g.add_node("generate_sql", generate_sql)
    g.add_node("execute_sql", execute_sql)
    g.add_node("bump_retry", bump_retry)
    g.add_node("analyze", run_analyze)
    g.add_node("fail", run_fail)

    g.add_edge(START, "route_query")
    g.add_conditional_edges(
        "route_query",
        route_after_classifier,
        {
            "rag": "answer_with_rag",
            "sql": "generate_sql",
        },
    )
    g.add_edge("generate_sql", "execute_sql")
    g.add_conditional_edges(
        "execute_sql",
        route_after_execute,
        {
            "retry": "bump_retry",
            "analyze": "analyze",
            "fail": "fail",
        },
    )
    g.add_edge("bump_retry", "generate_sql")
    g.add_edge("answer_with_rag", END)
    g.add_edge("analyze", END)
    g.add_edge("fail", END)

    return g
