import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from text2sql.llm_client import build_chat_model


_SQL_BLOCK = re.compile(r"```(?:sql)?\s*([\s\S]*?)```", re.IGNORECASE)

# ReAct 续轮：模型未给出 Final 时注入，促使其继续或收尾为 Final SQL
_REACT_OBSERVATION = (
    "Observation（系统）：上一轮未出现以 `Final:` 开头的最终 SQL。"
    "请继续 ReAct：输出 Thought / Action / Observation，"
    "若已确定查询则下一则回复必须以 `Final:` 开头并紧跟一条可执行的 SELECT（PostgreSQL），"
    "在此之前不要结束。"
)


def _extract_sql(text: str) -> str:
    text = text.strip()
    m = _SQL_BLOCK.search(text)
    if m:
        return m.group(1).strip()
    return text


def _extract_final_sql(text: str) -> str | None:
    """从 `Final:` 行（或段）解析最终 SQL；无则返回 None。"""
    lower = text.lower()
    key = "final:"
    idx = lower.rfind(key)
    if idx == -1:
        return None
    rest = text[idx + len(key) :].strip()
    if not rest:
        return None
    # 去掉可选的 markdown 围栏
    rest = re.sub(r"^```(?:sql)?\s*", "", rest, flags=re.IGNORECASE)
    rest = re.sub(r"\s*```\s*$", "", rest)
    rest = rest.strip()
    return rest or None


def _resolve_sql_output(text: str) -> str:
    """优先 Final:，否则兼容旧版 ```sql``` 或整段文本。"""
    final = _extract_final_sql(text)
    if final:
        return final
    return _extract_sql(text).strip()


class Text2sqlAgent:
    """根据自然语言与上下文生成只读 SQL，并在失败时结合错误信息修正。"""

    SYSTEM = """你是一个 PostgreSQL Text2SQL Agent，使用 ReAct（Thought / Action / Observation）模式工作。
你的目标：根据「数据库 schema」「对话历史」「用户问题」，生成最终可执行的**只读 SQL**。

你必须遵循以下硬性约束：
- 只能输出一条 SQL（最终答案必须是 SQL）。
- SQL 必须以 SELECT 开头。
- 禁止 INSERT/UPDATE/DELETE/DDL/ALTER/DROP/TRUNCATE/CREATE/事务控制/多语句/存储过程。
- 禁止输出 Markdown、解释文字、注释。
- 若信息不足，生成保守但可执行的查询（例如加 LIMIT、使用 ILIKE 模糊匹配）。
- 必须使用 PostgreSQL 语法。
- 优先使用 schema 中存在的表与字段，禁止臆造字段。
- 若用户追问指代前文，必须结合对话历史消歧后再思考。
- 若给出错误反馈（Observation），Thought 必须解释修正依据，再输出新 Action。

格式要求：
Thought: <分析需求、参照 schema 与对话历史、消歧、判断可行性、错误修正思路>
Action: 选择操作（如：识别表、识别字段、构建 join、添加过滤条件、添加聚合、加 limit、修复错误）
Observation: 检查 SQL 合法性、字段是否存在、语法是否正确、是否只读

重要输出格式规则：
- 你的最终输出必须是：
Final: <SQL>"""

    def __init__(self, model: Any | None = None, *, max_react_steps: int = 8):
        self._llm = model or build_chat_model()
        self._max_react_steps = max(1, max_react_steps)

    def generate(
        self,
        *,
        messages: list[BaseMessage],
        schema_text: str,
        previous_sql: str = "",
        last_error: str | None = None,
    ) -> str:
        extra: list[BaseMessage] = []
        if last_error:
            extra.append(
                HumanMessage(
                    content=(
                        f"上一次生成的 SQL 执行失败。\n错误信息：{last_error}\n"
                        f"上次 SQL：\n{previous_sql}\n\n"
                        "请按 Thought / Action / Observation 分析失败原因并修正；"
                        "最终必须以 `Final:` 开头输出一条可执行的 SELECT（PostgreSQL）。"
                    )
                )
            )
        trail: list[BaseMessage] = [
            SystemMessage(
                content=f"{self.SYSTEM}\n\n当前数据库 schema 摘要：\n{schema_text}"
            ),
            *messages,
            *extra,
        ]
        last_raw = ""
        for step in range(self._max_react_steps):
            resp = self._llm.invoke(trail)
            last_raw = (
                resp.content if hasattr(resp, "content") else str(resp)
            )
            last_raw = str(last_raw)
            sql = _extract_final_sql(last_raw)
            if sql:
                return sql.strip()
            # 未出现 Final:：继续 ReAct 轮次（最后一轮仍无 Final 则走兼容解析）
            if step < self._max_react_steps - 1:
                trail.append(AIMessage(content=last_raw))
                trail.append(HumanMessage(content=_REACT_OBSERVATION))
        return _resolve_sql_output(last_raw).strip()


def last_user_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            c = m.content
            return c if isinstance(c, str) else str(c)
    return ""
