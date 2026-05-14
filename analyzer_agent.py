import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from text2sql.config import get_settings
from text2sql.llm_client import build_chat_model
from text2sql.skill_runtime import find_skill_tool, invoke_skill_tool, load_skill_tools


class AnalyzerAgent:
    """对查询结果生成中文分析报告，并按需调用 skill 外部导出工具。"""

    SYSTEM = """你是数据分析助手。根据用户问题、执行的 SQL 与查询结果，写一份**专业、简洁**的中文分析报告。

要求：
- 用 Markdown 小标题与列表，先给结论再给依据。
- 若结果为空，说明可能原因与下一步建议。
- 不要编造数据中不存在的数字；统计量请基于给定结果。
- 篇幅适中，避免空话。"""

    _TOOL_ROUTER_SYSTEM = """你是导出工具路由器。请根据用户需求，判断是否需要调用数据导出技能。

你将收到可用技能列表（intent/name/description/triggers）与用户问题。
你的任务：
1) 只从可用 intent 中选择需要执行的项；
2) 如果不需要导出，返回空列表；
3) 输出必须是 JSON，格式：{"intents":["pdf","table"]}。

注意：
- 不要编造不存在的 intent；
- 若用户明确需要下载/导出文件，优先选择对应 intent；
- 若仅是普通分析问答，返回空列表。"""

    def __init__(self, model: Any | None = None, *, report_dir: str | Path | None = None):
        self._llm = model or build_chat_model()
        self._report_dir = self._resolve_report_dir(report_dir)
        settings = get_settings()
        self._enable_skill_tools = settings.analyzer_enable_skill_tools
        configured_root = Path(settings.analyzer_skills_dir).expanduser()
        if not configured_root.is_absolute():
            configured_root = Path(__file__).resolve().parents[1] / configured_root
        self._skills_root = configured_root.resolve()
        self._skill_tools = (
            load_skill_tools(self._skills_root) if self._enable_skill_tools else []
        )

    @staticmethod
    def _resolve_report_dir(report_dir: str | Path | None) -> Path:
        if report_dir is not None:
            p = Path(report_dir)
        else:
            # 默认路径：项目根目录下 outputs/reports
            p = Path(__file__).resolve().parents[1] / "outputs" / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _save_report_file(
        self,
        *,
        report_text: str,
        user_question: str,
        sql: str,
        columns: list[str],
        rows: list[tuple[Any, ...]],
    ) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid4().hex[:8]
        filename = f"analysis_{ts}_{suffix}.md"
        path = self._report_dir / filename
        content = (
            "# 分析报告\n\n"
            f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}\n"
            f"- 用户问题：{user_question}\n"
            f"- SQL：`{sql}`\n"
            f"- 返回列：{columns}\n"
            f"- 返回行数：{len(rows)}\n\n"
            "## 报告正文\n\n"
            f"{report_text}\n"
        )
        path.write_text(content, encoding="utf-8")
        return path

    def _detect_export_intents(self, user_question: str) -> list[str]:
        if not self._skill_tools:
            return []

        # 去重并保序，避免同一 intent 被重复执行。
        available_intents: list[str] = []
        for tool in self._skill_tools:
            if tool.intent not in available_intents:
                available_intents.append(tool.intent)

        tool_lines = []
        for tool in self._skill_tools:
            tool_lines.append(
                (
                    f"- intent={tool.intent}; name={tool.name}; "
                    f"description={tool.description}; triggers={','.join(tool.triggers)}"
                )
            )
        tool_summary = "\n".join(tool_lines)
        router_input = (
            f"用户问题：{user_question}\n\n"
            f"可用 intents：{available_intents}\n"
            f"可用技能明细：\n{tool_summary}\n\n"
            '请仅输出 JSON，例如：{"intents":["pdf"]} 或 {"intents":[]}'
        )
        resp = self._llm.invoke(
            [
                SystemMessage(content=self._TOOL_ROUTER_SYSTEM),
                HumanMessage(content=router_input),
            ]
        )
        raw = str(resp.content if hasattr(resp, "content") else resp).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # 兼容模型返回包裹说明文字的情况，尽量提取 JSON 主体。
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return []
            try:
                data = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return []

        intents_raw = data.get("intents") if isinstance(data, dict) else None
        if not isinstance(intents_raw, list):
            return []
        selected: list[str] = []
        for item in intents_raw:
            intent = str(item).strip().lower()
            if intent in available_intents and intent not in selected:
                selected.append(intent)
        return selected

    def _run_export_tools(
        self,
        *,
        user_question: str,
        sql: str,
        columns: list[str],
        rows: list[tuple[Any, ...]],
        report_text: str,
    ) -> list[str]:
        if not self._enable_skill_tools:
            return []
        intents = self._detect_export_intents(user_question)
        if not intents:
            return []

        outputs: list[str] = []
        payload = {
            "user_question": user_question,
            "sql": sql,
            "columns": columns,
            "rows": rows,
            "report_text": report_text,
            "report_dir": str(self._report_dir),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        for intent in intents:
            tool = find_skill_tool(user_question, self._skill_tools, intent)
            if not tool:
                outputs.append(f"- 未找到可用的 `{intent}` 导出 skill。")
                continue
            result = invoke_skill_tool(tool, payload=payload)
            if not result.get("ok"):
                outputs.append(
                    f"- `{intent}` 导出失败（skill: {tool.name}）：{result.get('error') or '未知错误'}"
                )
                continue
            data = result.get("result") or {}
            artifact = data.get("artifact_path")
            message = data.get("message") or "导出完成"
            if artifact:
                outputs.append(
                    f"- `{intent}` 导出成功（skill: {tool.name}）：{message}，文件：`{artifact}`"
                )
            else:
                outputs.append(f"- `{intent}` 导出成功（skill: {tool.name}）：{message}")
        return outputs

    def analyze(
        self,
        *,
        user_question: str,
        sql: str,
        columns: list[str],
        rows: list[tuple[Any, ...]],
        max_rows_in_prompt: int = 50,
    ) -> str:
        preview = rows[:max_rows_in_prompt]
        body = (
            f"用户问题：{user_question}\n\n"
            f"执行的 SQL：\n{sql}\n\n"
            f"列：{columns}\n"
            f"行（最多展示 {max_rows_in_prompt} 行）：\n{preview}"
        )
        if len(rows) > max_rows_in_prompt:
            body += f"\n… 共 {len(rows)} 行，其余已省略。"
        msgs = [
            SystemMessage(content=self.SYSTEM),
            HumanMessage(content=body),
        ]
        resp = self._llm.invoke(msgs)
        report_text = str(resp.content if hasattr(resp, "content") else resp).strip()

        tool_outputs = self._run_export_tools(
            user_question=user_question,
            sql=sql,
            columns=columns,
            rows=rows,
            report_text=report_text,
        )
        if tool_outputs:
            report_text = (
                f"{report_text}\n\n## 导出结果\n\n" + "\n".join(tool_outputs)
            )

        self._save_report_file(
            report_text=report_text,
            user_question=user_question,
            sql=sql,
            columns=columns,
            rows=rows,
        )
        return report_text
