import re
from dataclasses import dataclass
from typing import Any

import psycopg2
import psycopg2.extras

from text2sql.config import Settings, get_settings


class UnsafeSQLError(ValueError):
    """Raised when SQL fails static safety checks."""


_SELECT_START = re.compile(r"^\s*select\b", re.IGNORECASE | re.DOTALL)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|truncate|alter|create|grant|revoke|copy\s+from|;\s*select)\b",
    re.IGNORECASE,
)


def _assert_read_only_sql(sql: str) -> None:
    s = sql.strip()
    if not s:
        raise UnsafeSQLError("SQL 为空")
    if not _SELECT_START.search(s):
        raise UnsafeSQLError("仅允许以 SELECT 开头的只读查询")
    if ";" in s.rstrip(";").strip():
        raise UnsafeSQLError("不允许多条语句或分号拼接")
    if _FORBIDDEN.search(s):
        raise UnsafeSQLError("检测到可能的数据变更或危险关键字")


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    error: str | None = None


class QueryExecutor:
    """连接 PostgreSQL 执行只读 SQL。"""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()

    def execute(self, sql: str) -> QueryResult:
        _assert_read_only_sql(sql)
        try:
            conn = psycopg2.connect(
                self._settings.database_url,
                connect_timeout=self._settings.sql_timeout_seconds,
            )
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(f"SET statement_timeout = {self._settings.sql_timeout_seconds * 1000}")
                    cur.execute(sql)
                    rows = cur.fetchall()
                    cols = [d[0] for d in cur.description] if cur.description else []
                    tuples = [tuple(r[c] for c in cols) for r in rows]
                    return QueryResult(columns=cols, rows=tuples, error=None)
            finally:
                conn.close()
        except Exception as e:  # noqa: BLE001 — 执行层统一转为可展示错误
            return QueryResult(columns=[], rows=[], error=str(e))

    def fetch_schema_digest(self, table_filter: list[str] | None = None) -> str:
        """拉取 public 模式下表与列信息，供 Text2SQL 作为上下文。"""
        filter_clause = ""
        params: list[Any] = []
        if table_filter:
            filter_clause = " AND c.table_name = ANY(%s)"
            params.append(table_filter)
        sql = f"""
        SELECT c.table_name, c.column_name, c.data_type
        FROM information_schema.columns c
        WHERE c.table_schema = 'public'
        {filter_clause}
        ORDER BY c.table_name, c.ordinal_position
        """
        r = self.execute(sql)
        if r.error:
            return f"(无法读取 schema: {r.error})"
        lines: list[str] = []
        current = ""
        for row in r.rows:
            t, col, dtype = row[0], row[1], row[2]
            if t != current:
                current = t
                lines.append(f"\n表 {t}:")
            lines.append(f"  - {col} ({dtype})")
        return "\n".join(lines) if lines else "(public 下无表或无可读列)"
