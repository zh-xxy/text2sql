#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = REPO_ROOT.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_env_file(path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_wrapping_quotes(value.strip())
        loaded[key] = value
        os.environ.setdefault(key, value)

    return loaded


def describe_database_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or "(none)"
    port = parsed.port or "(default)"
    database = parsed.path.lstrip("/") or "(none)"
    user = parsed.username or "(none)"
    return f"{host}:{port}/{database} (user={user})"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the text2sql demo from a stable env file.",
    )
    parser.add_argument(
        "question",
        nargs="?",
        default="统计最近30天订单数",
        help="Natural-language question to send to Chatbot.",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV_FILE),
        help="Path to the env file to preload before importing text2sql.",
    )
    parser.add_argument(
        "--thread-id",
        default="demo",
        help="LangGraph thread id used for the session checkpoint.",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Print the schema digest and exit without calling the LLM.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the env file, database target, and model settings.",
    )
    parser.add_argument(
        "--disable-rag",
        action="store_true",
        help="Force RAG off for this run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON summary instead of the human-readable view.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    env_file = Path(args.env_file).expanduser().resolve()
    load_env_file(env_file)
    if args.disable_rag:
        os.environ["RAG_ENABLED"] = "false"

    if str(PACKAGE_PARENT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_PARENT))

    from text2sql.chatbot import Chatbot
    from text2sql.config import get_settings

    settings = get_settings()
    bot = Chatbot(thread_id=args.thread_id)

    if args.show_config:
        print(f"env_file: {env_file}")
        print(f"database: {describe_database_url(settings.database_url)}")
        print(f"llm_provider: {settings.llm_provider}")
        print(f"llm_model: {settings.llm_model}")
        print(f"rag_enabled: {settings.rag_enabled}")
        print()

    if args.schema_only:
        print(bot.refresh_schema())
        return 0

    result = bot.chat(args.question)
    summary = {
        "question": args.question,
        "route": result.get("route", ""),
        "sql": result.get("sql", ""),
        "columns": result.get("columns", []),
        "rows": result.get("rows", []),
        "error": result.get("error"),
        "analysis": result.get("analysis", ""),
        "rag_context": result.get("rag_context", []),
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0

    print(f"Question: {summary['question']}")
    print(f"Route: {summary['route'] or '(unknown)'}")
    print()
    print("SQL:")
    print(summary["sql"] or "(none)")
    print()
    if summary["error"]:
        print("Error:")
        print(summary["error"])
        print()
    if summary["rag_context"]:
        print("RAG Context:")
        for item in summary["rag_context"]:
            print(f"- {item}")
        print()
    print("Analysis:")
    print(summary["analysis"] or "(empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
