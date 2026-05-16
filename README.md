# text2sql

`text2sql` is a Python package-style repository that turns natural language
questions into read-only PostgreSQL queries, executes them, and returns a
Chinese analysis report. The current prototype combines LangGraph orchestration,
LangChain-compatible LLM clients, a PostgreSQL executor, optional RAG routing,
and skill-driven export hooks for analysis artifacts.

## Architecture

- `chatbot.py`: session wrapper that caches schema text and invokes the graph.
- `graph.py`: LangGraph workflow with SQL routing, execution, retry, and report
  generation.
- `text2sql_agent.py`: ReAct-style SQL generator that always tries to return a
  single `SELECT`.
- `query_executor.py`: read-only PostgreSQL executor plus schema introspection.
- `analyzer_agent.py`: turns result sets into Chinese Markdown reports and can
  call project-local export skills.
- `rag_engine.py`: optional knowledge-base routing and retrieval flow.
- `skill_runtime.py`: parses `skills/*/SKILL.md` and dispatches skill scripts.

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and fill in database and LLM settings.
4. Run from the parent directory of this package, or install it in editable
   mode, because modules import `text2sql.*`.

Example:

```bash
cd /home/ubuntu/my
python3 -m venv .venv
source .venv/bin/activate
pip install -r text2sql/requirements.txt
cp text2sql/.env.example text2sql/.env
python text2sql/scripts/run_demo.py --show-config --disable-rag "统计最近30天订单数"
```

The demo script preloads `text2sql/.env`, so it does not depend on your current
working directory when resolving settings.

## Current Gaps

- The repository does not currently include database bootstrap SQL.
- A working RAG flow still requires a populated `knowledge_base/` and reachable
  embedding / rerank endpoints.
