from __future__ import annotations
import json
import re
import shutil
from urllib import error as urlerror
from urllib import parse, request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from typing import cast

from text2sql.config import Settings, get_settings
from text2sql.llm_client import build_chat_model

ChunkMethod = Literal["fixed", "sentence", "paragraph", "semantic"]

try:
    from langchain_chroma import Chroma
except Exception:  # pragma: no cover
    Chroma = None  # type: ignore[assignment]

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:  # pragma: no cover
    OpenAIEmbeddings = None  # type: ignore[assignment]

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None  # type: ignore[assignment]

class ApiReranker:
    """OpenAI 兼容 rerank 客户端（优先调用 /rerank）。"""

    def __init__(self, *, model: str, base_url: str | None, api_key: str | None):
        self._model = model
        self._base_url = (base_url or "").rstrip("/")
        self._api_key = api_key or ""

    def predict(self, pairs: list[list[str]]) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        docs = [p[1] for p in pairs]
        if not self._base_url:
            return [_lexical_score(query, doc) for doc in docs]
        try:
            return self._predict_via_api(query=query, docs=docs)
        except Exception:
            # 接口失败时兜底，保证业务不中断
            return [_lexical_score(query, doc) for doc in docs]

    def _predict_via_api(self, *, query: str, docs: list[str]) -> list[float]:
        endpoint = f"{self._base_url}/rerank"
        body = json.dumps(
            {
                "model": self._model,
                "query": query,
                "documents": docs,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode("utf-8")
        except urlerror.HTTPError as e:
            detail = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"rerank 接口错误: {e.code} {detail}") from e
        data = json.loads(raw)
        return _extract_rerank_scores(data, doc_count=len(docs))


class LengthSafeEmbeddings:
    """为有严格 token 限制的 Embedding API 提供长度保护。"""

    def __init__(self, backend: Any, *, max_chars: int = 380):
        self._backend = backend
        self._max_chars = max(64, max_chars)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        safe_texts = [_truncate_for_embedding(t, max_chars=self._max_chars) for t in texts]
        return self._backend.embed_documents(safe_texts)

    def embed_query(self, text: str) -> list[float]:
        safe_text = _truncate_for_embedding(text, max_chars=self._max_chars)
        return self._backend.embed_query(safe_text)


@dataclass(frozen=True)
class RagResult:
    answer: str
    route: str
    contexts: list[Document]


class RagEngine:
    _ROUTER_SYSTEM = """你是查询路由器，只返回 JSON。
任务：把用户问题路由到 "sql" 或 "kb"。

规则：
1) 若问题明确要求数据库实时统计、订单明细、按字段筛选，优先 "sql"；
2) 若问题是政策、复盘、规划、培训、年度报告、未来年份(>=2026)经营结论，优先 "kb"；
3) 输出格式必须是 {"route":"sql"} 或 {"route":"kb"}。"""

    _QA_SYSTEM = """你是企业经营知识库问答助手。请基于给定检索片段回答：
- 先给简明结论，再给依据；
- 若证据不足要明确说明；
- 禁止编造不存在的事实。"""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        self._llm = build_chat_model(self._settings)
        self._kb_dir = self._resolve_path(self._settings.rag_kb_dir)
        self._store_dir = self._resolve_path(self._settings.rag_store_dir)
        self._require_rag_deps()
        embedding_api_key = (
            self._settings.rag_embedding_api_key or self._settings.llm_api_key or "not-set"
        )
        embedding_base_url = self._settings.rag_embedding_base_url or self._settings.llm_base_url
        self._embedding_max_chars = min(380, max(64, self._settings.rag_chunk_size))
        raw_embeddings = OpenAIEmbeddings(
            model=self._settings.rag_embedding_model,
            api_key=embedding_api_key,
            base_url=_join_openai_base_url(embedding_base_url),
            request_timeout=self._settings.rag_embedding_timeout_seconds,
            max_retries=self._settings.rag_embedding_max_retries,
            chunk_size=self._settings.rag_embedding_batch_size,
        )
        self._embeddings = LengthSafeEmbeddings(
            raw_embeddings,
            max_chars=self._embedding_max_chars,
        )
        self._reranker = ApiReranker(
            model=self._settings.rag_rerank_model,
            base_url=self._settings.rag_rerank_base_url or self._settings.llm_base_url,
            api_key=self._settings.rag_rerank_api_key or self._settings.llm_api_key,
        )

    @staticmethod
    def _require_rag_deps() -> None:
        if not all([Chroma, OpenAIEmbeddings, PdfReader]):
            raise RuntimeError(
                "RAG 依赖未安装，请先安装: langchain-chroma, langchain-openai, pypdf"
            )

    @staticmethod
    def _resolve_path(path: str) -> Path:
        p = Path(path).expanduser()
        if p.is_absolute():
            return p
        return (Path(__file__).resolve().parents[1] / p).resolve()

    def ensure_index(self, *, method: ChunkMethod | None = None, force_rebuild: bool = False) -> None:
        split_method = method or self._settings.rag_chunk_method
        index_dir = self._store_dir / split_method
        if force_rebuild and index_dir.exists():
            shutil.rmtree(index_dir, ignore_errors=True)
        if index_dir.exists() and self._index_has_data(split_method):
            return
        docs = self.load_kb_documents()
        chunks = split_documents(
            docs,
            method=split_method,
            chunk_size=self._settings.rag_chunk_size,
            overlap=self._settings.rag_chunk_overlap,
            embeddings=self._embeddings,
            embedding_max_chars=self._embedding_max_chars,
        )
        if not chunks:
            raise RuntimeError(
                "知识库切分后无可用 chunk，请检查 RAG_KB_DIR 下是否有非空 txt/pdf 文档"
            )
        self._precheck_embedding_api()
        index_dir.mkdir(parents=True, exist_ok=True)
        cast(Any, Chroma).from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=str(index_dir),
        )

    @property
    def embeddings(self) -> Any:
        return self._embeddings

    def _get_store(self, *, method: ChunkMethod | None = None) -> Any:
        split_method = method or self._settings.rag_chunk_method
        self.ensure_index(method=split_method)
        return cast(Any, Chroma)(
            persist_directory=str(self._store_dir / split_method),
            embedding_function=self._embeddings,
        )

    def _index_has_data(self, method: ChunkMethod) -> bool:
        index_dir = self._store_dir / method
        if not index_dir.exists():
            return False
        try:
            store = cast(Any, Chroma)(
                persist_directory=str(index_dir),
                embedding_function=self._embeddings,
            )
            collection = getattr(store, "_collection", None)
            if collection is None:
                return False
            count = int(collection.count())
            return count > 0
        except Exception:
            return False

    def route(self, question: str) -> str:
        if not self._settings.rag_enabled:
            return "sql"
        years = [int(y) for y in re.findall(r"\b(20\d{2})\b", question)]
        if years and max(years) >= self._settings.rag_router_year_threshold:
            return "kb"
        resp = self._llm.invoke(
            [
                SystemMessage(content=self._ROUTER_SYSTEM),
                HumanMessage(content=f"用户问题：{question}"),
            ]
        )
        raw = str(resp.content if hasattr(resp, "content") else resp)
        if '"route":"kb"' in raw.replace(" ", "").lower():
            return "kb"
        return "sql"

    def _precheck_embedding_api(self) -> None:
        try:
            self._embeddings.embed_query("ping")
        except Exception as e:
            raise RuntimeError(
                "Embedding API 连通性检查失败，请检查网络/网关地址/超时设置。"
                f" base_url={self._settings.rag_embedding_base_url or self._settings.llm_base_url},"
                f" timeout={self._settings.rag_embedding_timeout_seconds}s,"
                f" retries={self._settings.rag_embedding_max_retries},"
                f" batch={self._settings.rag_embedding_batch_size}. 详细错误: {e}"
            ) from e

    def ask(self, question: str) -> RagResult:
        store = self._get_store()
        candidates = store.similarity_search(question, k=self._settings.rag_top_k)
        ranked = rerank_documents(
            query=question,
            docs=candidates,
            reranker=self._reranker,
            top_n=self._settings.rag_top_n,
        )
        context_text = "\n\n".join(
            f"[片段{i + 1}] {d.page_content}" for i, d in enumerate(ranked)
        )
        prompt = f"用户问题：{question}\n\n检索上下文：\n{context_text}"
        resp = self._llm.invoke(
            [
                SystemMessage(content=self._QA_SYSTEM),
                HumanMessage(content=prompt),
            ]
        )
        answer = str(resp.content if hasattr(resp, "content") else resp).strip()
        return RagResult(answer=answer, route="kb", contexts=ranked)

    def load_kb_documents(self) -> list[Document]:
        docs: list[Document] = []
        if not self._kb_dir.exists():
            return docs
        for p in sorted(self._kb_dir.glob("**/*")):
            if p.suffix.lower() not in {".txt", ".pdf"} or not p.is_file():
                continue
            text = self._read_text_file(p) if p.suffix.lower() == ".txt" else self._read_pdf(p)
            if not text.strip():
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": str(p), "file_type": p.suffix.lower()},
                )
            )
        return docs

    @staticmethod
    def _read_text_file(path: Path) -> str:
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _read_pdf(path: Path) -> str:
        reader = cast(Any, PdfReader)(str(path))
        texts: list[str] = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)


def split_documents(
    docs: list[Document],
    *,
    method: ChunkMethod,
    chunk_size: int,
    overlap: int,
    embeddings: Any,
    embedding_max_chars: int | None = None,
) -> list[Document]:
    chunked: list[Document] = []
    for doc in docs:
        text = doc.page_content.strip()
        if not text:
            continue
        parts = split_text(
            text=text,
            method=method,
            chunk_size=chunk_size,
            overlap=overlap,
            embeddings=embeddings,
        )
        safe_parts = _split_for_embedding_limit(
            parts,
            max_chars=embedding_max_chars or chunk_size,
            overlap=min(overlap, max(0, (embedding_max_chars or chunk_size) // 8)),
        )
        for idx, part in enumerate(safe_parts):
            chunked.append(
                Document(
                    page_content=part,
                    metadata={**doc.metadata, "chunk_method": method, "chunk_index": idx},
                )
            )
    return chunked


def split_text(
    *,
    text: str,
    method: ChunkMethod,
    chunk_size: int,
    overlap: int,
    embeddings: Any,
) -> list[str]:
    if method == "fixed":
        return _fixed_chunks(text, chunk_size=chunk_size, overlap=overlap)
    if method == "sentence":
        return _sentence_chunks(text, chunk_size=chunk_size, overlap=overlap)
    if method == "paragraph":
        return _paragraph_chunks(text, chunk_size=chunk_size, overlap=overlap)
    return _semantic_chunks(text, chunk_size=chunk_size, overlap=overlap, embeddings=embeddings)


def _fixed_chunks(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    out: list[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(text), step):
        seg = text[i : i + chunk_size].strip()
        if seg:
            out.append(seg)
    return out


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;.\n])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def _sentence_chunks(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    sentences = _split_sentences(text)
    return _pack_units(sentences, chunk_size=chunk_size, overlap=overlap)


def _paragraph_chunks(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return _pack_units(paragraphs, chunk_size=chunk_size, overlap=overlap)


def _pack_units(units: list[str], *, chunk_size: int, overlap: int) -> list[str]:
    out: list[str] = []
    cur = ""
    for unit in units:
        candidate = f"{cur}\n{unit}".strip() if cur else unit
        if len(candidate) <= chunk_size:
            cur = candidate
            continue
        if cur:
            out.append(cur)
        if overlap > 0 and out:
            tail = out[-1][-overlap:]
            cur = f"{tail}\n{unit}".strip()
        else:
            cur = unit
    if cur:
        out.append(cur)
    return out


def _semantic_chunks(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    embeddings: Any,
) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) <= 2:
        return _sentence_chunks(text, chunk_size=chunk_size, overlap=overlap)
    vecs = embeddings.embed_documents(sentences)
    sims: list[float] = []
    for i in range(len(vecs) - 1):
        a, b = vecs[i], vecs[i + 1]
        denom = (_norm(a) * _norm(b)) or 1e-9
        sims.append(_dot(a, b) / denom)
    threshold = sorted(sims)[max(0, int(len(sims) * 0.25) - 1)] if sims else 0.6
    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = sims[i - 1] if i - 1 < len(sims) else 1.0
        if sim < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])
    units = [" ".join(g).strip() for g in groups if g]
    return _pack_units(units, chunk_size=chunk_size, overlap=overlap)


def rerank_documents(
    *,
    query: str,
    docs: list[Document],
    reranker: CrossEncoder,
    top_n: int,
) -> list[Document]:
    if not docs:
        return []
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(docs, scores, strict=False),
        key=lambda x: float(x[1]),
        reverse=True,
    )
    out: list[Document] = []
    for doc, score in ranked[:top_n]:
        meta = dict(doc.metadata)
        meta["rerank_score"] = float(score)
        out.append(Document(page_content=doc.page_content, metadata=meta))
    return out


def _join_openai_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = parse.urlparse(base_url)
    if not parsed.scheme:
        return base_url
    if parsed.path.rstrip("/").endswith("/v1"):
        return base_url
    trimmed = base_url.rstrip("/")
    return f"{trimmed}/v1"


def _extract_rerank_scores(payload: dict[str, Any], *, doc_count: int) -> list[float]:
    raw_results = payload.get("results") or payload.get("data") or []
    if not isinstance(raw_results, list):
        raise RuntimeError("rerank 响应格式异常：results/data 不是列表")
    scores = [0.0] * doc_count
    found = False
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        score = item.get("relevance_score", item.get("score"))
        if isinstance(idx, int) and 0 <= idx < doc_count and score is not None:
            scores[idx] = float(score)
            found = True
    if not found:
        raise RuntimeError("rerank 响应格式异常：未解析到任何分数")
    return scores


def _lexical_score(query: str, doc: str) -> float:
    q_tokens = {t for t in re.split(r"\W+", query.lower()) if t}
    d_tokens = {t for t in re.split(r"\W+", doc.lower()) if t}
    if not q_tokens or not d_tokens:
        return 0.0
    inter = len(q_tokens & d_tokens)
    union = len(q_tokens | d_tokens)
    return float(inter / union) if union else 0.0


def _truncate_for_embedding(text: str, *, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars]


def _split_for_embedding_limit(parts: list[str], *, max_chars: int, overlap: int) -> list[str]:
    out: list[str] = []
    for part in parts:
        text = part.strip()
        if not text:
            continue
        if len(text) <= max_chars:
            out.append(text)
            continue
        out.extend(_fixed_chunks(text, chunk_size=max_chars, overlap=overlap))
    return out


def _dot(a: list[float], b: list[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b, strict=False)))


def _norm(a: list[float]) -> float:
    return float(sum(x * x for x in a) ** 0.5)

