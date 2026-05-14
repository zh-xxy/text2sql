from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


Provider = Literal["doubao", "deepseek", "qwen", "openai"]


# 常见 OpenAI 兼容网关默认地址（可通过环境变量覆盖）
_DEFAULT_BASE_URLS: dict[str, str] = {
    "doubao": "https://ark.cn-beijing.volces.com/api/v3",
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "openai": "https://api.openai.com/v1",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql://user:pass@localhost:5432/dbname",
        description="PostgreSQL 连接串",
    )

    llm_provider: Provider = Field(default="deepseek", description="doubao | deepseek | qwen | openai")
    llm_api_key: str = Field(default="", description="LLM API Key")
    llm_base_url: str | None = Field(default=None, description="覆盖默认网关")
    llm_model: str = Field(default="deepseek-chat", description="模型名")

    max_sql_retries: int = Field(default=3, ge=1, le=10)
    sql_timeout_seconds: int = Field(default=60, ge=1)

    # 可选：限制只暴露这些表给模型（逗号分隔）；空表示从库中拉取 public 表清单
    schema_allow_tables: str = Field(default="", description="e.g. orders,users")
    analyzer_skills_dir: str = Field(
        default="skills",
        description="skills 根目录，按 <skill>/SKILL.md 组织",
    )
    analyzer_enable_skill_tools: bool = Field(
        default=True,
        description="是否启用分析阶段的 skill 外部工具调用",
    )

    rag_enabled: bool = Field(default=True, description="是否启用 RAG 路由")
    rag_kb_dir: str = Field(default="knowledge_base", description="知识库文档目录")
    rag_store_dir: str = Field(default="outputs/rag_store", description="向量库目录")
    rag_chunk_method: Literal["fixed", "sentence", "paragraph", "semantic"] = Field(
        default="semantic",
        description="切分方式：fixed|sentence|paragraph|semantic",
    )
    rag_chunk_size: int = Field(default=500, ge=50, le=4000)
    rag_chunk_overlap: int = Field(default=80, ge=0, le=1000)
    rag_top_k: int = Field(default=12, ge=1, le=50, description="向量召回候选数")
    rag_top_n: int = Field(default=5, ge=1, le=20, description="重排后保留数")
    rag_embedding_model: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        description="Embedding API 模型名（建议填写服务商已托管模型）",
    )
    rag_rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3",
        description="Rerank API 模型名（建议填写服务商已托管模型）",
    )
    rag_embedding_api_key: str | None = Field(default=None, description="Embedding API Key")
    rag_embedding_base_url: str | None = Field(
        default=None,
        description="Embedding API Base URL（OpenAI 兼容）",
    )
    rag_embedding_timeout_seconds: int = Field(
        default=20,
        ge=3,
        le=300,
        description="Embedding API 超时秒数（避免长时间卡住）",
    )
    rag_embedding_max_retries: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Embedding API 最大重试次数",
    )
    rag_embedding_batch_size: int = Field(
        default=8,
        ge=1,
        le=128,
        description="Embedding 批量大小（越小越稳，越大越快）",
    )
    rag_rerank_api_key: str | None = Field(default=None, description="Rerank API Key")
    rag_rerank_base_url: str | None = Field(
        default=None,
        description="Rerank API Base URL（优先调用 /rerank）",
    )
    rag_rerank_fallback_model: str = Field(
        default="qwen-plus",
        description="当 /rerank 不可用时，回退使用的廉价聊天模型",
    )
    rag_router_year_threshold: int = Field(
        default=2026,
        description="查询年份 >= 该值时优先走知识库",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_llm_base_url(settings: Settings) -> str:
    if settings.llm_base_url:
        return settings.llm_base_url
    return _DEFAULT_BASE_URLS.get(settings.llm_provider, _DEFAULT_BASE_URLS["openai"])
