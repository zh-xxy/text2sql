from langchain_openai import ChatOpenAI

from text2sql.config import Settings, get_settings, resolve_llm_base_url


def build_chat_model(settings: Settings | None = None) -> ChatOpenAI:
    """基于 LangChain OpenAI 兼容客户端，支持豆包 / DeepSeek / 千问等网关。"""
    s = settings or get_settings()
    base_url = resolve_llm_base_url(s)
    return ChatOpenAI(
        model=s.llm_model,
        api_key=s.llm_api_key or "not-set",
        base_url=base_url,
        temperature=0.1,
        timeout=120,
    )
