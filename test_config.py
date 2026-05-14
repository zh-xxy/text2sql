from config import get_settings

settings = get_settings()
print(f"Provider: {settings.llm_provider}")
print(f"Model: {settings.llm_model}")
print(f"DB URL: {settings.database_url}")
