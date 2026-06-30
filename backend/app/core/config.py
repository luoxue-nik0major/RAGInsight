from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Environment
    environment: str = "development"

    # DeepSeek API
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # Database
    database_url: str = "sqlite+aiosqlite:///./raginsight.db"

    # ChromaDB
    chroma_db_path: str = "./chroma_db"
    chroma_db_test_path: str = "./chroma_db_test"

    # Embedding
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # Cache TTL (seconds)
    retrieval_cache_ttl: int = 300
    answer_cache_ttl: int = 600
    embedding_cache_size: int = 1000

    # Router mode: "heuristic" or "learned"
    router_mode: str = "heuristic"

    class Config:
        env_prefix = "RAGINSIGHT_"
        env_file = ".env"
        env_file_encoding = "utf-8"

        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return init_settings, env_settings, file_secret_settings, file_secret_settings


@lru_cache()
def get_settings() -> Settings:
    return Settings()
