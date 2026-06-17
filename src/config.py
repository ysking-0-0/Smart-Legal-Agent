"""
配置管理模块
"""
import os
from pathlib import Path
from typing import Optional
import yaml
from dotenv import load_dotenv

# 项目根目录（必须在 load_dotenv 前确定）
ROOT_DIR = Path(__file__).parent.parent

# 加载环境变量（显式指定路径，不依赖 cwd）
load_dotenv(ROOT_DIR / ".env")

CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
VECTOR_DB_DIR = ROOT_DIR / "vector_db"


class Config:
    """配置管理类"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置

        Args:
            config_path: 配置文件路径，默认为 config/config.yaml
        """
        if config_path is None:
            config_path = CONFIG_DIR / "config.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default=None):
        """
        获取配置值，支持点号分隔的嵌套键

        Args:
            key: 配置键，如 "llm.openai.model"
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    @property
    def app_name(self) -> str:
        return self.get("app.name", "法律RAG知识库")

    @property
    def data_dir(self) -> Path:
        return ROOT_DIR / self.get("document.data_dir", "./data")

    @property
    def vector_db_dir(self) -> Path:
        return ROOT_DIR / self.get("vector_store.chroma.persist_dir", "./vector_db")

    @property
    def collection_name(self) -> str:
        return self.get("vector_store.chroma.collection_name", "legal_documents")

    @property
    def embedding_provider(self) -> str:
        return self.get("vector_store.embedding.provider", "chromadb")

    @property
    def embedding_model(self) -> str:
        return self.get("vector_store.embedding.model_name", "text-embedding-3-small")

    @property
    def embedding_device(self) -> str:
        return self.get("vector_store.embedding.device", "cpu")

    @property
    def chunk_size(self) -> int:
        return self.get("document.chunk.size", 500)

    @property
    def chunk_overlap(self) -> int:
        return self.get("document.chunk.overlap", 100)

    @property
    def retrieval_top_k(self) -> int:
        return self.get("retrieval.top_k", 5)

    @property
    def llm_provider(self) -> str:
        return self.get("llm.provider", "openai")

    @property
    def openai_model(self) -> str:
        return self.get("llm.openai.model", "gpt-3.5-turbo")

    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")

    @property
    def openai_base_url(self) -> str:
        return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    @property
    def openai_request_timeout(self) -> int:
        return self.get("llm.openai.request_timeout", 60)

    @property
    def openai_max_retries(self) -> int:
        return self.get("llm.openai.max_retries", 3)

    @property
    def ollama_model(self) -> str:
        return self.get("llm.ollama.model", "qwen2:7b")

    @property
    def ollama_base_url(self) -> str:
        return self.get("llm.ollama.base_url", "http://localhost:11434")

    @property
    def agent_max_steps(self) -> int:
        return self.get("agent.max_steps", 6)

    @property
    def agent_verbose(self) -> bool:
        return self.get("agent.verbose", True)

    @property
    def agent_default_mode(self) -> str:
        return self.get("agent.default_mode", "agent")

    @property
    def web_search_enabled(self) -> bool:
        return self.get("web_search.enabled", True)

    @property
    def api_host(self) -> str:
        return self.get("api.host", "0.0.0.0")

    @property
    def api_port(self) -> int:
        return self.get("api.port", 8000)


# 全局配置实例
config = Config()
