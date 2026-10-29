"""全局配置：从 .env 文件读取环境变量，集中管理模型与知识库参数。"""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    value = os.getenv(key, default)
    return value.strip() if isinstance(value, str) else default


@dataclass
class Settings:
    # ===== LLM（对话大模型）=====
    # openai_compatible: 任意兼容 OpenAI 接口的在线大模型（智谱 / DeepSeek / 通义 / Kimi / OpenAI ...）
    # ollama: 本地离线大模型（需先安装 Ollama 并拉取模型）
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai_compatible"))
    llm_api_base: str = field(default_factory=lambda: _env("LLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "glm-4-flash"))
    temperature: float = field(default_factory=lambda: float(_env("TEMPERATURE", "0.2")))

    # ===== Ollama 离线模型 =====
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:7b"))

    # ===== Embedding（向量化模型）=====
    # openai_compatible: 在线 Embedding 接口（默认复用 LLM 的 key 和地址，智谱 embedding-3 可直接使用）
    # huggingface: 本地离线 Embedding（BAAI/bge-small-zh-v1.5），需安装 requirements-offline.txt
    embedding_provider: str = field(default_factory=lambda: _env("EMBEDDING_PROVIDER", "openai_compatible"))
    embedding_api_base: str = field(default_factory=lambda: _env("EMBEDDING_API_BASE"))
    embedding_api_key: str = field(default_factory=lambda: _env("EMBEDDING_API_KEY"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "embedding-3"))

    # ===== 知识库 / 切分参数 =====
    chroma_dir: str = field(default_factory=lambda: _env("CHROMA_DIR", "./chroma_db"))
    collection_name: str = field(default_factory=lambda: _env("COLLECTION_NAME", "smartkb"))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "50")))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))

    @property
    def embedding_base_url(self) -> str:
        """Embedding 接口地址：未单独配置时复用 LLM 的地址。"""
        return self.embedding_api_base or self.llm_api_base

    @property
    def embedding_api_key_value(self) -> str:
        """Embedding API Key：未单独配置时复用 LLM 的 Key。"""
        return self.embedding_api_key or self.llm_api_key

    @property
    def llm_display_name(self) -> str:
        if self.llm_provider == "ollama":
            return f"Ollama / {self.ollama_model}（离线）"
        return f"{self.llm_api_base} / {self.llm_model}"

    @property
    def embedding_display_name(self) -> str:
        if self.embedding_provider == "huggingface":
            return f"本地 Embedding / {self.embedding_model}（离线）"
        return f"{self.embedding_base_url} / {self.embedding_model}"


settings = Settings()
