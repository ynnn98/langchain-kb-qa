"""模型工厂：根据配置创建大模型（Chat Model）与向量化模型（Embeddings）。

支持两类接入方式：
1. openai_compatible —— 任意兼容 OpenAI 协议的在线服务
   （智谱 BigModel / DeepSeek / 通义 DashScope / Kimi / OpenAI / SiliconFlow 等）
2. ollama —— 本地离线大模型（Ollama）
"""
from config import Settings


def get_llm(s: Settings):
    """创建对话大模型实例。"""
    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=s.ollama_base_url,
            model=s.ollama_model,
            temperature=s.temperature,
        )

    if s.llm_provider == "openai_compatible":
        if not s.llm_api_key:
            raise ValueError(
                "尚未配置大模型 API Key。请复制 .env.example 为 .env，"
                "并填写 LLM_API_KEY（如智谱 / DeepSeek 的 Key）；"
                "离线使用可设置 LLM_PROVIDER=ollama。"
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=s.llm_model,
            api_key=s.llm_api_key,
            base_url=s.llm_api_base,
            temperature=s.temperature,
        )

    raise ValueError(f"不支持的 LLM_PROVIDER: {s.llm_provider}（可选 openai_compatible / ollama）")


def get_embeddings(s: Settings):
    """创建向量化（Embedding）模型实例。"""
    if s.embedding_provider == "huggingface":
        # 本地离线 Embedding，需安装 requirements-offline.txt
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=s.embedding_model)

    if s.embedding_provider == "openai_compatible":
        if not s.embedding_api_key_value:
            raise ValueError(
                "尚未配置 Embedding API Key。请在 .env 中填写 EMBEDDING_API_KEY"
                "（未填写时将复用 LLM_API_KEY）；离线使用可设置 EMBEDDING_PROVIDER=huggingface。"
            )
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=s.embedding_model,
            api_key=s.embedding_api_key_value,
            base_url=s.embedding_base_url,
            # 第三方 OpenAI 兼容接口需关闭 token 预切分，直接发送原始文本
            check_embedding_ctx_length=False,
        )

    raise ValueError(f"不支持的 EMBEDDING_PROVIDER: {s.embedding_provider}（可选 openai_compatible / huggingface）")
