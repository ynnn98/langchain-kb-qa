"""离线冒烟测试：使用假模型（Fake LLM / Fake Embeddings）验证 RAG 全链路，不需要 API Key。

运行: python tests/test_smoke.py
"""
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_core.embeddings.fake import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from config import Settings
from core.chain import build_rag_chain
from core.loader import load_document, split_documents
from core.vectorstore import add_documents, clear_vectorstore, doc_count, get_vectorstore


def main() -> int:
    tmp_dir = tempfile.mkdtemp(prefix="smartkb_test_")
    s = Settings()
    s.chroma_dir = str(Path(tmp_dir) / "chroma_db")
    s.collection_name = "test_collection"

    try:
        # 1. 文档加载与切分
        sample = Path(__file__).parent.parent / "data" / "samples" / "langchain_intro.md"
        docs = load_document(sample)
        assert docs, "文档加载失败"
        splits = split_documents(docs, s)
        assert splits, "文档切分失败"
        print(f"[1/4] 文档加载与切分 OK：{len(docs)} 段原文 → {len(splits)} 个文档块")

        # 2. 向量入库（假 Embedding，本地无网络依赖）
        vs = get_vectorstore(s, DeterministicFakeEmbedding(size=64))
        added = add_documents(vs, splits)
        assert added == len(splits)
        assert doc_count(vs) == added
        # 重复入库应幂等（MD5 去重）
        add_documents(vs, splits)
        assert doc_count(vs) == added, "重复入库未去重"
        print(f"[2/4] 向量入库 OK：{doc_count(vs)} 个文档块，重复入库自动去重")

        # 3. 构建 RAG 链（假 LLM）
        llm = FakeListChatModel(responses=["根据知识库，RAG 是检索增强生成技术。"])
        chain = build_rag_chain(llm, vs.as_retriever(search_kwargs={"k": 2}))
        print("[3/4] RAG 链构建 OK")

        # 4. 端到端问答
        result = chain.invoke({"input": "什么是RAG？", "chat_history": []})
        assert result["answer"], "未返回回答"
        assert len(result["context"]) >= 1, "未检索到上下文"
        print(f"[4/4] 端到端问答 OK：answer={result['answer']!r}, 引用 {len(result['context'])} 个文档块")

        print("\n✅ 冒烟测试全部通过（离线模式，无需 API Key）")
        return 0
    finally:
        clear_vectorstore(s)
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
