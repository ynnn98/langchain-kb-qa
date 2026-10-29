"""向量知识库：基于 Chroma 的持久化向量存储与检索。"""
import hashlib
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config import Settings


def get_vectorstore(s: Settings, embeddings: Embeddings):
    """获取（或创建）持久化的 Chroma 向量库集合。"""
    from langchain_chroma import Chroma

    Path(s.chroma_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=s.collection_name,
        embedding_function=embeddings,
        persist_directory=s.chroma_dir,
    )


def add_documents(vs, splits: list[Document]) -> int:
    """向知识库写入文档块。

    以「内容 MD5」作为文档 ID，重复上传同一文件时自动覆盖旧条目，避免知识库膨胀。
    """
    if not splits:
        return 0
    ids = [hashlib.md5(doc.page_content.encode("utf-8")).hexdigest() for doc in splits]
    vs.add_documents(splits, ids=ids)
    return len(splits)


def doc_count(vs) -> int:
    """当前知识库中的文档块数量。"""
    return vs._collection.count()


def clear_vectorstore(s: Settings) -> None:
    """清空知识库：直接删除 Chroma 持久化目录。"""
    path = Path(s.chroma_dir)
    if path.exists():
        shutil.rmtree(path)
