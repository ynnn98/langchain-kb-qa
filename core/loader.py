"""文档加载与切分：支持 TXT / Markdown / PDF / Word 四种格式。"""
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def load_document(file_path: str | Path) -> list[Document]:
    """根据文件扩展名选择合适的 Loader 读取文档。"""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式: {suffix}（支持 {', '.join(sorted(SUPPORTED_EXTENSIONS))}）")

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(path))
    elif suffix == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader

        loader = Docx2txtLoader(str(path))
    else:
        # txt / md：自动识别编码（兼容 Windows 下常见的 GBK 文件）
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(str(path), autodetect_encoding=True)

    docs = loader.load()
    # 统一补充元数据，便于前端展示引用来源
    for doc in docs:
        doc.metadata.setdefault("source", path.name)
    return docs


def split_documents(docs: list[Document], s: Settings) -> list[Document]:
    """递归字符切分，适合中文文本；相邻块保留重叠以保持上下文连贯。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", ", ", " ", ""],
    )
    return splitter.split_documents(docs)


def ingest_file(file_path: str | Path, s: Settings) -> tuple[int, int]:
    """加载并切分单个文件（供 CLI 使用；Web 端由 app.py 直接处理上传字节流）。"""
    from core.vectorstore import add_documents, get_vectorstore

    docs = load_document(file_path)
    splits = split_documents(docs, s)
    vs = get_vectorstore(s)
    add_documents(vs, splits)
    return len(docs), len(splits)
