"""SmartKB · 智能知识库问答助手 —— Streamlit Web 界面。

运行: streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st

from config import settings
from core.chain import build_rag_chain, to_langchain_history
from core.loader import SUPPORTED_EXTENSIONS, load_document, split_documents
from core.llm import get_embeddings, get_llm
from core.vectorstore import add_documents, clear_vectorstore, doc_count, get_vectorstore

SAMPLE_DOC = Path(__file__).parent / "data" / "samples" / "langchain_intro.md"

st.set_page_config(page_title="SmartKB 智能知识库问答", page_icon="📚", layout="centered")
st.title("📚 SmartKB 智能知识库问答助手")
st.caption("基于 LangChain 的 RAG（检索增强生成）应用 · 支持在线 / 离线大模型")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


@st.cache_resource(show_spinner="正在初始化模型与向量知识库…")
def init_components():
    embeddings = get_embeddings(settings)
    vs = get_vectorstore(settings, embeddings)
    llm = get_llm(settings)
    retriever = vs.as_retriever(search_kwargs={"k": settings.top_k})
    chain = build_rag_chain(llm, retriever)
    return vs, chain


try:
    vs, chain = init_components()
    kb_count = doc_count(vs)
except Exception as e:
    st.error(f"初始化失败：{e}")
    st.info("请检查项目根目录下的 .env 配置（可参考 .env.example），或直接使用左侧「加载示例文档」。")
    st.stop()

# ========================= 侧边栏 =========================
with st.sidebar:
    st.header("⚙️ 模型配置")
    st.markdown(f"**对话模型**\n\n`{settings.llm_display_name}`")
    st.markdown(f"**向量化模型**\n\n`{settings.embedding_display_name}`")

    st.header("📂 知识库")
    st.metric("已入库文档块", kb_count)

    uploaded_files = st.file_uploader(
        "上传知识文档",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        help=f"支持格式：{', '.join(sorted(SUPPORTED_EXTENSIONS))}",
    )

    if st.button("🧪 加载示例文档", use_container_width=True, help="一键导入内置的 LangChain 介绍文档"):
        docs = load_document(SAMPLE_DOC)
        splits = split_documents(docs, settings)
        added = add_documents(vs, splits)
        st.cache_resource.clear()
        st.success(f"示例文档已入库：{len(docs)} 段原文 → {added} 个文档块")
        st.rerun()

    if kb_count > 0 and st.button("🗑️ 清空知识库", use_container_width=True):
        clear_vectorstore(settings)
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.markdown(
        "**使用流程**：上传文档 → 自动切分并向量化入库 → 在下方输入框提问，"
        "回答内容均来自知识库，并可展开查看引用原文。"
    )

# ========================= 文档入库 =========================
if uploaded_files:
    for upload in uploaded_files:
        key = (upload.name, upload.size)
        if key in st.session_state.processed_files:
            continue
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(upload.name).suffix
        ) as tmp:
            tmp.write(upload.getvalue())
            tmp_path = tmp.name
        try:
            docs = load_document(tmp_path)
            for doc in docs:  # 元数据记录用户上传的原始文件名
                doc.metadata["source"] = upload.name
            splits = split_documents(docs, settings)
            added = add_documents(vs, splits)
            st.session_state.processed_files.add(key)
            st.toast(f"《{upload.name}》已入库：{added} 个文档块", icon="✅")
        except Exception as e:
            st.error(f"处理《{upload.name}》失败：{e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    st.rerun()

# ========================= 对话界面 =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 引用来源"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{i}]** `{src}`")

if question := st.chat_input("请输入关于知识库内容的问题，例如：什么是 RAG？"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
        history = to_langchain_history(
            [m for m in st.session_state.messages[:-1] if m["role"] in ("user", "assistant")]
        )
        with st.spinner("检索知识库并生成回答…"):
            result = chain.invoke({"input": question, "chat_history": history})
        answer = result["answer"]
        sources = sorted(
            {
                f"{d.metadata.get('source', '未知来源')}"
                + (f" · 第{d.metadata['page']}页" if d.metadata.get("page") is not None else "")
                for d in result.get("context", [])
            }
        )
        st.markdown(answer)
        if sources:
            with st.expander("📎 引用来源"):
                for i, src in enumerate(sources, 1):
                    st.markdown(f"**[{i}]** `{src}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()

if not kb_count:
    st.info("💡 知识库当前为空。可点击左侧「加载示例文档」快速体验，或上传自己的 TXT / Markdown / PDF / Word 文档。")
