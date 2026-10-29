"""SmartKB 命令行入口：不依赖浏览器，在终端完成「入库 + 问答」。

用法:
    python cli.py                          # 交互式问答
    python cli.py --add data/samples/langchain_intro.md   # 追加文档入库
    python cli.py --ask "什么是RAG？"       # 单次提问
    python cli.py --reset                  # 清空知识库
"""
import argparse
import sys

from config import settings
from core.chain import build_rag_chain
from core.loader import ingest_file
from core.llm import get_embeddings, get_llm
from core.vectorstore import clear_vectorstore, doc_count, get_vectorstore


def build_components():
    embeddings = get_embeddings(settings)
    vs = get_vectorstore(settings, embeddings)
    llm = get_llm(settings)
    retriever = vs.as_retriever(search_kwargs={"k": settings.top_k})
    chain = build_rag_chain(llm, retriever)
    return vs, chain


def ask(chain, question: str, history: list | None = None) -> dict:
    return chain.invoke({"input": question, "chat_history": history or []})


def main() -> int:
    parser = argparse.ArgumentParser(description="SmartKB 命令行知识库问答")
    parser.add_argument("--add", metavar="FILE", help="将指定文档(txt/md/pdf/docx)加入知识库")
    parser.add_argument("--ask", metavar="QUESTION", help="单次提问后退出")
    parser.add_argument("--reset", action="store_true", help="清空知识库")
    args = parser.parse_args()

    if args.reset:
        clear_vectorstore(settings)
        print(f"✅ 知识库已清空（{settings.chroma_dir}）")
        if not (args.add or args.ask):
            return 0

    if args.add:
        docs, splits = ingest_file(args.add, settings)
        print(f"✅ 《{args.add}》入库完成：{docs} 段原文 → {splits} 个文档块")

    try:
        vs, chain = build_components()
    except Exception as e:
        print(f"❌ 初始化失败：{e}", file=sys.stderr)
        return 1

    print(f"📚 知识库文档块数量：{doc_count(vs)}")
    print(f"🤖 对话模型：{settings.llm_display_name}")

    if args.ask:
        result = ask(chain, args.ask)
        print(f"\n🤖 回答：{result['answer']}\n")
        return 0

    print("\n进入交互问答（输入 q 退出）。建议先执行: python cli.py --add data/samples/langchain_intro.md\n")
    history = []
    while True:
        try:
            question = input("🧑‍💻 你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not question:
            continue
        if question.lower() in ("q", "quit", "exit"):
            print("再见！")
            break
        result = ask(chain, question, history)
        print(f"\n🤖 答：{result['answer']}")
        sources = {d.metadata.get("source", "?") for d in result.get("context", [])}
        if sources:
            print(f"📎 引用来源：{', '.join(sorted(sources))}\n")
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": result["answer"]},
        ]
    return 0


if __name__ == "__main__":
    sys.exit(main())
