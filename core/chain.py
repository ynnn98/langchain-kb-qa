"""RAG 问答链：检索增强生成 + 多轮对话历史感知（基于 LangChain LCEL 构建）。"""
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

# —— 第一步 Prompt：结合对话历史，把追问改写成独立完整的问题再检索 ——
CONTEXTUALIZE_SYSTEM_PROMPT = (
    "你是一个检索查询改写助手。给定对话历史和用户的最新提问，"
    "如果最新提问依赖对话历史中的内容（如使用了代词或省略），"
    "请将其改写为一个不依赖历史也能理解的独立问题；否则原样输出。"
    "注意：只输出改写后的问题本身，不要回答问题。"
)

# —— 第二步 Prompt：基于检索到的知识库上下文回答 ——
QA_SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。请仅根据下面提供的上下文内容回答用户问题，"
    "回答使用中文、条理清晰。\n"
    "如果上下文中没有足够信息，请如实回答\"根据知识库中的资料，我无法回答这个问题\"，"
    "禁止编造答案。\n\n"
    "上下文：\n{context}"
)


def build_rag_chain(llm, retriever) -> Runnable:
    """构建带历史感知检索的 RAG 链。

    输入: {"input": 用户问题, "chat_history": 消息列表}
    输出: {"answer": 回答, "context": 引用的文档块, ...}
    """
    # 1) 历史感知检索器：多轮对话时先改写问题，再去做向量检索
    history_aware_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, history_aware_prompt
    )

    # 2) 阅读理解链：把检索到的文档块填充(stuff)进 Prompt，交由大模型生成回答
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 3) 检索链：检索 -> 生成 串联成完整 RAG 流程
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def to_langchain_history(chat_history: list[dict]) -> list:
    """把 UI 层的消息记录 [{'role': 'user'|'assistant', ...}] 转成 LangChain 消息对象。"""
    from langchain_core.messages import AIMessage, HumanMessage

    messages = []
    for item in chat_history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    return messages


__all__ = ["build_rag_chain", "to_langchain_history"]
