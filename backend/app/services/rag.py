"""RAG pipeline: retrieve relevant chunks, then generate an answer with the chat model."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import Settings
from app.schemas import ChatResponse, HistoryMessage, Source
from app.services.llm import get_chat_model
from app.services.vectorstore import VectorStoreManager

SYSTEM_PROMPT = (
    "You are a precise document question-answering assistant.\n"
    "Answer the user's question ONLY using the context below.\n"
    "If the context does not contain the answer, say you don't know based on the "
    "uploaded documents — do not invent facts.\n"
    "Cite the sources you used inline like [Source 1], [Source 2], etc.\n"
    "Keep the answer concise and directly responsive to the question."
)

SNIPPET_LEN = 280


def _format_context(chunks) -> str:
    parts = []
    for i, (doc, _score) in enumerate(chunks, start=1):
        filename = doc.metadata.get("source", "unknown")
        chunk_index = doc.metadata.get("chunk_index")
        loc = f"{filename}, chunk {chunk_index}" if chunk_index is not None else filename
        parts.append(f"[Source {i}] ({loc})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _history_to_messages(history: list[HistoryMessage]):
    messages = []
    for msg in history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))
    return messages


def answer_query(
    *,
    query: str,
    settings: Settings,
    store: VectorStoreManager,
    top_k: int | None = None,
    history: list[HistoryMessage] | None = None,
) -> ChatResponse:
    k = top_k or settings.top_k
    chunks = store.similarity_search_with_score(query, k=k)

    context = _format_context(chunks)
    human = (
        f"Context from the uploaded documents:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above."
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(_history_to_messages(history or []))
    messages.append(HumanMessage(content=human))

    model = get_chat_model(settings)
    result = model.invoke(messages)
    answer_text = result.content if isinstance(result.content, str) else str(result.content)

    sources = [
        Source(
            doc_id=doc.metadata.get("doc_id"),
            filename=doc.metadata.get("source", "unknown"),
            chunk_index=doc.metadata.get("chunk_index"),
            page=doc.metadata.get("page"),
            snippet=doc.page_content[:SNIPPET_LEN],
            score=float(score) if score is not None else None,
        )
        for doc, score in chunks
    ]

    return ChatResponse(answer=answer_text, sources=sources, provider=settings.provider, model=settings.chat_model)
