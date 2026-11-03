"""RAG pipeline: retrieve relevant chunks, then generate an answer with the chat model."""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import Settings
from app.schemas import ChatResponse, HistoryMessage, Source
from app.services.llm import get_chat_model
from app.services.vectorstore import EmptyRetrievalError, VectorStoreManager

logger = logging.getLogger("chatbot.rag")

SYSTEM_PROMPT = (
    "You are a precise document question-answering assistant.\n"
    "Answer the user's question ONLY using the context below.\n"
    "If the context does not contain the answer, say you don't know based on the "
    "uploaded documents — do not invent facts.\n"
    "Cite the sources you used inline like [Source 1], [Source 2], etc.\n"
    "Keep the answer concise and directly responsive to the question."
)

SNIPPET_LEN = 280


def _block_text(block) -> str:
    """Extract plain text from one message content block."""
    if isinstance(block, str):
        return block
    if isinstance(block, dict):
        # Gemini 3.x emits {"type": "text", "text": ..., "extras": {"signature": ...}}.
        # Skip non-text parts (e.g. reasoning/thinking blocks).
        if block.get("type", "text") != "text":
            return ""
        text = block.get("text")
        return text if isinstance(text, str) else ""
    text = getattr(block, "text", None)
    return text if isinstance(text, str) else ""


def _content_to_text(content) -> str:
    """Normalize model output to a plain string.

    Gemini 3.x models return `AIMessage.content` as a list of content blocks
    (optionally with thought signatures) instead of a plain string. Stringifying
    that list leaks Python repr into the API response.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _block_text(content)
    if isinstance(content, (list, tuple)):
        parts = [text for text in (_block_text(block) for block in content) if text]
        return "\n".join(parts).strip()
    return str(content)


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

    logger.info(
        "retrieved %d chunk(s) for query=%r (index=%s namespace=%r)",
        len(chunks),
        query[:80],
        settings.pinecone_index or settings.pinecone_host,
        settings.pinecone_namespace,
    )

    if not chunks:
        raise EmptyRetrievalError(
            "Pinecone returned no matches even though "
            f"{store.count_documents()} document(s) are recorded locally. The vectors were "
            "probably written to a different index or namespace (or removed). Inspect "
            "GET /api/diagnostics to compare the manifest with Pinecone's vector counts."
        )

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
    answer_text = _content_to_text(result.content)

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
