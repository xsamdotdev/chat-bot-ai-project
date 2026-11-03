"""RAG pipeline: retrieve relevant chunks, then generate an answer with the chat model."""

import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import Settings
from app.schemas import ChatResponse, HistoryMessage, Source
from app.services.llm import get_chat_model
from app.services.vectorstore import EmptyRetrievalError, VectorStoreManager

logger = logging.getLogger("chatbot.rag")

SYSTEM_PROMPT_TEMPLATE = (
    "You are {name}, a friendly, professional female AI chatbot. "
    "You help users understand and answer questions about the documents they have uploaded.\n\n"
    "Identity and tone:\n"
    "- You know you are a chatbot (an AI assistant), not a human. If asked who you are, "
    "whether you are a bot/human, or what you can do, introduce yourself as {name}, a "
    "female AI assistant that answers questions about the user's uploaded documents.\n"
    "- Be warm, polite and concise. Use natural conversational language, but never claim "
    "to be human, to have a body, or to have personal feelings and experiences.\n"
    "- If the user greets you or makes small talk (hello, thanks, how are you), reply "
    "naturally and briefly, then invite them to ask about their documents.\n"
    "- Stay {name} in every reply; never adopt another name or persona.\n\n"
    "Answering questions:\n"
    "- For questions about the document content, answer using ONLY the context below.\n"
    "- If the context does not contain the answer, say you couldn't find it in the "
    "uploaded documents — do not invent facts.\n"
    "- Cite the sources you used inline like [Source 1], [Source 2], etc.\n"
    "- Keep answers concise and directly responsive to the question."
)

SNIPPET_LEN = 280


def build_system_prompt(settings: Settings) -> str:
    """Persona prompt for the assistant (name configurable via CHATBOT_NAME)."""
    return SYSTEM_PROMPT_TEMPLATE.format(name=settings.chatbot_display_name)


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
    document_count = store.count_documents()

    chunks = []
    if document_count:
        chunks = store.similarity_search_with_score(query, k=k)

    logger.info(
        "retrieved %d chunk(s) for query=%r (index=%s namespace=%r, %d document(s) recorded)",
        len(chunks),
        query[:80],
        settings.pinecone_index or settings.pinecone_host,
        settings.pinecone_namespace,
        document_count,
    )

    if document_count and not chunks:
        raise EmptyRetrievalError(
            "Pinecone returned no matches even though "
            f"{document_count} document(s) are recorded locally. The vectors were "
            "probably written to a different index or namespace (or removed). Inspect "
            "GET /api/diagnostics to compare the manifest with Pinecone's vector counts."
        )

    if chunks:
        context = _format_context(chunks)
        human = (
            f"Context from the uploaded documents:\n\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above."
        )
    else:
        # No documents yet: stay in character and guide the user instead of failing.
        human = (
            "No documents have been uploaded yet, so there is no document context.\n\n"
            f"Question: {query}\n\n"
            "Reply as your persona: greet the user or answer questions about who you are "
            "naturally. If the user is asking about documents, explain that nothing has "
            "been uploaded yet and invite them to upload a file first."
        )

    messages = [SystemMessage(content=build_system_prompt(settings))]
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

    return ChatResponse(
        answer=answer_text,
        sources=sources,
        provider=settings.provider,
        model=settings.chat_model,
        chatbot_name=settings.chatbot_display_name,
    )
