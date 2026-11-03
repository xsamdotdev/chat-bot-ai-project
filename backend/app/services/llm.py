"""Provider-aware chat model factory (plus an offline echo model for `fake`)."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import Settings


class EchoChatModel(BaseChatModel):
    """Offline stub: answers with a deterministic message echoing the prompt.

    Exists so the full upload -> retrieve -> generate pipeline can be exercised
    without any external service or credentials. Dev/smoke tests only.
    """

    model_name: str = "fake-echo"

    @property
    def _llm_type(self) -> str:
        return "fake-echo"

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model_name}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        user_text = next((m.content for m in reversed(messages) if m.type == "human"), "")
        text = (
            "[fake-llm] Offline stub response.\n"
            f"Received prompt ({len(user_text)} chars). Here is the context you passed in:\n\n"
            f"{user_text}"
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def get_chat_model(settings: Settings) -> BaseChatModel:
    """Return the chat model for the configured provider."""
    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to backend/.env "
                "(or switch PROVIDER to 'ollama' or 'fake')."
            )
        kwargs: dict = {
            "model": settings.openai_chat_model,
            "api_key": settings.openai_api_key,
            "temperature": 0.1,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)

    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
            temperature=0.1,
        )

    if settings.provider == "mistral":
        from langchain_mistralai import ChatMistralAI

        if not settings.mistral_api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY is not set. Add it to backend/.env "
                "(or switch PROVIDER to 'openai', 'ollama' or 'fake')."
            )
        kwargs: dict = {
            "model": settings.mistral_chat_model,
            "mistral_api_key": settings.mistral_api_key,
            "temperature": 0.1,
        }
        if settings.mistral_base_url:
            kwargs["endpoint"] = settings.mistral_base_url
        return ChatMistralAI(**kwargs)

    if settings.provider == "fake":
        return EchoChatModel()

    raise RuntimeError(f"Unknown provider: {settings.provider!r}")
