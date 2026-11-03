"""Chat model factory: Google Gemini (plus an offline echo model for `fake`)."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import Settings


class EchoChatModel(BaseChatModel):
    """Offline stub: answers with a deterministic message echoing the prompt.

    Exists so the full upload -> retrieve -> generate pipeline can be exercised
    without any credentials. Tests/smoke checks only.
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
        reply = (
            "[fake-echo] No LLM is configured (PROVIDER=fake). "
            f"Your question was: {user_text}"
        )
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])


def get_chat_model(settings: Settings) -> BaseChatModel:
    """Return the chat model for the configured provider."""
    if settings.provider == "fake":
        return EchoChatModel()

    if settings.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.google_api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY (or GEMINI_API_KEY) is not set. Set it in the repo-root .env "
                "(Docker) or backend/.env (local uvicorn), then restart/recreate the container."
            )
        kwargs: dict = {
            "model": settings.google_chat_model,
            "google_api_key": settings.google_api_key,
            "temperature": 0.1,
        }
        if settings.google_base_url:
            kwargs["base_url"] = settings.google_base_url
        return ChatGoogleGenerativeAI(**kwargs)

    raise RuntimeError(f"Unknown LLM provider: {settings.provider!r}")
