import litellm

from app.config import settings
from app.schemas import ChatRequest


async def call_llm(req: ChatRequest) -> str:
    """
    Thin wrapper around LiteLLM so we're not locked into one provider.
    Phase 0: pure passthrough. Phase 1+ will insert sanitized/wrapped
    messages here instead of req.messages directly.
    """
    model = req.model or settings.default_model

    kwargs = {}
    if model.startswith("ollama/"):
        # No API key needed for local models — just point at the Ollama server.
        kwargs["api_base"] = settings.ollama_base_url

    response = await litellm.acompletion(
        model=model,
        messages=[m.model_dump() for m in req.messages],
        max_tokens=req.max_tokens,
        **kwargs,
    )

    return response["choices"][0]["message"]["content"]