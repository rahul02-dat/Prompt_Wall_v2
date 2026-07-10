from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Comma-separated list of valid API keys (MVP only — move to a proper
    # secrets store / hashed DB lookup before production).
    gateway_api_keys: str = "dev-key-change-me"

    # Default LLM model routed via LiteLLM.
    # Local (no API key needed): "ollama/llama3.1" (requires Ollama running locally)
    # Cloud: "anthropic/claude-sonnet-5", "openai/gpt-4o", etc.
    default_model: str = "ollama/llama3.1"

    # Ollama server address (only used for ollama/* models)
    ollama_base_url: str = "http://localhost:11434"

    # Local semantic safety classifier (Phase 1). Pull with:
    # `ollama pull llama-guard3` — or swap for another guard model.
    guardrail_model: str = "ollama/llama-guard3"

    # If True, block requests when the classifier itself is unreachable
    # instead of failing open. Off by default for MVP dev ergonomics.
    guardrail_fail_closed: bool = False

    # Rate limit: requests per minute per API key
    rate_limit_rpm: int = 60

    # LiteLLM / provider credentials are picked up from standard env vars
    # (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY) — do not put keys here.

    class Config:
        env_file = ".env"

    @property
    def valid_api_keys(self) -> set[str]:
        return {k.strip() for k in self.gateway_api_keys.split(",") if k.strip()}


settings = Settings()