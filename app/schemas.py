from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = Field(default=1000, le=4096)
    # Untrusted content the agent is grounding on: retrieved RAG docs,
    # scraped web pages, or tool-call outputs. Kept separate from
    # `messages` so the gateway can isolate/wrap it distinctly from
    # the user's direct instructions.
    untrusted_context: str | None = None
    untrusted_context_source: str = "retrieved_content"  # or "tool_output"


class ChatResponse(BaseModel):
    request_id: str
    content: str
    blocked: bool = False
    block_reason: str | None = None