from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = Field(default=1000, le=4096)


class ChatResponse(BaseModel):
    request_id: str
    content: str
