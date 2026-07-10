import logging
import time
import uuid
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.config import settings
from app.auth import verify_api_key
from app.rate_limit import check_rate_limit
from app.llm_client import call_llm
from app.schemas import ChatRequest, ChatResponse
from app.logging_utils import setup_logging, log_event

setup_logging()
logger = logging.getLogger("gateway")

app = FastAPI(title="AI Security Gateway", version="0.1.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    request_id = str(uuid.uuid4())
    start = time.time()

    check_rate_limit(api_key)

    log_event(
        request_id=request_id,
        event="request_received",
        api_key_hash=api_key[:8],  # never log full key
        model=req.model,
        message_count=len(req.messages),
    )

    try:
        # NOTE: This is Phase 0 — pure passthrough, no security filtering yet.
        # Phase 1 will insert sanitization + guardrail scanning here.
        result = await call_llm(req)
    except Exception as e:
        log_event(request_id=request_id, event="llm_call_failed", error=str(e))
        raise HTTPException(status_code=502, detail="LLM call failed")

    latency_ms = (time.time() - start) * 1000
    log_event(
        request_id=request_id,
        event="request_completed",
        latency_ms=round(latency_ms, 2),
    )

    return ChatResponse(request_id=request_id, content=result)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
