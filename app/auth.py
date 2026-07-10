from fastapi import Header, HTTPException

from app.config import settings


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """
    Zero-trust default: every request must present a valid key.
    MVP note: plaintext comparison against settings.valid_api_keys.
    Before production, swap to SHA-256 hashed key lookup against a DB,
    as called out in the original project plan (Phase 0 hardening item).
    """
    if x_api_key not in settings.valid_api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key
