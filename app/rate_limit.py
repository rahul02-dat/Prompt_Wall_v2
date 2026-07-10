import time
from collections import defaultdict, deque
from fastapi import HTTPException

from app.config import settings

# MVP: in-memory sliding-window counter per API key.
# Not multi-process safe — swap for Redis before scaling beyond one worker.
_request_log: dict[str, deque] = defaultdict(deque)


def check_rate_limit(api_key: str) -> None:
    now = time.time()
    window_seconds = 60
    limit = settings.rate_limit_rpm

    log = _request_log[api_key]
    while log and now - log[0] > window_seconds:
        log.popleft()

    if len(log) >= limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    log.append(now)
