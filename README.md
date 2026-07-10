# AI Security Gateway — Phase 0

Passthrough proxy with auth, rate limiting, and structured logging.
No security filtering yet — that's Phase 1.

## Run locally
```
pip install -r requirements.txt
cp .env.example .env   # fill in your ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

## Test
```
curl -X POST http://localhost:8000/v1/chat \
  -H "X-API-Key: dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

## Run via Docker
```
docker compose up --build
```
