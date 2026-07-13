# AI Security Gateway

An API gateway proxy for Large Language Models that adds authentication, rate limiting, structured logging, and robust security filtering.

## Features

- **Authentication & Rate Limiting**: Verifies API keys and enforces rate limits.
- **Character-Level Sanitization**: Strips invisible Unicode tags (used for smuggling), zero-width characters, and normalizes high-risk homoglyphs.
- **Deterministic Heuristics**: Fast regex-based scanning for known attacks (instruction overrides, data exfiltration, system prompt leaks) and suspicious base64 payloads.
- **Structural Isolation**: Wraps untrusted content in randomized XML-style tags to prevent boundary escape (prompt injection).
- **Semantic Guardrails**: Integration with local guard models (e.g., Llama Guard 3) via Ollama for deep semantic analysis of remaining inputs.

## Research & Evaluation

The repository includes research components to evaluate Prompt Injection and Hijacking vulnerabilities, specifically a tool-calling **MELON (Masked Extraction of Latent Objectives) Baseline Oracle** (`research/oracle/melon_baseline.py`).

- **Multi-Step Trajectory Evaluation**: Parses and evaluates structured JSON tool call sequences rather than just text.
- **Hijack Detection**: Uses the MELON method to mask user requests and determine if the model's plan is maliciously hijacked by untrusted content (indicated by high similarity between the original and masked plans).
- **Intelligent Similarity Scoring**: Employs robust token and JSON-step comparison, intentionally skipping the often-obvious first step of a sequence to avoid false positives.

## Run locally
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

## Test
```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "X-API-Key: dev-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hello"}]}'
```

## Run via Docker
```bash
docker compose up --build
```
