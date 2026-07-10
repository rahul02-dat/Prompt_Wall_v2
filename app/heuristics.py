"""
Phase 1: Fast, deterministic heuristic scanning.

This is intentionally cheap and low-latency — it catches known,
unsophisticated patterns before the (slower) semantic classifier runs.
It will produce false negatives against novel attacks; that's expected,
it's not the only layer.
"""
import base64
import re

_PATTERNS = [
    ("instruction_override", re.compile(
        r"ignore (all |the )?(previous|prior|above) (instructions|prompts|rules)",
        re.IGNORECASE,
    )),
    ("instruction_override_2", re.compile(
        r"disregard (your |the )?(system prompt|previous instructions|rules)",
        re.IGNORECASE,
    )),
    ("role_override", re.compile(
        r"you are now (a |an )?(dan|jailbroken|unrestricted|free from)",
        re.IGNORECASE,
    )),
    ("exfil_directive", re.compile(
        r"(send|forward|post|upload|exfiltrate) .{0,40}(data|information|contents|secrets|keys) to",
        re.IGNORECASE,
    )),
    ("markdown_image_exfil", re.compile(
        r"!\[.*?\]\((https?://[^)\s]+)\)",
    )),
    ("system_prompt_leak", re.compile(
        r"(reveal|print|output|repeat) (your |the )?(system prompt|instructions)",
        re.IGNORECASE,
    )),
    ("tag_boundary_escape", re.compile(
        r"</\s*(untrusted|system|instructions?)\s*>",
        re.IGNORECASE,
    )),
]

_MIN_B64_LEN = 40  # ignore short incidental base64-looking tokens


def _find_suspicious_base64(text: str) -> list[str]:
    hits = []
    for match in re.finditer(r"[A-Za-z0-9+/]{%d,}={0,2}" % _MIN_B64_LEN, text):
        candidate = match.group(0)
        try:
            base64.b64decode(candidate, validate=True)
            hits.append(candidate[:20] + "...")
        except Exception:
            continue
    return hits


def scan_text(text: str) -> dict:
    """
    Returns a findings dict: {pattern_name: [matched strings]} plus
    a 'base64_blobs' entry if any decodable base64 payloads were found.
    Empty dict means clean.
    """
    findings: dict = {}

    for name, pattern in _PATTERNS:
        matches = pattern.findall(text)
        if matches:
            findings[name] = matches

    b64_hits = _find_suspicious_base64(text)
    if b64_hits:
        findings["base64_blobs"] = b64_hits

    return findings