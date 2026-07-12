"""
Research R3 — cheap features for the proxy/student model.

These features are meant to cost orders of magnitude less than a full
MELON masked re-execution or ICON attention trace — think milliseconds
on CPU, not a second+ GPU forward pass. The research question is whether
a classifier trained on features this cheap can approximate the oracle
labels well enough to be useful.
"""
import re
from dataclasses import dataclass, asdict

# Reuse Phase 1 gateway heuristics as features rather than duplicating logic
from app.heuristics import scan_text
from app.sanitize import sanitize_text


@dataclass
class CheapFeatures:
    text_length: int
    heuristic_hit_count: int
    has_instruction_override: bool
    has_exfil_directive: bool
    has_markdown_image: bool
    has_base64_blob: bool
    sanitize_flag_count: int
    imperative_verb_density: float
    special_char_ratio: float
    uppercase_ratio: float


_IMPERATIVE_VERBS = {
    "ignore", "disregard", "forget", "override", "bypass", "reveal",
    "send", "forward", "execute", "run", "delete", "act", "pretend",
    "respond", "output", "print", "repeat", "disable",
}


def extract_features(text: str) -> CheapFeatures:
    heuristic_findings = scan_text(text)
    _, sanitize_findings = sanitize_text(text)

    words = re.findall(r"\b\w+\b", text.lower())
    imperative_count = sum(1 for w in words if w in _IMPERATIVE_VERBS)
    imperative_density = imperative_count / len(words) if words else 0.0

    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    special_ratio = special_chars / len(text) if text else 0.0

    upper_chars = sum(1 for c in text if c.isupper())
    upper_ratio = upper_chars / len(text) if text else 0.0

    return CheapFeatures(
        text_length=len(text),
        heuristic_hit_count=len(heuristic_findings),
        has_instruction_override="instruction_override" in heuristic_findings
        or "instruction_override_2" in heuristic_findings,
        has_exfil_directive="exfil_directive" in heuristic_findings,
        has_markdown_image="markdown_image_exfil" in heuristic_findings,
        has_base64_blob="base64_blobs" in heuristic_findings,
        sanitize_flag_count=len(sanitize_findings),
        imperative_verb_density=round(imperative_density, 4),
        special_char_ratio=round(special_ratio, 4),
        uppercase_ratio=round(upper_ratio, 4),
    )


def features_to_vector(features: CheapFeatures) -> list[float]:
    """Flatten to a numeric vector for sklearn-style classifiers."""
    d = asdict(features)
    return [
        d["text_length"],
        d["heuristic_hit_count"],
        float(d["has_instruction_override"]),
        float(d["has_exfil_directive"]),
        float(d["has_markdown_image"]),
        float(d["has_base64_blob"]),
        d["sanitize_flag_count"],
        d["imperative_verb_density"],
        d["special_char_ratio"],
        d["uppercase_ratio"],
    ]


FEATURE_NAMES = [
    "text_length", "heuristic_hit_count", "has_instruction_override",
    "has_exfil_directive", "has_markdown_image", "has_base64_blob",
    "sanitize_flag_count", "imperative_verb_density", "special_char_ratio",
    "uppercase_ratio",
]
