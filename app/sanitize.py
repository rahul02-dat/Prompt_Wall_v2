"""
Phase 1: Character-level sanitization.

Strips invisible Unicode Tag block characters (U+E0000-U+E007F) used to
smuggle hidden ASCII payloads, plus other zero-width/invisible characters
commonly used for obfuscation. Also does basic homoglyph normalization
for a small set of high-risk lookalikes.

Reference: original project plan, "Character-Level Obfuscation, Emojis,
and Unicode Smuggling" + AWS/Cisco Unicode tag smuggling writeups.
"""
import re
import unicodedata

# Unicode Tag block: originally for invisible language tagging / flag
# emoji composition. Used by attackers to smuggle hidden ASCII text.
_UNICODE_TAG_RANGE = re.compile(r"[\U000E0000-\U000E007F]")

# Zero-width and other invisible/formatting characters sometimes used
# to break up filtered keywords or hide payloads.
_INVISIBLE_CHARS = re.compile(
    "["
    "\u200b"  # zero-width space
    "\u200c"  # zero-width non-joiner
    "\u200d"  # zero-width joiner
    "\u200e"  # left-to-right mark
    "\u200f"  # right-to-left mark
    "\ufeff"  # BOM / zero-width no-break space
    "\u2060"  # word joiner
    "\u180e"  # Mongolian vowel separator
    "]"
)

# Small, high-confidence homoglyph map. Not exhaustive — this is a
# cheap first line of defense, not a full confusables normalizer.
_HOMOGLYPHS = {
    "\u0430": "a",  # Cyrillic а
    "\u0435": "e",  # Cyrillic е
    "\u043e": "o",  # Cyrillic о
    "\u0440": "p",  # Cyrillic р
    "\u0441": "c",  # Cyrillic с
    "\u0456": "i",  # Cyrillic і
    "\u04bb": "h",  # Cyrillic һ
}


def _decode_unicode_tags(text: str) -> str:
    """
    Detect and decode any Unicode Tag block smuggling. These characters
    map 1:1 onto ASCII (subtract 0xE0000 to get the ASCII code point),
    so a hidden payload can be reconstructed even though it never
    displays visually.
    Returns the decoded hidden text if found (for logging), stripped
    from the original string either way.
    """
    hidden = ""
    for ch in text:
        cp = ord(ch)
        if 0xE0000 <= cp <= 0xE007F:
            ascii_cp = cp - 0xE0000
            if 0x20 <= ascii_cp <= 0x7E:
                hidden += chr(ascii_cp)
    return hidden


def sanitize_text(text: str) -> tuple[str, dict]:
    """
    Returns (cleaned_text, findings) where findings notes anything
    suspicious that was stripped, for logging/alerting upstream.
    """
    findings = {}

    hidden = _decode_unicode_tags(text)
    if hidden:
        findings["hidden_unicode_tag_payload"] = hidden

    cleaned = _UNICODE_TAG_RANGE.sub("", text)
    cleaned = _INVISIBLE_CHARS.sub("", cleaned)

    # NFKC normalization collapses many visual/compatibility variants
    cleaned = unicodedata.normalize("NFKC", cleaned)

    homoglyph_hits = sum(1 for ch in cleaned if ch in _HOMOGLYPHS)
    if homoglyph_hits:
        findings["homoglyph_substitutions"] = homoglyph_hits
        cleaned = "".join(_HOMOGLYPHS.get(ch, ch) for ch in cleaned)

    return cleaned, findings