from __future__ import annotations

import re


TRUSTED_SOURCES = {
    "reuters",
    "associated press",
    "ap news",
    "bbc",
    "the hindu",
    "indian express",
    "ndtv",
    "who",
    "unicef",
    "imf",
    "world bank",
}

LOW_TRUST_CUES = {
    "viral post",
    "whatsapp forward",
    "telegram channel",
    "anonymous source",
    "forwarded as received",
    "breaking!!!",
}

HIGH_RISK_CLAIM_CUES = {
    "shocking",
    "must share",
    "they don't want you to know",
    "secret",
    "cover-up",
    "miracle cure",
    "guaranteed",
    "urgent",
    "immediately",
}

MONTH_PATTERN = r"(january|february|march|april|may|june|july|august|september|october|november|december)"
DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", flags=re.IGNORECASE),
    re.compile(rf"\b{MONTH_PATTERN}\s+\d{{1,2}}(?:,\s*\d{{4}})?\b", flags=re.IGNORECASE),
    re.compile(r"\b\d{4}\b"),
]
ENTITY_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")


def _contains_any(text: str, cues: set[str]) -> list[str]:
    lowered = text.lower()
    return sorted(cue for cue in cues if cue in lowered)


def analyze_fact_check_signals(text: str) -> dict:
    normalized = str(text or "").strip()
    lowered = normalized.lower()

    trusted_mentions = _contains_any(lowered, TRUSTED_SOURCES)
    low_trust_mentions = _contains_any(lowered, LOW_TRUST_CUES)
    risk_cues = _contains_any(lowered, HIGH_RISK_CLAIM_CUES)

    matched_dates: list[str] = []
    for pattern in DATE_PATTERNS:
        matched_dates.extend(match.group(0) for match in pattern.finditer(normalized))

    unique_entities = sorted(set(match.group(0) for match in ENTITY_PATTERN.finditer(normalized)))
    entity_density = len(unique_entities)

    if trusted_mentions and not low_trust_mentions:
        source_signal = "trusted_mentioned"
    elif low_trust_mentions and not trusted_mentions:
        source_signal = "low_trust_cues"
    else:
        source_signal = "unknown"

    if matched_dates:
        date_signal = "dated_claim"
    elif any(token in lowered for token in {"today", "tonight", "yesterday", "tomorrow"}):
        date_signal = "relative_date_only"
    else:
        date_signal = "no_date_context"

    if entity_density >= 4:
        entity_signal = "entity_rich"
    elif entity_density >= 1:
        entity_signal = "some_entities"
    else:
        entity_signal = "no_clear_entities"

    risk_level = "high" if len(risk_cues) >= 2 or low_trust_mentions else "medium" if risk_cues else "low"

    recommended_checks = []
    if source_signal != "trusted_mentioned":
        recommended_checks.append("Verify the publisher or original reporting source.")
    if date_signal != "dated_claim":
        recommended_checks.append("Check the exact event date and whether the claim is being reshared out of context.")
    if entity_signal != "entity_rich":
        recommended_checks.append("Cross-check people, organizations, and places against primary sources.")
    if risk_level != "low":
        recommended_checks.append("Look for independent confirmation before trusting this claim.")

    return {
        "source_signal": source_signal,
        "trusted_mentions": trusted_mentions,
        "low_trust_mentions": low_trust_mentions,
        "date_signal": date_signal,
        "date_mentions": matched_dates[:5],
        "entity_signal": entity_signal,
        "named_entities": unique_entities[:8],
        "risk_cues": risk_cues,
        "risk_level": risk_level,
        "recommended_checks": recommended_checks,
        "retrieval_needed": source_signal != "trusted_mentioned" or risk_level != "low",
    }
