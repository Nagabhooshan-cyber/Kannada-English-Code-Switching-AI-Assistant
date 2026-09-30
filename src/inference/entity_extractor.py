"""Conservative pattern-based entity extraction for common task-oriented intents."""

from __future__ import annotations

import re
from typing import Any

DATE_TERMS = {
    "today": ("today", "ivattu", "eevattu", "indu", "ಇಂದು", "ಇವತ್ತು"),
    "tomorrow": ("tomorrow", "naale", "nale", "nalle", "ನಾಳೆ"),
}
ACTIVITY_TERMS = (
    "assignment", "appointment", "class", "exam", "homework", "interview", "meeting",
    "office", "presentation", "project", "report", "shopping", "task", "work",
)
TOKEN_RE = re.compile(r"[\w]+(?:[-'][\w]+)*", flags=re.UNICODE)
TIME_RE = re.compile(
    r"(?<!\d)(?P<hour>\d{1,2})(?:(?::|\.)(?P<minute>[0-5]\d)|"
    r"\s*(?P<period>am|pm)\b|\s*(?P<unit>gantege|gantge|gante|ಗಂಟೆಗೆ)\b|"
    r"(?=\s*o'clock\b))(?P<post_period>am|pm)?\b",
    flags=re.IGNORECASE,
)
DESTINATION_RE = re.compile(r"(?<!\w)(?P<place>[\w]+)(?:-ge|\s+ge|ige)\b", flags=re.IGNORECASE | re.UNICODE)


def _find_term(text: str, terms: tuple[str, ...]) -> dict[str, Any] | None:
    for term in terms:
        match = re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, flags=re.IGNORECASE | re.UNICODE)
        if match:
            return {"value": term, "start": match.start(), "end": match.end()}
    return None


def _extract_date(text: str) -> dict[str, Any] | None:
    for value, terms in DATE_TERMS.items():
        found = _find_term(text, terms)
        if found:
            return {"value": value, "raw": text[found["start"] : found["end"]],
                    "start": found["start"], "end": found["end"]}
    return None


def _extract_time(text: str) -> dict[str, Any] | None:
    for match in TIME_RE.finditer(text):
        hour = int(match.group("hour"))
        period = match.group("period") or match.group("post_period")
        if hour > 23 or (period and not 1 <= hour <= 12):
            continue
        return {
            "raw": text[match.start() : match.end()].strip(),
            "hour": hour,
            "minute": int(match.group("minute")) if match.group("minute") else None,
            "period": period.lower() if period else None,
            "ambiguous": period is None and hour <= 12,
            "start": match.start(),
            "end": match.end(),
        }
    return None


def _extract_activity(text: str) -> dict[str, Any] | None:
    for term in ACTIVITY_TERMS:
        match = re.search(
            rf"(?<!\w){re.escape(term)}(?:[- ]?(?:ge|ige|alli|inda|na|annu))?(?!\w)",
            text,
            flags=re.IGNORECASE | re.UNICODE,
        )
        if match:
            return {"value": term, "raw": match.group(0), "start": match.start(), "end": match.end()}
    return None


def _extract_destination(text: str) -> dict[str, Any] | None:
    route_context = re.search(r"\b(?:hege\s+hog\w*|ellige\s+hog\w*|how\s+to\s+get\s+to|directions?\s+to)\b", text, re.IGNORECASE)
    if not route_context:
        return None
    match = DESTINATION_RE.search(text)
    if not match:
        return None
    return {"value": match.group("place"), "raw": match.group(0), "start": match.start(), "end": match.end()}


def _extract_topic(text: str) -> dict[str, Any] | None:
    topics = ("weather", "climate", "traffic", "schedule", "meeting", "bus", "train", "movie")
    found = _find_term(text, topics)
    if found:
        return {"value": found["value"], "start": found["start"], "end": found["end"]}
    return None


def extract_entities(text: str) -> dict[str, Any]:
    """Extract a small set of common entities without guessing missing details.

    Time values without AM/PM remain marked ambiguous. Unknown tokens are ignored.
    Character offsets refer to the original input string.
    """
    activity = _extract_activity(text)
    topic = _extract_topic(text)
    if activity and topic and activity["value"].casefold() == topic["value"].casefold():
        topic = None
    return {
        "date": _extract_date(text),
        "time": _extract_time(text),
        "activity": activity,
        "destination": _extract_destination(text),
        "topic": topic,
    }
