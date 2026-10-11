"""Turn unlabelled ASR segments into an ``Agent:``/``Prospect:`` transcript.

Whisper does not identify speakers. When an LLM is configured it labels each segment;
otherwise a deterministic heuristic assumes a speaker change at every pause of at least
``gap_seconds`` (two-party calls alternate). Heuristic labels are estimates: the transcript
can be corrected through the API/UI and the call re-scored.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from callinsights.ai.asr import Segment

Role = Literal["agent", "prospect"]
LABELS: dict[str, str] = {"agent": "Agent", "prospect": "Prospect"}


def other(role: Role) -> Role:
    return "prospect" if role == "agent" else "agent"


def heuristic_roles(
    segments: Sequence[Segment], gap_seconds: float = 0.8, first: Role = "agent"
) -> list[Role]:
    roles: list[Role] = []
    current = first
    for i, seg in enumerate(segments):
        if i > 0 and seg.start - segments[i - 1].end >= gap_seconds:
            current = other(current)
        roles.append(current)
    return roles


def render_transcript(segments: Sequence[Segment], roles: Sequence[Role]) -> str:
    """Merge consecutive segments by the same speaker into ``Speaker: text`` lines."""
    if len(segments) != len(roles):
        raise ValueError("segments and roles must have the same length")
    lines: list[tuple[Role, str]] = []
    for seg, role in zip(segments, roles, strict=True):
        text = seg.text.strip()
        if not text:
            continue
        if lines and lines[-1][0] == role:
            lines[-1] = (role, lines[-1][1] + " " + text)
        else:
            lines.append((role, text))
    return "\n".join(f"{LABELS[role]}: {text}" for role, text in lines)


def segments_to_transcript(
    segments: Sequence[Segment],
    *,
    llm_roles: Sequence[Role] | None = None,
    gap_seconds: float = 0.8,
    first: Role = "agent",
) -> tuple[str, str]:
    """Return ``(transcript, method)`` where method is ``"llm"`` or ``"heuristic"``."""
    if llm_roles is not None and len(llm_roles) == len(segments):
        return render_transcript(segments, llm_roles), "llm"
    return render_transcript(segments, heuristic_roles(segments, gap_seconds, first)), "heuristic"
