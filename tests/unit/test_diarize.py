from __future__ import annotations

import pytest

from callinsights.ai.asr import Segment
from callinsights.ai.diarize import (
    heuristic_roles,
    other,
    render_transcript,
    segments_to_transcript,
)


def seg(start: float, end: float, text: str = "x") -> Segment:
    return Segment(start, end, text)


def test_other() -> None:
    assert other("agent") == "prospect"
    assert other("prospect") == "agent"


def test_heuristic_switches_on_gap_at_threshold() -> None:
    segs = [seg(0, 1), seg(1.5, 2), seg(3, 4), seg(4, 5)]
    # gaps: 0.5 (no switch), 1.0 (switch, threshold is inclusive), 0.0 (no switch)
    assert heuristic_roles(segs, gap_seconds=1.0) == ["agent", "agent", "prospect", "prospect"]
    assert heuristic_roles(segs, gap_seconds=1.5) == ["agent"] * 4


def test_heuristic_first_speaker_and_alternation() -> None:
    segs = [seg(0, 1), seg(2, 3), seg(4, 5)]
    assert heuristic_roles(segs, 0.5, first="prospect") == ["prospect", "agent", "prospect"]
    assert heuristic_roles([], 0.5) == []


def test_heuristic_defaults() -> None:
    segs = [seg(0, 1), seg(2, 3), seg(3.5, 4)]
    assert heuristic_roles(segs) == ["agent", "prospect", "prospect"]


def test_render_merges_consecutive_and_skips_blank() -> None:
    segs = [seg(0, 1, " Hello "), seg(1, 2, "there."), seg(3, 4, "  "), seg(4, 5, "Hi!")]
    out = render_transcript(segs, ["agent", "agent", "prospect", "prospect"])
    assert out == "Agent: Hello there.\nProspect: Hi!"


def test_render_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        render_transcript([seg(0, 1)], [])


def test_segments_to_transcript_prefers_llm_roles() -> None:
    segs = [seg(0, 1, "a"), seg(1.0, 2, "b")]
    text, method = segments_to_transcript(segs, llm_roles=["agent", "prospect"])
    assert (text, method) == ("Agent: a\nProspect: b", "llm")


def test_segments_to_transcript_falls_back_on_bad_llm_roles() -> None:
    segs = [seg(0, 1, "a"), seg(5, 6, "b")]
    text, method = segments_to_transcript(segs, llm_roles=["agent"], gap_seconds=1.0)
    assert (text, method) == ("Agent: a\nProspect: b", "heuristic")
    text, method = segments_to_transcript(segs, gap_seconds=10.0, first="prospect")
    assert (text, method) == ("Prospect: a b", "heuristic")
    text, method = segments_to_transcript(segs)
    assert (text, method) == ("Agent: a\nProspect: b", "heuristic")
