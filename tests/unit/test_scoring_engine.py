"""Exact-value tests for the scoring engine's components, thresholds and helpers."""

from __future__ import annotations

from typing import Any

import pytest

from callinsights import scoring
from callinsights.scoring import component_scores, outcome_for


def turns(*pairs: tuple[str, str]) -> list[scoring.Turn]:
    return [scoring.Turn(speaker=s, role=scoring._role_for(s), text=t) for s, t in pairs]


# --- parsing -------------------------------------------------------------------------------


def test_parse_upload_srt_and_plain() -> None:
    srt = "1\n00:00:01,000 --> 00:00:02,000\nAgent: Hello\n\n2\n00:00:03,000 --> 00:00:04,000\nProspect: Hi\n"
    assert scoring.parse_upload("call.SRT", srt) == "Agent: Hello\nProspect: Hi"
    # The same content in a .txt is left untouched (apart from outer whitespace).
    assert scoring.parse_upload("call.txt", srt).startswith("1\n00:00:01,000")
    vtt = "WEBVTT\nNOTE comment\n\n00:00:00.000 --> 00:00:01.000\nAgent: Yo\n"
    assert scoring.parse_upload("a.vtt", vtt) == "Agent: Yo"
    # srt-style numbering in a .vtt is not stripped, vtt headers in .srt are kept
    assert scoring.parse_upload("a.vtt", "1\nAgent: x") == "1\nAgent: x"
    assert scoring.parse_upload("a.srt", "WEBVTT\nAgent: x") == "WEBVTT\nAgent: x"
    assert scoring.parse_upload("", "  Agent: x  ") == "Agent: x"


# --- counting helpers ----------------------------------------------------------------------


def test_word_counts_ignore_other_speakers() -> None:
    t = turns(("Agent", "one two three"), ("Prospect", "four five"), ("Narrator", "six seven"))
    assert scoring._word_counts(t) == (3, 2)


def test_count_questions_only_agent() -> None:
    t = turns(("Agent", "a? b?"), ("Prospect", "c?"), ("Agent", "d"))
    assert scoring._count_questions(t) == 2


def test_objection_must_be_answered_later() -> None:
    before = turns(("Agent", "We offer a free trial."), ("Prospect", "It looks expensive."))
    assert scoring._objections(before) == (1, 0)
    after = turns(("Prospect", "It looks expensive."), ("Agent", "We offer a free trial."))
    assert scoring._objections(after) == (1, 1)
    echoed = turns(("Prospect", "I am worried."), ("Agent", "Why worried?"))
    assert scoring._objections(echoed) == (1, 1)
    prospect_only = turns(("Prospect", "Too expensive."), ("Prospect", "We offer a trial."))
    assert scoring._objections(prospect_only) == (1, 0)
    agent_objection = turns(("Agent", "Is it too expensive?"))
    assert scoring._objections(agent_objection) == (0, 0)
    two = turns(
        ("Prospect", "Too expensive."),
        ("Agent", "Fair point."),
        ("Prospect", "And the contract worries me."),
    )
    assert scoring._objections(two) == (2, 1)


def test_sentiment_exact() -> None:
    assert scoring._sentiment(turns(("Agent", "great great"))) == 0.0
    assert scoring._sentiment(turns(("Prospect", "great great bad"))) == pytest.approx(1 / 3)
    assert scoring._sentiment(turns(("Prospect", "bad"), ("Prospect", "ok ok ok"))) == -0.25
    assert scoring._sentiment(turns(("Prospect", "great"))) == 1.0


def test_mentions_and_next_step() -> None:
    t = turns(("Agent", "What budget?"), ("Prospect", "Our CFO decides by Q3."))
    assert scoring._mentions(t) == (True, True, True)
    assert scoring._mentions(turns(("Agent", "Maybe whenever."))) == (False, False, False)
    assert scoring._next_step_set(turns(("Agent", "Let's schedule a demo."))) is True
    assert scoring._next_step_set(turns(("Agent", "Bye."))) is False


# --- components ----------------------------------------------------------------------------


def comps(**kw: Any) -> dict[str, float]:
    base: dict[str, Any] = {
        "questions": 0,
        "talk_ratio": 0.5,
        "qualified": 0,
        "objections_found": 0,
        "objections_handled": 0,
        "sentiment": 0.0,
    }
    base.update(kw)
    return component_scores(**base)


@pytest.mark.parametrize(("q", "pts"), [(0, 0.0), (1, 2.5), (3, 7.5), (8, 20.0), (12, 20.0)])
def test_question_component(q: int, pts: float) -> None:
    assert comps(questions=q)["questions"] == pts


@pytest.mark.parametrize(
    ("ratio", "pts"),
    [(0.5, 20.0), (0.25, 10.0), (0.75, 10.0), (0.0, 0.0), (1.0, 0.0), (0.6, 16.0)],
)
def test_talk_balance_component(ratio: float, pts: float) -> None:
    assert comps(talk_ratio=ratio)["talk_balance"] == pytest.approx(pts)


@pytest.mark.parametrize(("n", "pts"), [(0, 0.0), (1, 20 / 3), (2, 40 / 3), (3, 20.0)])
def test_qualification_component(n: int, pts: float) -> None:
    assert comps(qualified=n)["qualification"] == pytest.approx(pts)


@pytest.mark.parametrize(
    ("found", "handled", "pts"), [(0, 0, 20.0), (2, 1, 10.0), (4, 3, 15.0), (3, 0, 0.0)]
)
def test_objection_component(found: int, handled: int, pts: float) -> None:
    assert comps(objections_found=found, objections_handled=handled)["objection_handling"] == pts


@pytest.mark.parametrize(
    ("sentiment", "pts"),
    [(0.0, 10.0), (0.1, 15.0), (0.2, 20.0), (0.5, 20.0), (-0.1, 5.0), (-0.2, 0.0), (-0.9, 0.0)],
)
def test_sentiment_component(sentiment: float, pts: float) -> None:
    assert comps(sentiment=sentiment)["sentiment"] == pytest.approx(pts)


def test_components_have_fixed_keys() -> None:
    assert list(comps()) == [
        "questions",
        "talk_balance",
        "qualification",
        "objection_handling",
        "sentiment",
    ]


@pytest.mark.parametrize(
    ("total", "outcome"),
    [
        (100, "Interested"),
        (75, "Interested"),
        (74, "Follow Up"),
        (50, "Follow Up"),
        (49, "Not Interested"),
        (0, "Not Interested"),
    ],
)
def test_outcome_thresholds(total: int, outcome: str) -> None:
    assert outcome_for(total) == outcome


# --- coaching thresholds -------------------------------------------------------------------


def tips(**kw: Any) -> list[str]:
    base: dict[str, Any] = {
        "talk_ratio": 0.5,
        "spoken": 100,
        "questions": 5,
        "budget": True,
        "timeline": True,
        "authority": True,
        "objections_found": 0,
        "objections_handled": 0,
        "sentiment": 0.0,
        "next_step": True,
        "total": 60,
    }
    base.update(kw)
    return scoring.coaching_tips(**base)


def test_no_tips_for_a_clean_mid_call() -> None:
    assert tips() == []


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.71, "Let the prospect speak more: you talked 71% of the time."),
        (0.7, None),
        (0.25, None),
        (0.24, "You only talked 24% of the time."),
    ],
)
def test_talk_share_tips(ratio: float, expected: str | None) -> None:
    out = tips(talk_ratio=ratio)
    if expected is None:
        assert out == []
    else:
        assert len(out) == 1
        assert out[0].startswith(expected)


def test_low_talk_share_needs_speech() -> None:
    assert tips(talk_ratio=0.0, spoken=0) == []
    assert tips(talk_ratio=0.0, spoken=1)[0].startswith("You only talked 0%")


@pytest.mark.parametrize(("q", "count"), [(2, 1), (3, 0)])
def test_question_tip_threshold(q: int, count: int) -> None:
    out = tips(questions=q)
    assert len(out) == count
    if count:
        assert out[0] == "Ask more discovery questions (you asked 2). Strong calls ask 5 or more."


def test_qualification_tips() -> None:
    assert tips(budget=False) == [
        "Qualify budget earlier: ask about budget range before the pitch."
    ]
    assert tips(timeline=False) == ["Establish a timeline: ask when they need a solution live."]
    assert tips(authority=False) == ["Confirm authority: ask who else weighs in on the decision."]


def test_objection_tip() -> None:
    assert tips(objections_found=2, objections_handled=2) == []
    out = tips(objections_found=3, objections_handled=1)
    assert out == [
        "Address objections directly: 2 raised concern(s) went unanswered. "
        "Name the concern and offer a concrete fix."
    ]


@pytest.mark.parametrize(("s", "count"), [(-0.02, 0), (-0.021, 1)])
def test_sentiment_tip_threshold(s: float, count: int) -> None:
    assert len(tips(sentiment=s)) == count


def test_next_step_and_strong_call_tips() -> None:
    assert tips(next_step=False) == [
        "Always close with a concrete next step: a scheduled follow-up, a proposal, or a trial."
    ]
    assert tips(total=74) == []
    assert tips(total=75) == [
        "Strong call: keep this structure. Replicate the discovery cadence and the explicit close."
    ]


# --- score_call integration of the parts --------------------------------------------------


def test_score_call_exact_breakdown() -> None:
    t = turns(
        ("Agent", "What budget do you have? When do you go live? Who decides?"),
        ("Prospect", "Great question, ten thousand, March, the CFO."),
    )
    r = scoring.score_call(t)
    assert r["questions_asked"] == 3
    assert r["agent_words"] == 12
    assert r["prospect_words"] == 7
    assert r["talk_ratio"] == round(12 / 19, 3)
    assert r["talk_ratio_agent_pct"] == 63
    assert r["sentiment"] == round(1 / 7, 3)
    assert r["components"] == {
        "questions": 7.5,
        "talk_balance": round((1 - abs(12 / 19 - 0.5) / 0.5) * 20, 1),
        "qualification": 20.0,
        "objection_handling": 20.0,
        "sentiment": 17.1,
    }
    raw = 7.5 + (1 - abs(12 / 19 - 0.5) / 0.5) * 20 + 20 + 20 + min(20, 10 + (1 / 7) * 50)
    assert r["score"] == round(raw)
    assert r["outcome"] == outcome_for(r["score"])
    assert r["highlights"][3] == {
        "label": "Objections handled",
        "met": True,
        "detail": "No objections raised",
    }
    assert r["coaching"] == [
        "Always close with a concrete next step: a scheduled follow-up, a proposal, or a trial.",
        "Strong call: keep this structure. Replicate the discovery cadence and the explicit close.",
    ]


def test_score_call_rounds_components_to_one_decimal() -> None:
    t = turns(("Agent", "Budget?"), ("Prospect", "ok fine"))
    r = scoring.score_call(t)
    assert r["components"]["qualification"] == 6.7
    assert r["components"]["talk_balance"] == 13.3
    assert r["talk_ratio"] == 0.333


def test_score_call_partial_objections_highlight() -> None:
    t = turns(
        ("Prospect", "It is too expensive and I am worried about the contract."),
        ("Agent", "Fair point."),
        ("Prospect", "Still not sure."),
    )
    r = scoring.score_call(t)
    assert (r["objections_found"], r["objections_handled"]) == (2, 1)
    assert r["highlights"][3]["met"] is False
    assert r["highlights"][3]["detail"] == "1/2 addressed"


def test_score_call_empty() -> None:
    r = scoring.score_call([])
    assert r["talk_ratio"] == 0.0
    assert r["score"] == (0 + 0 + 0 + 20 + 10)
    assert r["outcome"] == "Not Interested"
