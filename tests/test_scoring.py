"""Unit tests for app/scoring.py (pure, deterministic)."""

from app import scoring


def _turns(*pairs):
    return [{"speaker": s, "role": scoring._role_for(s), "text": t} for s, t in pairs]


# --- parse_turns -----------------------------------------------------------


def test_parse_turns_basic():
    text = "Agent: Hello there.\nProspect: Hi, I have a minute.\nAgent: Great."
    turns = scoring.parse_turns(text)
    assert len(turns) == 3
    assert turns[0] == {"speaker": "Agent", "role": "agent", "text": "Hello there."}
    assert turns[1]["role"] == "prospect"


def test_parse_turns_continuation_lines_merge():
    text = "Agent: First line\ncontinued here\nProspect: Reply."
    turns = scoring.parse_turns(text)
    assert len(turns) == 2
    assert turns[0]["text"] == "First line continued here"


def test_parse_turns_aliases():
    turns = scoring.parse_turns("Sales Rep: Hi.\nCustomer: Hello.\nMe: Yo.")
    assert [t["role"] for t in turns] == ["agent", "prospect", "agent"]


def test_parse_turns_unknown_speaker_is_other():
    turns = scoring.parse_turns("Narrator: Scene one.\nAgent: Hi.")
    assert turns[0]["role"] == "other"


def test_parse_turns_empty():
    assert scoring.parse_turns("") == []
    assert scoring.parse_turns("\n  \n") == []


# --- parse_upload ----------------------------------------------------------


VTT = """WEBVTT

00:00:00.000 --> 00:00:02.000
Agent: Hello there.

00:00:02.500 --> 00:00:04.000
Prospect: Hi, one minute.
"""

SRT = """1
00:00:00,000 --> 00:00:02,000
Agent: Hello there.

2
00:00:02,500 --> 00:00:04,000
Prospect: Hi, one minute.
"""


def test_parse_upload_vtt_strips_timestamps():
    out = scoring.parse_upload("call.vtt", VTT)
    assert "00:00" not in out
    assert "WEBVTT" not in out
    turns = scoring.parse_turns(out)
    assert len(turns) == 2
    assert turns[0]["text"] == "Hello there."


def test_parse_upload_srt_strips_numbers_and_timestamps():
    out = scoring.parse_upload("call.srt", SRT)
    assert "00:00" not in out
    turns = scoring.parse_turns(out)
    assert len(turns) == 2
    assert turns[1]["role"] == "prospect"


def test_parse_upload_txt_passthrough():
    raw = "Agent: Hi.\nProspect: Yo."
    assert scoring.parse_upload("call.txt", raw) == raw


# --- score_call components -------------------------------------------------


def _balanced_call(**over):
    """A decent call the test can mutate: 4 questions, balanced talk,
    budget+timeline+authority, no objections, next step, positive tone."""
    turns = _turns(
        ("Agent", "Hi, thanks for your time. What is your biggest challenge right now?"),
        ("Prospect", "Ticket volume is great pain, but your demo was good."),
        ("Agent", "How many tickets do you handle each week?"),
        ("Prospect", "About three hundred."),
        ("Agent", "What budget have you set aside for tooling like this?"),
        ("Prospect", "Around two thousand a month."),
        ("Agent", "When would you want this live? Who signs off on the decision?"),
        ("Prospect", "Next quarter, and our COO approves."),
        ("Agent", "Perfect. I will send a proposal and we can schedule a follow up Thursday."),
        ("Prospect", "Sounds great, I like the plan."),
    )
    return turns


def test_score_call_strong_call_scores_high():
    r = scoring.score_call(_balanced_call())
    assert r["score"] >= 75
    assert r["outcome"] == "Interested"
    assert r["budget_mentioned"] and r["timeline_mentioned"] and r["authority_mentioned"]
    assert r["next_step_set"]
    assert r["objections_found"] == 0


def test_score_call_outcome_thresholds():
    good = scoring.score_call(_balanced_call())
    assert good["outcome"] == "Interested"

    # A terrible call: agent monologue, no questions, nothing qualified.
    bad_turns = _turns(
        ("Agent", "Hi let me tell you about our platform. " * 20),
        ("Prospect", "Not interested really."),
        ("Agent", "It is the best platform. " * 20),
    )
    bad = scoring.score_call(bad_turns)
    assert bad["score"] < 50
    assert bad["outcome"] == "Not Interested"

    # Middle call: some discovery, partial qualification.
    mid_turns = _turns(
        ("Agent", "What is your biggest challenge?"),
        ("Prospect", "Volume is high."),
        ("Agent", "How many tickets per week?"),
        ("Prospect", "Three hundred."),
        ("Agent", "What budget range are we in?"),
        ("Prospect", "About a thousand."),
        ("Agent", "Our tool helps a lot with volume."),
        ("Prospect", "Okay."),
    )
    mid = scoring.score_call(mid_turns)
    assert 50 <= mid["score"] < 75
    assert mid["outcome"] == "Follow Up"


def test_score_call_talk_ratio_sweet_spot():
    even = _turns(
        ("Agent", "one two three four five six seven eight nine ten?"),
        ("Prospect", "one two three four five six seven eight nine ten."),
    )
    r = scoring.score_call(even)
    assert r["talk_ratio"] == 0.5
    assert r["components"]["talk_balance"] == 20.0

    mono = _turns(
        ("Agent", "word " * 90),
        ("Prospect", "word " * 10),
    )
    r2 = scoring.score_call(mono)
    assert r2["talk_ratio"] == 0.9
    assert r2["components"]["talk_balance"] < 5


def test_score_call_questions_capped_at_eight():
    turns = [("Agent", "Question number one?")] * 12 + [("Prospect", "Sure.")]
    r = scoring.score_call(_turns(*turns))
    assert r["questions_asked"] == 12
    assert r["components"]["questions"] == 20.0

    few = _turns(("Agent", "One question?"), ("Prospect", "Yes."))
    r2 = scoring.score_call(few)
    assert r2["components"]["questions"] == 2.5


def test_score_call_qualification_thirds():
    only_budget = _turns(
        ("Agent", "What is your budget?"),
        ("Prospect", "A thousand."),
    )
    r = scoring.score_call(only_budget)
    assert r["components"]["qualification"] == round(20 / 3, 1)


def test_score_call_objection_handled_vs_unhandled():
    handled = _turns(
        ("Agent", "How does pricing sound?"),
        ("Prospect", "It looks too expensive for us."),
        ("Agent", "Fair point. What if we start with a pilot so the ROI is proven first?"),
    )
    r = scoring.score_call(handled)
    assert r["objections_found"] == 1
    assert r["objections_handled"] == 1
    assert r["components"]["objection_handling"] == 20.0

    unhandled = _turns(
        ("Agent", "How does pricing sound?"),
        ("Prospect", "It looks too expensive for us."),
        ("Agent", "Let me show you the feature list instead."),
    )
    r2 = scoring.score_call(unhandled)
    assert r2["objections_found"] == 1
    assert r2["objections_handled"] == 0
    assert r2["components"]["objection_handling"] == 0.0


def test_score_call_sentiment_direction():
    pos = _turns(
        ("Agent", "Hi."),
        ("Prospect", "This is great, I love it, excellent work, fantastic."),
    )
    neg = _turns(
        ("Agent", "Hi."),
        ("Prospect", "This is terrible, awful, horrible, frustrating."),
    )
    assert scoring.score_call(pos)["sentiment"] > scoring.score_call(neg)["sentiment"]
    assert scoring.score_call(pos)["sentiment"] > 0
    assert scoring.score_call(neg)["sentiment"] < 0


# --- coaching rules --------------------------------------------------------


def test_coaching_talk_too_much():
    turns = _turns(
        ("Agent", "word " * 80 + "?"),
        ("Prospect", "word " * 10),
    )
    r = scoring.score_call(turns)
    assert any("talked 89%" in c for c in r["coaching"])


def test_coaching_few_questions():
    r = scoring.score_call(_turns(("Agent", "Hi there."), ("Prospect", "Hello.")))
    assert any("discovery questions" in c for c in r["coaching"])


def test_coaching_missing_qualification():
    r = scoring.score_call(_turns(("Agent", "Hi?"), ("Prospect", "Hi.")))
    joined = " ".join(r["coaching"])
    assert "budget" in joined and "timeline" in joined and "authority" in joined


def test_coaching_unhandled_objection_and_next_step():
    turns = _turns(
        ("Agent", "Hi?"),
        ("Prospect", "This seems too expensive."),
        ("Agent", "Okay."),
    )
    r = scoring.score_call(turns)
    joined = " ".join(r["coaching"])
    assert "objection" in joined
    assert "next step" in joined


def test_coaching_strong_call_gets_praise():
    r = scoring.score_call(_balanced_call())
    assert any("Strong call" in c for c in r["coaching"])


# --- highlights ------------------------------------------------------------


def test_highlights_checklist_shape():
    r = scoring.score_call(_balanced_call())
    labels = [h["label"] for h in r["highlights"]]
    assert labels == [
        "Budget discussed",
        "Timeline established",
        "Decision maker confirmed",
        "Objections handled",
        "Next step set",
    ]
    assert all(h["met"] for h in r["highlights"])


def test_highlights_unmet_and_detail():
    turns = _turns(
        ("Agent", "Hi?"),
        ("Prospect", "I am worried about the switching effort."),
        ("Agent", "Okay."),
    )
    r = scoring.score_call(turns)
    by_label = {h["label"]: h for h in r["highlights"]}
    assert by_label["Budget discussed"]["met"] is False
    assert by_label["Objections handled"]["met"] is False
    assert by_label["Objections handled"]["detail"] == "0/1 addressed"


def test_score_dict_schema():
    r = scoring.score_call(_balanced_call())
    for key in (
        "score", "outcome", "talk_ratio", "questions_asked", "budget_mentioned",
        "timeline_mentioned", "authority_mentioned", "objections_found",
        "objections_handled", "sentiment", "next_step_set", "highlights",
        "coaching", "components",
    ):
        assert key in r, key
    assert 0 <= r["score"] <= 100
    assert r["outcome"] in ("Interested", "Follow Up", "Not Interested")
    assert r["coaching"], "coaching should never be empty"
