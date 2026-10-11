"""Deterministic call-scoring engine for CallInsights.

Pure functions: no DB, no network, no provider calls. Heavily unit-tested in
tests/unit/test_scoring.py. Summaries and narrative insights (LLM or local) live in
callinsights.ai.llm and are layered on top of these deterministic results.
"""

from __future__ import annotations

import re
from functools import cache
from typing import Any, TypedDict


class Turn(TypedDict):
    speaker: str
    role: str
    text: str


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

AGENT_SPEAKERS = {
    "agent",
    "rep",
    "sales",
    "sales rep",
    "salesrep",
    "sdr",
    "ae",
    "caller",
    "consultant",
    "advisor",
    "me",
}
PROSPECT_SPEAKERS = {
    "prospect",
    "customer",
    "client",
    "lead",
    "contact",
    "buyer",
    "them",
    "user",
}

BUDGET_WORDS = {
    "budget",
    "price",
    "pricing",
    "cost",
    "afford",
    "expensive",
    "quote",
    "quotation",
    "spend",
    "per-seat",
    "per seat",
    "invoice",
}
TIMELINE_WORDS = {
    "timeline",
    "deadline",
    "when",
    "quarter",
    "q1",
    "q2",
    "q3",
    "q4",
    "month",
    "week",
    "asap",
    "launch",
    "go-live",
    "go live",
    "rollout",
    "by end",
    "september",
    "october",
    "november",
    "december",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
}
AUTHORITY_WORDS = {
    "decision",
    "decide",
    "approver",
    "approval",
    "approve",
    "sign off",
    "sign-off",
    "boss",
    "manager",
    "ceo",
    "cto",
    "cfo",
    "coo",
    "owner",
    "stakeholder",
    "final say",
}
OBJECTION_WORDS = {
    "expensive",
    "too much",
    "cheaper",
    "competitor",
    "contract",
    "locked",
    "bad time",
    "not sure",
    "worried",
    "worry",
    "concern",
    "risk",
    "skeptical",
    "skeptic",
    "too pricey",
    "over budget",
    "no budget",
    "learning curve",
    "switching",
    "downtime",
    "too long",
    "not ready",
}
RESOLUTION_WORDS = {
    "understand",
    "great question",
    "fair",
    "fair point",
    "what if",
    "discount",
    "payment",
    "trial",
    "guarantee",
    "pilot",
    "offer",
    "address",
    "hear you",
    "makes sense",
    "roi",
    "save",
    "training",
    "onboarding",
    "walk away",
    "risk-free",
    "risk free",
    "success manager",
}
NEXT_STEP_WORDS = {
    "next step",
    "follow up",
    "follow-up",
    "schedule",
    "book",
    "send proposal",
    "proposal",
    "demo",
    "trial",
    "call back",
    "callback",
    "meeting",
    "calendar",
    "invite",
    "poc",
    "pilot",
    "sign up",
    "signup",
}
POSITIVE_WORDS = {
    "great",
    "good",
    "excellent",
    "love",
    "like",
    "perfect",
    "awesome",
    "happy",
    "impressed",
    "helpful",
    "works",
    "working",
    "interested",
    "excited",
    "fantastic",
    "wonderful",
    "glad",
    "pleased",
}
NEGATIVE_WORDS = {
    "bad",
    "terrible",
    "awful",
    "hate",
    "dislike",
    "horrible",
    "frustrated",
    "frustrating",
    "annoying",
    "problem",
    "issue",
    "broken",
    "slow",
    "worst",
    "disappointed",
    "concerned",
    "worried",
    "unhappy",
    "difficult",
    "pain",
}

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_SPEAKER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _\-.']{0,30}?)\s*:\s*(.*)$")
_VTT_TS_RE = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}")
_SRT_TS_RE = re.compile(r"^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}")
_SRT_NUM_RE = re.compile(r"^\d{1,5}$")


def _role_for(speaker: str) -> str:
    s = speaker.strip().lower()
    if s in AGENT_SPEAKERS:
        return "agent"
    if s in PROSPECT_SPEAKERS:
        return "prospect"
    return "other"


def parse_upload(filename: str, raw: str) -> str:
    """Strip .vtt/.srt caption formatting so parse_turns sees clean lines."""
    name = (filename or "").lower()
    lines = raw.splitlines()
    out: list[str] = []
    for line in lines:
        s = line.strip()
        if (
            name.endswith(".vtt")
            and (s in ("", "WEBVTT") or s.startswith("NOTE") or _VTT_TS_RE.match(s))
        ) or (name.endswith(".srt") and (s == "" or _SRT_TS_RE.match(s) or _SRT_NUM_RE.match(s))):
            continue
        out.append(line)
    # Remove speaker-less cue duplicates: keep every line; parse_turns merges.
    return "\n".join(out).strip()


def parse_turns(text: str) -> list[Turn]:
    """Parse "Agent: ..."/"Prospect: ..." transcript lines into turns.

    Returns list of {"speaker", "role", "text"}. Lines without a speaker
    prefix continue the previous turn; leading stray lines are dropped.
    """
    turns: list[Turn] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _SPEAKER_RE.match(line)
        if m:
            speaker, utter = m.group(1).strip(), m.group(2).strip()
            if not utter:
                continue
            turns.append(Turn(speaker=speaker, role=_role_for(speaker), text=utter))
        elif turns:
            turns[-1]["text"] += " " + line
    return turns


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


@cache
def _phrase_re(phrase: str) -> re.Pattern[str]:
    # Whole-word/phrase match so "may" does not fire on "maybe" or "when" on "whenever".
    return re.compile(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])")


def _contains_any(text: str, words: set[str]) -> str | None:
    """Return the longest lexicon phrase present in ``text`` as a whole word, else None."""
    low = text.lower()
    for w in sorted(words, key=lambda x: (-len(x), x)):
        if _phrase_re(w).search(low):
            return w
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _count_questions(turns: list[Turn]) -> int:
    return sum(t["text"].count("?") for t in turns if t["role"] == "agent")


def _word_counts(turns: list[Turn]) -> tuple[int, int]:
    """Words spoken by the agent and by the prospect (other speakers are ignored)."""
    agent = sum(len(_words(t["text"])) for t in turns if t["role"] == "agent")
    prospect = sum(len(_words(t["text"])) for t in turns if t["role"] == "prospect")
    return agent, prospect


def _mentions(turns: list[Turn]) -> tuple[bool, bool, bool]:
    full = " ".join(t["text"] for t in turns).lower()
    return (
        _contains_any(full, BUDGET_WORDS) is not None,
        _contains_any(full, TIMELINE_WORDS) is not None,
        _contains_any(full, AUTHORITY_WORDS) is not None,
    )


def _objections(turns: list[Turn]) -> tuple[int, int]:
    """Return ``(found, handled)``.

    A prospect turn containing an objection counts as handled when a *later* agent turn
    repeats the concern or uses a resolution phrase.
    """
    found = 0
    handled = 0
    for i, turn in enumerate(turns):
        if turn["role"] != "prospect":
            continue
        kw = _contains_any(turn["text"], OBJECTION_WORDS)
        if not kw:
            continue
        found += 1
        later_agent = " ".join(t["text"] for t in turns[i + 1 :] if t["role"] == "agent").lower()
        if _phrase_re(kw).search(later_agent) or _contains_any(later_agent, RESOLUTION_WORDS):
            handled += 1
    return found, handled


def _sentiment(turns: list[Turn]) -> float:
    """Net sentiment of prospect speech in [-1, 1]: (positive - negative) / words."""
    words: list[str] = []
    for t in turns:
        if t["role"] == "prospect":
            words.extend(_words(t["text"]))
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    return (pos - neg) / len(words)


def _next_step_set(turns: list[Turn]) -> bool:
    full = " ".join(t["text"] for t in turns).lower()
    return _contains_any(full, NEXT_STEP_WORDS) is not None


# Each component is worth up to 20 points; the total is their sum (0-100).
COMPONENT_MAX = 20.0
QUESTION_TARGET = 8
INTERESTED_AT = 75
FOLLOW_UP_AT = 50


def component_scores(
    questions: int,
    talk_ratio: float,
    qualified: int,
    objections_found: int,
    objections_handled: int,
    sentiment: float,
) -> dict[str, float]:
    """The five 0-20 point components, unrounded.

    ``qualified`` is how many of budget / timeline / authority (0-3) came up.
    """
    return {
        "questions": min(questions, QUESTION_TARGET) / QUESTION_TARGET * COMPONENT_MAX,
        "talk_balance": max(0.0, 1 - abs(talk_ratio - 0.5) / 0.5) * COMPONENT_MAX,
        "qualification": qualified / 3 * COMPONENT_MAX,
        "objection_handling": (
            objections_handled / objections_found * COMPONENT_MAX
            if objections_found
            else COMPONENT_MAX
        ),
        "sentiment": max(0.0, min(COMPONENT_MAX, 10 + sentiment * 50)),
    }


def outcome_for(total: int) -> str:
    if total >= INTERESTED_AT:
        return "Interested"
    if total >= FOLLOW_UP_AT:
        return "Follow Up"
    return "Not Interested"


def coaching_tips(
    *,
    talk_ratio: float,
    spoken: int,
    questions: int,
    budget: bool,
    timeline: bool,
    authority: bool,
    objections_found: int,
    objections_handled: int,
    sentiment: float,
    next_step: bool,
    total: int,
) -> list[str]:
    tips: list[str] = []
    pct = round(talk_ratio * 100)
    if talk_ratio > 0.7:
        tips.append(
            f"Let the prospect speak more: you talked {pct}% of the time. "
            "Aim for a 40-60% talk share."
        )
    elif talk_ratio < 0.25 and spoken > 0:
        tips.append(
            f"You only talked {pct}% of the time. Guide the conversation with "
            "targeted discovery questions."
        )
    if questions < 3:
        tips.append(
            f"Ask more discovery questions (you asked {questions}). Strong calls ask 5 or more."
        )
    if not budget:
        tips.append("Qualify budget earlier: ask about budget range before the pitch.")
    if not timeline:
        tips.append("Establish a timeline: ask when they need a solution live.")
    if not authority:
        tips.append("Confirm authority: ask who else weighs in on the decision.")
    if objections_found > objections_handled:
        tips.append(
            f"Address objections directly: {objections_found - objections_handled} raised "
            "concern(s) went unanswered. Name the concern and offer a concrete fix."
        )
    if sentiment < -0.02:
        tips.append(
            "Watch the prospect's tone: acknowledge their frustration explicitly before moving on."
        )
    if not next_step:
        tips.append(
            "Always close with a concrete next step: a scheduled follow-up, a proposal, or a trial."
        )
    if total >= INTERESTED_AT:
        tips.append(
            "Strong call: keep this structure. Replicate the discovery cadence "
            "and the explicit close."
        )
    return tips


def score_call(turns: list[Turn]) -> dict[str, Any]:
    """Score a parsed call. Returns a dict with score 0-100 and all parts."""
    questions = _count_questions(turns)
    agent_w, prospect_w = _word_counts(turns)
    spoken = agent_w + prospect_w
    talk_ratio = agent_w / spoken if spoken else 0.0
    budget, timeline, authority = _mentions(turns)
    obj_found, obj_handled = _objections(turns)
    sentiment = _sentiment(turns)
    next_step = _next_step_set(turns)

    parts = component_scores(
        questions,
        talk_ratio,
        int(budget) + int(timeline) + int(authority),
        obj_found,
        obj_handled,
        sentiment,
    )
    total = round(sum(parts.values()))
    highlights: list[dict[str, Any]] = [
        {"label": "Budget discussed", "met": budget},
        {"label": "Timeline established", "met": timeline},
        {"label": "Decision maker confirmed", "met": authority},
        {
            "label": "Objections handled",
            "met": obj_handled == obj_found,
            "detail": (
                f"{obj_handled}/{obj_found} addressed" if obj_found else "No objections raised"
            ),
        },
        {"label": "Next step set", "met": next_step},
    ]
    coaching = coaching_tips(
        talk_ratio=talk_ratio,
        spoken=spoken,
        questions=questions,
        budget=budget,
        timeline=timeline,
        authority=authority,
        objections_found=obj_found,
        objections_handled=obj_handled,
        sentiment=sentiment,
        next_step=next_step,
        total=total,
    )
    return {
        "score": total,
        "outcome": outcome_for(total),
        "talk_ratio": round(talk_ratio, 3),
        "talk_ratio_agent_pct": round(talk_ratio * 100),
        "questions_asked": questions,
        "agent_words": agent_w,
        "prospect_words": prospect_w,
        "budget_mentioned": budget,
        "timeline_mentioned": timeline,
        "authority_mentioned": authority,
        "objections_found": obj_found,
        "objections_handled": obj_handled,
        "sentiment": round(sentiment, 3),
        "next_step_set": next_step,
        "highlights": highlights,
        "coaching": coaching,
        "components": {k: round(v, 1) for k, v in parts.items()},
    }
