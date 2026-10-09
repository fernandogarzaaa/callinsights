"""Deterministic call-scoring engine for CallInsights.

Pure functions: no DB, no network, no provider calls. Heavily unit-tested in
tests/test_scoring.py. The summary step (LLM/extractive) lives in the caller
via app.ai.providers.get_provider().summarize().
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

AGENT_SPEAKERS = {
    "agent", "rep", "sales", "sales rep", "salesrep", "sdr", "ae", "caller",
    "consultant", "advisor", "me",
}
PROSPECT_SPEAKERS = {
    "prospect", "customer", "client", "lead", "contact", "buyer", "them",
    "user",
}

BUDGET_WORDS = {
    "budget", "price", "pricing", "cost", "afford", "expensive", "quote",
    "quotation", "spend", "per-seat", "per seat", "invoice",
}
TIMELINE_WORDS = {
    "timeline", "deadline", "when", "quarter", "q1", "q2", "q3", "q4", "month",
    "week", "asap", "launch", "go-live", "go live", "rollout", "by end",
    "september", "october", "november", "december", "january", "february",
    "march", "april", "may", "june", "july", "august",
}
AUTHORITY_WORDS = {
    "decision", "decide", "approver", "approval", "approve", "sign off",
    "sign-off", "boss", "manager", "ceo", "cto", "cfo", "coo", "owner",
    "stakeholder", "final say",
}
OBJECTION_WORDS = {
    "expensive", "too much", "cheaper", "competitor", "contract", "locked",
    "bad time", "not sure", "worried", "worry", "concern", "risk", "skeptical",
    "skeptic", "too pricey", "over budget", "no budget", "learning curve",
    "switching", "downtime", "too long", "not ready",
}
RESOLUTION_WORDS = {
    "understand", "great question", "fair", "fair point", "what if", "discount",
    "payment", "trial", "guarantee", "pilot", "offer", "address",
    "hear you", "makes sense", "roi", "save", "training", "onboarding",
    "walk away", "risk-free", "risk free", "success manager",
}
NEXT_STEP_WORDS = {
    "next step", "follow up", "follow-up", "schedule", "book", "send proposal",
    "proposal", "demo", "trial", "call back", "callback", "meeting",
    "calendar", "invite", "poc", "pilot", "sign up", "signup",
}
POSITIVE_WORDS = {
    "great", "good", "excellent", "love", "like", "perfect", "awesome",
    "happy", "impressed", "helpful", "works", "working", "interested",
    "excited", "fantastic", "wonderful", "glad", "pleased",
}
NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "hate", "dislike", "horrible", "frustrated",
    "frustrating", "annoying", "problem", "issue", "broken", "slow", "worst",
    "disappointed", "concerned", "worried", "unhappy", "difficult", "pain",
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
        if name.endswith(".vtt") and (
            s in ("", "WEBVTT") or s.startswith("NOTE") or _VTT_TS_RE.match(s)
        ) or name.endswith(".srt") and (
            s == "" or _SRT_TS_RE.match(s) or _SRT_NUM_RE.match(s)
        ):
            continue
        out.append(line)
    # Remove speaker-less cue duplicates: keep every line; parse_turns merges.
    return "\n".join(out).strip()


def parse_turns(text: str) -> list[dict]:
    """Parse "Agent: ..."/"Prospect: ..." transcript lines into turns.

    Returns list of {"speaker", "role", "text"}. Lines without a speaker
    prefix continue the previous turn; leading stray lines are dropped.
    """
    turns: list[dict] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _SPEAKER_RE.match(line)
        if m:
            speaker, utter = m.group(1).strip(), m.group(2).strip()
            if not utter:
                continue
            turns.append(
                {"speaker": speaker, "role": _role_for(speaker), "text": utter}
            )
        elif turns:
            turns[-1]["text"] += " " + line
    return turns


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _contains_any(text: str, words: set[str]) -> str | None:
    low = text.lower()
    for w in sorted(words, key=len, reverse=True):
        if w in low:
            return w
    return None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _count_questions(turns: list[dict]) -> int:
    return sum(t["text"].count("?") for t in turns if t["role"] == "agent")


def _word_counts(turns: list[dict]) -> tuple[int, int, int]:
    agent = sum(len(_words(t["text"])) for t in turns if t["role"] == "agent")
    prospect = sum(len(_words(t["text"])) for t in turns if t["role"] == "prospect")
    other = sum(len(_words(t["text"])) for t in turns if t["role"] == "other")
    return agent, prospect, other


def _mentions(turns: list[dict]) -> tuple[bool, bool, bool]:
    full = " ".join(t["text"] for t in turns).lower()
    return (
        _contains_any(full, BUDGET_WORDS) is not None,
        _contains_any(full, TIMELINE_WORDS) is not None,
        _contains_any(full, AUTHORITY_WORDS) is not None,
    )


def _objections(turns: list[dict]) -> tuple[int, int, list[str]]:
    """Return (found, handled, handled_keywords)."""
    found = 0
    handled = 0
    handled_kws: list[str] = []
    prospect_idxs = [i for i, t in enumerate(turns) if t["role"] == "prospect"]
    for i in prospect_idxs:
        kw = _contains_any(turns[i]["text"], OBJECTION_WORDS)
        if not kw:
            continue
        found += 1
        later_agent = " ".join(
            t["text"] for t in turns[i + 1 :] if t["role"] == "agent"
        ).lower()
        if kw in later_agent or _contains_any(later_agent, RESOLUTION_WORDS):
            handled += 1
            handled_kws.append(kw)
    return found, handled, handled_kws


def _sentiment(turns: list[dict]) -> float:
    """Net sentiment of prospect speech in [-1, 1]."""
    words: list[str] = []
    for t in turns:
        if t["role"] == "prospect":
            words.extend(_words(t["text"]))
    if not words:
        return 0.0
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    return max(-1.0, min(1.0, (pos - neg) / len(words)))


def _next_step_set(turns: list[dict]) -> bool:
    full = " ".join(t["text"] for t in turns).lower()
    return _contains_any(full, NEXT_STEP_WORDS) is not None


def plain_text(turns: list[dict]) -> str:
    """Turn texts without speaker labels, for clean extractive summaries."""
    return " ".join(t["text"] for t in turns)


def score_call(turns: list[dict]) -> dict:
    """Score a parsed call. Returns a dict with score 0-100 and all parts."""
    questions = _count_questions(turns)
    agent_w, prospect_w, _ = _word_counts(turns)
    spoken = agent_w + prospect_w
    talk_ratio = agent_w / spoken if spoken else 0.0
    budget, timeline, authority = _mentions(turns)
    obj_found, obj_handled, _ = _objections(turns)
    sentiment = _sentiment(turns)
    next_step = _next_step_set(turns)

    # Component scores, each 0-20.
    q_score = min(questions, 8) / 8 * 20
    balance = max(0.0, 1 - abs(talk_ratio - 0.5) / 0.5) * 20
    qual_score = (int(budget) + int(timeline) + int(authority)) / 3 * 20
    if obj_found:
        obj_score = obj_handled / obj_found * 20
    else:
        obj_score = 20.0
    sent_score = max(0.0, min(20.0, 10 + sentiment * 50))

    total = round(q_score + balance + qual_score + obj_score + sent_score)
    total = max(0, min(100, total))
    outcome = "Interested" if total >= 75 else ("Follow Up" if total >= 50 else "Not Interested")

    highlights = [
        {"label": "Budget discussed", "met": budget},
        {"label": "Timeline established", "met": timeline},
        {"label": "Decision maker confirmed", "met": authority},
        {
            "label": "Objections handled",
            "met": obj_found == 0 or obj_handled == obj_found,
            "detail": f"{obj_handled}/{obj_found} addressed" if obj_found else "No objections raised",
        },
        {"label": "Next step set", "met": next_step},
    ]

    coaching: list[str] = []
    pct = round(talk_ratio * 100)
    if talk_ratio > 0.7:
        coaching.append(
            f"Let the prospect speak more: you talked {pct}% of the time. "
            "Aim for a 40-60% talk share."
        )
    elif talk_ratio < 0.25 and spoken > 0:
        coaching.append(
            f"You only talked {pct}% of the time. Guide the conversation with "
            "targeted discovery questions."
        )
    if questions < 3:
        coaching.append(
            f"Ask more discovery questions (you asked {questions}). "
            "Strong calls ask 5 or more."
        )
    if not budget:
        coaching.append("Qualify budget earlier: ask about budget range before the pitch.")
    if not timeline:
        coaching.append("Establish a timeline: ask when they need a solution live.")
    if not authority:
        coaching.append(
            "Confirm authority: ask who else weighs in on the decision."
        )
    if obj_found > obj_handled:
        coaching.append(
            f"Address objections directly: {obj_found - obj_handled} raised "
            "concern(s) went unanswered. Name the concern and offer a concrete fix."
        )
    if sentiment < -0.02:
        coaching.append(
            "Watch the prospect's tone: acknowledge their frustration explicitly "
            "before moving on."
        )
    if not next_step:
        coaching.append(
            "Always close with a concrete next step: a scheduled follow-up, "
            "a proposal, or a trial."
        )
    if total >= 75:
        coaching.append(
            "Strong call: keep this structure. Replicate the discovery cadence "
            "and the explicit close."
        )

    return {
        "score": total,
        "outcome": outcome,
        "talk_ratio": round(talk_ratio, 3),
        "talk_ratio_agent_pct": pct,
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
        "components": {
            "questions": round(q_score, 1),
            "talk_balance": round(balance, 1),
            "qualification": round(qual_score, 1),
            "objection_handling": round(obj_score, 1),
            "sentiment": round(sent_score, 1),
        },
    }
