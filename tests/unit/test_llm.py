from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from callinsights import scoring
from callinsights.ai import llm
from callinsights.ai.llm import (
    Insights,
    LLMError,
    LocalProvider,
    OpenAICompatibleProvider,
    analytics_for_prompt,
    backoff_seconds,
    build_llm_provider,
    clean_list,
    extractive_summary,
    local_next_best_action,
    local_risks,
    local_strengths,
    parse_json_object,
    transcript_for_prompt,
)
from tests.conftest import GOOD_TRANSCRIPT


def completion(content: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})


class Recorder:
    """httpx MockTransport handler that replays queued responses and records requests."""

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def body(self, i: int = -1) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self.requests[i].content)
        return data


def provider(
    rec: Recorder, api_key: str = "sk-test", **kw: Any
) -> tuple[OpenAICompatibleProvider, list[float]]:
    sleeps: list[float] = []
    sleep: Callable[[float], None] = sleeps.append
    p = OpenAICompatibleProvider(
        "https://llm.example/v1/",
        api_key,
        "test-model",
        client=httpx.Client(transport=httpx.MockTransport(rec)),
        sleep=sleep,
        **kw,
    )
    return p, sleeps


TURNS = scoring.parse_turns(GOOD_TRANSCRIPT)
ANALYTICS = scoring.score_call(TURNS)


# --- local helpers ---------------------------------------------------------------------------


def test_extractive_summary_short_text_unchanged() -> None:
    assert extractive_summary("One. Two!  Three?", 3) == "One. Two! Three?"
    assert extractive_summary("", 3) == ""


def test_extractive_summary_picks_informative_sentences_in_order() -> None:
    text = (
        "Pricing matters to the team. Hello. Pricing and budget drive the pricing decision. "
        "Okay. The budget is set by finance."
    )
    out = extractive_summary(text, 2)
    assert out == "Pricing matters to the team. Pricing and budget drive the pricing decision."


def test_extractive_summary_ignores_stopword_only_sentences() -> None:
    text = "Budget budget budget. It is what it is. Budget again. And so on."
    assert extractive_summary(text, 2) == "Budget budget budget. Budget again."


def test_extractive_summary_normalizes_by_length() -> None:
    # A long sentence of rare words must not beat a short sentence of frequent ones.
    text = (
        "Alpha beta gamma delta epsilon zeta eta theta iota kappa. "
        "Renewal renewal. Renewal soon. Renewal now."
    )
    assert extractive_summary(text, 2) == "Renewal renewal. Renewal soon."


def test_local_strengths_full_and_empty() -> None:
    full = {
        "questions_asked": 5,
        "talk_ratio_agent_pct": 40,
        "budget_mentioned": True,
        "timeline_mentioned": True,
        "authority_mentioned": True,
        "objections_found": 2,
        "objections_handled": 2,
        "next_step_set": True,
    }
    assert local_strengths(full) == [
        "Strong discovery: asked 5 questions.",
        "Balanced conversation: agent talk share was 40%.",
        "Budget was discussed.",
        "A timeline was established.",
        "The decision process was explored.",
        "Handled every objection raised (2).",
        "Closed with a concrete next step.",
    ]
    assert local_strengths({}) == []


@pytest.mark.parametrize(("pct", "balanced"), [(39, False), (40, True), (60, True), (61, False)])
def test_local_strengths_talk_balance_bounds(pct: int, balanced: bool) -> None:
    out = local_strengths({"talk_ratio_agent_pct": pct})
    assert (out == [f"Balanced conversation: agent talk share was {pct}%."]) is balanced


def test_local_strengths_question_threshold_and_partial_objections() -> None:
    assert local_strengths({"questions_asked": 4}) == []
    assert local_strengths({"objections_found": 2, "objections_handled": 1}) == []


def test_local_risks() -> None:
    assert local_risks(
        {
            "objections_found": 3,
            "objections_handled": 1,
            "sentiment": -0.03,
            "next_step_set": False,
            "authority_mentioned": False,
            "budget_mentioned": False,
        }
    ) == [
        "2 objection(s) left unanswered.",
        "Prospect sentiment trended negative.",
        "No next step agreed; the deal may stall.",
        "Decision maker unknown.",
        "Budget not qualified.",
    ]
    clean = {"next_step_set": True, "authority_mentioned": True, "budget_mentioned": True}
    assert local_risks({**clean, "sentiment": -0.02}) == []


@pytest.mark.parametrize(
    ("analytics", "expected"),
    [
        ({"objections_found": 1, "objections_handled": 0}, "address the open objection"),
        ({}, "book a specific next meeting"),
        ({"next_step_set": True}, "who else is involved"),
        ({"next_step_set": True, "authority_mentioned": True}, "Qualify budget"),
        (
            {"next_step_set": True, "authority_mentioned": True, "budget_mentioned": True},
            "Confirm the go-live date",
        ),
        (
            {
                "next_step_set": True,
                "authority_mentioned": True,
                "budget_mentioned": True,
                "timeline_mentioned": True,
            },
            "Send the agreed follow-up",
        ),
    ],
)
def test_local_next_best_action(analytics: dict[str, Any], expected: str) -> None:
    assert expected in local_next_best_action(analytics)


def test_local_provider_insights_are_deterministic() -> None:
    p = LocalProvider()
    a = p.insights(TURNS, ANALYTICS)
    b = p.insights(TURNS, ANALYTICS)
    assert a == b
    assert a.source == "local"
    assert a.coaching == ANALYTICS["coaching"]
    assert a.summary == extractive_summary(" ".join(t["text"] for t in TURNS), 4)
    assert a.strengths == local_strengths(ANALYTICS)
    assert a.risks == local_risks(ANALYTICS)
    assert a.next_best_action == local_next_best_action(ANALYTICS)
    assert p.summarize("A. B. C. D. E. F.", 2) == extractive_summary("A. B. C. D. E. F.", 2)
    assert p.label_speakers(["x"]) is None
    assert p.name == "local"


# --- JSON / prompt helpers ------------------------------------------------------------------


def test_clean_list() -> None:
    assert clean_list("nope") == []
    assert clean_list([" a ", "", "  ", 3]) == ["a", "3"]
    assert clean_list([str(i) for i in range(10)]) == [str(i) for i in range(6)]
    assert clean_list(["x" * 500]) == ["x" * 400]


@pytest.mark.parametrize(
    "content",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Sure! Here you go: {"a": 1} Hope that helps.',
    ],
)
def test_parse_json_object_tolerant(content: str) -> None:
    assert parse_json_object(content) == {"a": 1}


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("no json here", "no JSON object"),
        ("[1, 2]", "not an object"),
        ("{bad json}", "not valid JSON"),
        ("} {", "no JSON object"),
    ],
)
def test_parse_json_object_errors(content: str, message: str) -> None:
    with pytest.raises(LLMError, match=message):
        parse_json_object(content)


def test_transcript_for_prompt_truncates_middle() -> None:
    turns = scoring.parse_turns("Agent: hello\nProspect: hi")
    assert transcript_for_prompt(turns, 1000) == "Agent (agent): hello\nProspect (prospect): hi"
    long_turns = scoring.parse_turns("\n".join(f"Agent: line {i}" for i in range(200)))
    out = transcript_for_prompt(long_turns, 100)
    assert "[... middle of call omitted for length ...]" in out
    assert out.startswith("Agent (agent): line 0")
    assert out.endswith("line 199")
    head, tail = out.split("\n[... middle of call omitted for length ...]\n")
    assert len(head) == 50
    assert len(tail) == 50


def test_transcript_for_prompt_exact_budget_not_truncated() -> None:
    turns = scoring.parse_turns("Agent: hello")
    text = "Agent (agent): hello"
    assert transcript_for_prompt(turns, len(text)) == text


def test_analytics_for_prompt_selects_keys() -> None:
    out = analytics_for_prompt(ANALYTICS)
    assert set(out) == set(llm.PROMPT_ANALYTICS_KEYS)
    assert "coaching" not in out
    assert analytics_for_prompt({"score": 1, "extra": 2}) == {"score": 1}


def test_backoff_seconds() -> None:
    assert [backoff_seconds(i) for i in (1, 2, 3, 4, 5, 6)] == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]


# --- OpenAI-compatible provider -------------------------------------------------------------


def test_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        OpenAICompatibleProvider("", "k", "m")


def test_chat_request_shape_and_headers() -> None:
    rec = Recorder(completion("  Hello there  "))
    p, sleeps = provider(rec)
    assert p.chat([{"role": "user", "content": "hi"}]) == "Hello there"
    req = rec.requests[0]
    assert str(req.url) == "https://llm.example/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    assert req.headers["content-type"] == "application/json"
    assert rec.body() == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
    }
    assert sleeps == []


def test_chat_without_api_key_sends_no_authorization() -> None:
    rec = Recorder(completion("ok"))
    p, _ = provider(rec, api_key="")
    p.chat([])
    assert "authorization" not in rec.requests[0].headers


def test_chat_json_mode_and_fallback_when_rejected() -> None:
    rec = Recorder(httpx.Response(400, text="unsupported"), completion("{}"))
    p, sleeps = provider(rec)
    assert p.chat([], json_mode=True) == "{}"
    assert rec.body(0)["response_format"] == {"type": "json_object"}
    assert "response_format" not in rec.body(1)
    assert sleeps == []


def test_chat_400_without_json_mode_is_error() -> None:
    rec = Recorder(httpx.Response(400, text="bad request"))
    p, _ = provider(rec)
    with pytest.raises(LLMError, match="HTTP 400: bad request"):
        p.chat([])


def test_chat_retries_retryable_status_then_succeeds() -> None:
    rec = Recorder(httpx.Response(429), httpx.Response(503), completion("done"))
    p, sleeps = provider(rec, max_retries=2)
    assert p.chat([]) == "done"
    assert sleeps == [1.0, 2.0]
    assert len(rec.requests) == 3


def test_chat_gives_up_after_max_retries() -> None:
    rec = Recorder(httpx.Response(500, text="boom"), httpx.Response(500, text="boom"))
    p, sleeps = provider(rec, max_retries=1)
    with pytest.raises(LLMError, match="HTTP 500"):
        p.chat([])
    assert sleeps == [1.0]


def test_chat_non_retryable_error_fails_fast() -> None:
    rec = Recorder(httpx.Response(401, text="nope"))
    p, sleeps = provider(rec, max_retries=3)
    with pytest.raises(LLMError, match="HTTP 401"):
        p.chat([])
    assert sleeps == []


def test_chat_transport_errors_retry_then_raise() -> None:
    rec = Recorder(httpx.ConnectError("down"), completion("back"))
    p, sleeps = provider(rec, max_retries=1)
    assert p.chat([]) == "back"
    assert sleeps == [1.0]
    rec2 = Recorder(httpx.ConnectError("down"), httpx.ConnectError("down"))
    p2, _ = provider(rec2, max_retries=1)
    with pytest.raises(LLMError, match="transport error"):
        p2.chat([])


def test_chat_zero_retries() -> None:
    rec = Recorder(httpx.ConnectError("down"))
    p, sleeps = provider(rec, max_retries=0)
    with pytest.raises(LLMError):
        p.chat([])
    assert sleeps == []


@pytest.mark.parametrize(
    "resp",
    [
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"nope": 1}),
        httpx.Response(200, text="not json"),
        completion(""),
        completion("   "),
        completion(None),
        completion(42),
    ],
)
def test_chat_bad_payloads(resp: httpx.Response) -> None:
    p, _ = provider(Recorder(resp))
    with pytest.raises(LLMError):
        p.chat([])


def test_summarize_remote_and_fallback() -> None:
    rec = Recorder(completion("Remote summary."))
    p, _ = provider(rec, max_transcript_chars=1000)
    assert p.summarize("x" * 2000, 3) == "Remote summary."
    msgs = rec.body()["messages"]
    assert "at most 3 sentences" in msgs[0]["content"]
    assert len(msgs[1]["content"]) == 1000

    p2, _ = provider(Recorder(httpx.Response(401)))
    text = "One. Two. Three. Four. Five."
    assert p2.summarize(text, 2) == extractive_summary(text, 2)


def test_insights_from_model() -> None:
    reply = json.dumps(
        {
            "summary": "Good call.",
            "coaching": ["Ask about timeline", ""],
            "strengths": ["Rapport"],
            "risks": ["Price"],
            "next_best_action": "Send proposal",
        }
    )
    rec = Recorder(completion(reply))
    p, _ = provider(rec)
    out = p.insights(TURNS, ANALYTICS)
    assert out == Insights(
        summary="Good call.",
        coaching=["Ask about timeline"],
        strengths=["Rapport"],
        risks=["Price"],
        next_best_action="Send proposal",
        source="llm:test-model",
    )
    body = rec.body()
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][0]["content"] == llm.INSIGHTS_SYSTEM_PROMPT
    user = body["messages"][1]["content"]
    assert user.startswith("Analytics:\n{")
    assert '"score": ' in user
    assert "\n\nTranscript:\nAgent (agent): Hi, thanks" in user


def test_insights_fills_missing_sections_from_local() -> None:
    p, _ = provider(Recorder(completion('{"strengths": ["x"]}')))
    out = p.insights(TURNS, ANALYTICS)
    local = LocalProvider().insights(TURNS, ANALYTICS)
    assert out.summary == local.summary
    assert out.coaching == local.coaching
    assert out.next_best_action == local.next_best_action
    assert out.strengths == ["x"]
    assert out.risks == []
    assert out.source == "llm:test-model"


def test_insights_truncates_next_best_action() -> None:
    p, _ = provider(Recorder(completion(json.dumps({"next_best_action": "  " + "y" * 900}))))
    assert p.insights(TURNS, ANALYTICS).next_best_action == "y" * 400


@pytest.mark.parametrize(
    "resp", [httpx.Response(500), completion("not json at all"), completion("[1]")]
)
def test_insights_fall_back_to_local_on_failure(resp: httpx.Response) -> None:
    p, _ = provider(Recorder(resp), max_retries=0)
    assert p.insights(TURNS, ANALYTICS) == LocalProvider().insights(TURNS, ANALYTICS)


def test_label_speakers_success() -> None:
    rec = Recorder(completion('{"roles": ["Agent", " prospect "]}'))
    p, _ = provider(rec)
    assert p.label_speakers(["hi", "hello"]) == ["agent", "prospect"]
    body = rec.body()
    assert body["messages"][0]["content"] == llm.SPEAKER_SYSTEM_PROMPT
    assert body["messages"][1]["content"] == "1. hi\n2. hello"
    assert body["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "content",
    ['{"roles": ["agent"]}', '{"roles": ["agent", "boss"]}', '{"roles": "agent"}', '{"x": 1}'],
)
def test_label_speakers_rejects_bad_output(content: str) -> None:
    p, _ = provider(Recorder(completion(content)))
    assert p.label_speakers(["a", "b"]) is None


def test_label_speakers_disabled_empty_or_failing() -> None:
    p, _ = provider(Recorder(), speaker_labeling=False)
    assert p.label_speakers(["a"]) is None
    p2, _ = provider(Recorder())
    assert p2.label_speakers([]) is None
    p3, _ = provider(Recorder(httpx.Response(401)))
    assert p3.label_speakers(["a"]) is None


def test_build_llm_provider(tmp_path: Any) -> None:
    from tests.conftest import make_settings

    assert isinstance(build_llm_provider(make_settings(tmp_path)), LocalProvider)
    s = make_settings(
        tmp_path,
        LLM_PROVIDER="auto",
        LLM_BASE_URL="http://localhost:11434/v1/",
        LLM_MODEL="llama3.1",
        LLM_MAX_RETRIES=4,
        LLM_SPEAKER_LABELING=False,
        LLM_MAX_TRANSCRIPT_CHARS=5000,
    )
    p = build_llm_provider(s)
    assert isinstance(p, OpenAICompatibleProvider)
    assert p.base_url == "http://localhost:11434/v1"
    assert p.model == "llama3.1"
    assert p.max_retries == 4
    assert p.speaker_labeling is False
    assert p.max_transcript_chars == 5000
    assert p.name == "openai-compatible"
