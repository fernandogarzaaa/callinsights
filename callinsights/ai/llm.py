"""Narrative AI layer: call summaries, coaching insights and speaker labelling.

Two providers implement the same interface:

* ``LocalProvider`` - deterministic and offline. Extractive summaries and rule-based
  insights derived from the scoring engine. Every feature works in this mode, and it is
  what the test-suite uses, so no API keys are ever needed to run tests.
* ``OpenAICompatibleProvider`` - any server that speaks the OpenAI Chat Completions API:
  OpenAI, Groq, OpenRouter, Together, Ollama (``http://localhost:11434/v1``), vLLM,
  LM Studio... On any remote failure it logs a warning and falls back to the local
  result, so a provider outage never blocks call scoring.

The numeric score is always computed by :mod:`callinsights.scoring`; the LLM only writes
the narrative (summary, coaching, strengths, risks, next best action).
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from collections.abc import Callable
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, Field

from callinsights.scoring import Turn

logger = logging.getLogger(__name__)

Role = Literal["agent", "prospect"]
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
MAX_ITEMS = 6
MAX_ITEM_CHARS = 400
MAX_BACKOFF_SECONDS = 8.0


class Insights(BaseModel):
    summary: str = ""
    coaching: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    next_best_action: str = ""
    source: str = "local"


class LLMError(RuntimeError):
    """A remote model call failed or returned unusable output."""


class LLMProvider(Protocol):
    name: str

    def summarize(self, text: str, max_sentences: int = 4) -> str: ...

    def insights(self, turns: list[Turn], analytics: dict[str, Any]) -> Insights: ...

    def label_speakers(self, segments: list[str]) -> list[Role] | None: ...


# ---------------------------------------------------------------------------------------------
# Local, deterministic provider
# ---------------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "do",
        "for",
        "from",
        "have",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "what",
        "with",
        "you",
        "your",
        "our",
        "us",
        "they",
        "them",
        "he",
        "she",
        "not",
        "just",
        "can",
        "will",
        "would",
    }
)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]


def extractive_summary(text: str, max_sentences: int = 4) -> str:
    """Pick the ``max_sentences`` highest-information sentences, kept in original order."""
    sents = _sentences(text)
    if len(sents) <= max_sentences:
        return " ".join(sents)
    freq = Counter(t for t in _tokens(text) if t not in STOPWORDS)

    def weight(i: int) -> float:
        toks = {t for t in _tokens(sents[i]) if t not in STOPWORDS}
        if not toks:
            return 0.0
        return float(sum(freq[t] for t in toks)) / math.sqrt(len(toks))

    ranked = sorted(range(len(sents)), key=lambda i: (-weight(i), i))
    keep = sorted(ranked[:max_sentences])
    return " ".join(sents[i] for i in keep)


def _unhandled(analytics: dict[str, Any]) -> int:
    return int(analytics.get("objections_found", 0)) - int(analytics.get("objections_handled", 0))


def local_strengths(analytics: dict[str, Any]) -> list[str]:
    out: list[str] = []
    questions = int(analytics.get("questions_asked", 0))
    if questions >= 5:
        out.append(f"Strong discovery: asked {questions} questions.")
    pct = int(analytics.get("talk_ratio_agent_pct", 0))
    if 40 <= pct <= 60:
        out.append(f"Balanced conversation: agent talk share was {pct}%.")
    if analytics.get("budget_mentioned"):
        out.append("Budget was discussed.")
    if analytics.get("timeline_mentioned"):
        out.append("A timeline was established.")
    if analytics.get("authority_mentioned"):
        out.append("The decision process was explored.")
    found = int(analytics.get("objections_found", 0))
    if found and _unhandled(analytics) == 0:
        out.append(f"Handled every objection raised ({found}).")
    if analytics.get("next_step_set"):
        out.append("Closed with a concrete next step.")
    return out


def local_risks(analytics: dict[str, Any]) -> list[str]:
    out: list[str] = []
    unhandled = _unhandled(analytics)
    if unhandled > 0:
        out.append(f"{unhandled} objection(s) left unanswered.")
    if float(analytics.get("sentiment", 0.0)) < -0.02:
        out.append("Prospect sentiment trended negative.")
    if not analytics.get("next_step_set"):
        out.append("No next step agreed; the deal may stall.")
    if not analytics.get("authority_mentioned"):
        out.append("Decision maker unknown.")
    if not analytics.get("budget_mentioned"):
        out.append("Budget not qualified.")
    return out


def local_next_best_action(analytics: dict[str, Any]) -> str:
    if _unhandled(analytics) > 0:
        return "Follow up in writing to address the open objection(s) with a concrete answer."
    if not analytics.get("next_step_set"):
        return "Reach out within 24 hours to book a specific next meeting."
    if not analytics.get("authority_mentioned"):
        return "Ask who else is involved in the decision and offer to include them next time."
    if not analytics.get("budget_mentioned"):
        return "Qualify budget before sending a proposal."
    if not analytics.get("timeline_mentioned"):
        return "Confirm the go-live date the prospect is working toward."
    return "Send the agreed follow-up materials and confirm the scheduled next step."


class LocalProvider:
    """Deterministic, offline provider. Same input always yields the same output."""

    name = "local"

    def summarize(self, text: str, max_sentences: int = 4) -> str:
        return extractive_summary(text, max_sentences)

    def insights(self, turns: list[Turn], analytics: dict[str, Any]) -> Insights:
        text = " ".join(t["text"] for t in turns)
        return Insights(
            summary=extractive_summary(text, 4),
            coaching=[str(c) for c in analytics.get("coaching", [])],
            strengths=local_strengths(analytics),
            risks=local_risks(analytics),
            next_best_action=local_next_best_action(analytics),
            source="local",
        )

    def label_speakers(self, segments: list[str]) -> list[Role] | None:
        # The deterministic heuristic in callinsights.ai.diarize handles labelling.
        return None


# ---------------------------------------------------------------------------------------------
# OpenAI-compatible provider
# ---------------------------------------------------------------------------------------------

INSIGHTS_SYSTEM_PROMPT = """You are an expert B2B sales coach reviewing a recorded sales call.
You receive the transcript and deterministic analytics already computed for it.
Reply with ONLY a JSON object, no prose and no code fences, with exactly these keys:
  "summary": string, 2-4 sentences, what happened and where the deal stands,
  "coaching": array of 2-5 short, specific, actionable coaching tips for the agent,
  "strengths": array of 0-4 short things the agent did well,
  "risks": array of 0-4 short deal risks,
  "next_best_action": string, the single most valuable next step for the agent.
Ground every statement in the transcript. Do not invent facts, names or numbers."""

SPEAKER_SYSTEM_PROMPT = """You label the speakers of a two-party sales call transcribed without
diarization. One speaker is the sales agent (seller), the other is the prospect (buyer).
You receive numbered segments. Reply with ONLY a JSON object: {"roles": [...]} where the array has
exactly one entry per segment, in order, each either "agent" or "prospect"."""


def clean_list(value: Any) -> list[str]:
    """Coerce model output to a short list of trimmed, non-empty strings."""
    if not isinstance(value, list):
        return []
    out = [str(v).strip()[:MAX_ITEM_CHARS] for v in value if str(v).strip()]
    return out[:MAX_ITEMS]


def parse_json_object(content: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences and surrounding prose."""
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError("model reply contained no JSON object") from None
        try:
            data = json.loads(text[start : end + 1])
        except ValueError as exc:
            raise LLMError(f"model reply was not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise LLMError("model reply JSON was not an object")
    return data


def transcript_for_prompt(turns: list[Turn], max_chars: int) -> str:
    """Render turns as ``Speaker (role): text`` lines, keeping head and tail if over budget."""
    text = "\n".join(f"{t['speaker']} ({t['role']}): {t['text']}" for t in turns)
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n[... middle of call omitted for length ...]\n" + text[-half:]


PROMPT_ANALYTICS_KEYS = (
    "score",
    "outcome",
    "talk_ratio_agent_pct",
    "questions_asked",
    "budget_mentioned",
    "timeline_mentioned",
    "authority_mentioned",
    "objections_found",
    "objections_handled",
    "sentiment",
    "next_step_set",
)


def analytics_for_prompt(analytics: dict[str, Any]) -> dict[str, Any]:
    return {k: analytics[k] for k in PROMPT_ANALYTICS_KEYS if k in analytics}


def backoff_seconds(attempt: int) -> float:
    """Exponential backoff: 1s, 2s, 4s, ... capped at MAX_BACKOFF_SECONDS."""
    return min(2.0 ** (attempt - 1), MAX_BACKOFF_SECONDS)


class OpenAICompatibleProvider:
    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
        max_transcript_chars: int = 24_000,
        speaker_labeling: bool = True,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required for the OpenAI-compatible provider")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.max_transcript_chars = max_transcript_chars
        self.speaker_labeling = speaker_labeling
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._headers = headers
        self._client = client or httpx.Client(timeout=timeout)
        self._sleep = sleep
        self._local = LocalProvider()

    # -- transport ------------------------------------------------------------------------
    def chat(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> str:
        """POST /chat/completions with retries on transport errors, 429 and 5xx."""
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.2}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        url = f"{self.base_url}/chat/completions"
        attempt = 0
        while True:
            try:
                resp = self._client.post(url, headers=self._headers, json=body)
            except httpx.TransportError as exc:
                if attempt >= self.max_retries:
                    raise LLMError(f"transport error: {exc}") from exc
            else:
                if resp.status_code == 400 and "response_format" in body:
                    # Some OpenAI-compatible servers reject response_format; retry without it.
                    body.pop("response_format")
                    continue
                if resp.status_code not in RETRYABLE_STATUS or attempt >= self.max_retries:
                    if resp.is_error:
                        raise LLMError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    return self.content(resp)
            attempt += 1
            self._sleep(backoff_seconds(attempt))

    @staticmethod
    def content(resp: httpx.Response) -> str:
        try:
            content = resp.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {exc!r}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("empty completion")
        return content.strip()

    # -- features -------------------------------------------------------------------------
    def summarize(self, text: str, max_sentences: int = 4) -> str:
        try:
            return self.chat(
                [
                    {
                        "role": "system",
                        "content": f"Summarize this sales call in at most {max_sentences} "
                        "sentences. Plain text only.",
                    },
                    {"role": "user", "content": text[: self.max_transcript_chars]},
                ]
            )
        except LLMError as exc:
            logger.warning("LLM summarize failed; using local summary", extra={"error": str(exc)})
            return self._local.summarize(text, max_sentences)

    def insights(self, turns: list[Turn], analytics: dict[str, Any]) -> Insights:
        fallback = self._local.insights(turns, analytics)
        user = (
            "Analytics:\n"
            + json.dumps(analytics_for_prompt(analytics))
            + "\n\nTranscript:\n"
            + transcript_for_prompt(turns, self.max_transcript_chars)
        )
        try:
            data = parse_json_object(
                self.chat(
                    [
                        {"role": "system", "content": INSIGHTS_SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                    ],
                    json_mode=True,
                )
            )
        except LLMError as exc:
            logger.warning("LLM insights failed; using local insights", extra={"error": str(exc)})
            return fallback
        summary = str(data.get("summary") or "").strip()
        coaching = clean_list(data.get("coaching"))
        action = str(data.get("next_best_action") or "").strip()[:MAX_ITEM_CHARS]
        # Fill any section the model left empty with the deterministic version.
        return Insights(
            summary=summary or fallback.summary,
            coaching=coaching or fallback.coaching,
            strengths=clean_list(data.get("strengths")),
            risks=clean_list(data.get("risks")),
            next_best_action=action or fallback.next_best_action,
            source=f"llm:{self.model}",
        )

    def label_speakers(self, segments: list[str]) -> list[Role] | None:
        if not self.speaker_labeling or not segments:
            return None
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(segments))
        try:
            data = parse_json_object(
                self.chat(
                    [
                        {"role": "system", "content": SPEAKER_SYSTEM_PROMPT},
                        {"role": "user", "content": numbered[: self.max_transcript_chars]},
                    ],
                    json_mode=True,
                )
            )
        except LLMError as exc:
            logger.warning("LLM speaker labelling failed", extra={"error": str(exc)})
            return None
        roles = data.get("roles")
        if not isinstance(roles, list) or len(roles) != len(segments):
            logger.warning("LLM speaker labelling returned the wrong number of roles")
            return None
        normalized = [str(r).strip().lower() for r in roles]
        if any(r not in ("agent", "prospect") for r in normalized):
            logger.warning("LLM speaker labelling returned unknown roles")
            return None
        return ["agent" if r == "agent" else "prospect" for r in normalized]


def build_llm_provider(settings: Any, client: httpx.Client | None = None) -> LLMProvider:
    """Create the provider selected by ``LLM_PROVIDER`` (``auto`` uses openai if a URL is set)."""
    if settings.llm_mode == "openai":
        return OpenAICompatibleProvider(
            settings.LLM_BASE_URL,
            settings.LLM_API_KEY.get_secret_value(),
            settings.LLM_MODEL,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
            max_transcript_chars=settings.LLM_MAX_TRANSCRIPT_CHARS,
            speaker_labeling=settings.LLM_SPEAKER_LABELING,
            client=client,
        )
    return LocalProvider()
