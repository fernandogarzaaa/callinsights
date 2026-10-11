"""Speech-to-text for uploaded call recordings.

Providers (selected with ``ASR_PROVIDER``):

* ``none``           - transcription disabled; audio is stored and played back only.
* ``mock``           - deterministic offline transcript (for demos and tests, no model needed).
* ``faster-whisper`` - local Whisper via CTranslate2. ``pip install 'callinsights[asr]'``.
* ``openai``         - any OpenAI-compatible ``/audio/transcriptions`` endpoint
                       (OpenAI ``whisper-1``, Groq ``whisper-large-v3``, a self-hosted
                       faster-whisper-server, ...).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


class TranscriptionError(RuntimeError):
    pass


class Transcriber(Protocol):
    name: str

    def transcribe(self, path: Path) -> list[Segment]: ...


# ---------------------------------------------------------------------------------------------
# Mock (deterministic, offline)
# ---------------------------------------------------------------------------------------------

MOCK_SCRIPTS: tuple[tuple[str, ...], ...] = (
    (
        "Hi, thanks for taking the time today. What prompted you to look at call coaching tools?",
        "Our reps are inconsistent and we have no visibility into what happens on calls.",
        "How many reps are on the team, and who would sign off on a purchase like this?",
        "Twelve reps. Our VP of sales makes the final decision.",
        "What budget range have you set aside, and when do you want this live?",
        "Around two thousand a month, ideally live before next quarter.",
        "Great. I will send a proposal and schedule a demo with your VP on Thursday.",
        "Sounds good, send it over.",
    ),
    (
        "Hello, I wanted to walk you through our platform and all of its features in detail.",
        "Honestly I am not sure this is a good time, and it looks expensive.",
        "It does a lot, the analytics suite is best in class and it integrates with everything.",
        "We are locked into a contract with a competitor until next year.",
        "Okay, well, I will let you go then.",
    ),
)
MOCK_PAUSE_SECONDS = 1.0
MOCK_SECONDS_PER_WORD = 0.4
MOCK_MIN_SEGMENT_SECONDS = 1.5


def mock_segments(script: tuple[str, ...]) -> list[Segment]:
    segments: list[Segment] = []
    t = 0.0
    for line in script:
        dur = max(MOCK_MIN_SEGMENT_SECONDS, len(line.split()) * MOCK_SECONDS_PER_WORD)
        segments.append(Segment(start=t, end=t + dur, text=line))
        t += dur + MOCK_PAUSE_SECONDS
    return segments


class MockTranscriber:
    """Returns one of a few fixed scripts, chosen by the file's content hash."""

    name = "mock"

    def transcribe(self, path: Path) -> list[Segment]:
        try:
            digest = hashlib.sha256(path.read_bytes()).digest()
        except OSError as exc:
            raise TranscriptionError(f"cannot read audio file: {exc}") from exc
        return mock_segments(MOCK_SCRIPTS[digest[0] % len(MOCK_SCRIPTS)])


# ---------------------------------------------------------------------------------------------
# faster-whisper (local)
# ---------------------------------------------------------------------------------------------


class FasterWhisperTranscriber:
    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        language: str | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._factory = model_factory
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                factory = self._factory
                if factory is None:
                    try:
                        from faster_whisper import WhisperModel
                    except ImportError as exc:
                        raise TranscriptionError(
                            "ASR_PROVIDER=faster-whisper requires the 'asr' extra: "
                            "pip install 'callinsights[asr]'"
                        ) from exc
                    factory = WhisperModel
                logger.info("loading whisper model", extra={"model": self.model_size})
                self._model = factory(
                    self.model_size, device=self.device, compute_type=self.compute_type
                )
            return self._model

    def transcribe(self, path: Path) -> list[Segment]:
        model = self._load()
        try:
            raw_segments, _info = model.transcribe(
                str(path), language=self.language, vad_filter=True, beam_size=5
            )
            segments = [
                Segment(start=float(s.start), end=float(s.end), text=str(s.text).strip())
                for s in raw_segments
            ]
        except Exception as exc:
            raise TranscriptionError(f"faster-whisper failed: {exc}") from exc
        return [s for s in segments if s.text]


# ---------------------------------------------------------------------------------------------
# OpenAI-compatible HTTP endpoint
# ---------------------------------------------------------------------------------------------


class OpenAITranscriber:
    name = "openai"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "whisper-1",
        language: str | None = None,
        timeout: float = 600.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("ASR base URL is required for ASR_PROVIDER=openai")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.language = language
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout)

    def transcribe(self, path: Path) -> list[Segment]:
        data: dict[str, str] = {"model": self.model, "response_format": "verbose_json"}
        if self.language:
            data["language"] = self.language
        try:
            with path.open("rb") as fh:
                resp = self._client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers=self._headers,
                    data=data,
                    files={"file": (path.name, fh, "application/octet-stream")},
                )
        except OSError as exc:
            raise TranscriptionError(f"cannot read audio file: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"transcription request failed: {exc}") from exc
        if resp.is_error:
            raise TranscriptionError(f"transcription HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            body = resp.json()
        except ValueError as exc:
            raise TranscriptionError("transcription response was not JSON") from exc
        return segments_from_openai(body)


def segments_from_openai(body: Any) -> list[Segment]:
    """Parse an OpenAI ``verbose_json`` (or plain ``json``) transcription response."""
    if not isinstance(body, dict):
        raise TranscriptionError("transcription response was not a JSON object")
    raw = body.get("segments")
    if isinstance(raw, list) and raw:
        out: list[Segment] = []
        for s in raw:
            if not isinstance(s, dict):
                continue
            text = str(s.get("text") or "").strip()
            if text:
                out.append(
                    Segment(
                        start=float(s.get("start") or 0.0),
                        end=float(s.get("end") or 0.0),
                        text=text,
                    )
                )
        return out
    text = str(body.get("text") or "").strip()
    if not text:
        return []
    return [Segment(start=0.0, end=0.0, text=text)]


def build_transcriber(settings: Any, client: httpx.Client | None = None) -> Transcriber | None:
    provider = settings.ASR_PROVIDER
    if provider == "none":
        return None
    if provider == "mock":
        return MockTranscriber()
    if provider == "faster-whisper":
        return FasterWhisperTranscriber(
            settings.WHISPER_MODEL_SIZE,
            settings.WHISPER_DEVICE,
            settings.WHISPER_COMPUTE_TYPE,
            settings.ASR_LANGUAGE,
        )
    return OpenAITranscriber(
        settings.asr_base_url,
        settings.asr_api_key,
        settings.ASR_MODEL,
        settings.ASR_LANGUAGE,
        settings.ASR_TIMEOUT_SECONDS,
        client=client,
    )
