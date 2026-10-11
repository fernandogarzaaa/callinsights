from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest

from callinsights.ai import asr
from callinsights.ai.asr import (
    FasterWhisperTranscriber,
    MockTranscriber,
    OpenAITranscriber,
    Segment,
    TranscriptionError,
    build_transcriber,
    mock_segments,
    segments_from_openai,
)
from tests.conftest import make_settings


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    p = tmp_path / "call.wav"
    p.write_bytes(b"RIFF" + b"\x01" * 64)
    return p


def test_mock_segments_timing() -> None:
    segs = mock_segments(("one two three four five", "hi"))
    # 5 words * 0.4 = 2.0s; "hi" -> minimum 1.5s; 1s pause between.
    assert segs == [Segment(0.0, 2.0, "one two three four five"), Segment(3.0, 4.5, "hi")]


def test_mock_transcriber_is_deterministic_and_content_based(tmp_path: Path) -> None:
    t = MockTranscriber()
    seen = set()
    for i in range(16):
        p = tmp_path / f"{i}.wav"
        p.write_bytes(bytes([i]) * 10)
        first = t.transcribe(p)
        assert first == t.transcribe(p)
        seen.add(first[0].text)
        expected = asr.MOCK_SCRIPTS[asr.hashlib.sha256(p.read_bytes()).digest()[0] % 2]
        assert [s.text for s in first] == list(expected)
    assert len(seen) == 2
    assert t.name == "mock"


def test_mock_transcriber_missing_file(tmp_path: Path) -> None:
    with pytest.raises(TranscriptionError, match="cannot read"):
        MockTranscriber().transcribe(tmp_path / "missing.wav")


class FakeWhisperSegment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start, self.end, self.text = start, end, text


class FakeWhisperModel:
    instances: list[FakeWhisperModel] = []

    def __init__(self, size: str, device: str, compute_type: str) -> None:
        self.args = (size, device, compute_type)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        FakeWhisperModel.instances.append(self)

    def transcribe(self, path: str, **kw: Any) -> tuple[Any, None]:
        self.calls.append((path, kw))
        segs = [FakeWhisperSegment(0, 1.5, " Hello. "), FakeWhisperSegment(2, 3, "  ")]
        return iter(segs), None


def test_faster_whisper_with_factory(audio: Path) -> None:
    FakeWhisperModel.instances.clear()
    t = FasterWhisperTranscriber("small", "cpu", "int8", "en", model_factory=FakeWhisperModel)
    assert t.transcribe(audio) == [Segment(0.0, 1.5, "Hello.")]
    t.transcribe(audio)
    assert len(FakeWhisperModel.instances) == 1  # model is loaded once and cached
    model = FakeWhisperModel.instances[0]
    assert model.args == ("small", "cpu", "int8")
    assert model.calls[0] == (str(audio), {"language": "en", "vad_filter": True, "beam_size": 5})
    assert t.name == "faster-whisper"


def test_faster_whisper_imports_real_module(monkeypatch: pytest.MonkeyPatch, audio: Path) -> None:
    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    assert FasterWhisperTranscriber().transcribe(audio)[0].text == "Hello."


def test_faster_whisper_missing_extra(monkeypatch: pytest.MonkeyPatch, audio: Path) -> None:
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(TranscriptionError, match=r"callinsights\[asr\]"):
        FasterWhisperTranscriber().transcribe(audio)


def test_faster_whisper_wraps_model_errors(audio: Path) -> None:
    class Broken(FakeWhisperModel):
        def transcribe(self, path: str, **kw: Any) -> tuple[Any, None]:
            raise RuntimeError("decoder exploded")

    t = FasterWhisperTranscriber(model_factory=Broken)
    with pytest.raises(TranscriptionError, match="faster-whisper failed: decoder exploded"):
        t.transcribe(audio)


def openai_t(handler: Any, **kw: Any) -> OpenAITranscriber:
    return OpenAITranscriber(
        "https://asr.example/v1/",
        kw.pop("api_key", "sk-asr"),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kw,
    )


def test_openai_transcriber_request_and_segments(audio: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(
            200,
            json={
                "text": "ignored",
                "segments": [
                    {"start": 0, "end": 1.25, "text": " Hi "},
                    {"start": 2, "end": 3, "text": ""},
                    "junk",
                    {"start": None, "end": None, "text": "there"},
                ],
            },
        )

    t = openai_t(handler, model="whisper-large-v3", language="en")
    assert t.transcribe(audio) == [Segment(0.0, 1.25, "Hi"), Segment(0.0, 0.0, "there")]
    req = seen[0]
    assert str(req.url) == "https://asr.example/v1/audio/transcriptions"
    assert req.headers["authorization"] == "Bearer sk-asr"
    body = req.content.decode("latin-1")
    assert 'name="model"\r\n\r\nwhisper-large-v3' in body
    assert 'name="response_format"\r\n\r\nverbose_json' in body
    assert 'name="language"\r\n\r\nen' in body
    assert 'filename="call.wav"' in body
    assert t.name == "openai"


def test_openai_transcriber_no_key_no_language(audio: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"text": "plain text"})

    t = openai_t(handler, api_key="")
    assert t.transcribe(audio) == [Segment(0.0, 0.0, "plain text")]
    assert "authorization" not in seen[0].headers
    assert 'name="language"' not in seen[0].content.decode("latin-1")
    assert t.model == "whisper-1"


def test_openai_transcriber_requires_url() -> None:
    with pytest.raises(ValueError, match="base URL"):
        OpenAITranscriber("", "k")


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(401, text="bad key"), "HTTP 401: bad key"),
        (httpx.Response(200, text="<html>"), "not JSON"),
        (httpx.Response(200, json=[1]), "not a JSON object"),
    ],
)
def test_openai_transcriber_errors(audio: Path, response: httpx.Response, message: str) -> None:
    t = openai_t(lambda req: response)
    with pytest.raises(TranscriptionError, match=message):
        t.transcribe(audio)


def test_openai_transcriber_transport_and_file_errors(audio: Path, tmp_path: Path) -> None:
    def boom(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(TranscriptionError, match="request failed"):
        openai_t(boom).transcribe(audio)
    with pytest.raises(TranscriptionError, match="cannot read"):
        openai_t(boom).transcribe(tmp_path / "missing.wav")


def test_segments_from_openai_edge_cases() -> None:
    assert segments_from_openai({"text": "   "}) == []
    assert segments_from_openai({}) == []
    assert segments_from_openai({"segments": [], "text": "fallback"}) == [
        Segment(0.0, 0.0, "fallback")
    ]


def test_build_transcriber(tmp_path: Path) -> None:
    assert build_transcriber(make_settings(tmp_path, ASR_PROVIDER="none")) is None
    assert isinstance(build_transcriber(make_settings(tmp_path)), MockTranscriber)
    fw = build_transcriber(
        make_settings(
            tmp_path,
            ASR_PROVIDER="faster-whisper",
            WHISPER_MODEL_SIZE="tiny",
            WHISPER_DEVICE="cpu",
            WHISPER_COMPUTE_TYPE="float32",
            ASR_LANGUAGE="es",
        )
    )
    assert isinstance(fw, FasterWhisperTranscriber)
    assert (fw.model_size, fw.device, fw.compute_type, fw.language) == (
        "tiny",
        "cpu",
        "float32",
        "es",
    )
    oa = build_transcriber(
        make_settings(
            tmp_path,
            ASR_PROVIDER="openai",
            LLM_BASE_URL="https://api.groq.com/openai/v1",
            ASR_MODEL="whisper-large-v3",
        )
    )
    assert isinstance(oa, OpenAITranscriber)
    assert oa.base_url == "https://api.groq.com/openai/v1"
    assert oa.model == "whisper-large-v3"
