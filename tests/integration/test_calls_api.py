"""Call intake -> scoring -> retrieval -> editing -> export, end to end over HTTP."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from callinsights.ai.asr import MockTranscriber
from tests.conftest import GOOD_TRANSCRIPT, build_client, login, make_settings

pytestmark = pytest.mark.integration

VTT = (
    "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n"
    "Agent: Hi, what is your biggest challenge?\n\n"
    "00:00:02.500 --> 00:00:04.000\n"
    "Prospect: Volume is too high.\n"
)


def create(client: TestClient, **fields: object) -> dict[str, object]:
    body = {"contact_name": "Test Prospect", "agent": "Test Agent", **fields}
    body.setdefault("transcript_text", GOOD_TRANSCRIPT)
    r = client.post("/api/calls", json=body)
    assert r.status_code == 201, r.text
    data: dict[str, object] = r.json()
    return data


def test_create_json_scores_and_persists(client: TestClient) -> None:
    data = create(
        client,
        contact_phone="+1 555 0100",
        started_at="2026-10-01T09:30:00+08:00",
        duration_sec=300,
    )
    assert data["status"] == "scored"
    assert isinstance(data["score"], int)
    assert 0 <= data["score"] <= 100
    assert data["outcome"] in ("Interested", "Follow Up", "Not Interested")
    assert data["url"] == f"/calls/{data['id']}"
    assert data["started_at"] == "2026-10-01T01:30:00"
    assert data["duration_sec"] == 300
    assert data["coaching"]
    assert data["summary"]
    assert data["insights_source"] == "local"

    got = client.get(f"/api/calls/{data['id']}").json()
    assert got["score"] == data["score"]
    assert got["transcript"] == GOOD_TRANSCRIPT

    analysis = client.get(f"/api/calls/{data['id']}/analysis").json()
    assert analysis["analytics"]["score"] == data["score"]
    assert analysis["turns"][0]["role"] == "agent"


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"contact_name": "Nobody"}, 400),
        ({"transcript_text": "no speakers here"}, 400),
        ({"transcript_text": "Agent: hi", "duration_sec": -1}, 422),
        ({"transcript_text": "Agent: hi", "unknown": 1}, 422),
        ({"transcript_text": "Agent: hi", "contact_name": "x" * 300}, 422),
    ],
)
def test_create_validation(client: TestClient, body: dict[str, object], status: int) -> None:
    assert client.post("/api/calls", json=body).status_code == status


def test_create_rejects_non_json_body(client: TestClient) -> None:
    r = client.post("/api/calls", content=b"{oops", headers={"Content-Type": "application/json"})
    assert r.status_code == 400


def test_transcript_too_long(tmp_path: Path) -> None:
    with build_client(make_settings(tmp_path, MAX_TRANSCRIPT_CHARS=1000)) as c:
        login(c)
        r = c.post("/api/calls", json={"transcript_text": "Agent: " + "x" * 2000})
        assert r.status_code == 400
        assert "exceeds 1000" in r.json()["detail"]


def test_multipart_vtt_upload_with_audio(client: TestClient, settings: object) -> None:
    r = client.post(
        "/api/calls",
        data={"contact_name": "VTT Prospect", "agent": "Test Agent", "duration_sec": "75"},
        files={
            "transcript_file": ("call.vtt", VTT, "text/vtt"),
            "audio_file": ("rec.mp3", b"\xff\xfb" + b"\x00" * 200, "audio/mpeg"),
        },
    )
    assert r.status_code == 201, r.text
    data = r.json()
    assert data["status"] == "scored"
    assert data["has_audio"] is True
    assert data["duration_sec"] == 75
    assert data["transcript"].startswith("Agent: Hi, what is your biggest challenge?")
    audio = client.get(f"/api/calls/{data['id']}/audio")
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/mpeg"
    assert audio.content.startswith(b"\xff\xfb")
    assert client.get(f"/audio/{data['id']}").status_code == 200  # legacy URL
    page = client.get(f"/calls/{data['id']}")
    assert "wavesurfer" in page.text.lower()


def test_multipart_validation(client: TestClient) -> None:
    bad_type = client.post(
        "/api/calls", files={"transcript_file": ("x.pdf", b"%PDF", "application/pdf")}
    )
    assert bad_type.status_code == 400
    bad_audio = client.post(
        "/api/calls",
        data={"transcript_text": GOOD_TRANSCRIPT},
        files={"audio_file": ("x.exe", b"MZ", "application/octet-stream")},
    )
    assert bad_audio.status_code == 400
    bad_field = client.post("/api/calls", data={"transcript_text": "Agent: x", "duration_sec": "abc"})
    assert bad_field.status_code == 422


def test_audio_only_is_transcribed_in_background(client: TestClient) -> None:
    r = client.post(
        "/api/calls",
        data={"agent": "Maya"},
        files={"audio_file": ("call.wav", b"RIFF" + b"\x00" * 100, "audio/wav")},
    )
    assert r.status_code == 202
    assert r.json()["status"] == "queued"
    cid = r.json()["id"]
    got = client.get(f"/api/calls/{cid}").json()  # TestClient runs background tasks inline
    assert got["status"] == "scored"
    assert got["transcript_source"] == "asr:mock:heuristic"
    assert got["transcript"].startswith("Agent: ")
    assert got["score"] > 0

    again = client.post(f"/api/calls/{cid}/transcribe")
    assert again.status_code == 202
    assert client.get(f"/api/calls/{cid}").json()["status"] == "scored"


def test_audio_only_without_asr_is_rejected(tmp_path: Path) -> None:
    s = make_settings(tmp_path, ASR_PROVIDER="none")
    with build_client(s) as c:
        login(c)
        r = c.post("/api/calls", files={"audio_file": ("c.wav", b"RIFF", "audio/wav")})
        assert r.status_code == 400
        assert "ASR_PROVIDER" in r.json()["detail"]
        assert not list((tmp_path / "data" / "uploads").glob("*"))
        call = c.post(
            "/api/calls",
            data={"transcript_text": GOOD_TRANSCRIPT},
            files={"audio_file": ("c.wav", b"RIFF", "audio/wav")},
        ).json()
        assert c.post(f"/api/calls/{call['id']}/transcribe").status_code == 409


def test_upload_too_large(tmp_path: Path) -> None:
    with build_client(make_settings(tmp_path, MAX_UPLOAD_MB=1), MockTranscriber()) as c:
        login(c)
        r = c.post(
            "/api/calls",
            data={"transcript_text": GOOD_TRANSCRIPT},
            files={"audio_file": ("c.wav", b"0" * (1024 * 1024 + 10), "audio/wav")},
        )
        assert r.status_code == 413
        huge = c.post(
            "/api/calls",
            content=b"x",
            headers={"Content-Type": "application/json", "Content-Length": str(50 * 1024 * 1024)},
        )
        assert huge.status_code == 413


def test_transcribe_conflicts(client: TestClient) -> None:
    call = create(client)
    assert client.post(f"/api/calls/{call['id']}/transcribe").status_code == 409  # no audio
    assert client.post("/api/calls/999/transcribe").status_code == 404


def test_list_filters_and_pagination(seeded: TestClient) -> None:
    all_calls = seeded.get("/api/calls").json()
    assert all_calls["total"] == 5
    assert all_calls["limit"] == 50
    assert all_calls["offset"] == 0
    assert {c["contact_name"] for c in all_calls["calls"]} >= {"Dana Whitfield", "Tom Beck"}
    maya = seeded.get("/api/calls", params={"agent": "Maya Santos"}).json()
    assert maya["total"] == 2
    assert all(c["agent"] == "Maya Santos" for c in maya["calls"])
    page = seeded.get("/api/calls", params={"limit": 2, "offset": 4}).json()
    assert len(page["calls"]) == 1
    hi = seeded.get("/api/calls", params={"min_score": 75}).json()
    assert all(c["score"] >= 75 for c in hi["calls"])
    assert seeded.get("/api/calls", params={"q": "COO"}).json()["total"] >= 1
    assert seeded.get("/api/calls", params={"status": "queued"}).json()["total"] == 0
    assert seeded.get("/api/calls", params={"limit": 0}).status_code == 422
    assert seeded.get("/api/calls", params={"min_score": 101}).status_code == 422
    assert seeded.get("/api/agents").json()["agents"] == ["Jake Rivera", "Maya Santos", "Priya Nair"]


def test_analytics_endpoint(seeded: TestClient) -> None:
    a = seeded.get("/api/analytics").json()
    assert a["total_calls"] == 5
    assert sum(a["outcomes"].values()) == 5
    assert len(a["leaderboard"]) == 3
    assert a["leaderboard"][0]["avg_score"] >= a["leaderboard"][-1]["avg_score"]
    assert seeded.get("/api/analytics", params={"agent": "Priya Nair"}).json()["total_calls"] == 1
    assert seeded.get("/api/analytics", params={"days": 1}).json()["total_calls"] == 0


def test_score_endpoint_is_stateless(client: TestClient) -> None:
    r = client.post("/api/score", json={"transcript_text": GOOD_TRANSCRIPT})
    assert r.status_code == 200
    body = r.json()
    assert body["insights"]["source"] == "local"
    assert set(body["components"]) == {
        "questions",
        "talk_balance",
        "qualification",
        "objection_handling",
        "sentiment",
    }
    assert client.get("/api/calls").json()["total"] == 0
    assert client.post("/api/score", json={"transcript_text": "nothing"}).status_code == 400


def test_patch_rescore_delete(client: TestClient) -> None:
    call = create(client, transcript_text="Agent: Hello.\nProspect: Bye.")
    cid = call["id"]
    r = client.patch(f"/api/calls/{cid}", json={"agent": "New Agent", "transcript_text": GOOD_TRANSCRIPT})
    assert r.status_code == 200
    assert r.json()["agent"] == "New Agent"
    assert r.json()["transcript_source"] == "edited"
    assert r.json()["score"] > call["score"]  # type: ignore[operator]
    assert client.patch(f"/api/calls/{cid}", json={"transcript_text": "nope"}).status_code == 400
    assert client.patch(f"/api/calls/{cid}", json={"bogus": 1}).status_code == 422
    assert client.patch("/api/calls/999", json={"agent": "x"}).status_code == 404
    re = client.post(f"/api/calls/{cid}/rescore")
    assert re.status_code == 200
    assert re.json()["score"] == r.json()["score"]
    assert client.post("/api/calls/999/rescore").status_code == 404
    assert client.delete(f"/api/calls/{cid}").status_code == 204
    assert client.get(f"/api/calls/{cid}").status_code == 404
    assert client.delete(f"/api/calls/{cid}").status_code == 404


def test_rescore_without_transcript(tmp_path: Path) -> None:
    class Silent:
        name = "silent"

        def transcribe(self, path: Path) -> list[object]:
            return []

    with build_client(make_settings(tmp_path), Silent()) as c:  # type: ignore[arg-type]
        login(c)
        r = c.post("/api/calls", files={"audio_file": ("c.wav", b"RIFF", "audio/wav")})
        cid = r.json()["id"]
        failed = c.get(f"/api/calls/{cid}").json()
        assert failed["status"] == "failed"
        assert "no speech" in failed["error"]
        assert c.post(f"/api/calls/{cid}/rescore").status_code == 400
        assert "Processing failed" in c.get(f"/calls/{cid}").text


def test_member_can_only_delete_own_calls(client: TestClient) -> None:
    admin_call = create(client)
    client.post("/api/users", json={"email": "rep@example.com", "password": "Rep-passw0rd"})
    rep = TestClient(client.app, headers={"Origin": "http://testserver"})
    login(rep, "rep@example.com", "Rep-passw0rd")
    own = create(rep)
    assert rep.delete(f"/api/calls/{admin_call['id']}").status_code == 403
    assert rep.delete(f"/api/calls/{own['id']}").status_code == 204
    assert client.delete(f"/api/calls/{admin_call['id']}").status_code == 204


def test_export_csv(seeded: TestClient) -> None:
    r = seeded.get("/api/calls/export.csv", params={"agent": "Maya Santos"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("id,created_at,started_at,agent")
    assert len(lines) == 3


def test_audio_missing(client: TestClient) -> None:
    call = create(client)
    assert client.get(f"/api/calls/{call['id']}/audio").status_code == 404
    assert client.get("/api/calls/999/audio").status_code == 404
    assert client.get("/audio/999").status_code == 404
    anon = TestClient(client.app)
    assert anon.get(f"/audio/{call['id']}").status_code == 401
