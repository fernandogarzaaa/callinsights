"""Product flow tests: intake -> scoring -> dashboard/detail/audio."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

TRANSCRIPT = """Agent: Hi, thanks for taking the time. What is your biggest support challenge right now?
Prospect: Ticket volume. We are drowning.
Agent: How many tickets per week?
Prospect: About three hundred.
Agent: And what budget have you set aside for tooling?
Prospect: Around a thousand a month. But your pricing looks expensive.
Agent: Fair point. What if we start with a pilot so the ROI is proven first?
Prospect: That sounds good. What is the next step?
Agent: I will send a proposal and we can schedule a follow up on Thursday."""


def _login():
    r = client.post(
        "/api/auth/login",
        json={"email": "admin@clusterx.local", "password": "ChangeMe123!"},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def test_dashboard_renders_with_seed_data():
    _login()  # cookie jar carries the session
    r = client.get("/")
    assert r.status_code == 200
    assert "Call Analytics" in r.text
    assert "Dana Whitfield" in r.text  # seeded call


def test_agent_filter():
    _login()
    r = client.get("/", params={"agent": "Maya Santos"})
    assert r.status_code == 200
    assert "Dana Whitfield" in r.text
    assert "Tom Beck" not in r.text


def test_create_call_json_scores_and_persists():
    _login()
    r = client.post(
        "/api/calls",
        json={
            "contact_name": "Test Prospect",
            "contact_phone": "+1 555 0100",
            "agent": "Test Agent",
            "transcript_text": TRANSCRIPT,
        },
    )
    assert r.status_code == 201, r.text
    data = r.json()
    assert 0 <= data["score"] <= 100
    assert data["outcome"] in ("Interested", "Follow Up", "Not Interested")
    assert data["url"].startswith("/calls/")

    call_id = data["id"]
    d = client.get(f"/calls/{call_id}")
    assert d.status_code == 200
    assert "Test Prospect" in d.text
    assert "Coaching recommendations" in d.text
    assert "Key highlights" in d.text
    assert str(data["score"]) in d.text


def test_create_call_requires_transcript():
    _login()
    r = client.post("/api/calls", json={"contact_name": "Nobody"})
    assert r.status_code == 400


def test_create_call_vtt_upload_and_audio():
    _login()
    vtt = (
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n"
        "Agent: Hi, what is your biggest challenge?\n\n"
        "00:00:02.500 --> 00:00:04.000\n"
        "Prospect: Volume is too high.\n"
    )
    r = client.post(
        "/api/calls",
        data={"contact_name": "VTT Prospect", "agent": "Test Agent"},
        files={
            "transcript_file": ("call.vtt", vtt, "text/vtt"),
            "audio_file": ("rec.mp3", b"\xff\xfb" + b"\x00" * 200, "audio/mpeg"),
        },
    )
    assert r.status_code == 201, r.text
    call_id = r.json()["id"]
    a = client.get(f"/audio/{call_id}")
    assert a.status_code == 200
    assert a.headers["content-type"].startswith("audio/")
    d = client.get(f"/calls/{call_id}")
    assert d.status_code == 200
    assert "wavesurfer" in d.text.lower()


def test_api_list_and_filter():
    _login()
    r = client.get("/api/calls")
    assert r.status_code == 200
    calls = r.json()["calls"]
    assert len(calls) >= 5  # seeded
    assert all("score" in c and "outcome" in c for c in calls)

    r2 = client.get("/api/calls", params={"agent": "Priya Nair"})
    assert r2.status_code == 200
    names = [c["agent"] for c in r2.json()["calls"]]
    assert names and all(n == "Priya Nair" for n in names)


def test_new_call_page_and_settings():
    _login()
    assert client.get("/calls/new").status_code == 200
    s = client.get("/settings")
    assert s.status_code == 200
    assert "AI provider" in s.text


def test_api_requires_auth():
    anon = TestClient(app)
    assert anon.get("/api/calls").status_code == 401
    assert anon.post("/api/calls", json={}).status_code == 401
