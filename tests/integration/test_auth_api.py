"""Authentication, sessions, CSRF, API keys, rate limiting and user administration."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from callinsights.ai.asr import MockTranscriber
from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD, build_client, login, make_settings

pytestmark = pytest.mark.integration


def test_login_returns_token_and_cookie(anon: TestClient) -> None:
    r = anon.post(
        "/api/auth/login", json={"email": ADMIN_EMAIL.upper(), "password": ADMIN_PASSWORD}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 720 * 60
    cookie = r.headers["set-cookie"]
    assert "access_token=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Secure" not in cookie  # ENV=test without COOKIE_SECURE


def test_secure_cookie_when_configured(tmp_path: Path) -> None:
    with build_client(make_settings(tmp_path, COOKIE_SECURE=True)) as c:
        r = c.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
        assert "Secure" in r.headers["set-cookie"]
        assert r.headers["strict-transport-security"].startswith("max-age=31536000")


def test_login_failures(anon: TestClient) -> None:
    assert (
        anon.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "nope"}).status_code
        == 401
    )
    assert (
        anon.post("/api/auth/login", json={"email": "x@y.zz", "password": "nope"}).status_code
        == 401
    )
    assert anon.post("/api/auth/login", json={"email": ADMIN_EMAIL}).status_code == 422
    assert (
        anon.post(
            "/api/auth/login", json={"email": ADMIN_EMAIL, "password": "x", "extra": 1}
        ).status_code
        == 422
    )


def test_bearer_token_and_me(anon: TestClient) -> None:
    token = login(anon)
    anon.cookies.clear()
    r = anon.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.json()["email"] == ADMIN_EMAIL
    assert r.json()["role"] == "admin"
    assert anon.get("/api/auth/me", headers={"Authorization": "Bearer junk"}).status_code == 401
    assert anon.get("/api/auth/me", headers={"Authorization": "Bearer "}).status_code == 401
    r = anon.get("/api/auth/me")
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


def test_api_requires_auth(anon: TestClient) -> None:
    for method, path in [
        ("GET", "/api/calls"),
        ("POST", "/api/calls"),
        ("GET", "/api/analytics"),
        ("GET", "/api/users"),
        ("GET", "/api/keys"),
        ("POST", "/api/score"),
        ("GET", "/api/calls/1/audio"),
    ]:
        assert anon.request(method, path).status_code == 401, path


def test_csrf_cookie_writes_need_same_origin(client: TestClient) -> None:
    payload = {"transcript_text": "Agent: hi?\nProspect: yes"}
    ok = client.post("/api/score", json=payload)
    assert ok.status_code == 200
    evil = client.post("/api/score", json=payload, headers={"Origin": "https://evil.example"})
    assert evil.status_code == 403
    assert "CSRF" in evil.json()["detail"]
    client.headers.pop("Origin")
    assert client.post("/api/score", json=payload).status_code == 403
    referer = client.post("/api/score", json=payload, headers={"Referer": "http://testserver/x"})
    assert referer.status_code == 200
    assert client.post("/api/score", json=payload, headers={"Origin": "null"}).status_code == 403
    # Reads are never blocked.
    assert client.get("/api/calls").status_code == 200


def test_trusted_origins(tmp_path: Path) -> None:
    s = make_settings(tmp_path, TRUSTED_ORIGINS="https://app.example.com")
    with build_client(s) as c:
        login(c)
        payload = {"transcript_text": "Agent: hi?\nProspect: yes"}
        r = c.post("/api/score", json=payload, headers={"Origin": "https://app.example.com"})
        assert r.status_code == 200
        r = c.post("/api/score", json=payload, headers={"Origin": "https://other.example.com"})
        assert r.status_code == 403


def test_bearer_writes_skip_csrf(anon: TestClient) -> None:
    token = login(anon)
    anon.cookies.clear()
    r = anon.post(
        "/api/score",
        json={"transcript_text": "Agent: hi?\nProspect: yes"},
        headers={"Authorization": f"Bearer {token}", "Origin": "https://evil.example"},
    )
    assert r.status_code == 200


def test_rate_limit(tmp_path: Path) -> None:
    with build_client(make_settings(tmp_path, LOGIN_MAX_ATTEMPTS=2)) as c:
        bad = {"email": ADMIN_EMAIL, "password": "wrong-one"}
        assert c.post("/api/auth/login", json=bad).status_code == 401
        assert c.post("/api/auth/login", json=bad).status_code == 401
        r = c.post("/api/auth/login", json=bad)
        assert r.status_code == 429
        assert int(r.headers["retry-after"]) >= 1
        good = {"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
        assert c.post("/api/auth/login", json=good).status_code == 429
        form = c.post("/login", data=good, follow_redirects=False)
        assert form.headers["location"] == "/login?error=rate"


def test_successful_login_resets_limiter(tmp_path: Path) -> None:
    with build_client(make_settings(tmp_path, LOGIN_MAX_ATTEMPTS=2)) as c:
        bad = {"email": ADMIN_EMAIL, "password": "wrong-one"}
        assert c.post("/api/auth/login", json=bad).status_code == 401
        login(c)  # success clears the earlier failure
        assert c.post("/api/auth/login", json=bad).status_code == 401
        assert c.post("/api/auth/login", json=bad).status_code == 401
        assert c.post("/api/auth/login", json=bad).status_code == 429


def test_html_login_logout(anon: TestClient) -> None:
    assert anon.get("/", follow_redirects=False).headers["location"] == "/login"
    page = anon.get("/login")
    assert page.status_code == 200
    assert "Sign in" in page.text
    assert "Invalid email" in anon.get("/login?error=1").text
    assert "Too many attempts" in anon.get("/login?error=rate").text
    bad = anon.post("/login", data={"email": ADMIN_EMAIL, "password": "x"}, follow_redirects=False)
    assert bad.headers["location"] == "/login?error=1"
    ok = anon.post(
        "/login", data={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}, follow_redirects=False
    )
    assert ok.status_code == 303
    assert ok.headers["location"] == "/"
    assert anon.get("/login", follow_redirects=False).headers["location"] == "/"
    out = anon.get("/logout", follow_redirects=False)
    assert out.headers["location"] == "/login"
    assert anon.get("/api/auth/me").status_code == 401


def test_logout_api(client: TestClient) -> None:
    r = client.post("/api/auth/logout")
    assert r.status_code == 204
    assert 'access_token=""' in r.headers["set-cookie"]


def test_change_password(client: TestClient) -> None:
    bad = client.post(
        "/api/auth/password", json={"current_password": "nope", "new_password": "N3w-password"}
    )
    assert bad.status_code == 400
    weak = client.post(
        "/api/auth/password", json={"current_password": ADMIN_PASSWORD, "new_password": "short"}
    )
    assert weak.status_code == 400
    ok = client.post(
        "/api/auth/password",
        json={"current_password": ADMIN_PASSWORD, "new_password": "N3w-password"},
    )
    assert ok.status_code == 204
    login(client, ADMIN_EMAIL, "N3w-password")


def test_api_keys_flow(client: TestClient) -> None:
    created = client.post("/api/keys", json={"name": "dialer"})
    assert created.status_code == 201
    key = created.json()
    assert key["key"].startswith("ci_")
    assert key["prefix"] == key["key"][:10]
    listed = client.get("/api/keys").json()
    assert [k["name"] for k in listed] == ["dialer"]
    assert "key" not in listed[0]

    machine = TestClient(client.app)
    auth = {"Authorization": f"Bearer {key['key']}"}
    assert machine.get("/api/calls", headers=auth).status_code == 200
    assert client.get("/api/keys").json()[0]["last_used_at"] is not None

    assert client.delete(f"/api/keys/{key['id']}").status_code == 204
    assert machine.get("/api/calls", headers=auth).status_code == 401
    assert client.delete("/api/keys/9999").status_code == 404
    assert client.post("/api/keys", json={"name": ""}).status_code == 422


def test_user_admin(client: TestClient) -> None:
    r = client.post(
        "/api/users",
        json={"email": "Rep@Example.com", "password": "Rep-passw0rd", "name": "Rep"},
    )
    assert r.status_code == 201
    rep = r.json()
    assert rep["email"] == "rep@example.com"
    assert rep["role"] == "member"
    dup = client.post("/api/users", json={"email": "rep@example.com", "password": "Rep-passw0rd"})
    assert dup.status_code == 409
    assert client.post("/api/users", json={"email": "bad", "password": "x"}).status_code == 422
    weak = client.post("/api/users", json={"email": "w@example.com", "password": "weak"})
    assert weak.status_code == 400
    assert len(client.get("/api/users").json()) == 2

    upd = client.patch(f"/api/users/{rep['id']}", json={"name": "Rep 2", "role": "admin"})
    assert upd.json()["name"] == "Rep 2"
    assert upd.json()["role"] == "admin"
    client.patch(f"/api/users/{rep['id']}", json={"role": "member", "password": "An0ther-pass"})
    assert client.patch(f"/api/users/{rep['id']}", json={"password": "weak"}).status_code == 400
    assert client.patch("/api/users/999", json={"name": "x"}).status_code == 404

    member = TestClient(client.app, headers={"Origin": "http://testserver"})
    login(member, "rep@example.com", "An0ther-pass")
    assert member.get("/api/users").status_code == 403
    assert member.get("/api/auth/me").json()["role"] == "member"

    disabled = client.patch(f"/api/users/{rep['id']}", json={"is_active": False})
    assert disabled.json()["is_active"] is False
    assert member.get("/api/auth/me").status_code == 401
    assert (
        member.post(
            "/api/auth/login", json={"email": "rep@example.com", "password": "An0ther-pass"}
        ).status_code
        == 401
    )
    assert client.delete(f"/api/users/{rep['id']}").status_code == 204
    assert client.delete(f"/api/users/{rep['id']}").status_code == 404


def test_last_admin_is_protected(client: TestClient) -> None:
    me = client.get("/api/auth/me").json()
    assert client.patch(f"/api/users/{me['id']}", json={"role": "member"}).status_code == 409
    assert client.patch(f"/api/users/{me['id']}", json={"is_active": False}).status_code == 409
    assert client.delete(f"/api/users/{me['id']}").status_code == 409
    second = client.post(
        "/api/users",
        json={"email": "a2@example.com", "password": "Adm1n-two-pw", "role": "admin"},
    ).json()
    assert client.patch(f"/api/users/{me['id']}", json={"role": "member"}).status_code == 200
    assert client.delete(f"/api/users/{second['id']}").status_code == 403  # no longer admin


def test_no_admin_configured_logs_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    s = make_settings(tmp_path, ADMIN_EMAIL=None, ADMIN_PASSWORD=None)
    with caplog.at_level("WARNING"), build_client(s, MockTranscriber()) as c:
        assert c.get("/healthz").status_code == 200
    assert any("no users exist" in r.getMessage() for r in caplog.records)
