from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from callinsights import security

SECRET = "x" * 40


@pytest.mark.parametrize(
    ("password", "message"),
    [
        ("short1!", "at least 10"),
        ("a" * 9 + "1", None),
        ("abcdefghij", "mix letters"),
        ("1234567890", "mix letters"),
        ("é" * 36 + "1", "at most 72"),
    ],
)
def test_validate_password(password: str, message: str | None) -> None:
    if message is None:
        security.validate_password(password)
        return
    with pytest.raises(security.PasswordPolicyError, match=message):
        security.validate_password(password)


def test_password_length_boundaries() -> None:
    security.validate_password("abcdefghi1")  # exactly 10 chars
    with pytest.raises(security.PasswordPolicyError):
        security.validate_password("abcdefgh1")  # 9 chars
    security.validate_password("a" * 71 + "1")  # exactly 72 bytes
    with pytest.raises(security.PasswordPolicyError):
        security.validate_password("a" * 72 + "1")  # 73 bytes


def test_hash_and_verify_roundtrip() -> None:
    hashed = security.hash_password("Correct-horse-1", rounds=4)
    assert hashed.startswith("$2b$04$")
    assert security.verify_password("Correct-horse-1", hashed) is True
    assert security.verify_password("wrong-horse-1", hashed) is False


def test_hash_uses_requested_rounds() -> None:
    assert security.hash_password("pw-123456789", rounds=5).startswith("$2b$05$")


def test_verify_rejects_malformed_hash() -> None:
    assert security.verify_password("anything", "not-a-bcrypt-hash") is False


def test_dummy_verify_does_not_raise() -> None:
    assert security.dummy_verify("whatever") is None


def test_token_roundtrip_and_claims() -> None:
    now = datetime.now(UTC)
    token = security.create_access_token("a@b.co", SECRET, 30, now=now)
    payload = jwt.decode(token, SECRET, algorithms=["HS256"])
    assert payload["sub"] == "a@b.co"
    assert payload["typ"] == "access"
    assert payload["iat"] == int(now.timestamp())
    assert payload["exp"] == int((now + timedelta(minutes=30)).timestamp())
    assert security.decode_access_token(token, SECRET) == "a@b.co"


def test_token_default_now_is_current() -> None:
    before = int(datetime.now(UTC).timestamp())
    payload = jwt.decode(
        security.create_access_token("a@b.co", SECRET, 1), SECRET, algorithms=["HS256"]
    )
    assert before <= payload["iat"] <= before + 5
    assert payload["exp"] - payload["iat"] == 60


def test_token_wrong_secret_is_rejected() -> None:
    token = security.create_access_token("a@b.co", SECRET, 30)
    assert security.decode_access_token(token, "y" * 40) is None


def test_expired_token_is_rejected() -> None:
    past = datetime.now(UTC) - timedelta(hours=2)
    token = security.create_access_token("a@b.co", SECRET, 30, now=past)
    assert security.decode_access_token(token, SECRET) is None


def test_token_with_wrong_type_or_subject_is_rejected() -> None:
    exp = int((datetime.now(UTC) + timedelta(minutes=5)).timestamp())
    refresh = jwt.encode({"sub": "a@b.co", "exp": exp, "typ": "refresh"}, SECRET, "HS256")
    assert security.decode_access_token(refresh, SECRET) is None
    untyped = jwt.encode({"sub": "a@b.co", "exp": exp}, SECRET, "HS256")
    assert security.decode_access_token(untyped, SECRET) is None
    empty_sub = jwt.encode({"sub": "", "exp": exp, "typ": "access"}, SECRET, "HS256")
    assert security.decode_access_token(empty_sub, SECRET) is None
    no_exp = jwt.encode({"sub": "a@b.co", "typ": "access"}, SECRET, "HS256")
    assert security.decode_access_token(no_exp, SECRET) is None


def test_token_with_other_algorithm_is_rejected() -> None:
    exp = int((datetime.now(UTC) + timedelta(minutes=5)).timestamp())
    token = jwt.encode({"sub": "a@b.co", "exp": exp, "typ": "access"}, SECRET, "HS512")
    assert security.decode_access_token(token, SECRET) is None


def test_garbage_token_is_rejected() -> None:
    assert security.decode_access_token("not.a.jwt", SECRET) is None


def test_generate_api_key() -> None:
    raw, prefix, digest = security.generate_api_key()
    assert raw.startswith("ci_")
    assert len(raw) > 40
    assert prefix == raw[:10]
    assert len(prefix) == 10
    assert digest == security.hash_api_key(raw)
    assert len(digest) == 64
    other, _, _ = security.generate_api_key()
    assert other != raw


def test_hash_api_key_is_sha256() -> None:
    assert security.hash_api_key("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_is_api_key() -> None:
    assert security.is_api_key("ci_abc") is True
    assert security.is_api_key("eyJhbGciOi") is False
