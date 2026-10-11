"""Password hashing, JWT session tokens and API keys."""

from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

ALGORITHM = "HS256"
API_KEY_PREFIX = "ci_"
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_BYTES = 72  # bcrypt's hard input limit


class PasswordPolicyError(ValueError):
    pass


def validate_password(password: str) -> None:
    """Enforce a minimal, explicit password policy."""
    if len(password) < MIN_PASSWORD_LENGTH:
        raise PasswordPolicyError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise PasswordPolicyError(f"Password must be at most {MAX_PASSWORD_BYTES} bytes")
    if password.isalpha() or password.isdigit():
        raise PasswordPolicyError("Password must mix letters with digits or symbols")


def hash_password(password: str, rounds: int = 12) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=rounds)).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("ascii"))
    except ValueError:
        # Malformed hash or an over-long password: never authenticate.
        return False


# A real bcrypt hash used to equalize timing when the user does not exist.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16), rounds=4)


def dummy_verify(plain: str) -> None:
    verify_password(plain, _DUMMY_HASH)


def create_access_token(
    subject: str, secret: str, expires_minutes: int, now: datetime | None = None
) -> str:
    issued = now or datetime.now(UTC)
    payload = {
        "sub": subject,
        "iat": int(issued.timestamp()),
        "exp": int((issued + timedelta(minutes=expires_minutes)).timestamp()),
        "typ": "access",
    }
    return jwt.encode(payload, secret, algorithm=ALGORITHM)


def decode_access_token(token: str, secret: str) -> str | None:
    """Return the subject of a valid, unexpired access token, else ``None``."""
    try:
        payload = jwt.decode(
            token, secret, algorithms=[ALGORITHM], options={"require": ["exp", "sub"]}
        )
    except jwt.PyJWTError:
        return None
    if payload.get("typ") != "access":
        return None
    sub = payload.get("sub")
    return sub if isinstance(sub, str) and sub else None


def generate_api_key() -> tuple[str, str, str]:
    """Return ``(plaintext, prefix, sha256_hex)``. Only the hash is stored."""
    raw = API_KEY_PREFIX + secrets.token_urlsafe(32)
    return raw, raw[:10], hash_api_key(raw)


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def is_api_key(token: str) -> bool:
    return token.startswith(API_KEY_PREFIX)
