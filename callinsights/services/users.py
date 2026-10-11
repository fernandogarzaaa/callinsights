"""User accounts, authentication and API keys."""

from __future__ import annotations

import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from callinsights import models, security
from callinsights.schemas import normalize_email

logger = logging.getLogger(__name__)


class UserError(ValueError):
    pass


class DuplicateUserError(UserError):
    pass


def get_by_email(db: Session, email: str) -> models.User | None:
    return db.scalar(select(models.User).where(models.User.email == email.strip().lower()))


def create_user(
    db: Session,
    email: str,
    password: str,
    *,
    name: str = "",
    role: str = "member",
    rounds: int = 12,
) -> models.User:
    try:
        email = normalize_email(email)
    except ValueError as exc:
        raise UserError(str(exc)) from exc
    if role not in models.ROLES:
        raise UserError(f"role must be one of {models.ROLES}")
    try:
        security.validate_password(password)
    except security.PasswordPolicyError as exc:
        raise UserError(str(exc)) from exc
    if get_by_email(db, email):
        raise DuplicateUserError("a user with that email already exists")
    user = models.User(
        email=email,
        name=name.strip(),
        role=role,
        hashed_password=security.hash_password(password, rounds),
        is_active=True,
    )
    db.add(user)
    db.flush()
    logger.info("user created", extra={"user_id": user.id, "role": role})
    return user


def set_password(user: models.User, password: str, rounds: int = 12) -> None:
    try:
        security.validate_password(password)
    except security.PasswordPolicyError as exc:
        raise UserError(str(exc)) from exc
    user.hashed_password = security.hash_password(password, rounds)


def authenticate(db: Session, email: str, password: str) -> models.User | None:
    user = get_by_email(db, email)
    if user is None:
        security.dummy_verify(password)
        return None
    if not security.verify_password(password, user.hashed_password) or not user.is_active:
        return None
    return user


def active_admin_count(db: Session) -> int:
    count = db.scalar(
        select(func.count())
        .select_from(models.User)
        .where(models.User.role == "admin", models.User.is_active.is_(True))
    )
    return int(count or 0)


def ensure_admin(db: Session, email: str | None, password: str | None, rounds: int = 12) -> bool:
    """Create the bootstrap admin from settings if configured and absent. Returns True if made."""
    if not email or not password:
        return False
    if get_by_email(db, email):
        return False
    create_user(db, email, password, name="Admin", role="admin", rounds=rounds)
    return True


def create_api_key(db: Session, user: models.User, name: str) -> tuple[models.ApiKey, str]:
    raw, prefix, digest = security.generate_api_key()
    key = models.ApiKey(user_id=user.id, name=name.strip(), prefix=prefix, key_hash=digest)
    db.add(key)
    db.flush()
    logger.info("api key created", extra={"user_id": user.id, "key_id": key.id})
    return key, raw


def user_for_api_key(db: Session, raw: str) -> models.User | None:
    key = db.scalar(
        select(models.ApiKey).where(models.ApiKey.key_hash == security.hash_api_key(raw))
    )
    if key is None or key.revoked_at is not None:
        return None
    user = key.user
    if not user.is_active:
        return None
    key.last_used_at = models.utcnow()
    return user


def revoke_api_key(db: Session, user: models.User, key_id: int) -> bool:
    key = db.get(models.ApiKey, key_id)
    if key is None or (key.user_id != user.id and not user.is_admin):
        return False
    if key.revoked_at is None:
        key.revoked_at = models.utcnow()
    return True
