"""User management (admin only) and personal API keys."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select

from callinsights import models
from callinsights.schemas import (
    ApiKeyCreate,
    ApiKeyCreated,
    ApiKeyOut,
    UserCreate,
    UserOut,
    UserUpdate,
)
from callinsights.services import users as user_service
from callinsights.web.deps import AdminDep, DbDep, SettingsDep, UserDep

router = APIRouter(prefix="/api")


@router.get("/users", response_model=list[UserOut], tags=["users"])
def list_users(_: AdminDep, db: DbDep) -> list[models.User]:
    return list(db.scalars(select(models.User).order_by(models.User.id)))


@router.post("/users", response_model=UserOut, status_code=201, tags=["users"])
def create_user(payload: UserCreate, _: AdminDep, db: DbDep, settings: SettingsDep) -> models.User:
    try:
        return user_service.create_user(
            db,
            payload.email,
            payload.password,
            name=payload.name,
            role=payload.role,
            rounds=settings.BCRYPT_ROUNDS,
        )
    except user_service.DuplicateUserError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except user_service.UserError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


def _get_user(db: DbDep, user_id: int) -> models.User:
    user = db.get(models.User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return user


def _guard_last_admin(db: DbDep, user: models.User) -> None:
    if user.is_admin and user.is_active and user_service.active_admin_count(db) <= 1:
        raise HTTPException(status.HTTP_409_CONFLICT, "Cannot remove the last active admin")


@router.patch("/users/{user_id}", response_model=UserOut, tags=["users"])
def update_user(
    user_id: int, payload: UserUpdate, _: AdminDep, db: DbDep, settings: SettingsDep
) -> models.User:
    user = _get_user(db, user_id)
    if payload.role == "member" or payload.is_active is False:
        _guard_last_admin(db, user)
    if payload.name is not None:
        user.name = payload.name
    if payload.role is not None:
        user.role = payload.role
    if payload.is_active is not None:
        user.is_active = payload.is_active
    if payload.password is not None:
        try:
            user_service.set_password(user, payload.password, settings.BCRYPT_ROUNDS)
        except user_service.UserError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    db.flush()
    return user


@router.delete("/users/{user_id}", status_code=204, tags=["users"])
def delete_user(user_id: int, _: AdminDep, db: DbDep) -> Response:
    user = _get_user(db, user_id)
    _guard_last_admin(db, user)
    db.delete(user)
    db.flush()
    return Response(status_code=204)


@router.get("/keys", response_model=list[ApiKeyOut], tags=["api-keys"])
def list_keys(user: UserDep, db: DbDep) -> list[models.ApiKey]:
    return list(
        db.scalars(
            select(models.ApiKey).where(models.ApiKey.user_id == user.id).order_by(models.ApiKey.id)
        )
    )


@router.post("/keys", response_model=ApiKeyCreated, status_code=201, tags=["api-keys"])
def create_key(payload: ApiKeyCreate, user: UserDep, db: DbDep) -> ApiKeyCreated:
    key, raw = user_service.create_api_key(db, user, payload.name)
    return ApiKeyCreated(
        id=key.id,
        name=key.name,
        prefix=key.prefix,
        created_at=key.created_at,
        last_used_at=key.last_used_at,
        revoked_at=key.revoked_at,
        key=raw,
    )


@router.delete("/keys/{key_id}", status_code=204, tags=["api-keys"])
def revoke_key(key_id: int, user: UserDep, db: DbDep) -> Response:
    if not user_service.revoke_api_key(db, user, key_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "API key not found")
    return Response(status_code=204)
