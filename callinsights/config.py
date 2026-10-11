"""Application configuration, loaded from environment variables (and an optional .env file).

Every setting can be overridden by an environment variable of the same name.
See ``.env.example`` for a documented template.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

INSECURE_SECRET_KEYS = frozenset({"", "change-me", "change-me-in-production", "secret", "changeme"})
INSECURE_PASSWORDS = frozenset({"ChangeMe123!", "admin", "password", "changeme"})
MIN_PRODUCTION_SECRET_LENGTH = 32


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=True)

    # --- Runtime -----------------------------------------------------------------------------
    ENV: Literal["development", "production", "test"] = "development"
    APP_NAME: str = "CallInsights"
    HOST: str = "0.0.0.0"  # noqa: S104 - container default; bind narrower via env if needed
    PORT: int = Field(default=8003, ge=1, le=65535)
    WORKERS: int = Field(default=1, ge=1, le=64)
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FORMAT: Literal["json", "text"] = "json"

    # --- Storage -----------------------------------------------------------------------------
    DATABASE_URL: str = "sqlite:///./data/callinsights.db"
    AUTO_MIGRATE: bool = True
    DATA_DIR: Path = Path("./data")
    MAX_UPLOAD_MB: int = Field(default=200, ge=1, le=4096)
    MAX_TRANSCRIPT_CHARS: int = Field(default=500_000, ge=1_000)

    # --- Security ----------------------------------------------------------------------------
    SECRET_KEY: SecretStr = SecretStr("change-me-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=720, ge=5, le=60 * 24 * 30)
    COOKIE_SECURE: bool | None = None  # defaults to True in production
    BCRYPT_ROUNDS: int = Field(default=12, ge=4, le=16)
    LOGIN_MAX_ATTEMPTS: int = Field(default=10, ge=1)
    LOGIN_WINDOW_SECONDS: int = Field(default=300, ge=1)
    TRUSTED_ORIGINS: str = ""  # comma-separated extra origins allowed for cookie-auth writes

    # --- Bootstrap ---------------------------------------------------------------------------
    ADMIN_EMAIL: str | None = None
    ADMIN_PASSWORD: SecretStr | None = None
    SEED_DEMO_DATA: bool = False

    # --- LLM (OpenAI-compatible: OpenAI, Groq, OpenRouter, Ollama, vLLM, LM Studio...) ---------
    LLM_PROVIDER: Literal["auto", "local", "openai"] = "auto"
    LLM_BASE_URL: str = ""
    LLM_API_KEY: SecretStr = SecretStr("")
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)
    LLM_MAX_RETRIES: int = Field(default=2, ge=0, le=10)
    LLM_MAX_TRANSCRIPT_CHARS: int = Field(default=24_000, ge=1_000)
    LLM_SPEAKER_LABELING: bool = True

    # --- Speech-to-text ----------------------------------------------------------------------
    ASR_PROVIDER: Literal["none", "mock", "faster-whisper", "openai"] = "none"
    ASR_BASE_URL: str = ""  # falls back to LLM_BASE_URL
    ASR_API_KEY: SecretStr = SecretStr("")  # falls back to LLM_API_KEY
    ASR_MODEL: str = "whisper-1"
    ASR_LANGUAGE: str | None = None
    ASR_TIMEOUT_SECONDS: float = Field(default=600.0, gt=0)
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "auto"
    WHISPER_COMPUTE_TYPE: str = "int8"
    DIARIZATION_GAP_SECONDS: float = Field(default=0.8, ge=0)
    ASR_FIRST_SPEAKER: Literal["agent", "prospect"] = "agent"

    @field_validator("DATABASE_URL")
    @classmethod
    def _normalize_db_url(cls, value: str) -> str:
        # Heroku/Render style URLs use the legacy "postgres://" scheme; we ship psycopg 3.
        if value.startswith("postgres://"):
            value = "postgresql+psycopg://" + value[len("postgres://") :]
        elif value.startswith("postgresql://"):
            value = "postgresql+psycopg://" + value[len("postgresql://") :]
        return value

    @field_validator("LLM_BASE_URL", "ASR_BASE_URL")
    @classmethod
    def _strip_slash(cls, value: str) -> str:
        return value.strip().rstrip("/")

    @field_validator("ADMIN_EMAIL")
    @classmethod
    def _normalize_email(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return value.strip().lower()

    @field_validator("ADMIN_PASSWORD")
    @classmethod
    def _empty_password_is_none(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None or not value.get_secret_value():
            return None
        return value

    @model_validator(mode="after")
    def _provider_requirements(self) -> Settings:
        if self.LLM_PROVIDER == "openai" and not self.LLM_BASE_URL:
            raise ValueError("LLM_PROVIDER=openai requires LLM_BASE_URL")
        if self.ASR_PROVIDER == "openai" and not (self.ASR_BASE_URL or self.LLM_BASE_URL):
            raise ValueError("ASR_PROVIDER=openai requires ASR_BASE_URL (or LLM_BASE_URL)")
        return self

    @model_validator(mode="after")
    def _production_guards(self) -> Settings:
        if self.ENV != "production":
            return self
        secret = self.SECRET_KEY.get_secret_value()
        if secret in INSECURE_SECRET_KEYS or len(secret) < MIN_PRODUCTION_SECRET_LENGTH:
            raise ValueError(
                "SECRET_KEY must be set to a random value of at least "
                f"{MIN_PRODUCTION_SECRET_LENGTH} characters when ENV=production "
                "(generate one with: python -c 'import secrets; print(secrets.token_urlsafe(48))')"
            )
        if self.ADMIN_PASSWORD is not None and (
            self.ADMIN_PASSWORD.get_secret_value() in INSECURE_PASSWORDS
        ):
            raise ValueError("ADMIN_PASSWORD uses a well-known default; choose a strong password")
        return self

    # --- Derived values ------------------------------------------------------------------------
    @property
    def upload_dir(self) -> Path:
        return self.DATA_DIR / "uploads"

    @property
    def cookie_secure(self) -> bool:
        if self.COOKIE_SECURE is not None:
            return self.COOKIE_SECURE
        return self.ENV == "production"

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    @property
    def llm_mode(self) -> Literal["local", "openai"]:
        if self.LLM_PROVIDER == "auto":
            return "openai" if self.LLM_BASE_URL else "local"
        return self.LLM_PROVIDER

    @property
    def trusted_origins(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.TRUSTED_ORIGINS.split(",") if o.strip()]

    @property
    def asr_base_url(self) -> str:
        return self.ASR_BASE_URL or self.LLM_BASE_URL

    @property
    def asr_api_key(self) -> str:
        return self.ASR_API_KEY.get_secret_value() or self.LLM_API_KEY.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton. Tests call ``get_settings.cache_clear()``."""
    return Settings()
