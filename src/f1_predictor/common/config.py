"""
Pydantic-based settings loader.
Reads configs/project.yaml and exposes a typed singleton `settings`.

Usage:
    from f1_predictor.common.config import settings
    year = settings.current_year
"""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


def _find_project_root() -> Path:
    """
    Walk up from this file until we find a directory containing
    'configs/project.yaml'.  Raises FileNotFoundError if not found.
    """
    candidate = Path(__file__).resolve()
    for _ in range(10):          # max 10 levels up
        candidate = candidate.parent
        if (candidate / "configs" / "project.yaml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate configs/project.yaml. "
        "Ensure you are running from the project root."
    )


_PROJECT_ROOT = _find_project_root()
_CONFIG_FILE  = _PROJECT_ROOT / "configs" / "project.yaml"


def _load_yaml() -> dict:
    """Load raw YAML config file."""
    if not _CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {_CONFIG_FILE}")
    with open(_CONFIG_FILE, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class DatabaseSettings(BaseSettings):
    url: str = "sqlite:///data/f1_platform.db"
    echo_sql: bool = False
    pool_size: int = 5

    model_config = {"extra": "ignore"}


class AuthSettings(BaseSettings):
    cookie_name: str = "f1_auth_token"
    cookie_expiry_days: int = 30

    model_config = {"extra": "ignore"}


class CalendarSettings(BaseSettings):
    cache_ttl_hours: int = 24
    lock_buffer_minutes: int = 0   # 0 = lock exactly at FP1 start

    model_config = {"extra": "ignore"}


class ModelSettings(BaseSettings):
    random_state: int = 42
    test_fraction: float = 0.15
    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05

    model_config = {"extra": "ignore"}


class PathSettings(BaseSettings):
    bronze:   str = "data/bronze/historical"
    silver:   str = "data/silver/historical.parquet"
    gold:     str = "data/gold/features.parquet"
    models:   str = "models"
    outputs:  str = "outputs"
    logs:     str = "outputs/logs"
    backups:  str = "outputs/backups"

    model_config = {"extra": "ignore"}


class Settings(BaseSettings):
    project_name: str = "f1_predictor"
    season: str = "dynamic"          # "dynamic" → current year, or e.g. "2026"

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    calendar: CalendarSettings = Field(default_factory=CalendarSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    paths: PathSettings = Field(default_factory=PathSettings)

    feature_windows: list[int] = [3, 5, 10]
    monte_carlo_runs: int = 1000
    top_n_drivers: int = 10

    model_config = {"extra": "ignore"}

    @field_validator("feature_windows", mode="before")
    @classmethod
    def parse_windows(cls, v: object) -> list[int]:
        if isinstance(v, list):
            return [int(x) for x in v]
        return [3, 5, 10]

    @property
    def current_year(self) -> int:
        """Derive the active F1 season year dynamically."""
        if str(self.season).strip().lower() == "dynamic":
            return datetime.now().year
        return int(self.season)

    @property
    def bronze_path(self) -> Path:
        return Path(self.paths.bronze)

    @property
    def gold_path(self) -> Path:
        return Path(self.paths.gold)

    @property
    def models_path(self) -> Path:
        return Path(self.paths.models)


def _get_db_url_from_secrets() -> str | None:
    """
    Check Streamlit secrets for a database URL override.
    This is how the cloud deployment (Streamlit Community Cloud + Supabase)
    injects the PostgreSQL connection string without touching project.yaml.

    Returns None if:
    - Not running in Streamlit context
    - No [database] section in secrets.toml
    - No url key in that section
    """
    try:
        import streamlit as st
        url = st.secrets.get("database", {}).get("url", None)
        if url:
            return str(url)
    except Exception:
        pass  # Not in Streamlit context (e.g. running train.py directly)
    return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Priority order for database URL:
      1. Streamlit secrets  → [database] url  (Streamlit Cloud + Supabase)
      2. project.yaml       → database.url    (local SQLite default)

    This means:
      - Local dev:  project.yaml wins → SQLite, zero config
      - Cloud:      secrets.toml wins → PostgreSQL, zero code change
    """
    raw = _load_yaml()

    # Base database config from YAML
    db_raw = raw.get("database", {})

    # Override with Streamlit secrets if available (cloud deployment)
    cloud_url = _get_db_url_from_secrets()
    if cloud_url:
        db_raw = {**db_raw, "url": cloud_url}

    return Settings(
        project_name=raw.get("project_name", "f1_predictor"),
        season=raw.get("season", "dynamic"),
        database=DatabaseSettings(**db_raw),
        auth=AuthSettings(**raw.get("auth", {})),
        calendar=CalendarSettings(**raw.get("calendar", {})),
        model=ModelSettings(**raw.get("model", {})),
        paths=PathSettings(**raw.get("paths", {})),
        feature_windows=raw.get("features", {}).get("windows", [3, 5, 10]),
        monte_carlo_runs=raw.get("simulation", {}).get("monte_carlo_runs", 1000),
        top_n_drivers=raw.get("reporting", {}).get("top_n_drivers", 10),
    )


# Public singleton — import this everywhere
settings: Settings = get_settings()

