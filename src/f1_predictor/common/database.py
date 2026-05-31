"""
SQLAlchemy ORM models and session factory for the F1 Prediction Platform.

Database: SQLite (local) — switchable to PostgreSQL/Supabase via project.yaml
          by changing database.url to a postgresql:// connection string.

Tables:
    users           — registered accounts with role-based access
    predictions     — locked race predictions with post-race scoring
    model_run_log   — training run history + accuracy metrics
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey,
                        Integer, String, Text, create_engine, event)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from f1_predictor.common.config import settings
from f1_predictor.common.logger import get_logger

log = get_logger(__name__)


# ── Engine ────────────────────────────────────────────────────────────────────

def _build_engine():
    url = settings.database.url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    engine = create_engine(
        url,
        echo=settings.database.echo_sql,
        connect_args=connect_args,
    )
    # Enable WAL mode for SQLite — better concurrent read performance
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def set_wal(conn, _record):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
    log.info(f"Database engine created: {url.split('///')[0]}")
    return engine


engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# ── Base ──────────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


# ── ORM Models ────────────────────────────────────────────────────────────────

class User(Base):
    """Registered platform user."""
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(50),  unique=True, nullable=False, index=True)
    email         = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    role          = Column(String(20),  nullable=False, default="user")
    # "user" | "admin"
    is_active     = Column(Boolean, default=True, nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login    = Column(DateTime, nullable=True)

    predictions   = relationship("Prediction", back_populates="user",
                                 cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User {self.username!r} role={self.role!r}>"


class Prediction(Base):
    """A user's race prediction, lockable before FP1."""
    __tablename__ = "predictions"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    race_name        = Column(String(100), nullable=False)   # OfficialEventName from FastF1
    year             = Column(Integer,     nullable=False)
    round_number     = Column(Integer,     nullable=False)
    predicted_at     = Column(DateTime,    default=datetime.utcnow, nullable=False)
    is_locked        = Column(Boolean,     default=False, nullable=False)
    locked_at        = Column(DateTime,    nullable=True)
    fp1_start_utc    = Column(DateTime,    nullable=True)    # snapshot of lock deadline

    # Prediction payload — stored as JSON strings
    top10_json       = Column(Text, nullable=True)   # [{"pos":1,"driver":"X","prob":0.33},...]
    win_probs_json   = Column(Text, nullable=True)   # {"DriverName": probability, ...}

    # Post-race actuals (None until scored)
    actual_top10_json   = Column(Text,    nullable=True)
    winner_correct      = Column(Boolean, nullable=True)
    podium_hits         = Column(Integer, nullable=True)   # 0-3
    top10_hits          = Column(Integer, nullable=True)   # 0-10
    position_mae        = Column(Float,   nullable=True)   # mean |pred - actual| position
    spearman_rho        = Column(Float,   nullable=True)   # rank correlation coefficient
    scored_at           = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="predictions")

    # ── Convenience helpers ───────────────────────────────────────────────────

    def set_top10(self, data: list[dict]) -> None:
        self.top10_json = json.dumps(data)

    def get_top10(self) -> list[dict]:
        return json.loads(self.top10_json) if self.top10_json else []

    def set_win_probs(self, data: dict[str, float]) -> None:
        self.win_probs_json = json.dumps(data)

    def get_win_probs(self) -> dict[str, float]:
        return json.loads(self.win_probs_json) if self.win_probs_json else {}

    def set_actual_top10(self, data: list[str]) -> None:
        self.actual_top10_json = json.dumps(data)

    def get_actual_top10(self) -> list[str]:
        return json.loads(self.actual_top10_json) if self.actual_top10_json else []

    def __repr__(self) -> str:
        return (f"<Prediction user={self.user_id} race={self.race_name!r} "
                f"year={self.year} locked={self.is_locked}>")


class ModelRunLog(Base):
    """Records every model training run for auditability."""
    __tablename__ = "model_run_log"

    id             = Column(Integer, primary_key=True, index=True)
    run_at         = Column(DateTime, default=datetime.utcnow, nullable=False)
    triggered_by   = Column(String(50), nullable=True)   # "admin:<username>" or "scheduler"
    race_name      = Column(String(100), nullable=True)  # race that prompted retraining
    year           = Column(Integer,     nullable=True)
    gold_row_count = Column(Integer,     nullable=True)
    features_used  = Column(Text,        nullable=True)  # JSON list
    winner_auc     = Column(Float,       nullable=True)
    pole_auc       = Column(Float,       nullable=True)
    top3_auc       = Column(Float,       nullable=True)
    winner_brier   = Column(Float,       nullable=True)
    training_sec   = Column(Float,       nullable=True)

    def set_features(self, features: list[str]) -> None:
        self.features_used = json.dumps(features)

    def get_features(self) -> list[str]:
        return json.loads(self.features_used) if self.features_used else []

    def __repr__(self) -> str:
        return f"<ModelRunLog run_at={self.run_at} winner_auc={self.winner_auc}>"


# ── Schema creation ───────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    import os
    # Ensure data directory exists for SQLite
    db_url = settings.database.url
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "")
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    Base.metadata.create_all(bind=engine)
    log.info("Database tables verified / created.")


# ── Session context manager ───────────────────────────────────────────────────

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Provide a transactional database session.

    Usage:
        with get_db() as db:
            user = db.query(User).filter_by(username="alice").first()
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
