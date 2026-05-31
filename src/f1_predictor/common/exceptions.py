"""
Custom exception hierarchy for the F1 Prediction Platform.
All application exceptions inherit from F1PredictorError so callers
can catch the whole family with a single except clause when needed.
"""
from __future__ import annotations


class F1PredictorError(Exception):
    """Base exception for all F1 Predictor errors."""


# ── Calendar / Schedule ───────────────────────────────────────────────────────

class CalendarFetchError(F1PredictorError):
    """Raised when the race schedule cannot be fetched from FastF1 or OpenF1."""


class RaceNotFoundError(F1PredictorError):
    """Raised when a requested race does not exist in the current schedule."""


class SessionNotFoundError(F1PredictorError):
    """Raised when a specific session (FP1, Q, R) cannot be found."""


# ── Data / ETL ────────────────────────────────────────────────────────────────

class BronzeDataMissingError(F1PredictorError):
    """Raised when no bronze CSV files are available for the requested scope."""


class GoldFeaturesMissingError(F1PredictorError):
    """Raised when features.parquet does not exist or is stale."""


class IngestError(F1PredictorError):
    """Raised when data ingestion from FastF1 or OpenF1 fails."""


# ── Model ─────────────────────────────────────────────────────────────────────

class ModelNotFoundError(F1PredictorError):
    """Raised when a required model JSON file is missing from models/."""


class ModelFeatureMismatchError(F1PredictorError):
    """Raised when inference features don't match the model's training features."""


class PredictionError(F1PredictorError):
    """Raised when the prediction engine encounters an unrecoverable error."""


# ── Auth ──────────────────────────────────────────────────────────────────────

class AuthError(F1PredictorError):
    """Base auth exception."""


class UserAlreadyExistsError(AuthError):
    """Raised when a username or email is already registered."""


class InvalidCredentialsError(AuthError):
    """Raised when username/password do not match."""


class InsufficientPermissionsError(AuthError):
    """Raised when a user attempts an action requiring a higher role."""


class AccountDisabledError(AuthError):
    """Raised when a deactivated account tries to log in."""


# ── Prediction Store ──────────────────────────────────────────────────────────

class PredictionLockedError(F1PredictorError):
    """Raised when attempting to modify a prediction that is already locked."""


class PredictionNotFoundError(F1PredictorError):
    """Raised when a prediction record cannot be found in the database."""


class PredictionAlreadyScoredError(F1PredictorError):
    """Raised when trying to score a prediction that has already been scored."""


class RaceAlreadyLockedError(F1PredictorError):
    """Raised when trying to create a prediction after FP1 has started."""
