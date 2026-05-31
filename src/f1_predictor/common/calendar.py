"""
Dynamic F1 race calendar module.

All race schedule data is fetched live from FastF1 (primary)
or OpenF1 REST API (fallback). Zero hardcoded dates or race names.

Public API:
    get_season_schedule(year)    → full schedule DataFrame
    get_upcoming_races(year)     → upcoming races, calendar order
    get_next_race(year)          → single dict for the next race
    get_fp1_start(year, round)   → FP1 session start datetime (UTC)
    is_prediction_locked(year, round) → bool
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from typing import Optional

import pandas as pd
import requests
import streamlit as st

from f1_predictor.common.config import settings
from f1_predictor.common.exceptions import CalendarFetchError, RaceNotFoundError, SessionNotFoundError
from f1_predictor.common.logger import get_logger

log = get_logger(__name__)

_OPENF1_BASE = "https://api.openf1.org/v1"


# ── Primary: FastF1 ───────────────────────────────────────────────────────────

def _fetch_via_fastf1(year: int) -> pd.DataFrame:
    """
    Fetch season schedule from FastF1.
    Returns DataFrame with standardised columns.
    """
    import fastf1  # lazy import — heavy dependency

    cache_dir = os.path.join("data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    schedule = fastf1.get_event_schedule(year, include_testing=False)

    # Standardise column names
    schedule = schedule.rename(columns={
        "OfficialEventName": "race_name",
        "EventDate":         "event_date",
        "Country":           "country",
        "Location":          "location",
        "RoundNumber":       "round_number",
    })

    # Determine sprint flag dynamically
    if "EventFormat" in schedule.columns:
        schedule["is_sprint"] = schedule["EventFormat"].str.lower().str.contains("sprint", na=False)
    else:
        schedule["is_sprint"] = False

    # Ensure event_date is timezone-aware UTC
    schedule["event_date"] = pd.to_datetime(schedule["event_date"], utc=True)

    keep_cols = ["round_number", "race_name", "country", "location",
                 "event_date", "is_sprint"]
    return schedule[[c for c in keep_cols if c in schedule.columns]].copy()


# ── Fallback: OpenF1 ─────────────────────────────────────────────────────────

def _fetch_via_openf1(year: int) -> pd.DataFrame:
    """
    Fallback: fetch meeting list from OpenF1 REST API.
    Maps to the same standardised columns as _fetch_via_fastf1.
    """
    url = f"{_OPENF1_BASE}/meetings?year={year}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise CalendarFetchError(
            f"OpenF1 fallback also failed for year {year}: {exc}"
        ) from exc

    rows = []
    for i, m in enumerate(data, start=1):
        rows.append({
            "round_number": m.get("meeting_key", i),
            "race_name":    m.get("meeting_official_name", m.get("meeting_name", "")),
            "country":      m.get("country_name", ""),
            "location":     m.get("location", ""),
            "event_date":   pd.Timestamp(m["date_start"], tz="UTC") if "date_start" in m else None,
            "is_sprint":    False,
        })

    df = pd.DataFrame(rows)
    df["event_date"] = pd.to_datetime(df["event_date"], utc=True)
    return df.sort_values("round_number").reset_index(drop=True)


# ── Cached public functions ───────────────────────────────────────────────────

@st.cache_data(ttl=settings.calendar.cache_ttl_hours * 3600, show_spinner=False)
def get_season_schedule(year: int) -> pd.DataFrame:
    """
    Return the full race schedule for *year*, sorted by round_number.

    Columns: round_number, race_name, country, location, event_date, is_sprint

    Caches result for `calendar.cache_ttl_hours` hours (default 24h).
    Primary source: FastF1.  Fallback: OpenF1.
    """
    log.info(f"Fetching {year} season schedule…")
    try:
        df = _fetch_via_fastf1(year)
        log.info(f"FastF1: {len(df)} rounds fetched for {year}")
        return df.sort_values("round_number").reset_index(drop=True)
    except Exception as exc:
        log.warning(f"FastF1 schedule fetch failed ({exc}), trying OpenF1…")

    try:
        df = _fetch_via_openf1(year)
        log.info(f"OpenF1: {len(df)} rounds fetched for {year}")
        return df.sort_values("round_number").reset_index(drop=True)
    except CalendarFetchError:
        raise
    except Exception as exc:
        raise CalendarFetchError(
            f"Both FastF1 and OpenF1 failed to provide schedule for {year}: {exc}"
        ) from exc


def get_upcoming_races(year: int) -> pd.DataFrame:
    """
    Return only races whose event_date (Race Sunday) is today or in the future.
    Rows are sorted by round_number — guaranteed calendar order, never alphabetical.
    """
    schedule = get_season_schedule(year)
    today_utc = datetime.now(timezone.utc)
    upcoming = schedule[schedule["event_date"] >= today_utc].copy()
    return upcoming.sort_values("round_number").reset_index(drop=True)


def get_next_race(year: int) -> Optional[dict]:
    """
    Return the single next upcoming race as a plain dict, or None if season is over.
    """
    upcoming = get_upcoming_races(year)
    if upcoming.empty:
        return None
    row = upcoming.iloc[0]
    return row.to_dict()


@st.cache_data(ttl=settings.calendar.cache_ttl_hours * 3600, show_spinner=False)
def get_fp1_start(year: int, round_number: int) -> Optional[datetime]:
    """
    Return the FP1 session start time (UTC-aware) for the given round.
    Used as the prediction lock deadline.

    Source: fastf1.get_session(year, round_number, 'FP1').date
    Falls back to event_date - 2 days (Friday approximation) if FP1 unavailable.
    """
    import fastf1

    cache_dir = os.path.join("data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)

    try:
        session = fastf1.get_session(year, round_number, "FP1")
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        fp1_dt = session.date
        if fp1_dt is not None:
            # Ensure UTC-aware
            if fp1_dt.tzinfo is None:
                fp1_dt = fp1_dt.replace(tzinfo=timezone.utc)
            log.info(f"FP1 start for round {round_number}/{year}: {fp1_dt}")
            return fp1_dt
    except Exception as exc:
        log.warning(f"Could not fetch FP1 time for round {round_number}/{year}: {exc}")

    # Fallback: event_date (Race Sunday) minus 2 days = Friday
    schedule = get_season_schedule(year)
    row = schedule[schedule["round_number"] == round_number]
    if row.empty:
        return None
    race_day = row.iloc[0]["event_date"]
    fp1_approx = race_day - timedelta(days=2)
    log.warning(f"Using approximate FP1 time (race - 2 days): {fp1_approx}")
    return fp1_approx


def is_prediction_locked(year: int, round_number: int) -> bool:
    """
    Return True if the current UTC time is at or past the FP1 start time
    (minus any configured buffer minutes).

    This is the gate that prevents new predictions after FP1 begins.
    """
    fp1_start = get_fp1_start(year, round_number)
    if fp1_start is None:
        return False   # Cannot determine → allow prediction (fail open)

    buffer = timedelta(minutes=settings.calendar.lock_buffer_minutes)
    lock_deadline = fp1_start - buffer
    now_utc = datetime.now(timezone.utc)

    locked = now_utc >= lock_deadline
    if locked:
        log.debug(f"Prediction locked for round {round_number}/{year} "
                  f"(FP1 started at {fp1_start})")
    return locked


def get_race_by_name(year: int, race_name: str) -> dict:
    """
    Return a race dict by exact or partial name match.
    Raises RaceNotFoundError if no match.
    """
    schedule = get_season_schedule(year)
    mask = schedule["race_name"].str.contains(race_name, case=False, na=False)
    matches = schedule[mask]
    if matches.empty:
        raise RaceNotFoundError(
            f"No race matching '{race_name}' found in {year} schedule."
        )
    return matches.iloc[0].to_dict()
