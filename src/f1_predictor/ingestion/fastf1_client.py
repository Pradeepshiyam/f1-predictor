import fastf1
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FastF1Client:
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(self.cache_dir)
        
    def get_race_results(self, year: int, gp_name: str) -> pd.DataFrame:
        """Fetch race results for a specific GP."""
        logger.info(f"Fetching results for {year} {gp_name}")
        session = fastf1.get_session(year, gp_name, 'R')
        # Optimized: Only load results and laps, skip heavy telemetry/weather
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        return session.results

    def get_lap_data(self, year: int, gp_name: str) -> pd.DataFrame:
        """Fetch all lap data for a specific GP."""
        logger.info(f"Fetching lap data for {year} {gp_name}")
        session = fastf1.get_session(year, gp_name, 'R')
        # Optimized: Skip telemetry for speed
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        return session.laps

    def get_qualifying_results(self, year: int, gp_name: str) -> pd.DataFrame:
        """Fetch qualifying results (pole setter etc)."""
        logger.info(f"Fetching qualifying for {year} {gp_name}")
        session = fastf1.get_session(year, gp_name, 'Q')
        session.load()
        return session.results

    def fetch_season_data(self, year: int):
        """Fetch data for an entire season."""
        # Get event schedule
        schedule = fastf1.get_event_schedule(year)
        # Filter for races only
        races = schedule[schedule['EventFormat'] != 'testing']
        return races
