"""
F1 Prediction Platform — Main Entry Point

Responsibilities:
  1. Initialize database tables on first run
  2. Render login / register page for unauthenticated users
  3. Show sidebar navigation and user info for authenticated users
  4. Route to the correct page (handled by Streamlit multi-page)

Run:
    streamlit run app.py
"""
import sys
import os

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

from f1_predictor.common.auth import (
    is_authenticated,
    is_admin,
    current_user,
    current_role,
    logout,
    render_login_page,
)
from f1_predictor.common.config import settings
from f1_predictor.common.database import init_db
from f1_predictor.common.logger import get_logger
from f1_predictor.models.inference import models_available

log = get_logger(__name__)

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Prediction Center",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Dark sidebar accent */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Header strip */
    .f1-header {
        background: linear-gradient(90deg, #e10600 0%, #1a1a2e 60%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        color: white;
        font-size: 1.6rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1e1e2e;
        border: 1px solid #2d2d3d;
        border-radius: 10px;
        padding: 0.8rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(225,6,0,0.3); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Bootstrap DB on first run ─────────────────────────────────────────────────
init_db()

# ── Auth gate ─────────────────────────────────────────────────────────────────
if not is_authenticated():
    render_login_page()
    st.stop()

# ── Authenticated: Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div style="text-align:center; padding: 1rem 0;">
            <div style="font-size:2.5rem;">🏎️</div>
            <div style="font-size:1.1rem; font-weight:700; color:#e10600;">F1 Prediction Center</div>
            <div style="font-size:0.8rem; color:#888; margin-top:4px;">Season {settings.current_year}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # User info
    role_badge = "👑 Admin" if is_admin() else "👤 User"
    st.markdown(
        f"""
        <div style="font-size:0.85rem; color:#aaa;">
            Logged in as<br>
            <b style="color:#fff; font-size:1rem;">{current_user()}</b>
            &nbsp;<span style="color:#e10600;">{role_badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Navigation links
    st.page_link("pages/1_Prediction.py",       label="🔮 Race Prediction",       icon=None)
    st.page_link("pages/2_Season.py",            label="📅 Season Tracker",         icon=None)
    st.page_link("pages/3_Drivers.py",           label="🏎️ Driver Analysis",       icon=None)
    st.page_link("pages/4_Constructors.py",      label="🏗️ Constructor Analysis",  icon=None)
    st.page_link("pages/5_My_Predictions.py",    label="🔬 My Predictions",         icon=None)
    st.page_link("pages/6_Historical.py",        label="📊 Historical Deep Dive",   icon=None)
    if is_admin():
        st.page_link("pages/7_Admin.py",         label="👑 Admin Panel",            icon=None)

    st.markdown("---")

    # Model status indicator
    st.markdown("**Model Status**")
    avail = models_available()
    for task, ok in avail.items():
        icon = "🟢" if ok else "🔴"
        st.markdown(f"{icon} `{task}`")

    st.markdown("---")

    if st.button("🚪 Logout", use_container_width=True):
        logout()
        st.rerun()

# ── Landing page content (shown on app.py itself) ────────────────────────────
st.markdown(
    f'<div class="f1-header">🏎️ F1 Prediction Center &nbsp;·&nbsp; '
    f'Season {settings.current_year}</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Season", settings.current_year)
with col2:
    models_ok = sum(models_available().values())
    st.metric("Models Ready", f"{models_ok}/3")
with col3:
    from f1_predictor.common.calendar import get_next_race
    next_race = get_next_race(settings.current_year)
    st.metric("Next Race", next_race["race_name"].split("Grand Prix")[0].strip() if next_race else "Season Over")
with col4:
    from f1_predictor.common.database import SessionLocal, Prediction
    with SessionLocal() as db:
        locked_count = db.query(Prediction).filter_by(is_locked=True).count()
    st.metric("Locked Predictions", locked_count)

st.markdown("---")
st.info("👈 Use the sidebar to navigate between pages.")
