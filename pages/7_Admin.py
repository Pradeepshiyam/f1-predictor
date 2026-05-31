"""Page 7 — Admin Panel (admin role only)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import pandas as pd

from f1_predictor.common.auth import require_auth, is_admin, render_admin_panel
from f1_predictor.common.config import settings
from f1_predictor.common.database import ModelRunLog, SessionLocal
from f1_predictor.models.inference import models_available, reload_models
from f1_predictor.prediction.store import get_all_predictions_admin

require_auth()

if not is_admin():
    st.error("🔒 Admin access required.")
    st.stop()

st.markdown("## 👑 Admin Panel")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["👥 User Management", "📋 All Predictions", "🤖 Model Logs"])

with tab1:
    render_admin_panel()

with tab2:
    st.markdown("### All User Predictions")
    all_preds = get_all_predictions_admin()
    if all_preds.empty:
        st.info("No predictions yet.")
    else:
        show_cols = [c for c in ["username","race_name","year","is_locked","winner_correct",
                                  "podium_hits","top10_hits","position_mae","spearman_rho","scored_at"]
                     if c in all_preds.columns]
        st.dataframe(all_preds[show_cols], use_container_width=True, hide_index=True)

with tab3:
    st.markdown("### Model Training History")
    with SessionLocal() as db:
        logs = db.query(ModelRunLog).order_by(ModelRunLog.run_at.desc()).all()
        rows = [{"Run At": l.run_at, "Race": l.race_name, "Year": l.year,
                 "Gold Rows": l.gold_row_count, "Winner AUC": l.winner_auc,
                 "Pole AUC": l.pole_auc, "Top3 AUC": l.top3_auc,
                 "Train Sec": l.training_sec} for l in logs]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No training runs logged yet.")

    st.markdown("---")
    if st.button("🔄 Reload Models from Disk", type="primary"):
        reload_models()
        st.success("Model registry cleared — will reload on next prediction.")

    avail = models_available()
    for task, ok in avail.items():
        st.markdown(f"{'🟢' if ok else '🔴'} `{task}_model.json` — {'Found' if ok else 'Missing'}")
