# 🏎️ F1 Prediction Center

AI-powered Formula 1 race prediction platform. Predicts pole, winner, top-10 finishing order, and constructor points — updated every race weekend.

**Stack:** Python · Streamlit · XGBoost · FastF1 · SQLAlchemy  
**Cost:** $0 forever

---

## 🚀 Run Locally

```powershell
# 1. Install dependencies
.venv\Scripts\pip.exe install -r requirements.txt

# 2. Build features
.venv\Scripts\python.exe src/f1_predictor/features/pandas_processor.py

# 3. Train models
.venv\Scripts\python.exe src/f1_predictor/models/train.py

# 4. Launch app
.venv\Scripts\streamlit.exe run app.py
```

---

## ☁️ Free Cloud Deployment

### Step 1 — Create Supabase Database (Free, No Credit Card)
1. Go to [supabase.com](https://supabase.com) → Sign up
2. **New Project** → choose a name and password
3. Go to **Settings → Database → Connection string (URI)**
4. Copy it — looks like:  
   `postgresql://postgres:YOUR_PASS@db.XXXX.supabase.co:5432/postgres`

### Step 2 — Push to GitHub
```powershell
git init
git add .
git commit -m "F1 Prediction Platform"
# Create repo at github.com first, then:
git remote add origin https://github.com/YOUR_USERNAME/f1-predictor.git
git push -u origin main
```

### Step 3 — Deploy to Streamlit Community Cloud (Free)
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. **New app** → Select your GitHub repo → Main file: `app.py`
3. Go to **Advanced settings → Secrets** and paste:
```toml
[database]
url = "postgresql://postgres:YOUR_PASS@db.XXXX.supabase.co:5432/postgres"
```
4. Click **Deploy** ✅

### Step 4 — Add Supabase Keep-Alive (Prevents Free Tier Pausing)
1. In GitHub repo → **Settings → Secrets → Actions**
2. Add secret: `SUPABASE_DB_URL` = your Supabase connection string
3. The `.github/workflows/keepalive.yml` will auto-ping every 5 days — free

---

## 📅 Weekly Workflow (Every Race Weekend)

### Before FP1 (Friday)
- Predictions auto-lock at FP1 start time

### Monday After Race
```powershell
.venv\Scripts\python.exe scripts/post_race_update.py
```
Auto-ingests, retrains, scores all predictions. Done.

---

## 📁 Project Structure

```
app.py                          ← Entry point (auth wall)
pages/
  1_Prediction.py               ← Race prediction + lock
  2_Season.py                   ← Season standings
  3_Drivers.py                  ← Driver analysis
  4_Constructors.py             ← Constructor analysis
  5_My_Predictions.py           ← History + leaderboard
  6_Historical.py               ← 2021–now deep dive
  7_Admin.py                    ← Admin panel
src/f1_predictor/
  common/                       ← config, auth, database, calendar, logger
  prediction/                   ← engine, store, scorer
  models/                       ← train, inference
  features/                     ← ETL pipeline
scripts/
  post_race_update.py           ← One-command post-race pipeline
configs/project.yaml            ← All settings (season: dynamic)
data/f1_platform.db             ← SQLite (local) / Supabase (cloud)
```

---

## 🔐 Auth

- Anyone can register
- Admin can deactivate users, promote to admin, view all predictions
- Prediction lock: automatically at FP1 start (live from FastF1 API)
