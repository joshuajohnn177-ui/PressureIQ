# 🏏 PressureIQ: AI/ML Player Pressure Intelligence Engine
### **Rajasthan Royals — SupeRR Selector Season 2 Hackathon**
*Identifying the Most Reliable T20 Players Under Pressure (2022–2026)*

[![Tests](https://img.shields.io/badge/pytest-28%2F28%20passing-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](.venv/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.67-red)](app/)
[![ML Model](https://img.shields.io/badge/ROC--AUC-0.816-gold)](outputs/)
[![Brier Score](https://img.shields.io/badge/Brier%20Score-0.173-purple)](outputs/)

---

## 📌 Executive Summary

In high-stakes T20 cricket, traditional statistics (raw averages, aggregate runs, economy rates) fail to capture **clutch performance**. An extra 15 runs in a dead rubber means far less than 12 runs off the final over of an IPL chase.

**PressureIQ** solves this problem with an end-to-end, mathematically rigorous analytics engine:
1. **Calibrated Machine Learning Win Probability ($WP$):** Dual-Innings Ensemble model combining regularized Logistic Regression and XGBoost ($ROC\text{-}AUC = 0.816$, $\text{Brier} = 0.173$).
2. **Win Probability Added ($WPA$) & Leverage Index ($LI$):** Quantifies the exact match equity added or lost on all 199,029 historical deliveries.
3. **Situational Differentials ($\Delta \text{SR}, \Delta \text{Econ}$):** Measures how players perform under pressure compared to their own baseline.
4. **Empirical Bayes Shrinkage Regularization:** Regularizes player rate metrics with credibility weighting ($\lambda_i = \frac{n_i}{n_i + 75}$) and a strict 50-ball minimum threshold, eliminating "small-sample hero" distortion.
5. **Interactive Scouting Dashboard:** A Streamlit application built for Rajasthan Royals scouts and performance analysts.

---

## 🏆 Official Top 10 Rankings

### Top 10 Batters Under High Pressure
| Rank | Player Name | Pressure Matches | Pressure Balls | Pressure Runs | Baseline SR | Pressure SR | SR Diff (+/-) | Total Pressure WPA | PressureIQ Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Suryakumar Yadav** | 59 | 827 | 1583 | 166.46 | 191.41 | **+24.96** | **+3.3509** | **403.24** |
| **2** | **Rinku Singh** | 72 | 769 | 1349 | 152.49 | 175.42 | **+22.93** | **+2.5896** | **311.27** |
| **3** | **Tim David** | 71 | 677 | 1308 | 177.65 | 193.21 | **+15.55** | **+2.7263** | **301.94** |
| **4** | **Shreyas Iyer** | 42 | 467 | 854 | 153.47 | 182.87 | **+29.40** | **+1.9699** | **246.38** |
| **5** | **Dinesh Karthik** | 55 | 527 | 961 | 167.29 | 182.35 | **+15.06** | **+2.1657** | **232.74** |
| **6** | **Ashutosh Sharma** | 25 | 291 | 526 | 167.50 | 180.76 | **+13.26** | **+2.3568** | **224.35** |
| **7** | **David Miller** | 74 | 868 | 1379 | 149.30 | 158.87 | **+9.57** | **+1.9162** | **202.07** |
| **8** | **Tilak Varma** | 77 | 993 | 1644 | 143.27 | 165.56 | **+22.29** | **+1.3802** | **180.28** |
| **9** | **Najibullah Zadran** | 14 | 146 | 265 | 138.56 | 181.51 | **+42.94** | **+1.4578** | **173.00** |
| **10** | **Naman Dhir** | 26 | 251 | 496 | 166.04 | 197.61 | **+31.57** | **+1.3854** | **169.83** |

### Top 10 Bowlers Under High Pressure
| Rank | Player Name | Pressure Matches | Pressure Balls | Pressure Wickets | Baseline Econ | Pressure Econ | Econ Diff (+/-) | Total Pressure WPA | PressureIQ Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Rashid Khan** | 91 | 955 | 84 | 7.51 | 8.12 | +0.61 | **+2.5981** | **301.18** |
| **2** | **Sunil Narine** | 47 | 484 | 32 | 6.92 | 6.79 | **-0.13** | **+2.1415** | **265.14** |
| **3** | **Jasprit Bumrah** | 66 | 717 | 57 | 7.28 | 7.51 | **+0.22** | **+1.5978** | **205.39** |
| **4** | **Varun Chakravarthy** | 64 | 752 | 62 | 8.06 | 8.72 | +0.66 | **+1.5536** | **183.66** |
| **5** | **Josh Hazlewood** | 57 | 576 | 44 | 8.37 | 9.20 | +0.83 | **+1.5730** | **174.08** |
| **6** | **Adam Zampa** | 37 | 362 | 36 | 8.43 | 8.85 | +0.42 | **+1.4005** | **162.32** |
| **7** | **Mitchell Santner** | 42 | 445 | 31 | 7.50 | 7.31 | **-0.20** | **+1.1021** | **161.90** |
| **8** | **Abrar Ahmed** | 24 | 252 | 20 | 6.75 | 5.88 | **-0.87** | **+0.9855** | **159.85** |
| **9** | **Sufiyan Muqeem** | 13 | 186 | 17 | 6.12 | 5.84 | **-0.28** | **+1.2252** | **149.03** |
| **10** | **Taskin Ahmed** | 39 | 310 | 28 | 8.10 | 9.12 | +1.01 | **+1.5048** | **146.33** |

---

## 🚀 Quickstart & Dashboard Launch

### 1. One-Click Launch (Windows)
Double-click [`run_dashboard.bat`](run_dashboard.bat) from the project directory.

### 2. Command-Line Launch
Run the following in PowerShell or Command Prompt:

```powershell
cd "d:\1 and only me\PressureIQ"
.venv\Scripts\python.exe -m streamlit run app/app.py --browser.gatherUsageStats false
```

Open your browser at: **`http://localhost:8501`**

---

## 🧪 Running Automated Tests

PressureIQ includes 28 test cases verifying data leakage prevention, extras accounting, WPA continuity, Bayesian shrinkage, and dashboard components.

Execute the full suite using the project `.venv`:

```powershell
cd "d:\1 and only me\PressureIQ"
.venv\Scripts\python.exe -m pytest tests/ -v
```

Expected result:
```
============================= 28 passed in 5.26s ==============================
```

---

## 📂 Repository Structure

```
PressureIQ/
├── app/
│   └── app.py                      # Production Streamlit Dashboard (6 Modules)
├── src/
│   ├── config.py                   # Central parameters, thresholds & paths
│   ├── data_loader.py              # Data loading, validation & joining
│   ├── features.py                 # Leak-free pre-delivery state feature engineering
│   ├── win_prob_model.py           # Dual-Innings ML Model (Ensemble LR + XGB)
│   ├── wpa_engine.py               # WPA & Sensitivity Leverage Index Engine
│   └── player_analytics.py         # Empirical Bayes shrinkage & composite PressureIQ
├── scripts/
│   ├── run_phase1_pipeline.py      # Preprocessing & feature extraction
│   ├── run_phase2_training.py      # Win probability training & WPA pipeline
│   ├── run_phase3_rankings.py      # Player intelligence & rankings generator
│   └── audit_phase3.py             # Phase 3 technical audit runner
├── tests/
│   ├── test_phase1.py              # Phase 1: Preprocessing & leakage tests (4 tests)
│   ├── test_phase2.py              # Phase 2: ML, WPA attribution & LI tests (7 tests)
│   ├── test_phase3.py              # Phase 3: Shrinkage, volume bias & sync tests (8 tests)
│   └── test_dashboard.py           # Phase 4: Streamlit components & data tests (9 tests)
├── outputs/
│   ├── top10_batters.csv           # Official Top 10 Batters Submission
│   ├── top10_bowlers.csv           # Official Top 10 Bowlers Submission
│   ├── all_batters_pressure_metrics.parquet # 288 eligible batters dataset
│   ├── all_bowlers_pressure_metrics.parquet # 248 eligible bowlers dataset
│   ├── deliveries_with_wpa.parquet # Master delivery dataset with WPA & LI
│   ├── processed_deliveries.parquet# Preprocessed deliveries with pre-* features
│   ├── win_prob_model.joblib       # Serialized production Ensemble model
│   └── figures/
│       └── calibration_curves.png  # Innings 1 & 2 Reliability Diagrams
├── methodology_report.md           # Comprehensive Technical Methodology Report
├── run_dashboard.bat               # Windows one-click launcher
└── README.md                       # Project documentation
```

---

## 🔬 Methodology Highlights

- **Dual-Innings Architecture:** Innings 1 models par score projection; Innings 2 models target chase dynamics without cross-innings feature leakage.
- **Strict Chronological Match Splitting:** 589 older matches (Train: 2022–Jun 2025), 126 matches (Validation), 127 matches (Chronological Holdout Test: 2026). Never randomly splits deliveries from the same match.
- **Accurate Extras Accounting:** Wides add 1 extra run without consuming legal balls. Oversteps credit boundary runs off the bat plus penalty. Legbyes and byes are decoupled from batter/bowler WPA.
- **Composite PressureIQ Scoring Formula:**
  $$\text{PressureIQ}_{\text{Batter}} = 60.0 \cdot Z(\text{Shrunk Total WPA}) + 25.0 \cdot Z(\Delta \text{SR}_{\text{shrunk}}) + 15.0 \cdot \lambda_i$$
  $$\text{PressureIQ}_{\text{Bowler}} = 60.0 \cdot Z(\text{Shrunk Total WPA}) + 25.0 \cdot Z(-\Delta \text{Econ}_{\text{shrunk}}) + 15.0 \cdot \lambda_i$$

For the full statistical treatise, see [`methodology_report.md`](methodology_report.md).

