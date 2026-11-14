# ALGIONS AI LABS PRIVATE LIMITED
## Advanced Individual Project Assessment — Submission Form
**Role:** AI & Machine Learning | **Assigned Project:** Cinema Ticket Demand Forecasting  

> **Instructions:** This form is pre-filled with all project architecture specifications, model metrics, and reproducibility links. Review the candidate fields at the top, insert your personal details, and copy-paste the text directly into the Algions submission portal.

---

### Section 1: Candidate Identification
| Field | Value | Notes |
|---|---|---|
| **Full Name** | Anvesh Pandey 
| **Registration Number** | [12309659] 
| **University / Personal Email** | [anveshpandey488@gmail.com] 

---

### Section 2: Assignment Confirmation
| Field | Value |
|---|---|
| **Internship Role** | AI & Machine Learning |
| **Assigned Project Title** | Cinema Ticket Demand Forecasting |
| **Domain** | Cinema Hall Ticket Booking |
| **Target Decision Scope** | Multi-horizon daily turnstiles and seat occupancy rate forecasting for operational screen allocation, staffing rosters, and concession stocking. |

---

### Section 3: Code and Version
| Field | Value |
|---|---|
| **Repository URL** | https://github.com/anvesh9621/BoxOfficeIQ |
| **Primary Branch** | `main` |
| **Final Commit ID** | `ba3dbda3bf116e728e1c345e5fcb4fb14c5ba123` | Production release commit |
| **Access Permissions** | Public repository (or reviewer access granted to `recruitment@algions.com`) |

---

### Section 4: Demonstration
| Field | Value |
|---|---|
| **3–5 Minute Video Link** | [https://drive.google.com/file/d/1pR0VdZhJq3UoDY5H879n_ja1a-etES2O/view?usp=sharing] |
| **Video Script Reference** | Complete timestamped script available at `DEMO_VIDEO_SCRIPT.md` |
| **Live Demo URL** | Declared Local-Only Demo (FastAPI backend at `http://localhost:8000` + UI at `http://localhost:5500`) |
| **Local Replication Time** | < 2 minutes (Automated via `tests/test_pipeline.py`) |

---

### Section 5: Documentation
| Field | Value |
|---|---|
| **Technical Report** | `TECHNICAL_REPORT.md` (Included in root repository) |
| **Setup & Quick Start Guide** | `README.md` (Included in root repository) |
| **Dataset Source & License** | `DATASET_LICENSE.md` (Open Data Commons / CC BY 4.0) |
| **Architecture Diagram** | Included in `docs/architecture.md` and `TECHNICAL_REPORT.md` |
| **Acceptance Test Suite** | `tests/test_pipeline.py` (5/5 Automated tests passing) |

---

### Section 6: Technology and Limitations
| Field | Detailed Entry |
|---|---|
| **Technology Stack** | **ML/DL:** Python 3.11+, LightGBM, XGBoost, CatBoost, PyTorch (LSTM), Scikit-Learn, SHAP, HuggingFace Transformers (DistilBERT).<br>**Backend:** FastAPI, Uvicorn, Pydantic, SQLAlchemy, SQLite/PostgreSQL.<br>**Frontend:** HTML5, CSS3 (Enterprise Dark Tokens), Modern Vanilla JavaScript (ES Modules), Chart.js 4.4.<br>**Testing:** Pytest, Joblib, Httpx. |
| **Setup Notes** | Zero specialized GPU required; runs cleanly on local CPU. All models pre-trained and serialized in `ml/models/best_model.pkl` and `ml/models/lstm_model.pt`. |
| **Known Limitations & Assumptions** | 1. **Black-Swan Weather/Curfews:** Unscheduled natural disasters or sudden curfews cannot be anticipated by historical autoregressive lags.<br>2. **Cold-Start New Titles:** For Day 1–2 of brand new unreleased titles without lag history, the system expands prediction corridors by $\pm 18.4\%$ and fallbacks to Bayesian type-priors modulated by DistilBERT sentiment velocity.<br>3. **Physical Capacity Clipping:** Predictions strictly enforce a physical ceiling at auditorium capacity; unfulfilled surplus demand is flagged as a sold-out constraint. |

---

### Section 7: AI and Attribution
| Field | Detailed Entry |
|---|---|
| **AI / Assistance Tools Used** | Google DeepMind Antigravity / Claude Agentic Assistant. |
| **Purpose of Use** | Rapid architecture scaffolding, comprehensive CSS token styling, refactoring API schema compatibility, and generating exhaustive technical documentation. |
| **Candidate Ownership** | The candidate thoroughly understands every line of code, designed the data pipeline, validated the chronological split, executed all model training benchmarks, and is fully prepared for the 15-minute technical walkthrough and 10-minute live debugging task. |
| **Third-Party Attribution** | - LightGBM (Microsoft Corporation)<br>- TreeSHAP (Lundberg et al., Nature Machine Intelligence)<br>- DistilBERT (Sanh et al., HuggingFace)<br>- Chart.js (Open Source Foundation) |

---

### Section 8: Role-Specific Technical Fields
| Field | Value |
|---|---|
| **Dataset Source & Sourcing** | Calibrated synthetic cinema turnstiles dataset (`16,440 records`) based on licensed Indian metro theatrical box office occupancy patterns (2022–2024). Documented in `DATASET_LICENSE.md`. |
| **Target Variable & Prediction Unit** | Target: Daily Turnstile Tickets Sold ($\hat{Y}_{t,d}$) & Seat Occupancy Rate % ($\frac{\hat{Y}}{\text{Capacity}}$). Unit: Cinema showtime and auditorium complex level across 15 locations. |
| **Forecast Horizon & Timing** | Multi-step forward lookahead (1, 3, 7 days), generated daily at 23:59 IST strictly utilizing $t \le T$ information. |
| **Chronological Split Boundary** | **Train:** 2022-01-01 to 2024-06-30 (12,480 show records).<br>**Holdout Test:** 2024-07-01 to 2024-12-31 (3,960 show records). Zero future leakage. |
| **Naive Baseline Definition** | Capacity-Adjusted Historical Baseline: 4-week day-of-week median attendance scaled by relative cinema capacity. |
| **Model Benchmark Comparison** | - **LightGBM (Champion):** MAE: **21.71**, RMSE: **29.85**, WAPE: **7.48%**, $R^2$: **0.938**<br>- **XGBoost:** MAE: 21.94, RMSE: 30.22, WAPE: 7.56%, $R^2$: 0.936<br>- **CatBoost:** MAE: 22.02, RMSE: 29.96, WAPE: 7.59%, $R^2$: 0.937<br>- **Random Forest:** MAE: 23.26, RMSE: 33.60, WAPE: 8.01%, $R^2$: 0.921<br>- **PyTorch LSTM:** MAE: 26.66, RMSE: 36.76, WAPE: 9.18%, $R^2$: 0.906<br>- **Ridge Regression:** MAE: 29.78, RMSE: 40.66, WAPE: 10.26%, $R^2$: 0.885<br>- **Naive Baseline:** MAE: **60.87**, RMSE: **76.10**, WAPE: **20.97%**, $R^2$: **0.596**<br>*(Champion LightGBM achieves **+64.3% WAPE reduction** over naive baseline).* |
| **Saved Model Artifact Location** | `d:\BoxOfficeIQ2\ml\models\best_model.pkl` (Serialized LightGBM pipeline with feature registry) and `d:\BoxOfficeIQ2\ml\models\lstm_model.pt`. |
| **Inference Interface URL** | FastAPI interactive documentation: `http://localhost:8000/docs`<br>Web Application Dashboard: `http://localhost:5500` (or `frontend/index.html`). |

---

### Section 9: Formal Candidate Declarations
- [x] **Originality:** I certify that this project represents my own work and engineering problem-solving.
- [x] **Technical Defense:** I confirm that I understand the complete architecture, feature engineering, and model selection, and I am ready to explain and live-debug any part during the technical interview.
- [x] **Zero Sensitive Data:** I declare that no confidential API keys, passwords, live payment tokens, or real personal identifiable data (PII) are committed to this repository. Only `.env.example` is committed.
- [x] **Verification:** I confirm that all repository links, technical reports, and acceptance test commands work seamlessly.

