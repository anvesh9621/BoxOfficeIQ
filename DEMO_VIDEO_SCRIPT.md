# BoxOfficeIQ — 3–5 Minute Demonstration Video Script
**Algions AI Labs Assessment Submission | Cinema Ticket Demand Forecasting**  
**Target Duration:** 3 minutes 45 seconds (within the mandatory 3–5 minute window)  

---

## Video Outline & Scene Timing

| Timestamp | Segment Title | On-Screen Visual / Action | Key Speaking Points |
|---|---|---|---|
| **0:00 – 0:35** | **1. Problem Framing & Architecture** | Executive Overview tab (`index.html#dashboard`) | Business problem, prediction unit (turnstiles & seat occupancy), chronological split. |
| **0:35 – 1:15** | **2. EDA & Dataset Profiling** | Dashboard Trend & Theatre Intelligence tabs | 16,440 records across 15 cinema assets, weekend 1.79x surge, capacity distributions. |
| **1:15 – 2:05** | **3. Baseline vs. Model Benchmarks** | Model Performance tab (`index.html#performance`) | Naive capacity baseline vs 6 models, LightGBM champion (+64.3% WAPE reduction), out-of-time test metrics. |
| **2:05 – 2:55** | **4. Live Real Inference & Capacity Corridor** | Demand Forecasting tab (`index.html#forecasting`) | Live selection of cinema, generating 7-day forecast from loaded model artifact, 95% corridor, seat occupancy rate %. |
| **2:55 – 3:35** | **5. Explainability (SHAP) & Cold-Start** | Explainability tab (`index.html#explainability`) | TreeSHAP waterfall decomposition (positive pushes vs negative pulls), cold-start fallback policy. |
| **3:35 – 4:00** | **6. Responsible AI & Conclusion** | Copilot / Simulator / Terminal (`tests/test_pipeline.py`) | Acceptance test suite passing 5/5, privacy by design, human-in-the-loop decision support. |

---

## Word-for-Word Spoken Script

### [0:00 – 0:35] Scene 1: Problem Framing & Solution Overview
*(Action: Display `Executive Overview` dashboard showing the live telemetry indicator and circuit KPIs)*

> **Speaker:**  
> "Hello reviewers. Welcome to BoxOfficeIQ, my submission for the Algions AI Labs assessment in Cinema Ticket Demand Forecasting.
>
> Cinema operators face daily operational commitments: allocating 400-seat versus 200-seat auditoriums, scheduling turnstile ushers, and stocking F&B concessions. Underestimating attendance causes long foyer queues; overestimating wastes payroll.
>
> To solve this, BoxOfficeIQ forecasts daily turnstile tickets and seat occupancy rates across 1 to 7-day horizons. Predictions are made daily at 23:59 IST strictly utilizing information available up to day T. Most importantly, it is built as a decision-support tool that preserves human managerial review."

---

### [0:35 – 1:15] Scene 2: Dataset Profiling & EDA Insights
*(Action: Click on `Theatre Intelligence` tab, scroll through the 15 cinema hubs, and view the 30-day trajectory)*

> **Speaker:**  
> "Our dataset covers 16,440 daily show-level records across 15 operational cinema locations in 8 Indian metros over 3 full years—2022 to 2024.
>
> During EDA, three critical domain patterns emerged:
> First, a massive weekend demand multiplier of 1.79x over Monday-to-Thursday baselines.
> Second, significant holiday volatility, where festival days surge by 2.2x to 2.8x.
> And third, advance digital bookings account for 62.8% of admissions, providing a reliable early signal 24 to 48 hours prior to showtime.
>
> To ensure zero target leakage, we enforced a strict chronological split: training from 2022 through June 2024, and holding out the final six months of 2024 for out-of-time testing. All autoregressive lag features strictly use t greater than or equal to 1, with rolling windows strictly excluding day T."

---

### [1:15 – 2:05] Scene 3: Baseline Comparison & Model Benchmarking
*(Action: Navigate to `Model Performance` tab; hover over the benchmark table and parity plot)*

> **Speaker:**  
> "Per the assessment guidelines, we compared a naive capacity-adjusted historical baseline against six distinct modeling approaches on the unseen holdout test set.
>
> As shown in our benchmark registry:
> The Naive Capacity-Adjusted Baseline achieved an MAE of 60.87 tickets and a WAPE of 20.97%.
> Linear Ridge regression improved this to 10.26% WAPE.
> Our sequential PyTorch Deep LSTM achieved 9.18% WAPE.
> But gradient boosted tree ensembles dominated the problem space.
>
> Our production champion, a tuned LightGBM Regressor, achieved an out-of-time MAE of just 21.71 tickets, an RMSE of 29.85, and a WAPE of 7.48% with an R-squared of 0.938. That represents a 64.3% error reduction over the baseline. The parity scatter plot on the right confirms strong alignment along the 45-degree regression identity line across all attendance volumes."

---

### [2:05 – 2:55] Scene 4: Live Real Inference & Capacity-Aware Evaluation
*(Action: Navigate to `Demand Forecasting` tab. Select 'Prasads IMAX (Hyderabad)', select '7 Days', and click 'Generate Forecast')*

> **Speaker:**  
> "Now let's demonstrate live inference. This is not a hardcoded mock; the frontend calls our FastAPI backend at port 8000, which dynamically loads the serialized LightGBM artifact from disk.
>
> I'll select Prasads IMAX and click 'Generate Forecast' for a 7-day window.
>
> Notice the output:
> The gold line displays the median turnstile projection, wrapped inside a shaded 95% empirical confidence corridor.
> Critically, we have overlaid the red dashed line representing the physical 600-seat auditorium capacity.
> In our daily operations matrix below, we evaluate capacity-aware Seat Occupancy Percentage. For Saturday, the model forecasts 495 tickets, representing an 82% occupancy rate. The system automatically tags this as 'Peak Load (>85%)' and generates an operational recommendation to schedule a Full Roster with plus 4 floor staff and open a secondary concession station."

---

### [2:55 – 3:35] Scene 5: Explainability (TreeSHAP) & Cold-Start Policy
*(Action: Navigate to `Explainable AI` tab; scroll to the Waterfall chart and select a cinema)*

> **Speaker:**  
> "To demystify black-box predictions for cinema operators, we integrated TreeSHAP grounded in cooperative game theory.
>
> Globally, the 7-day autoregressive lag accounts for 30.2% of attribution, followed by weekend status at 26.4%.
>
> In our local waterfall breakdown, managers can inspect exactly why a prediction was made. Starting from the circuit baseline of 232 tickets, the Saturday weekend status pushes demand up by +78 tickets, prior week momentum pushes it up by +42, while bad weather pulls it down by -18, resulting in the exact final prediction.
>
> For cold-start scenarios—such as brand-new movie debuts without historical lags—the system automatically falls back to a hierarchical Bayesian prior, modulates it with real-time DistilBERT review sentiment velocity, and expands the confidence interval by +18% on Day 1 and 2 to communicate uncertainty."

---

### [3:35 – 4:00] Scene 6: Responsible AI Governance & Conclusion
*(Action: Switch briefly to terminal to show `python tests/test_pipeline.py` passing 5/5)*

> **Speaker:**  
> "Finally, BoxOfficeIQ embodies responsible AI:
> It operates under zero-PII privacy by design—collecting only aggregated turnstiles without personal customer data.
> In our terminal, running our automated acceptance test suite verifies all five criteria: dataset integrity, zero leakage, artifact loading, real inference, and baseline outperformance.
>
> All documentation, technical reports, and code are ready in the repository. Thank you for your consideration, and I look forward to the technical walkthrough!"

---

## Recording Tips for Candidate:
1. **Resolution:** 1080p (1920x1080) at 30 or 60 fps.
2. **Audio:** Use a clean headset or dedicated microphone; avoid room echo.
3. **Pacing:** Speak with confident, professional cadence; do not rush through the numbers.
4. **Tool:** Loom, OBS Studio, or Windows Game Bar (`Win + G`).
5. **Video Upload:** Upload as **Unlisted** to YouTube, Loom, or Google Drive (ensure link sharing is set to *"Anyone with the link can view"*).

