# BoxOfficeIQ — Cinema Manager's User Guide

> **Intelligent Cinema Demand Forecasting & Decision Support System**  
> *A practical guide for cinema general managers, programmers, and operations teams.*

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Dashboard Overview (8 Core Pages)](#dashboard-overview)
3. [Reading Attendance Forecasts](#reading-attendance-forecasts)
4. [Using the What-If Simulator](#using-the-what-if-simulator)
5. [Understanding SHAP Feature Importance in Plain English](#understanding-shap-feature-importance)
6. [Using the AI Assistant](#using-the-ai-assistant)
7. [The Sentiment Analysis Module](#the-sentiment-analysis-module)
8. [Plain English Glossary](#plain-english-glossary)

---

## Getting Started

BoxOfficeIQ is designed to give you actionable foresight into theatre attendance so you can:
- **Optimise Staffing:** Avoid over-rostering concession and floor staff on slow Tuesdays, and prevent understaffing during holiday spikes.
- **Plan Inventory:** Stock appropriate volumes of popcorn, beverages, and 3D glasses based on predicted headcounts.
- **Screen Programming:** Dynamically reassign blockbuster titles to higher-capacity auditoriums when demand surges.

### Accessing the System
1. Open your browser and navigate to `http://localhost:5500` (or your internal company link).
2. Use the left navigation sidebar to switch between operational pages.

---

## Dashboard Overview

The platform is organized into **8 specialized modules**:

| Module | Purpose | Daily Operational Use |
|---|---|---|
| **1. Overview** | Executive Summary | Quick pulse check on overall chain attendance, today's targets, top movies. |
| **2. Forecasting** | 14-Day Demand Projections | Primary planning screen. Check expected attendance per theatre over the next two weeks. |
| **3. Analytics** | Historical Deep-Dive | Analyse past performance, compare theatres, review month-over-month growth. |
| **4. Heatmap** | Visual Scheduling Grid | Spot high-density days and seasonal patterns across theatres at a glance. |
| **5. Sentiment** | Audience Reception Hub | Track early word-of-mouth for currently running films before box office numbers react. |
| **6. Model Metrics** | AI Performance Health | Verify forecasting accuracy, check current error margins (MAE, WAPE). |
| **7. Simulator** | "What-If" Planning Tool | Test operational scenarios (e.g., ticket price tweaks, adding a late-night show). |
| **8. AI Assistant** | Operational Chatbot | Ask direct questions about your data or operational guidance in natural language. |

---

## Reading Attendance Forecasts

When you select a theatre on the **Forecasting** page, you will see a trend line accompanied by a shaded band:

```
Attendance
   ▲
500│                     ┌── Upper Bound (90%)
400│           .- - - - -o - - - - - - - - 
300│──────────●──────────●── Forecast (Point Estimate)
200│         /           o - - - - - - - - 
100│        /            └── Lower Bound (90%)
  0└───────┴─────────────┴────────────────► Days Ahead
        Historical     Forecast Horizon
```

### The Three Numbers That Matter:
1. **Forecast Point Estimate (Solid Line):** The most likely daily attendance count expected by our AI models.
2. **Lower Bound (Bottom of Shaded Band):** A conservative estimate (90% confidence floor). Base your minimum required staffing and food prep on this floor to avoid waste.
3. **Upper Bound (Top of Shaded Band):** A peak estimate (90% confidence ceiling). Ensure you have on-call staff and sufficient stock if attendance reaches this level.

### What Does "MAE = 28" Mean for You?
**MAE** stands for **Mean Absolute Error**. If your theatre shows an MAE of 28, it means that historically, on average, the forecast is within **±28 people** of actual ticket sales. If the forecast says 300, actual attendance typically falls between 272 and 328.

---

## Using the What-If Simulator

The **Simulator** page lets you model hypothetical operational changes before committing real resources.

### Available Controls:
- **Ticket Price Adjustment (±%):** See how price increases or discounts impact demand based on historical elasticity.
- **Holiday Override:** Test what happens if an unexpected regional holiday or school break is declared.
- **Additional Showings (+1 / +2 shows):** Estimate the attendance increase from adding early morning or midnight screenings.
- **Marketing Push:** Simulate the effect of a local promotional blitz.

### Simulator Limitations:
- The simulator provides estimates based on historical statistical correlations.
- Uncontrollable external shocks (e.g. extreme storm, transit strikes) cannot be anticipated by the simulator.

---

## Understanding SHAP Feature Importance

On the Forecasting page, BoxOfficeIQ displays a chart labelled **"Top Factors Driving This Forecast" (SHAP)**. 

Instead of treating the AI as an opaque black box, SHAP explains *why* the model predicted a particular number:

- **Green / Positive Bars (Pushing Attendance Up):**
  - E.g., `is_weekend`: Today is Saturday (+85 people).
  - E.g., `lag_7`: The theatre was packed last Saturday (+45 people).
  - E.g., `is_holiday`: Republic Day or Eid festival (+120 people).
- **Red / Negative Bars (Pushing Attendance Down):**
  - E.g., `rolling_mean_7`: Attendance has softened overall this week (-25 people).
  - E.g., `ticket_price`: Premium pricing applied (-15 people).

Use this to double-check AI reasoning against your local intuition!

---

## Using the AI Assistant

The **AI Assistant** tab is your interactive operational advisor. It reads from real attendance databases and internal methodology documentation.

### Great Questions to Ask:
- *"Which theatre had the highest attendance last weekend?"*
- *"Why is my Saturday forecast higher than usual?"*
- *"Explain the difference between MAE and RMSE in plain words."*
- *"What should I stock if Delhi Central is forecasted for 420 people?"*

### What It Cannot Do:
- Cannot perform destructive actions (it has read-only access; it cannot edit ticket prices or modify schedules directly).
- Cannot give accurate advice on events not recorded in the database (e.g. unannounced local street festivals).

---

## The Sentiment Analysis Module

The **Sentiment** page tracks audience reaction to movies currently in theatres using our fine-tuned language model:

- **Positive Ratio > 75%:** Strong audience word-of-mouth! Expect the film to have lower-than-average drop-off in weeks 2 and 3. Consider retaining the film on the primary screen.
- **Positive Ratio < 50%:** Critical or negative audience reception. Anticipate a steep drop-off next week. Plan to shift secondary showtimes to other releases.

*Remember:* High sentiment does not always equal mass appeal (an indie drama can have 90% positive sentiment with a niche audience size).

---

## Plain English Glossary

| Term | What It Means in Everyday Cinema Language |
|---|---|
| **MAE (Mean Absolute Error)** | The average number of seats our prediction was off by. Lower is better. |
| **RMSE (Root Mean Squared Error)** | Similar to MAE, but heavily penalizes rare, large forecasting misses. |
| **WAPE (Weighted Absolute Percentage Error)** | Your total percentage error over the whole cinema chain. A WAPE of 8% means our forecasts are 92% accurate overall. |
| **SHAP (Shapley Additive exPlanations)** | A fair scorecard showing which factors (weekends, holidays, price) boosted or dragged down the prediction. |
| **LSTM (Long Short-Term Memory)** | A deep neural network designed to recognise repeating cycles over the past 28 days. |
| **RAG (Retrieval-Augmented Generation)** | The technology powering our AI Assistant: it looks up verified numbers in our database and manuals before answering. |
| **DistilBERT** | A lightweight language AI that reads audience reviews and classifies them as positive or negative. |
