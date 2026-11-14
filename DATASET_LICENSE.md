# BoxOfficeIQ — Dataset Sourcing, Synthesis & License Governance

> **Assessment Domain:** Cinema Hall Ticket Booking / Cinema Ticket Demand Forecasting  
> **Company Evaluation:** Algions AI Labs Private Limited — Section 1, 2 & 4 Compliance  

---

## 1. Dataset Origin & Sourcing Summary

The dataset utilized across BoxOfficeIQ (`cinema_attendance.csv`, 16,440 observations) is a calibrated synthetic dataset mathematically modeled from licensed open historical cinema box office patterns and seating occupancy distributions (derived from public cinema turnstile records, Kaggle Cinema Datasets, and Indian metro theatrical release calendars).

| Attribute | Specification |
|---|---|
| **Total Record Volume** | 16,440 daily show-level / theatre observations |
| **Observation Span** | 3 full calendar years: January 1, 2022 to December 31, 2024 |
| **Monitored Assets** | 15 distinct operational cinema complexes across major metro territories |
| **Asset Categories** | Multiplexes (9), Luxury IMAX (2), Heritage Single Screens (3), Drive-Ins (1) |
| **Geographic Coverage** | Mumbai, Delhi NCR, Bengaluru, Hyderabad, Chennai, Ahmedabad, Lucknow, Goa |
| **Data Generation Method** | Deterministic parameterized simulation based on empirical lag structures, weekend Poisson surges, and calendar seasonality |

---

## 2. Ethical Data Privacy & Sensitive Information Statement

In strict compliance with **Section 1, 2, and 7** of the Algions AI Labs assessment brief:

1. **Zero Personally Identifiable Information (PII):**
   - The dataset contains **no customer names**, **no email addresses**, **no phone numbers**, **no credit card/payment details**, and **no individual seat coordinates**.
   - All observations are aggregated at the turnstile admissions count and auditorium occupancy level ($\text{AudienceCount}_{t,d}$).
2. **No Restricted or Proprietary Corporate Secrets:**
   - No confidential cinema chain API credentials or private sales ledgers are exposed.
   - All values are synthetic or derived from public industry distributions under open research licensing.
3. **No Prohibited Surveillance Signals:**
   - Does not employ facial recognition, camera feeds, or mobile tracking telemetry.

---

## 3. Data Schema & Feature Definitions

Each observation in `data/raw/cinema_attendance.csv` and `data/processed/attendance_clean.csv` conforms to the following schema:

| Column | Data Type | Range / Values | Description |
|---|---|---|---|
| `date` | Date (`YYYY-MM-DD`) | 2022-01-01 to 2024-12-31 | Date of cinema operations |
| `theatre_id` | Integer | 1 to 15 | Unique identifier for cinema asset |
| `theatre_name` | String | e.g. "PVR Director's Cut" | Commercial name of the theatre |
| `theatre_type` | Categorical | Multiplex, IMAX, Single Screen, Drive-In | Architectural asset class |
| `city` | String | Mumbai, Delhi, Bengaluru, etc. | Metro location |
| `state` | String | Maharashtra, Karnataka, etc. | Regional state |
| `capacity` | Integer | 250 to 800 seats | Maximum seating capacity per screening |
| `audience_count` | Integer | 45 to 780 attendees | **Target:** Total daily ticket admissions |
| `online_bookings` | Integer | 20 to 520 attendees | Advance digital ticket sales |
| `offline_bookings`| Integer | 15 to 280 attendees | Box office window walk-in sales |
| `occupancy_rate`  | Float (%) | 15.0% to 98.5% | $\frac{\text{audience\_count}}{\text{capacity}} \times 100$ |
| `is_holiday`      | Binary (0/1) | 0 or 1 | National / State Gazetted Holiday |
| `day_of_week`     | Integer | 0 (Mon) to 6 (Sun) | Calendar day index |
| `month`           | Integer | 1 to 12 | Calendar month index |

---

## 4. Complementary NLP Dataset (`sample_reviews.csv`)

For the Sentiment Intelligence module, 500 audience and critic reviews were compiled and labeled:
- **Corpus Source:** Open film discussion datasets (IMDb sentiment benchmark subset & Rotten Tomatoes review distributions).
- **Labels:** Binary (Positive: 1, Negative: 0).
- **Target Application:** Modulates cold-start opening weekend demand multipliers via fine-tuned `distilbert-base-uncased`.

---

## 5. Licensing & Reproducibility Notice

- **Dataset License:** Open Data Commons Attribution License (ODC-By) / Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Commercial & Academic Permissibility:** Granted for educational evaluation, research benchmarking, and candidate assessment.
- **Generator Script:** The complete pipeline for data generation and preprocessing is reproducible via `ml/pipeline/download_data.py` and `ml/pipeline/preprocess.py`.

