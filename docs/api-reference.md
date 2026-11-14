# BoxOfficeIQ — REST API Reference

> **Base URL:** `http://localhost:8000`  
> **API Version:** `v1` (`/api`)  
> **Interactive Documentation (Swagger):** `http://localhost:8000/docs`  
> **Alternative Documentation (ReDoc):** `http://localhost:8000/redoc`

---

## Table of Contents

1. [Overview & Standards](#overview--standards)
2. [Health Check](#health-check)
3. [Theatres](#theatres)
4. [Movies](#movies)
5. [Attendance & Analytics](#attendance--analytics)
6. [Forecasting & SHAP](#forecasting--shap)
7. [Sentiment Analysis](#sentiment-analysis)
8. [Model Metrics](#model-metrics)
9. [Simulator](#simulator)
10. [AI Assistant & RAG](#ai-assistant--rag)
11. [Error Codes & Formats](#error-codes--formats)

---

## Overview & Standards

### Authentication
Currently, no authentication is enforced for development (`None`). In production environments, Bearer JWT authentication should be passed in the `Authorization: Bearer <token>` header.

### Request & Response Formats
- Content-Type: `application/json` (unless specified otherwise)
- Date Format: ISO 8601 strings (`YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SSZ`)
- Standard Success Response: Direct JSON array, object, or wrapped payload.
- Standard Error Format:
  ```json
  {
    "detail": "Error description message"
  }
  ```

---

## Health Check

### `GET /api/health`
Checks whether backend API services and database connections are operational.

- **Query Parameters:** None
- **Response Schema:**
  ```json
  {
    "status": "healthy",
    "timestamp": "2024-11-13T08:00:00Z",
    "database": "connected",
    "version": "1.0.0"
  }
  ```

---

## Theatres

### `GET /api/theatres`
Returns a list of all 15 cinema theatres and metadata.

- **Query Parameters:**
  - `city` *(optional, string)*: Filter by city (e.g., `Delhi`, `Mumbai`, `Bengaluru`).
  - `theatre_type` *(optional, string)*: Filter by type (`Multiplex`, `SingleScreen`, `IMAX`, `DriveIn`).
- **Response Schema:** Array of Theatre objects:
  ```json
  [
    {
      "theatre_id": "T001",
      "name": "PVR Cinemas – Delhi Central",
      "city": "Delhi",
      "state": "Delhi",
      "theatre_type": "Multiplex",
      "capacity": 450,
      "screens": 5,
      "latitude": 28.6139,
      "longitude": 77.2090
    }
  ]
  ```

### `GET /api/theatres/{theatre_id}`
Returns metadata for a single theatre.

- **Path Parameters:**
  - `theatre_id` *(required, string)*: E.g., `T001`
- **Response:** Single Theatre object.
- **Errors:** `404 Not Found` if `theatre_id` is invalid.

---

## Movies

### `GET /api/movies`
Returns a list of movies tracked in the system.

- **Query Parameters:**
  - `genre` *(optional, string)*: Filter by genre (e.g., `Action`, `Sci-Fi`).
  - `limit` *(optional, int, default: 50)*: Number of records to return.
- **Response Schema:**
  ```json
  [
    {
      "movie_id": "M001",
      "title": "Cosmic Horizon",
      "genre": "Sci-Fi",
      "release_date": "2023-05-15",
      "runtime": 142,
      "popularity": 8.4,
      "avg_rating": 7.8,
      "vote_count": 14500,
      "overview": "A crew ventures into an uncharted wormhole..."
    }
  ]
  ```

---

## Attendance & Analytics

### `GET /api/attendance`
Returns filtered historical daily attendance records.

- **Query Parameters:**
  - `theatre_id` *(optional, string)*: E.g., `T001`
  - `start_date` *(optional, string YYYY-MM-DD)*: E.g., `2024-01-01`
  - `end_date` *(optional, string YYYY-MM-DD)*: E.g., `2024-06-30`
  - `limit` *(optional, int, default: 500)*: Max records returned
- **Response Schema:**
  ```json
  [
    {
      "date": "2024-06-01",
      "theatre_id": "T001",
      "movie_id": "M003",
      "audience_count": 380,
      "online_bookings": 240,
      "offline_bookings": 140,
      "ticket_price": 250.0,
      "show_count": 5
    }
  ]
  ```

### `GET /api/attendance/summary`
Returns high-level KPI aggregations for dashboard summary cards.

- **Query Parameters:**
  - `start_date` *(optional, string)*: Filter window start
  - `end_date` *(optional, string)*: Filter window end
- **Response Schema:**
  ```json
  {
    "total_attendance": 4825900,
    "average_daily_attendance": 293.5,
    "total_online_bookings": 2980100,
    "total_offline_bookings": 1845800,
    "top_performing_theatre": {
      "theatre_id": "T002",
      "name": "IMAX Grand Bengaluru",
      "total_attendance": 420150
    },
    "top_grossing_movie": {
      "movie_id": "M001",
      "title": "Cosmic Horizon",
      "estimated_attendance": 312000
    }
  }
  ```

---

## Forecasting & SHAP

### `GET /api/forecasts/{theatre_id}`
Retrieves 14-day future attendance forecasts with confidence intervals for a given theatre.

- **Path Parameters:**
  - `theatre_id` *(required, string)*: E.g., `T001`
- **Query Parameters:**
  - `model` *(optional, string, default: "ensemble")*: `xgboost`, `lightgbm`, `lstm`, or `ensemble`
- **Response Schema:**
  ```json
  {
    "theatre_id": "T001",
    "model_name": "xgboost",
    "forecast": [
      {
        "forecast_date": "2024-11-14",
        "predicted_count": 312.4,
        "lower_bound": 248.0,
        "upper_bound": 376.8,
        "confidence_level": 0.90,
        "horizon_days": 1
      },
      {
        "forecast_date": "2024-11-15",
        "predicted_count": 425.8,
        "lower_bound": 352.1,
        "upper_bound": 499.5,
        "confidence_level": 0.90,
        "horizon_days": 2
      }
    ]
  }
  ```

### `POST /api/forecasts/predict`
Calculates an ad-hoc real-time prediction using input feature values.

- **Request Body:**
  ```json
  {
    "theatre_id": "T001",
    "theatre_type": "Multiplex",
    "date": "2024-11-20",
    "ticket_price": 250.0,
    "show_count": 4,
    "is_holiday": 0,
    "lag_1": 280,
    "lag_7": 310,
    "lag_14": 295,
    "rolling_mean_7": 290.0,
    "rolling_mean_14": 285.0,
    "rolling_std_7": 35.0
  }
  ```
- **Response Schema:**
  ```json
  {
    "theatre_id": "T001",
    "predicted_attendance": 305.2,
    "lower_bound": 242.0,
    "upper_bound": 368.4,
    "model_used": "xgboost"
  }
  ```

### `GET /api/shap/{theatre_id}`
Returns global and recent local SHAP values explaining model predictions.

- **Path Parameters:**
  - `theatre_id` *(required, string)*: E.g., `T001`
- **Response Schema:**
  ```json
  {
    "theatre_id": "T001",
    "base_value": 289.4,
    "feature_importances": [
      {"feature": "lag_7", "importance": 82.4},
      {"feature": "rolling_mean_7", "importance": 61.3},
      {"feature": "is_weekend", "importance": 45.7},
      {"feature": "is_holiday", "importance": 38.2},
      {"feature": "ticket_price", "importance": 22.1}
    ],
    "latest_local_explanation": {
      "date": "2024-11-14",
      "predicted_value": 312.4,
      "contributions": [
        {"feature": "is_weekend", "value": 0, "contribution": -15.2},
        {"feature": "lag_7", "value": 320, "contribution": 24.8},
        {"feature": "is_holiday", "value": 0, "contribution": -4.1}
      ]
    }
  }
  ```

---

## Sentiment Analysis

### `GET /api/sentiment/{movie_id}`
Returns aggregated audience sentiment metrics for a specific movie.

- **Path Parameters:**
  - `movie_id` *(required, string)*: E.g., `M001`
- **Response Schema:**
  ```json
  {
    "movie_id": "M001",
    "title": "Cosmic Horizon",
    "total_reviews": 45,
    "positive_reviews": 38,
    "negative_reviews": 7,
    "positive_ratio": 0.844,
    "average_confidence": 0.912,
    "sentiment_summary": "Predominantly Positive",
    "sample_reviews": [
      {
        "review_text": "Visual masterpiece with breathtaking sequences.",
        "predicted_label": "positive",
        "confidence": 0.985
      }
    ]
  }
  ```

### `POST /api/sentiment/analyze`
Accepts raw text and returns live DistilBERT sentiment classification.

- **Request Body:**
  ```json
  {
    "text": "The pacing felt dragged and the ending was completely unconvincing."
  }
  ```
- **Response Schema:**
  ```json
  {
    "label": "negative",
    "confidence": 0.942,
    "probabilities": {
      "positive": 0.058,
      "negative": 0.942
    }
  }
  ```

---

## Model Metrics

### `GET /api/metrics`
Returns performance evaluation metrics across trained models on the validation split.

- **Query Parameters:**
  - `split` *(optional, string, default: "validation")*: `validation` or `test`
- **Response Schema:**
  ```json
  [
    {
      "model_name": "xgboost",
      "split": "validation",
      "mae": 26.8,
      "rmse": 39.4,
      "wape": 0.082,
      "r2": 0.923,
      "mape": 0.089
    },
    {
      "model_name": "lightgbm",
      "split": "validation",
      "mae": 28.1,
      "rmse": 40.8,
      "wape": 0.086,
      "r2": 0.917,
      "mape": 0.093
    },
    {
      "model_name": "lstm",
      "split": "validation",
      "mae": 34.5,
      "rmse": 48.2,
      "wape": 0.105,
      "r2": 0.884,
      "mape": 0.114
    }
  ]
  ```

---

## Simulator

### `POST /api/simulator`
Simulates attendance variation under counterfactual scenario adjustments.

- **Request Body:**
  ```json
  {
    "theatre_id": "T001",
    "date": "2024-11-20",
    "ticket_price_adjustment_pct": -10.0,
    "holiday_override": 1,
    "additional_shows": 1,
    "marketing_lift_factor": 1.15
  }
  ```
- **Response Schema:**
  ```json
  {
    "baseline_predicted_attendance": 290.0,
    "simulated_predicted_attendance": 378.5,
    "net_change": 88.5,
    "pct_change": 30.5,
    "estimated_revenue_baseline": 72500.0,
    "estimated_revenue_simulated": 85162.5,
    "revenue_difference": 12662.5
  }
  ```

---

## AI Assistant & RAG

### `POST /api/assistant/chat`
Answers user questions using the hybrid RAG (docs retrieval) + SQL data query pipeline.

- **Request Body:**
  ```json
  {
    "message": "Which theatre had the highest average attendance in 2024?",
    "history": [
      {"role": "user", "content": "Hi"},
      {"role": "assistant", "content": "Hello! How can I assist with cinema analytics?"}
    ]
  }
  ```
- **Response Schema:**
  ```json
  {
    "reply": "In 2024, IMAX Grand Bengaluru (T002) recorded the highest average attendance with 385 daily attendees.",
    "sources": [
      {
        "type": "database_query",
        "details": "Aggregated attendance grouped by theatre_id for 2024"
      },
      {
        "type": "rag_document",
        "doc": "data-dictionary.md",
        "chunk_id": 14
      }
    ]
  }
  ```

---

## Error Codes & Formats

| HTTP Status | Meaning | Typical Cause |
|---|---|---|
| `200 OK` | Success | Normal successful query |
| `400 Bad Request` | Invalid parameters | Date format invalid, missing mandatory field |
| `404 Not Found` | Entity not found | Theatre ID or Movie ID does not exist |
| `422 Unprocessable` | Validation Error | Pydantic schema validation failure |
| `500 Internal Error`| Server Error | Database failure or unhandled exception |

### Example 422 Error Response
```json
{
  "detail": [
    {
      "loc": ["body", "theatre_id"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```
