-- ============================================================
-- BoxOfficeIQ Database Schema
-- PostgreSQL 14+
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- THEATRES
-- ============================================================
CREATE TABLE IF NOT EXISTS theatres (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    theatre_id      INTEGER UNIQUE,            -- numeric id from synthetic data
    name            VARCHAR(255) NOT NULL,
    theatre_type    VARCHAR(50)  NOT NULL CHECK (theatre_type IN ('Multiplex','SingleScreen','IMAX','DriveIn')),
    city            VARCHAR(100) NOT NULL,
    state           VARCHAR(100),
    capacity        INTEGER      NOT NULL CHECK (capacity > 0),
    is_active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_theatres_city         ON theatres (city);
CREATE INDEX IF NOT EXISTS idx_theatres_theatre_type ON theatres (theatre_type);
CREATE INDEX IF NOT EXISTS idx_theatres_is_active    ON theatres (is_active);

-- ============================================================
-- DAILY ATTENDANCE
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_attendance (
    id               UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    theatre_id       UUID    NOT NULL REFERENCES theatres (id) ON DELETE CASCADE,
    date             DATE    NOT NULL,
    audience_count   INTEGER NOT NULL CHECK (audience_count >= 0),
    online_bookings  INTEGER NOT NULL CHECK (online_bookings >= 0),
    offline_bookings INTEGER NOT NULL CHECK (offline_bookings >= 0),
    occupancy_rate   NUMERIC(5,4),
    is_holiday       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at       TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_attendance_theatre_date UNIQUE (theatre_id, date)
);

CREATE INDEX IF NOT EXISTS idx_attendance_theatre_id  ON daily_attendance (theatre_id);
CREATE INDEX IF NOT EXISTS idx_attendance_date        ON daily_attendance (date);
CREATE INDEX IF NOT EXISTS idx_attendance_theatre_date ON daily_attendance (theatre_id, date);

-- ============================================================
-- MOVIES
-- ============================================================
CREATE TABLE IF NOT EXISTS movies (
    id           UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    movie_id     INTEGER UNIQUE,              -- numeric id from synthetic data
    tmdb_id      INTEGER,
    title        VARCHAR(255) NOT NULL,
    genre        VARCHAR(100),
    release_date DATE,
    runtime      INTEGER CHECK (runtime > 0),
    popularity   NUMERIC(10,4),
    avg_rating   NUMERIC(4,2) CHECK (avg_rating BETWEEN 0 AND 10),
    vote_count   INTEGER CHECK (vote_count >= 0),
    overview     TEXT,
    poster_url   TEXT,
    created_at   TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_movies_release_date ON movies (release_date);
CREATE INDEX IF NOT EXISTS idx_movies_genre        ON movies (genre);
CREATE INDEX IF NOT EXISTS idx_movies_tmdb_id      ON movies (tmdb_id);

-- ============================================================
-- MOVIE REVIEWS
-- ============================================================
CREATE TABLE IF NOT EXISTS movie_reviews (
    id              UUID       PRIMARY KEY DEFAULT uuid_generate_v4(),
    review_id       INTEGER    UNIQUE,        -- numeric id from synthetic data
    movie_id        UUID       NOT NULL REFERENCES movies (id) ON DELETE CASCADE,
    review_text     TEXT       NOT NULL,
    label           VARCHAR(10) NOT NULL CHECK (label IN ('positive','negative','neutral')),
    sentiment_score NUMERIC(6,4),
    created_at      TIMESTAMP  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_reviews_movie_id ON movie_reviews (movie_id);
CREATE INDEX IF NOT EXISTS idx_reviews_label    ON movie_reviews (label);
CREATE INDEX IF NOT EXISTS idx_reviews_created  ON movie_reviews (created_at);

-- ============================================================
-- SENTIMENT RESULTS  (aggregated per movie)
-- ============================================================
CREATE TABLE IF NOT EXISTS sentiment_results (
    id           UUID     PRIMARY KEY DEFAULT uuid_generate_v4(),
    movie_id     UUID     NOT NULL REFERENCES movies (id) ON DELETE CASCADE,
    positive_pct NUMERIC(6,4) NOT NULL CHECK (positive_pct BETWEEN 0 AND 1),
    negative_pct NUMERIC(6,4) NOT NULL CHECK (negative_pct BETWEEN 0 AND 1),
    review_count INTEGER  NOT NULL CHECK (review_count >= 0),
    avg_score    NUMERIC(6,4),
    computed_at  TIMESTAMP NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_sentiment_movie UNIQUE (movie_id)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_movie_id  ON sentiment_results (movie_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_computed  ON sentiment_results (computed_at);

-- ============================================================
-- FORECASTS
-- ============================================================
CREATE TABLE IF NOT EXISTS forecasts (
    id                 UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    theatre_id         UUID        NOT NULL REFERENCES theatres (id) ON DELETE CASCADE,
    forecast_date      DATE        NOT NULL,
    model_name         VARCHAR(100) NOT NULL,
    predicted_audience INTEGER     NOT NULL CHECK (predicted_audience >= 0),
    lower_bound        INTEGER     CHECK (lower_bound >= 0),
    upper_bound        INTEGER     CHECK (upper_bound >= 0),
    created_at         TIMESTAMP   NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_forecast_theatre_date_model UNIQUE (theatre_id, forecast_date, model_name)
);

CREATE INDEX IF NOT EXISTS idx_forecasts_theatre_id    ON forecasts (theatre_id);
CREATE INDEX IF NOT EXISTS idx_forecasts_forecast_date ON forecasts (forecast_date);
CREATE INDEX IF NOT EXISTS idx_forecasts_model_name    ON forecasts (model_name);
CREATE INDEX IF NOT EXISTS idx_forecasts_created       ON forecasts (created_at);

-- ============================================================
-- MODEL EXPERIMENTS
-- ============================================================
CREATE TABLE IF NOT EXISTS model_experiments (
    id               UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_name  VARCHAR(255) NOT NULL,
    model_name       VARCHAR(100) NOT NULL,
    run_id           VARCHAR(255),
    trained_at       TIMESTAMP   NOT NULL DEFAULT NOW(),
    training_start   DATE,
    training_end     DATE,
    validation_start DATE,
    validation_end   DATE,
    notes            TEXT
);

CREATE INDEX IF NOT EXISTS idx_experiments_model_name ON model_experiments (model_name);
CREATE INDEX IF NOT EXISTS idx_experiments_trained_at ON model_experiments (trained_at);

-- ============================================================
-- MODEL METRICS
-- ============================================================
CREATE TABLE IF NOT EXISTS model_metrics (
    id            UUID     PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID     NOT NULL REFERENCES model_experiments (id) ON DELETE CASCADE,
    metric_name   VARCHAR(100) NOT NULL,
    metric_value  NUMERIC(20,8) NOT NULL,
    horizon_days  INTEGER,
    theatre_id    UUID     REFERENCES theatres (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_experiment_id ON model_metrics (experiment_id);
CREATE INDEX IF NOT EXISTS idx_metrics_metric_name   ON model_metrics (metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_theatre_id    ON model_metrics (theatre_id);

-- ============================================================
-- VIEWS (convenience)
-- ============================================================

CREATE OR REPLACE VIEW vw_latest_forecasts AS
SELECT
    f.theatre_id,
    t.name            AS theatre_name,
    t.city,
    f.forecast_date,
    f.model_name,
    f.predicted_audience,
    f.lower_bound,
    f.upper_bound,
    f.created_at
FROM forecasts f
JOIN theatres  t ON t.id = f.theatre_id
WHERE f.created_at = (
    SELECT MAX(f2.created_at)
    FROM   forecasts f2
    WHERE  f2.theatre_id   = f.theatre_id
      AND  f2.forecast_date = f.forecast_date
      AND  f2.model_name    = f.model_name
);

CREATE OR REPLACE VIEW vw_theatre_summary AS
SELECT
    t.id,
    t.name,
    t.theatre_type,
    t.city,
    t.capacity,
    COUNT(DISTINCT da.date)        AS data_days,
    AVG(da.audience_count)         AS avg_daily_audience,
    MAX(da.audience_count)         AS max_daily_audience,
    AVG(da.occupancy_rate)         AS avg_occupancy
FROM theatres t
LEFT JOIN daily_attendance da ON da.theatre_id = t.id
GROUP BY t.id, t.name, t.theatre_type, t.city, t.capacity;
