"""
download_data.py
================
Generates a realistic synthetic cinema dataset for BoxOfficeIQ.

Outputs (saved to ml/data/raw/):
  - cinema_attendance.csv   (~16425 rows: 15 theatres × 1095 days)
  - movies.csv              (50 movies)
  - sample_reviews.csv      (500 reviews, balanced pos/neg)

Usage:
    python ml/pipeline/download_data.py
"""

import os
import sys
import random
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Theatre definitions ───────────────────────────────────────────────────────
THEATRES = [
    {"theatre_id": 1,  "theatre_name": "PVR Cinemas Juhu",           "type": "Multiplex",    "city": "Mumbai",    "state": "Maharashtra", "capacity": 350},
    {"theatre_id": 2,  "theatre_name": "INOX Nariman Point",          "type": "Multiplex",    "city": "Mumbai",    "state": "Maharashtra", "capacity": 320},
    {"theatre_id": 3,  "theatre_name": "Cinepolis Andheri",           "type": "Multiplex",    "city": "Mumbai",    "state": "Maharashtra", "capacity": 290},
    {"theatre_id": 4,  "theatre_name": "Regal Cinema Colaba",         "type": "SingleScreen", "city": "Mumbai",    "state": "Maharashtra", "capacity": 600},
    {"theatre_id": 5,  "theatre_name": "IMAX Wadala",                 "type": "IMAX",         "city": "Mumbai",    "state": "Maharashtra", "capacity": 500},
    {"theatre_id": 6,  "theatre_name": "PVR Select Citywalk",         "type": "Multiplex",    "city": "Delhi",     "state": "Delhi",       "capacity": 310},
    {"theatre_id": 7,  "theatre_name": "INOX Nehru Place",            "type": "Multiplex",    "city": "Delhi",     "state": "Delhi",       "capacity": 280},
    {"theatre_id": 8,  "theatre_name": "Delite Cinema Daryaganj",     "type": "SingleScreen", "city": "Delhi",     "state": "Delhi",       "capacity": 550},
    {"theatre_id": 9,  "theatre_name": "PVR Forum Bengaluru",         "type": "Multiplex",    "city": "Bengaluru", "state": "Karnataka",   "capacity": 300},
    {"theatre_id": 10, "theatre_name": "Cinepolis Orion Mall",        "type": "Multiplex",    "city": "Bengaluru", "state": "Karnataka",   "capacity": 270},
    {"theatre_id": 11, "theatre_name": "IMAX UB City",                "type": "IMAX",         "city": "Bengaluru", "state": "Karnataka",   "capacity": 480},
    {"theatre_id": 12, "theatre_name": "Sathyam Cinemas",             "type": "Multiplex",    "city": "Chennai",   "state": "Tamil Nadu",  "capacity": 340},
    {"theatre_id": 13, "theatre_name": "Rohini Silver Screens",       "type": "SingleScreen", "city": "Chennai",   "state": "Tamil Nadu",  "capacity": 580},
    {"theatre_id": 14, "theatre_name": "Prasads Multiplex IMAX",      "type": "IMAX",         "city": "Hyderabad", "state": "Telangana",   "capacity": 510},
    {"theatre_id": 15, "theatre_name": "Star Drive-In Ahmedabad",     "type": "DriveIn",      "city": "Ahmedabad", "state": "Gujarat",     "capacity": 200},
]

# Base audience occupancy (fraction of capacity) by theatre type
THEATRE_BASE_OCCUPANCY = {
    "Multiplex":    0.55,
    "SingleScreen": 0.45,
    "IMAX":         0.65,
    "DriveIn":      0.40,
}

# ── Indian public holidays 2022-2024 ──────────────────────────────────────────
HOLIDAYS = {
    # 2022
    date(2022, 1, 1), date(2022, 1, 26), date(2022, 3, 1),  # Maha Shivratri
    date(2022, 3, 18), date(2022, 4, 10), date(2022, 4, 14), date(2022, 4, 15),
    date(2022, 5, 3), date(2022, 8, 9), date(2022, 8, 15), date(2022, 10, 2),
    date(2022, 10, 5), date(2022, 10, 24), date(2022, 11, 8), date(2022, 12, 25),
    # 2023
    date(2023, 1, 1), date(2023, 1, 14), date(2023, 1, 26), date(2023, 2, 18),
    date(2023, 3, 7), date(2023, 3, 22), date(2023, 3, 30), date(2023, 4, 4),
    date(2023, 4, 7), date(2023, 4, 14), date(2023, 5, 5), date(2023, 6, 29),
    date(2023, 8, 15), date(2023, 9, 7), date(2023, 9, 19), date(2023, 10, 2),
    date(2023, 10, 24), date(2023, 11, 13), date(2023, 11, 27), date(2023, 12, 25),
    # 2024
    date(2024, 1, 1), date(2024, 1, 14), date(2024, 1, 22), date(2024, 1, 26),
    date(2024, 2, 24), date(2024, 3, 25), date(2024, 3, 29), date(2024, 4, 9),
    date(2024, 4, 14), date(2024, 4, 17), date(2024, 4, 21), date(2024, 5, 23),
    date(2024, 6, 17), date(2024, 7, 17), date(2024, 8, 15), date(2024, 10, 2),
    date(2024, 10, 12), date(2024, 10, 13), date(2024, 11, 1), date(2024, 11, 15),
    date(2024, 12, 25),
}


def seasonal_factor(d: date) -> float:
    """Return a multiplier for seasonal peaks (summer, Diwali, Christmas)."""
    month, day = d.month, d.day
    # Summer blockbuster season (May–July)
    if month in (5, 6, 7):
        return np.random.uniform(1.25, 1.50)
    # Diwali season (Oct-Nov)
    if month in (10, 11):
        return np.random.uniform(1.20, 1.45)
    # Christmas/New Year
    if month == 12 and day >= 20:
        return np.random.uniform(1.30, 1.55)
    if month == 1 and day <= 7:
        return np.random.uniform(1.20, 1.40)
    # Holi / spring (March)
    if month == 3:
        return np.random.uniform(1.05, 1.20)
    # Lean months (Feb, Sep)
    if month in (2, 9):
        return np.random.uniform(0.80, 0.95)
    return np.random.uniform(0.95, 1.10)


def weekend_factor(d: date) -> float:
    """Return a multiplier for weekends."""
    if d.weekday() == 4:   # Friday (new releases)
        return np.random.uniform(1.30, 1.55)
    if d.weekday() == 5:   # Saturday
        return np.random.uniform(1.40, 1.70)
    if d.weekday() == 6:   # Sunday
        return np.random.uniform(1.35, 1.65)
    return np.random.uniform(0.85, 1.05)


def holiday_factor(d: date) -> float:
    """Extra bump on public holidays."""
    return np.random.uniform(1.30, 1.60) if d in HOLIDAYS else 1.0


# ── Generate cinema_attendance.csv ───────────────────────────────────────────
def generate_attendance() -> pd.DataFrame:
    print("Generating cinema_attendance.csv ...")
    start = date(2022, 1, 1)
    end   = date(2024, 12, 31)
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    records = []
    for theatre in THEATRES:
        tid   = theatre["theatre_id"]
        tname = theatre["theatre_name"]
        ttype = theatre["type"]
        city  = theatre["city"]
        state = theatre["state"]
        cap   = theatre["capacity"]
        base_occ = THEATRE_BASE_OCCUPANCY[ttype]

        # Slight per-theatre random multiplier for variety
        theatre_bias = np.random.uniform(0.90, 1.10)

        for d in dates:
            sf = seasonal_factor(d)
            wf = weekend_factor(d)
            hf = holiday_factor(d)

            occupancy = base_occ * sf * wf * hf * theatre_bias
            # Add Gaussian noise (±10%)
            occupancy += np.random.normal(0, 0.05)
            occupancy = np.clip(occupancy, 0.05, 1.0)

            audience_count = int(round(cap * occupancy))
            audience_count = max(0, audience_count)

            # Online bookings ~55–65% of total
            online_ratio = np.random.uniform(0.50, 0.70)
            online_bookings  = int(round(audience_count * online_ratio))
            offline_bookings = audience_count - online_bookings

            records.append({
                "date":             d.strftime("%Y-%m-%d"),
                "theatre_id":       tid,
                "theatre_name":     tname,
                "theatre_type":     ttype,
                "city":             city,
                "state":            state,
                "capacity":         cap,
                "audience_count":   audience_count,
                "online_bookings":  online_bookings,
                "offline_bookings": offline_bookings,
                "occupancy_rate":   round(audience_count / cap, 4),
                "is_holiday":       int(d in HOLIDAYS),
            })

    df = pd.DataFrame(records)
    out = RAW_DIR / "cinema_attendance.csv"
    df.to_csv(out, index=False)
    print(f"  → {len(df):,} rows saved to {out}")
    return df


# ── Movie genres & data ───────────────────────────────────────────────────────
GENRES = ["Action", "Comedy", "Drama", "Thriller", "Romance", "Horror",
          "Sci-Fi", "Animation", "Biography", "Crime", "Adventure", "Fantasy"]

MOVIE_TITLES = [
    "Kalki 2898 AD", "Pushpa: The Rise", "RRR", "Brahmastra", "Pathaan",
    "Jawan", "Animal", "Dunki", "Tiger 3", "Fighter",
    "Yodha", "Bade Miyan Chote Miyan", "Maidaan", "Crew", "Srikanth",
    "GOAT", "Kalki Chronicles", "Stree 2", "Bhool Bhulaiyaa 3", "Singham Returns",
    "Welcome 3", "Merry Christmas", "Aa Okkati Adakku", "Devara", "Lucky Baskhar",
    "The Sabarmati Report", "Vettaiyan", "Kanguva", "Amaran", "Singham Again",
    "Pushpa 2 The Rule", "Deva", "Sky Force", "Emergency", "Chhaava",
    "Fateh", "Azaad", "Game Changer", "Daaku Maharaaj", "Sankranthiki Vastunnam",
    "Infinity", "Nexus", "Dark Matter", "Rising Sun", "Storm Breaker",
    "Crimson Peak", "The Last Stand", "Echoes of Tomorrow", "Silent Thunder", "Phantom City",
]

OVERVIEWS = [
    "An epic saga set in a dystopian future where civilisations collide.",
    "A heartwarming tale of family bonds tested by extraordinary circumstances.",
    "A gripping thriller that keeps you on the edge of your seat until the last frame.",
    "An action-packed adventure featuring stunning visuals and breathtaking stunts.",
    "A comedy of errors that unfolds across multiple cities in hilarious fashion.",
    "A romantic journey through monsoon-drenched landscapes and unspoken emotions.",
    "A biographical drama recounting the struggles of an unsung national hero.",
    "A supernatural horror that blends folklore with modern psychological terror.",
    "A sci-fi spectacle exploring the boundaries of artificial intelligence and humanity.",
    "A crime thriller set in the underbelly of a rapidly changing megacity.",
]


def generate_movies() -> pd.DataFrame:
    print("Generating movies.csv ...")
    records = []
    release_start = date(2021, 1, 1)
    release_end   = date(2024, 12, 31)
    span_days = (release_end - release_start).days

    for i, title in enumerate(MOVIE_TITLES[:50]):
        movie_id    = i + 1
        genre       = np.random.choice(GENRES)
        rel_offset  = np.random.randint(0, span_days)
        release_dt  = (release_start + timedelta(days=int(rel_offset))).strftime("%Y-%m-%d")
        runtime     = int(np.random.randint(95, 185))
        popularity  = round(float(np.random.uniform(30, 95)), 2)
        avg_rating  = round(float(np.random.uniform(5.5, 9.0)), 1)
        vote_count  = int(np.random.randint(1500, 150000))
        overview    = OVERVIEWS[i % len(OVERVIEWS)]

        records.append({
            "movie_id":    movie_id,
            "title":       title,
            "genre":       genre,
            "release_date": release_dt,
            "runtime":     runtime,
            "popularity":  popularity,
            "avg_rating":  avg_rating,
            "vote_count":  vote_count,
            "overview":    overview,
        })

    df = pd.DataFrame(records)
    out = RAW_DIR / "movies.csv"
    df.to_csv(out, index=False)
    print(f"  → {len(df):,} rows saved to {out}")
    return df


# ── Review templates ──────────────────────────────────────────────────────────
POSITIVE_REVIEWS = [
    "Absolutely loved every minute of this film! The performances were stellar and the direction was top-notch.",
    "One of the best movies I have seen in years. The story was engaging from start to finish.",
    "The cinematography was breathtaking and the music score perfectly complemented every scene.",
    "An incredible cinematic experience. The lead actor delivered a career-best performance.",
    "Brilliant screenplay backed by exceptional acting. This movie deserves every award.",
    "I went in with low expectations but came out thoroughly impressed. A hidden gem!",
    "The visual effects were jaw-dropping and the action sequences were phenomenal.",
    "A perfect blend of emotion and entertainment. Cried and laughed in equal measure.",
    "The director has outdone themselves this time. Every frame is a work of art.",
    "Outstanding film that leaves a lasting impression. Already planning to watch it again.",
    "Gripping plot with unexpected twists. The climax was absolutely mind-blowing.",
    "The chemistry between the lead actors was electric. A joy to watch throughout.",
    "A cinematic masterpiece that redefines Indian filmmaking. Truly world-class.",
    "Powerful storytelling with nuanced characters. This is what cinema should be.",
    "The soundtrack is phenomenal and the choreography in the dance sequences is unmatched.",
    "Superb production values. The sets and costumes transported me to another era.",
    "The script is tightly written with no dull moments. Bravo to the entire team.",
    "Emotional, engaging, and thoroughly entertaining. A must-watch for the whole family.",
    "The director has created something truly special. This film will stand the test of time.",
    "Remarkable performances all around. The supporting cast was equally impressive.",
]

NEGATIVE_REVIEWS = [
    "Hugely disappointing. The trailer promised much more than the movie delivered.",
    "The plot was full of holes and the acting felt forced and unconvincing.",
    "Wasted two and a half hours of my life. The story made no sense whatsoever.",
    "Over-hyped and under-delivered. The special effects looked cheap and rushed.",
    "The pacing was unbearably slow and the climax was a complete letdown.",
    "Poor screenplay with cardboard characters who had zero development throughout.",
    "The dialogue was cringe-worthy and the comedy fell completely flat.",
    "Predictable from the first scene. No surprises and no emotional depth.",
    "The music was jarring and completely out of place with the narrative.",
    "A mess of a movie that could not decide what it wanted to be.",
    "The lead performance was wooden and the supporting cast was forgettable.",
    "Despite a big budget, the film felt hollow and emotionally empty.",
    "Too long and too loud. Sensory overload with no substance underneath.",
    "The logic gaps were infuriating. The audience deserves better.",
    "Yet another formulaic entertainer with zero originality or creativity.",
    "The first half showed promise but the second half was an absolute disaster.",
    "Excessive CGI could not hide the weak story and thin characters.",
    "A complete waste of a talented cast. The script let everyone down.",
    "The climax was rushed and unsatisfying after hours of tedious build-up.",
    "Overly melodramatic with scenes that dragged on way past their welcome.",
]


def generate_reviews(movies_df: pd.DataFrame) -> pd.DataFrame:
    print("Generating sample_reviews.csv ...")
    records = []
    movie_ids = movies_df["movie_id"].tolist()

    for i in range(500):
        review_id   = i + 1
        movie_id    = np.random.choice(movie_ids)
        is_positive = i < 250  # First 250 positive, next 250 negative (then shuffle)
        label       = "positive" if is_positive else "negative"

        if is_positive:
            base_text = POSITIVE_REVIEWS[i % len(POSITIVE_REVIEWS)]
        else:
            base_text = NEGATIVE_REVIEWS[(i - 250) % len(NEGATIVE_REVIEWS)]

        # Add minor variation to avoid exact duplicates
        suffixes_pos = [
            " Highly recommended!", " Do not miss it!", " Watch it on the big screen.",
            " Take the whole family.", " A cinematic treat.", "",
        ]
        suffixes_neg = [
            " Save your money.", " Skip this one.", " Very disappointing.",
            " Not worth the ticket price.", " Expected so much more.", "",
        ]
        suffix = np.random.choice(suffixes_pos if is_positive else suffixes_neg)
        review_text = base_text + suffix

        records.append({
            "review_id":   review_id,
            "movie_id":    int(movie_id),
            "review_text": review_text,
            "label":       label,
        })

    # Shuffle to interleave pos/neg
    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["review_id"] = df.index + 1

    out = RAW_DIR / "sample_reviews.csv"
    df.to_csv(out, index=False)
    print(f"  → {len(df):,} rows saved to {out}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BoxOfficeIQ — Synthetic Dataset Generator")
    print("=" * 60)
    print(f"Output directory: {RAW_DIR}\n")

    attendance_df = generate_attendance()
    movies_df     = generate_movies()
    reviews_df    = generate_reviews(movies_df)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"cinema_attendance.csv : {len(attendance_df):>7,} rows × {len(attendance_df.columns)} cols")
    print(f"movies.csv            : {len(movies_df):>7,} rows × {len(movies_df.columns)} cols")
    print(f"sample_reviews.csv    : {len(reviews_df):>7,} rows × {len(reviews_df.columns)} cols")
    print("\nAll files generated successfully.")


if __name__ == "__main__":
    main()
