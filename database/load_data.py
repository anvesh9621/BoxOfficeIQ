"""BoxOfficeIQ Data Ingestion Script.
Loads synthetic CSV data into PostgreSQL (or configured SQLAlchemy database).

Usage:
    python database/load_data.py
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(WORKSPACE_ROOT / ".env")

from sqlalchemy import text
from backend.app.config import settings
from backend.app.database import engine, SessionLocal, init_db
from backend.app.models.theatre import Theatre
from backend.app.models.attendance import DailyAttendance
from backend.app.models.movie import Movie, MovieReview, SentimentResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("load_data")


def find_data_file(filename: str) -> Path:
    """Search for data file in standard locations (data/raw or ml/data/raw)."""
    candidates = [
        WORKSPACE_ROOT / "data" / "raw" / filename,
        WORKSPACE_ROOT / "ml" / "data" / "raw" / filename,
        WORKSPACE_ROOT / "data" / "processed" / filename,
        WORKSPACE_ROOT / "ml" / "data" / "processed" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not locate '{filename}' in candidate paths.")


def load_theatres(session) -> Dict[int, uuid.UUID]:
    """Extract and upsert theatres from cinema_attendance.csv.
    Returns mapping: int(theatre_id) -> UUID(id).
    """
    attendance_csv = find_data_file("cinema_attendance.csv")
    logger.info(f"Reading theatres from: {attendance_csv}")

    theatres_data: Dict[int, Dict[str, Any]] = {}
    with open(attendance_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["theatre_id"])
            if tid not in theatres_data:
                theatres_data[tid] = {
                    "theatre_id": tid,
                    "name": row["theatre_name"],
                    "theatre_type": row["theatre_type"],
                    "city": row["city"],
                    "state": row.get("state", ""),
                    "capacity": int(row["capacity"]),
                    "is_active": True,
                }

    theatre_map: Dict[int, uuid.UUID] = {}
    inserted = 0
    updated = 0

    for tid, data in sorted(theatres_data.items()):
        existing = session.query(Theatre).filter(Theatre.theatre_id == tid).first()
        if existing:
            existing.name = data["name"]
            existing.theatre_type = data["theatre_type"]
            existing.city = data["city"]
            existing.state = data["state"]
            existing.capacity = data["capacity"]
            existing.is_active = data["is_active"]
            theatre_map[tid] = existing.id
            updated += 1
        else:
            new_id = uuid.uuid4()
            th = Theatre(
                id=new_id,
                theatre_id=data["theatre_id"],
                name=data["name"],
                theatre_type=data["theatre_type"],
                city=data["city"],
                state=data["state"],
                capacity=data["capacity"],
                is_active=data["is_active"],
            )
            session.add(th)
            theatre_map[tid] = new_id
            inserted += 1

    session.flush()
    logger.info(f"Theatres processed: {len(theatre_map)} ({inserted} inserted, {updated} updated)")
    return theatre_map


def load_attendance(session, theatre_map: Dict[int, uuid.UUID], batch_size: int = 2000) -> int:
    """Load daily attendance records using batch insertion."""
    attendance_csv = find_data_file("cinema_attendance.csv")
    logger.info(f"Reading attendance from: {attendance_csv}")

    # Check existing count
    existing_count = session.query(DailyAttendance).count()
    if existing_count > 0:
        logger.info(f"Found {existing_count} existing attendance records in database.")

    total_inserted = 0
    batch = []

    with open(attendance_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["theatre_id"])
            theatre_uuid = theatre_map.get(tid)
            if not theatre_uuid:
                continue

            att_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
            att_count = int(row["audience_count"])
            online = int(row.get("online_bookings", 0))
            offline = int(row.get("offline_bookings", 0))
            occupancy = float(row["occupancy_rate"]) if row.get("occupancy_rate") else None
            is_hol = str(row.get("is_holiday", "0")).lower() in ("1", "true", "t", "yes")

            batch.append({
                "id": uuid.uuid4(),
                "theatre_id": theatre_uuid,
                "date": att_date,
                "audience_count": att_count,
                "online_bookings": online,
                "offline_bookings": offline,
                "occupancy_rate": occupancy,
                "is_holiday": is_hol,
                "created_at": datetime.utcnow(),
            })

            if len(batch) >= batch_size:
                session.bulk_insert_mappings(DailyAttendance, batch)
                total_inserted += len(batch)
                batch.clear()

        if batch:
            session.bulk_insert_mappings(DailyAttendance, batch)
            total_inserted += len(batch)
            batch.clear()

    session.flush()
    logger.info(f"Daily Attendance inserted: {total_inserted} rows")
    return total_inserted


def load_movies(session) -> Dict[int, uuid.UUID]:
    """Load movies from movies.csv.
    Returns mapping: int(movie_id) -> UUID(id).
    """
    movies_csv = find_data_file("movies.csv")
    logger.info(f"Reading movies from: {movies_csv}")

    movie_map: Dict[int, uuid.UUID] = {}
    inserted = 0

    with open(movies_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = int(row["movie_id"])
            existing = session.query(Movie).filter(Movie.movie_id == mid).first()
            if existing:
                movie_map[mid] = existing.id
                continue

            rel_date = None
            if row.get("release_date"):
                try:
                    rel_date = datetime.strptime(row["release_date"], "%Y-%m-%d").date()
                except ValueError:
                    rel_date = None

            new_id = uuid.uuid4()
            m = Movie(
                id=new_id,
                movie_id=mid,
                tmdb_id=int(row["tmdb_id"]) if row.get("tmdb_id") else 1000 + mid,
                title=row["title"],
                genre=row.get("genre"),
                release_date=rel_date,
                runtime=int(row["runtime"]) if row.get("runtime") else None,
                popularity=float(row["popularity"]) if row.get("popularity") else None,
                avg_rating=float(row["avg_rating"]) if row.get("avg_rating") else None,
                vote_count=int(row["vote_count"]) if row.get("vote_count") else None,
                overview=row.get("overview"),
                poster_url=row.get("poster_url"),
            )
            session.add(m)
            movie_map[mid] = new_id
            inserted += 1

    session.flush()
    logger.info(f"Movies processed: {len(movie_map)} ({inserted} inserted)")
    return movie_map


def load_reviews(session, movie_map: Dict[int, uuid.UUID], batch_size: int = 1000) -> int:
    """Load reviews from sample_reviews.csv using batch insertion."""
    reviews_csv = find_data_file("sample_reviews.csv")
    logger.info(f"Reading reviews from: {reviews_csv}")

    total_inserted = 0
    batch = []

    with open(reviews_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = int(row["movie_id"])
            movie_uuid = movie_map.get(mid)
            if not movie_uuid:
                continue

            rid = int(row["review_id"]) if row.get("review_id") else None
            existing = session.query(MovieReview).filter(MovieReview.review_id == rid).first() if rid else None
            if existing:
                continue

            score = 0.9 if row["label"] == "positive" else 0.1
            batch.append({
                "id": uuid.uuid4(),
                "review_id": rid,
                "movie_id": movie_uuid,
                "review_text": row["review_text"],
                "label": row["label"],
                "sentiment_score": score,
                "created_at": datetime.utcnow(),
            })

            if len(batch) >= batch_size:
                session.bulk_insert_mappings(MovieReview, batch)
                total_inserted += len(batch)
                batch.clear()

        if batch:
            session.bulk_insert_mappings(MovieReview, batch)
            total_inserted += len(batch)
            batch.clear()

    session.flush()
    logger.info(f"Movie Reviews inserted: {total_inserted} rows")
    return total_inserted


def run_pipeline() -> None:
    """Main execution wrapper with transaction management and rollback."""
    logger.info("=" * 60)
    logger.info("Starting BoxOfficeIQ Data Ingestion Pipeline")
    logger.info(f"Target Database URL: {settings.database_url}")
    logger.info("=" * 60)

    try:
        # 1. Initialize schema / create tables
        init_db()

        # 2. Open session and insert data
        session = SessionLocal()
        try:
            theatre_map = load_theatres(session)
            
            # Check if attendance is already loaded
            count_att = session.query(DailyAttendance).count()
            if count_att == 0:
                load_attendance(session, theatre_map)
            else:
                logger.info(f"Daily Attendance already has {count_att} records. Skipping attendance insertion.")

            movie_map = load_movies(session)

            count_rev = session.query(MovieReview).count()
            if count_rev == 0:
                load_reviews(session, movie_map)
            else:
                logger.info(f"Movie Reviews already has {count_rev} records. Skipping review insertion.")

            # Commit all operations
            session.commit()
            logger.info("Transaction committed successfully.")

            # 3. Print summary verification
            theatre_count = session.query(Theatre).count()
            att_count = session.query(DailyAttendance).count()
            movie_count = session.query(Movie).count()
            review_count = session.query(MovieReview).count()

            print("\n" + "=" * 50)
            print("  DATABASE INGESTION SUMMARY REPORT")
            print("=" * 50)
            print(f"  Theatres:         {theatre_count:>8,}")
            print(f"  Daily Attendance: {att_count:>8,}")
            print(f"  Movies:           {movie_count:>8,}")
            print(f"  Movie Reviews:    {review_count:>8,}")
            print("=" * 50 + "\n")

        except Exception as e:
            session.rollback()
            logger.error(f"Error during data insertion, transaction rolled back: {e}", exc_info=True)
            raise
        finally:
            session.close()

    except Exception as e:
        logger.error(f"Data ingestion pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
