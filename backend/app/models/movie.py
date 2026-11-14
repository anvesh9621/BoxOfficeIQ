"""Movie, MovieReview, and SentimentResult SQLAlchemy Models."""

from __future__ import annotations

import uuid
from datetime import datetime, date
from typing import List, Optional

from sqlalchemy import Column, String, Integer, Date, DateTime, Numeric, Text, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import relationship

from backend.app.database import Base, GUID


class Movie(Base):
    __tablename__ = "movies"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    movie_id = Column(Integer, unique=True, index=True, nullable=True)  # synthetic movie id
    tmdb_id = Column(Integer, index=True, nullable=True)
    title = Column(String(255), nullable=False)
    genre = Column(String(100), index=True, nullable=True)
    release_date = Column(Date, index=True, nullable=True)
    runtime = Column(Integer, nullable=True)
    popularity = Column(Numeric(10, 4), nullable=True)
    avg_rating = Column(Numeric(4, 2), nullable=True)
    vote_count = Column(Integer, nullable=True)
    overview = Column(Text, nullable=True)
    poster_url = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now(), default=datetime.utcnow)

    # Relationships
    reviews = relationship("MovieReview", back_populates="movie", cascade="all, delete-orphan")
    sentiment_result = relationship("SentimentResult", back_populates="movie", uselist=False, cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Movie(id={self.id}, title='{self.title}', genre='{self.genre}', rating={self.avg_rating})>"

    def to_dict(self) -> dict:
        return {
            "id": str(self.id) if self.id else None,
            "movie_id": self.movie_id,
            "tmdb_id": self.tmdb_id,
            "title": self.title,
            "genre": self.genre,
            "release_date": self.release_date.isoformat() if isinstance(self.release_date, (date, datetime)) else str(self.release_date) if self.release_date else None,
            "runtime": self.runtime,
            "popularity": float(self.popularity) if self.popularity is not None else None,
            "avg_rating": float(self.avg_rating) if self.avg_rating is not None else None,
            "vote_count": self.vote_count,
            "overview": self.overview,
            "poster_url": self.poster_url,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class MovieReview(Base):
    __tablename__ = "movie_reviews"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    review_id = Column(Integer, unique=True, index=True, nullable=True)
    movie_id = Column(GUID, ForeignKey("movies.id", ondelete="CASCADE"), nullable=False, index=True)
    review_text = Column(Text, nullable=False)
    label = Column(String(10), nullable=False, index=True)  # positive, negative, neutral
    sentiment_score = Column(Numeric(6, 4), nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now(), default=datetime.utcnow)

    # Relationship
    movie = relationship("Movie", back_populates="reviews")

    def __repr__(self) -> str:
        return f"<MovieReview(id={self.id}, movie_id={self.movie_id}, label='{self.label}')>"

    def to_dict(self) -> dict:
        return {
            "id": str(self.id) if self.id else None,
            "review_id": self.review_id,
            "movie_id": str(self.movie_id) if self.movie_id else None,
            "review_text": self.review_text,
            "label": self.label,
            "sentiment_score": float(self.sentiment_score) if self.sentiment_score is not None else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SentimentResult(Base):
    __tablename__ = "sentiment_results"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    movie_id = Column(GUID, ForeignKey("movies.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    positive_pct = Column(Numeric(6, 4), nullable=False)
    negative_pct = Column(Numeric(6, 4), nullable=False)
    review_count = Column(Integer, nullable=False, default=0)
    avg_score = Column(Numeric(6, 4), nullable=True)
    computed_at = Column(DateTime, nullable=False, server_default=func.now(), default=datetime.utcnow)

    # Relationship
    movie = relationship("Movie", back_populates="sentiment_result")

    def __repr__(self) -> str:
        return f"<SentimentResult(movie_id={self.movie_id}, pos={self.positive_pct}, neg={self.negative_pct})>"

    def to_dict(self) -> dict:
        return {
            "id": str(self.id) if self.id else None,
            "movie_id": str(self.movie_id) if self.movie_id else None,
            "positive_pct": float(self.positive_pct) if self.positive_pct is not None else 0.0,
            "negative_pct": float(self.negative_pct) if self.negative_pct is not None else 0.0,
            "review_count": self.review_count,
            "avg_score": float(self.avg_score) if self.avg_score is not None else None,
            "computed_at": self.computed_at.isoformat() if self.computed_at else None,
        }
