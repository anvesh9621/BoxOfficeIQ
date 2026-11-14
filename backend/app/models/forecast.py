"""Forecast SQLAlchemy Model."""

from __future__ import annotations

import uuid
from datetime import datetime, date
from typing import TYPE_CHECKING

from sqlalchemy import Column, String, Integer, Date, DateTime, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import relationship

from backend.app.database import Base, GUID

if TYPE_CHECKING:
    from backend.app.models.theatre import Theatre


class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    theatre_id = Column(GUID, ForeignKey("theatres.id", ondelete="CASCADE"), nullable=False, index=True)
    forecast_date = Column(Date, nullable=False, index=True)
    model_name = Column(String(100), nullable=False, index=True)
    predicted_audience = Column(Integer, nullable=False)
    lower_bound = Column(Integer, nullable=True)
    upper_bound = Column(Integer, nullable=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now(), default=datetime.utcnow)

    # Unique constraint per theatre, date, and model
    __table_args__ = (
        UniqueConstraint("theatre_id", "forecast_date", "model_name", name="uq_forecast_theatre_date_model"),
    )

    # Relationships
    theatre = relationship("Theatre", back_populates="forecasts")

    def __repr__(self) -> str:
        return f"<Forecast(theatre_id={self.theatre_id}, date={self.forecast_date}, model='{self.model_name}', pred={self.predicted_audience})>"

    def to_dict(self) -> dict:
        return {
            "id": str(self.id) if self.id else None,
            "theatre_id": str(self.theatre_id) if self.theatre_id else None,
            "forecast_date": self.forecast_date.isoformat() if isinstance(self.forecast_date, (date, datetime)) else str(self.forecast_date),
            "model_name": self.model_name,
            "predicted_audience": self.predicted_audience,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
