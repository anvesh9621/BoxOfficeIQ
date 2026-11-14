"""Theatre SQLAlchemy Model."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, TYPE_CHECKING

from sqlalchemy import Column, String, Integer, Boolean, DateTime, func
from sqlalchemy.orm import relationship

from backend.app.database import Base, GUID

if TYPE_CHECKING:
    from backend.app.models.attendance import DailyAttendance
    from backend.app.models.forecast import Forecast
    from backend.app.models.experiment import ModelMetric


class Theatre(Base):
    __tablename__ = "theatres"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    theatre_id = Column(Integer, unique=True, index=True, nullable=True)  # synthetic sequence id (1-15)
    name = Column(String(255), nullable=False)
    theatre_type = Column(String(50), nullable=False, index=True)  # Multiplex, SingleScreen, IMAX, DriveIn
    city = Column(String(100), nullable=False, index=True)
    state = Column(String(100), nullable=True)
    capacity = Column(Integer, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True, index=True)
    created_at = Column(DateTime, nullable=False, server_default=func.now(), default=datetime.utcnow)

    # Relationships
    attendances = relationship("DailyAttendance", back_populates="theatre", cascade="all, delete-orphan")
    forecasts = relationship("Forecast", back_populates="theatre", cascade="all, delete-orphan")
    metrics = relationship("ModelMetric", back_populates="theatre")

    def __repr__(self) -> str:
        return f"<Theatre(id={self.id}, name='{self.name}', city='{self.city}', capacity={self.capacity})>"

    def to_dict(self) -> dict:
        return {
            "id": str(self.id) if self.id else None,
            "theatre_id": self.theatre_id,
            "name": self.name,
            "theatre_type": self.theatre_type,
            "city": self.city,
            "state": self.state,
            "capacity": self.capacity,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
