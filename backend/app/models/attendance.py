"""Daily Attendance SQLAlchemy Model."""

from __future__ import annotations

import uuid
from datetime import datetime, date
from typing import TYPE_CHECKING

from sqlalchemy import Column, Integer, Date, Boolean, DateTime, Numeric, ForeignKey, UniqueConstraint, func
from sqlalchemy.orm import relationship

from backend.app.database import Base, GUID

if TYPE_CHECKING:
    from backend.app.models.theatre import Theatre


class DailyAttendance(Base):
    __tablename__ = "daily_attendance"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    theatre_id = Column(GUID, ForeignKey("theatres.id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    audience_count = Column(Integer, nullable=False)
    online_bookings = Column(Integer, nullable=False, default=0)
    offline_bookings = Column(Integer, nullable=False, default=0)
    occupancy_rate = Column(Numeric(5, 4), nullable=True)
    is_holiday = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, server_default=func.now(), default=datetime.utcnow)

    # Composite Unique Constraint
    __table_args__ = (
        UniqueConstraint("theatre_id", "date", name="uq_attendance_theatre_date"),
    )

    # Relationships
    theatre = relationship("Theatre", back_populates="attendances")

    def __repr__(self) -> str:
        return f"<DailyAttendance(theatre_id={self.theatre_id}, date={self.date}, audience={self.audience_count})>"

    def to_dict(self) -> dict:
        return {
            "id": str(self.id) if self.id else None,
            "theatre_id": str(self.theatre_id) if self.theatre_id else None,
            "date": self.date.isoformat() if isinstance(self.date, (date, datetime)) else str(self.date),
            "audience_count": self.audience_count,
            "online_bookings": self.online_bookings,
            "offline_bookings": self.offline_bookings,
            "occupancy_rate": float(self.occupancy_rate) if self.occupancy_rate is not None else None,
            "is_holiday": self.is_holiday,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
