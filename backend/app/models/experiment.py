"""ModelExperiment and ModelMetric SQLAlchemy Models."""

from __future__ import annotations

import uuid
from datetime import datetime, date
from typing import List, TYPE_CHECKING

from sqlalchemy import Column, String, Integer, Date, DateTime, Numeric, Text, ForeignKey, func
from sqlalchemy.orm import relationship

from backend.app.database import Base, GUID

if TYPE_CHECKING:
    from backend.app.models.theatre import Theatre


class ModelExperiment(Base):
    __tablename__ = "model_experiments"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    experiment_name = Column(String(255), nullable=False)
    model_name = Column(String(100), nullable=False, index=True)
    run_id = Column(String(255), nullable=True)
    trained_at = Column(DateTime, nullable=False, server_default=func.now(), default=datetime.utcnow, index=True)
    training_start = Column(Date, nullable=True)
    training_end = Column(Date, nullable=True)
    validation_start = Column(Date, nullable=True)
    validation_end = Column(Date, nullable=True)
    notes = Column(Text, nullable=True)

    # Relationships
    metrics = relationship("ModelMetric", back_populates="experiment", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<ModelExperiment(id={self.id}, model='{self.model_name}', run_id='{self.run_id}')>"

    def to_dict(self) -> dict:
        return {
            "id": str(self.id) if self.id else None,
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "run_id": self.run_id,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "training_start": self.training_start.isoformat() if isinstance(self.training_start, (date, datetime)) else str(self.training_start) if self.training_start else None,
            "training_end": self.training_end.isoformat() if isinstance(self.training_end, (date, datetime)) else str(self.training_end) if self.training_end else None,
            "validation_start": self.validation_start.isoformat() if isinstance(self.validation_start, (date, datetime)) else str(self.validation_start) if self.validation_start else None,
            "validation_end": self.validation_end.isoformat() if isinstance(self.validation_end, (date, datetime)) else str(self.validation_end) if self.validation_end else None,
            "notes": self.notes,
        }


class ModelMetric(Base):
    __tablename__ = "model_metrics"

    id = Column(GUID, primary_key=True, default=uuid.uuid4)
    experiment_id = Column(GUID, ForeignKey("model_experiments.id", ondelete="CASCADE"), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_value = Column(Numeric(20, 8), nullable=False)
    horizon_days = Column(Integer, nullable=True)
    theatre_id = Column(GUID, ForeignKey("theatres.id", ondelete="SET NULL"), nullable=True, index=True)

    # Relationships
    experiment = relationship("ModelExperiment", back_populates="metrics")
    theatre = relationship("Theatre", back_populates="metrics")

    def __repr__(self) -> str:
        return f"<ModelMetric(experiment_id={self.experiment_id}, name='{self.metric_name}', val={self.metric_value})>"

    def to_dict(self) -> dict:
        return {
            "id": str(self.id) if self.id else None,
            "experiment_id": str(self.experiment_id) if self.experiment_id else None,
            "metric_name": self.metric_name,
            "metric_value": float(self.metric_value) if self.metric_value is not None else None,
            "horizon_days": self.horizon_days,
            "theatre_id": str(self.theatre_id) if self.theatre_id else None,
        }
