from __future__ import annotations

import os
from functools import lru_cache
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# Preload .env if it exists
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import field_validator

    class Settings(BaseSettings):
        """Application settings loaded from environment variables / .env file."""

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",
        )

        # Database
        database_url: str = os.getenv("DATABASE_URL", "sqlite:///./boxofficeiq.db")

        # External API Keys
        openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        tmdb_api_key: str = os.getenv("TMDB_API_KEY", "")

        # Security
        secret_key: str = os.getenv("SECRET_KEY", "change-me-in-production")

        # MLflow
        mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

        # App
        environment: str = os.getenv("ENVIRONMENT", "development")
        debug: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "t")

        # CORS
        cors_origins: List[str] = [
            "http://localhost:3000",
            "http://localhost:5500",
            "http://127.0.0.1:5500",
            "http://127.0.0.1:3000",
            "null",
        ]

        @field_validator("cors_origins", mode="before")
        @classmethod
        def parse_cors_origins(cls, v):
            if isinstance(v, str):
                return [origin.strip() for origin in v.split(",") if origin.strip()]
            return v

        @property
        def is_production(self) -> bool:
            return self.environment.lower() == "production"

        @property
        def is_development(self) -> bool:
            return self.environment.lower() == "development"

except ImportError:
    from pydantic import BaseModel, Field

    class Settings(BaseModel):
        """Fallback application settings loaded using standard Pydantic and os.environ."""

        database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./boxofficeiq.db"))
        openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
        tmdb_api_key: str = Field(default_factory=lambda: os.getenv("TMDB_API_KEY", ""))
        secret_key: str = Field(default_factory=lambda: os.getenv("SECRET_KEY", "change-me-in-production"))
        mlflow_tracking_uri: str = Field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
        debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "true").lower() in ("true", "1", "t"))
        cors_origins: List[str] = Field(default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5500",
            "http://127.0.0.1:5500",
            "http://127.0.0.1:3000",
            "null",
        ])

        @property
        def is_production(self) -> bool:
            return self.environment.lower() == "production"

        @property
        def is_development(self) -> bool:
            return self.environment.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


settings = get_settings()
