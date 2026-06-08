"""Centralised settings loaded from environment / .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Data
    data_source: str = "synthetic"  # "synthetic" | "nrel"
    nrel_api_key: str = "DEMO_KEY"
    pvdaq_system_id: int = 2

    # Paths (relative to repo root — resolved at runtime)
    model_dir: Path = Path("artifacts/models")
    data_processed_dir: Path = Path("data/processed")

    # Anomaly detection
    anomaly_threshold_sigma: float = 2.0

    # Train/test split
    train_test_split_date: str = "2023-01-01"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    allowed_origins: str = "http://localhost:3000"  # comma-separated, or "*"


settings = Settings()
