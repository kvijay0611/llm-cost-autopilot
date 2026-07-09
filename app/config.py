"""
Central configuration. Reads from environment / .env file.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    OLLAMA_ENABLED: bool = _bool("OLLAMA_ENABLED", False)
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    FORCE_MOCK_MODE: bool = _bool("FORCE_MOCK_MODE", True)

    DB_PATH: str = os.getenv("DB_PATH", "./data/autopilot.db")
    # If set (e.g. on Render, a Postgres connection string), Postgres is used
    # instead of SQLite so multiple services can share one database over the
    # network. Leave unset for local dev — SQLite just works with zero setup.
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    VERIFIER_ENABLED: bool = _bool("VERIFIER_ENABLED", True)
    AUTO_ESCALATE: bool = _bool("AUTO_ESCALATE", True)
    ESCALATION_SIMILARITY_THRESHOLD: float = float(
        os.getenv("ESCALATION_SIMILARITY_THRESHOLD", "0.55")
    )

    ROUTING_CONFIG_PATH: str = os.getenv(
        "ROUTING_CONFIG_PATH", "./app/routing_config.yaml"
    )
    CLASSIFIER_MODEL_PATH: str = os.getenv(
        "CLASSIFIER_MODEL_PATH", "./app/classifier/model.joblib"
    )


settings = Settings()