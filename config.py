"""Central configuration for the Financial Statement RAG Assistant."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Reads project settings from environment variables."""

    data_dir: Path = BASE_DIR / os.getenv("DATA_DIRECTORY", "data")
    vectorstore_dir: Path = BASE_DIR / os.getenv(
        "VECTORSTORE_DIRECTORY",
        "vectorstore",
    )
    log_dir: Path = BASE_DIR / "logs"
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    default_currency: str = os.getenv("DEFAULT_CURRENCY", "USD").upper()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")

    def ensure_directories(self) -> None:
        """Create required runtime folders if they do not exist."""
        for directory in (self.data_dir, self.vectorstore_dir, self.log_dir):
            directory.mkdir(parents=True, exist_ok=True)

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "embeddinggemma",
    )

    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )

    chat_model: str = os.getenv(
        "CHAT_MODEL",
        "llama3.2",
    )


settings = Settings()