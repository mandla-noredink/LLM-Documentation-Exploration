import os
from enum import Enum

from __version__ import __version__
from pydantic_settings import BaseSettings


class Mode(Enum):
    CHATBOT = "query"
    DOC_SEARCH = "search"


class Settings(BaseSettings):

    environment: str = os.getenv("ENV", "development")
    app_id: str = f"llm-doc-exp:{__version__}"
    base_api_url: str = "http://127.0.0.1:8080"
    query_endpoint: str = "/query/stream_events/"
    search_endpoint: str = "/search/invoke/"
    feedback_endpoint: str = "/feedback/"
    default_mode: Mode = Mode.CHATBOT

    min_num_sources: int = 2
    max_num_sources: int = 8
    default_num_sources: int = 5

    max_score_threshold: float = 2.0
    min_score_threshold: float = 0.0
    default_score_threshold: float = 1.0
    score_threshold_step: float = 0.1


settings = Settings()
