from enum import Enum
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from logging.config import dictConfig
from pathlib import Path
import logging
import os

_DOTENV_PATH = Path(__file__).parent.joinpath(".env")
_BASE_FOLDER = os.path.dirname(os.path.realpath(__file__))
_LOG_FOLDER = ".logs/"
_LOG_PACKAGE_BLOCKLIST = [
    "unstructured",
    "markdown",
]


def create_file(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    filename = Path(file_path)
    filename.touch(exist_ok=True)


class Storage(Enum):
    LOCAL = "FAISS"
    REMOTE = "PGVECTOR"


class LogConfig(BaseModel):
    LOGGER_NAME: str = "llmdocexp"
    LOG_FORMAT: str = "%(levelprefix)s | %(name)s | %(asctime)s | %(message)s"
    LOG_LEVEL: str = "INFO"

    # Logging config
    version: int = 1
    disable_existing_loggers: bool = False
    formatters: dict = {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": LOG_FORMAT,
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    }
    handlers: dict = {
        "debug_console_handler": {
            "level": "DEBUG",
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "info_rotating_file_handler": {
            "level": "INFO",
            "formatter": "default",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": f"{_LOG_FOLDER}/info.log",
            "mode": "a",
            "maxBytes": 1048576,
            "backupCount": 5,
        },
        "error_file_handler": {
            "level": "WARNING",
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": f"{_LOG_FOLDER}/error.log",
            "mode": "a",
        },
    }
    loggers: dict = {
        "": {
            "handlers": [
                "debug_console_handler",
                "info_rotating_file_handler",
                "error_file_handler",
            ],
            "level": LOG_LEVEL,
        },
    }


class Settings(BaseSettings):

    ollama_llm: str = "llama3"
    ollama_embeddings_model: str = "nomic-embed-text"
    dropbox_remote_folder: str = "LLM Doc Exp Test Content"
    download_folder: str = f"{_BASE_FOLDER}/.content/"
    vector_store_folder: str = f"{_BASE_FOLDER}/.vectorstore/"
    docstore_folder: str = f"{_BASE_FOLDER}/.docstore/"
    cache_folder: str = f"{_BASE_FOLDER}/.cache/"
    temp_folder: str = f"{_BASE_FOLDER}/.tmp/"

    pre_rerank_doc_retrieval_num: int = 12
    default_reranked_top_n: int = 3
    optimize_by_default: bool = False
    parent_chunk_size: int = 1000
    parent_chunk_overlap: int = 100
    child_chunk_size: int = 1000
    child_chunk_overlap: int = 100

    # For PG Vector Store
    pg_host: str = "pg_host"
    pg_port: str = "pg_port"
    pg_user: str = "pg_user"
    pg_password: str = "pg_password"
    pg_dbname: str = "pg_dbname"

    vector_store_conn_name: str = "llm_doc_exp__vectors"
    doc_store_conn_name: str = "llm_doc_exp__documents"
    flashrank_model_name: str = "ms-marco-MiniLM-L-12-v2"

    # Vector DB selection
    storage: Storage = Storage.REMOTE

    model_config = SettingsConfigDict(env_file=_DOTENV_PATH)

    def db_conn_string(self):
        return f"postgresql+psycopg://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_dbname}"

    def rel_path(self, path):
        return os.path.relpath(path, _BASE_FOLDER)


create_file(f"{_LOG_FOLDER}info.log")
create_file(f"{_LOG_FOLDER}error.log")
dictConfig(LogConfig().model_dump())
for module in _LOG_PACKAGE_BLOCKLIST:
    logging.getLogger(module).setLevel(logging.WARNING)


def get_logger(name):
    return logging.getLogger(name)


settings = Settings()
