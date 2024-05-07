from enum import Enum

from pydantic_settings import BaseSettings


class VectorDB(Enum):
    LOCAL = "FAISS"
    REMOTE = "PGVECTOR"


class Settings(BaseSettings):

    ollama_llm: str = "llama3"
    ollama_embeddings_model: str = "nomic-embed-text"
    vector_store_path: str = ".vector_store"
    dropbox_remote_folder: str = "LLM Doc Exp Test Content"
    download_folder: str = ".content/"
    cache_folder: str = ".cache/"
    chunk_size: int = 1000
    chunk_overlap: int = 100

    # For PG Vector Store
    pg_host: str = "pg_host"
    pg_port: str = "pg_port"
    pg_user: str = "pg_user"
    pg_password: str = "pg_password"
    pg_dbname: str = "pg_dbname"

    # Vector DB selection
    vector_db: VectorDB = VectorDB.LOCAL


settings = Settings()
