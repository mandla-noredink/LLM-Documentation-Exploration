from pydantic_settings import BaseSettings
from enum import Enum


class Settings(BaseSettings):

    ollama_llm: str = "llama3"
    ollama_embeddings_model: str = "nomic-embed-text"
    vector_store_path: str = ".vector_store"
    dropbox_remote_folder: str = "LLM Doc Exp Test Content"
    download_folder: str = ".content/"
    cache_folder: str = ".cache/"
    chunk_size: int = 1000
    chunk_overlap: int = 100

settings = Settings()