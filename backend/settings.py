from enum import Enum

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    ollama_llm: str = "llama3"
    ollama_embeddings_model: str = "nomic-embed-text"
    dropbox_remote_folder: str = "LLM Doc Exp Test Content"
    download_folder: str = ".content/"
    vector_store_path: str = ".vectorstore/"
    docstore_folder: str = ".docstore/"
    cache_folder: str = ".cache/"
    chunk_size: int = 1000
    chunk_overlap: int = 100

    optimize_by_default: bool = False
    pre_rerank_doc_retrieval_num: int = 12

settings = Settings()