from enum import Enum
from pydantic_settings import BaseSettings


class Storage(Enum):
    LOCAL = "FAISS"
    REMOTE = "PGVECTOR"


class Settings(BaseSettings):

    ollama_llm: str = "llama3"
    ollama_embeddings_model: str = "nomic-embed-text"
    dropbox_remote_folder: str = "LLM Doc Exp Test Content"
    download_folder: str = ".content/"
    vector_store_path: str = ".vectorstore/"
    docstore_folder: str = ".docstore/"
    cache_folder: str = ".cache/"
    temp_folder: str = ".tmp/"

    pre_rerank_doc_retrieval_num: int = 12
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

    def db_conn_string(self):
        return f"postgresql+psycopg://{self.pg_user}:{self.pg_password}@{self.pg_host}:{self.pg_port}/{self.pg_dbname}"


settings = Settings()
