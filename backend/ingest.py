import shutil
from typing import List

from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.markdown import \
    UnstructuredMarkdownLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from settings import VectorDB, settings

CONNECTION_STRING = f"postgresql+psycopg://{settings.pg_user}:{settings.pg_password}@{settings.pg_host}:{settings.pg_port}/{settings.pg_dbname}"
COLLECTION_NAME = "llm_documentation"


def _clear_vector_store() -> None:
    shutil.rmtree(settings.vector_store_path, ignore_errors=True)
    shutil.rmtree(settings.cache_folder, ignore_errors=True)


def _load_docs() -> List[Document]:
    loader = DirectoryLoader(
        settings.download_folder, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
    )
    return loader.load()


def _chunk_docs(raw_documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    return text_splitter.split_documents(raw_documents)


def get_embeddings_model():
    return OllamaEmbeddings(model=settings.ollama_embeddings_model)


def _get_embeddings() -> Embeddings:
    store = LocalFileStore(settings.cache_folder)
    core_embeddings_model = get_embeddings_model()

    return CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model, store, namespace=core_embeddings_model.model
    )


def _build_local_vector_store() -> None:
    _clear_vector_store()
    raw_documents = _load_docs()
    documents = _chunk_docs(raw_documents)
    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(
        documents,
        embeddings,
        normalize_L2=True,
    )
    vector_store.save_local(settings.vector_store_path)


def _build_pg_vector_store() -> None:
    raw_documents = _load_docs()
    documents = _chunk_docs(raw_documents)
    embeddings = _get_embeddings()
    PGVector.from_documents(
        embedding=embeddings,
        documents=documents,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )


def _build_vector_store() -> None:
    store_builders = {
        VectorDB.LOCAL: _build_local_vector_store,
        VectorDB.REMOTE: _build_pg_vector_store,
    }
    store_builders[settings.vector_db]()


def _load_pg_vector_store(embeddings: Embeddings) -> PGVector:
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )


def _load_local_vector_store(embeddings: Embeddings) -> FAISS:
    return FAISS.load_local(
        settings.vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True,
        normalize_L2=True,
    )


def load_vector_store() -> PGVector | FAISS:
    embeddings = _get_embeddings()
    store_loaders = {
        VectorDB.LOCAL: _load_local_vector_store,
        VectorDB.REMOTE: _load_pg_vector_store,
    }
    return store_loaders[settings.vector_db](embeddings)


if __name__ == "__main__":
    _build_vector_store()
