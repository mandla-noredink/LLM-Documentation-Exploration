from typing import List

from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

from retrieval.retrievers import get_base_retriever
from settings import Storage, settings, get_logger
from utils import clear_folder

logger = get_logger(__name__)


def _load_docs() -> List[Document]:
    loader = DirectoryLoader(
        settings.rel_path(settings.download_folder),
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
    )
    return loader.load()


def _get_embeddings() -> Embeddings:
    store = LocalFileStore(settings.cache_folder)
    core_embeddings_model = OllamaEmbeddings(model=settings.ollama_embeddings_model)

    return CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model,
        store,
        namespace=core_embeddings_model.model,
    )


def _delete_local_vector_store():
    clear_folder(settings.cache_folder)
    clear_folder(settings.vector_store_folder)
    clear_folder(settings.docstore_folder)


def _build_local_vector_store() -> VectorStore:
    _delete_local_vector_store()
    raw_documents = _load_docs()
    return FAISS.from_documents(
        [raw_documents[0]],
        _get_embeddings(),
        normalize_L2=True,
    )


def _build_pg_vector_store() -> VectorStore:
    raw_documents = _load_docs()
    return PGVector.from_documents(
        embedding=_get_embeddings(),
        documents=[raw_documents[0]],
        collection_name=settings.vector_store_conn_name,
        connection=settings.db_conn_string(),
        pre_delete_collection=True,
        use_jsonb=True,
    )


def _build_vector_store() -> VectorStore:
    match settings.storage:
        case Storage.LOCAL:
            return _build_local_vector_store()
        case Storage.REMOTE:
            return _build_pg_vector_store()


def _load_pg_vector_store(embeddings: Embeddings) -> PGVector:
    # NOTE: It seems like this already handles storing some metadata, like the document id and source.
    return PGVector(
        embeddings=embeddings,
        collection_name=settings.vector_store_conn_name,
        connection=settings.db_conn_string(),
        use_jsonb=True,
    )


def _load_local_vector_store(embeddings: Embeddings) -> FAISS:
    return FAISS.load_local(
        settings.vector_store_folder,
        embeddings,
        allow_dangerous_deserialization=True,
        normalize_L2=True,
    )


def load_vector_store() -> VectorStore:
    embeddings = _get_embeddings()
    match settings.storage:
        case Storage.LOCAL:
            return _load_local_vector_store(embeddings)
        case Storage.REMOTE:
            return _load_pg_vector_store(embeddings)


def ingest_documents() -> None:
    # https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/
    # https://stackoverflow.com/a/77865835
    # https://medium.com/@guilhem.cheron35/sql-storage-langchain-rags-inmemorystore-alternative-ex-with-parentdocumentretriever-pgvector-5cc162950d77

    logger.debug("Loading raw documents")
    raw_documents = _load_docs()
    logger.debug("Building vector store")
    vector_store = _build_vector_store()
    logger.debug("Getting retriever")
    retriever = get_base_retriever(vector_store)
    logger.debug("Adding documents to retriever")
    retriever.add_documents(raw_documents)
    if settings.storage is Storage.LOCAL:
        logger.debug("Saving vector store")
        assert type(vector_store) == FAISS
        vector_store.save_local(settings.vector_store_folder)
    logger.debug("Document ingestion complete")
