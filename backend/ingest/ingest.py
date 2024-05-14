from typing import List

from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.markdown import \
    UnstructuredMarkdownLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from retrieve.retrievers import get_base_retriever
from settings import settings


def _load_docs() -> List[Document]:
    loader = DirectoryLoader(
        settings.download_folder, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
    )
    return loader.load()


def _get_embeddings() -> Embeddings:
    store = LocalFileStore(settings.cache_folder)
    core_embeddings_model = OllamaEmbeddings(model=settings.ollama_embeddings_model)

    return CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model, store, namespace=core_embeddings_model.model
    )


def load_vector_store() -> VectorStore:
    embeddings = _get_embeddings()
    return FAISS.load_local(
        settings.vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True,
        normalize_L2=True,
    )


def ingest_documents() -> None:
    # https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/
    # https://stackoverflow.com/a/77865835
    # https://medium.com/@guilhem.cheron35/sql-storage-langchain-rags-inmemorystore-alternative-ex-with-parentdocumentretriever-pgvector-5cc162950d77

    raw_documents = _load_docs()
    vector_store = FAISS.from_documents(
        [raw_documents[0]], _get_embeddings(), normalize_L2=True
    )
    retriever = get_base_retriever(vector_store)
    retriever.add_documents(raw_documents)
    vector_store.save_local(settings.vector_store_path)


if __name__ == "__main__":
    ingest_documents()
