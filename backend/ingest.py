import shutil

from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.markdown import \
    UnstructuredMarkdownLoader
from langchain_community.vectorstores.faiss import FAISS
from settings import settings


def clear_vector_store():
    shutil.rmtree(settings.vector_store_path, ignore_errors=True)
    shutil.rmtree(settings.cache_folder, ignore_errors=True)


def load_docs():
    loader = DirectoryLoader(
        settings.download_folder, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
    )
    return loader.load()


def chunk_docs(raw_documents):
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    return text_splitter.split_documents(raw_documents)


def get_embeddings_model():
    return OllamaEmbeddings(model=settings.ollama_embeddings_model)


def get_embeddings():
    store = LocalFileStore(settings.cache_folder)
    core_embeddings_model = get_embeddings_model()

    return CacheBackedEmbeddings.from_bytes_store(
        core_embeddings_model, store, namespace=core_embeddings_model.model
    )


def ingest_docs(documents):
    embeddings = get_embeddings()
    return FAISS.from_documents(
        documents,
        embeddings,
        normalize_L2=True,
    )


def build_vector_store():
    clear_vector_store()
    raw_documents = load_docs()
    documents = chunk_docs(raw_documents)
    vector_store = ingest_docs(documents)
    vector_store.save_local(settings.vector_store_path)


def load_vector_store():
    embeddings = get_embeddings()
    return FAISS.load_local(
        settings.vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True,
        normalize_L2=True,
    )


if __name__ == "__main__":
    build_vector_store()
