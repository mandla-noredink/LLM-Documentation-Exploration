import shutil

from langchain.embeddings import CacheBackedEmbeddings
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.markdown import \
    UnstructuredMarkdownLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain.storage import InMemoryStore

from parent_document_preprocess_retriever import ParentDocumentPreprocessRetriever
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

def ingest_docs_with_preprocess_retriever():
    # https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/
    # https://stackoverflow.com/a/77865835
    # https://medium.com/@guilhem.cheron35/sql-storage-langchain-rags-inmemorystore-alternative-ex-with-parentdocumentretriever-pgvector-5cc162950d77

    parent_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    child_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )

    # TODO: get stores if exists else create
    raw_documents = load_docs()
    store = FAISS.from_documents([raw_documents[0]], get_embeddings(), normalize_L2=True)
    docstore = InMemoryStore()

    retriever = ParentDocumentPreprocessRetriever(
        vectorstore=store,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 12},
    )

    # TODO: ONLY if stores don't exist, add documents
    retriever.add_documents(raw_documents)

    # TODO: save stores if they don't exist

    return retriever
    

def get_reranker_retriever(base_retriever):
    # Reranking
    # https://python.langchain.com/v0.1/docs/integrations/retrievers/flashrank-reranker/

    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import FlashrankRerank
    from flashrank import Ranker
    # from langchain_openai import ChatOpenAI

    # llm = ChatOpenAI(temperature=0)
    # ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".opt")
    ranker = Ranker()
    # print(ranker)
    compressor = FlashrankRerank(client=ranker)
    compressor.client = ranker  # NOTE: This is required as the assignment on initialization doesn't seem to work
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    return compression_retriever


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

retriever = ingest_docs_with_preprocess_retriever()

if __name__ == "__main__":
    build_vector_store()
