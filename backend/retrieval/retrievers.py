from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import (
    Language,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from langchain_core.vectorstores import VectorStore
from langchain_core.stores import BaseStore
from storage.local_base_store import LocalBaseStore
from storage.sql_store import SQLDocStore
from retrieval.parent_document_preprocess_retriever import (
    ParentDocumentPreprocessRetriever,
)
from settings import Storage, settings


def _get_splitter(is_parent: bool) -> TextSplitter:
    if is_parent:
        chunk_size = settings.parent_chunk_size
        chunk_overlap = settings.parent_chunk_overlap
    else:
        chunk_size = settings.child_chunk_size
        chunk_overlap = settings.child_chunk_overlap

    return RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )


def get_docstore() -> BaseStore:
    match settings.storage:
        case Storage.LOCAL:
            return LocalBaseStore(settings.docstore_folder)
        case Storage.REMOTE:
            return SQLDocStore(
                collection_name=settings.doc_store_conn_name,
                connection_string=settings.db_conn_string(),
            )


def get_base_retriever(vector_store: VectorStore) -> ParentDocumentPreprocessRetriever:
    return ParentDocumentPreprocessRetriever(
        vectorstore=vector_store,
        docstore=get_docstore(),
        child_splitter=_get_splitter(is_parent=False),
        parent_splitter=_get_splitter(is_parent=True),
        search_kwargs={"k": settings.pre_rerank_doc_retrieval_num},
    )


def get_reranker_retriever(vector_store: VectorStore) -> ContextualCompressionRetriever:
    # Reranking
    # https://python.langchain.com/v0.1/docs/integrations/retrievers/flashrank-reranker/

    base_retriever = get_base_retriever(vector_store)
    ranker = Ranker(model_name=settings.flashrank_model_name)
    compressor = FlashrankRerank(client=ranker, model=settings.flashrank_model_name)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    return compression_retriever
