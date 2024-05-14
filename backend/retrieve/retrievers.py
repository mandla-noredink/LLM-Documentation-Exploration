from flashrank import Ranker
from ingest.local_base_store import LocalBaseStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.text_splitter import (Language, RecursiveCharacterTextSplitter,
                                     TextSplitter)
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from retrieve.parent_document_preprocess_retriever import \
    ParentDocumentPreprocessRetriever
from settings import settings


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


def get_base_retriever(vector_store: VectorStore) -> ParentDocumentPreprocessRetriever:
    return ParentDocumentPreprocessRetriever(
        vectorstore=vector_store,
        docstore=LocalBaseStore(settings.docstore_folder),
        child_splitter=_get_splitter(is_parent=False),
        parent_splitter=_get_splitter(is_parent=True),
        search_kwargs={"k": settings.pre_rerank_doc_retrieval_num},
    )


def get_reranker_retriever(vector_store: VectorStore) -> ContextualCompressionRetriever:
    # Reranking
    # https://python.langchain.com/v0.1/docs/integrations/retrievers/flashrank-reranker/

    base_retriever = get_base_retriever(vector_store)
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=".opt")
    compressor = FlashrankRerank(client=ranker)
    compressor.client = ranker  # NOTE: This is required as the assignment on initialization doesn't seem to work
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    return compression_retriever
