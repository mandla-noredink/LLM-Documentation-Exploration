from typing import Optional
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
from retrieval.llm_compressor import LLMCompressor
from retrieval.llm import llm


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


def get_reranker_retriever(
    vector_store: VectorStore,
    top_n: Optional[int] = None,
) -> ContextualCompressionRetriever:
    # Reranking
    # https://python.langchain.com/v0.1/docs/integrations/retrievers/flashrank-reranker/

    base_retriever = get_base_retriever(vector_store)
    ranker = Ranker(model_name=settings.flashrank_model_name, cache_dir=".opt")
    # compressor = FlashrankRerank(client=ranker)
    # compressor.client = ranker
    # compressor.top_n = top_n or settings.default_reranked_top_n
    compressor = LLMCompressor(llm=llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )

    return compression_retriever

# from langchain_core.prompts import ChatPromptTemplate

# rerank_system_prompt = '''You are an Assistant responsible for helping detect whether the retrieved context is relevant to the engineering question. For a given input, you need to output a single token: "Yes" or "No" indicating the retrieved content is relevant to the engineering question. Examples are below:

# Engineering Question: What caused the merge fire?
# Context: """Jenkins was unable to coordinate merges and was throwing out memory errors. This caused multiple days of merge queue outages."""
# Relevant: Yes

# Engineering Question: What caused the merge fire?
# Context: """Question ID 15232 was reported as taking more than 60 seconds to submit. I'm making a pull request to stop showing new topics."""
# Relevant: No

# Engineering Question: What can we do if the database maxes out CPU resources?
# Context: """The quiz engine is the largest consumer of database resources, so reducing its resources is the main way to reduce database load."""
# Relevant: Yes

# Engineering Question: What can we do if the database maxes out CPU resources?
# Context: """error: var "firecrackers" is not defined. Question ID 23342352354 unable to perform celebration animation. root/directory/jobs/haskell_in_ruby."""
# Relevant: No'''

# rerank_user_prompt = '''Engineering Question: {question}
# Context: """{context}"""
# Relevant:
# '''

# rerank_prompt = ChatPromptTemplate.from_messages([
#     ("system", rerank_system_prompt),
#     ("user", rerank_user_prompt)
# ])

# def get_relevant_contexts(llm, question, contexts, rerank_prompt=rerank_prompt):
#     """Pass question and contexts to llm to get reranked context using LLM as decider of relevance"""
#     final_contexts = []
#     for context in contexts:
#         chain = rerank_prompt | llm
#         reply = chain.invoke({'question': question, 'context': context})
#         if reply == 'Yes':
#             final_contexts.append(context)
#     return final_contexts

# # Path is:
# # "Take user question" -> "Get initial relevant docs from Vector DB" -> "Rerank docs with Yes/No relevance and keep all the Yes" -> Pass retained docs in as context to LLM for final question
# # But I can't find out how to fit into the infrastructure with the ContextualCompressionRetriever and RunnablePassthrough's