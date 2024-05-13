from typing import Any, Dict, Callable, List, Protocol
from operator import itemgetter
import types

from ingest import get_embeddings_model, load_vector_store
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.llms import Ollama
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough, RunnableParallel
from langsmith import Client
from settings import settings
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.retrieval import create_retrieval_chain
from langchain.retrievers import ParentDocumentRetriever

from ingest import load_preprocess_retriever, get_reranker_retriever
from preprocess import pre_embedding_process
# https://github.com/langchain-ai/langchain/issues/14191
# https://stackoverflow.com/questions/77352474/langchain-how-to-get-complete-prompt-retrievalqa-from-chain-type

PROMPT = """Use the following context to give a COMPREHENSIVE and DETAILED answer to the question at the end.
The contextual information is delimited by triple backticks (```).
Be clear, direct, and concise - do not repeat the question or use phrases like 'according to the context' in your answer.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
```
{context}
```

Question: {question}

Helpful Answer:"""

DOCUMENT_PROMPT = """DOCUMENT SOURCE: {source}
DOCUMENT CONTENT:
{page_content}
END OF DOCUMENT
"""


# Create a PromptTemplate instance with your custom template
# custom_prompt = PromptTemplate(
#     template=PROMPT,
#     input_variables=["context", "question"],
# )

document_prompt = PromptTemplate(
    template=DOCUMENT_PROMPT,
    input_variables=["source", "page_content"]
)

def dict_subset(di, keys):
    return {key: di[key] for key in keys if key in di}

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document

# class VectorStoreRetrieverCallable(Protocol):
#     def __call__(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]: ...

# self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
# def _get_relevant_documents(query: str, *, run_manager: CallbackManagerForRetrieverRun, f: VectorStoreRetrieverCallable):
#     query = pre_embedding_process(query)
#     print(f"Query: {query}")
#     return f(query, run_manager=run_manager)
# client = Client()
from langchain.chains.combine_documents import create_stuff_documents_chain, collapse_docs
from langchain.chains import create_retrieval_chain

from langchain.chains.summarize import load_summarize_chain

from stuff_refine_documents_chain import create_stuff_refine_documents_chain


llm = Ollama(model=settings.ollama_llm, temperature=0)
handler = StdOutCallbackHandler()
vector_store = load_vector_store()

summarize_chain = load_summarize_chain(llm, chain_type="refine", verbose=False)

# vector_store_retriever = vector_store.as_retriever()
# funcType = type(vector_store_retriever._get_relevant_documents)
# _f = lambda query, run_manager: _get_relevant_documents(query, run_manager=run_manager, f=vector_store_retriever._get_relevant_documents)
# setattr(vector_store_retriever, "_get_relevant_documents", types.MethodType(_f, vector_store_retriever))

# chain = RunnablePassthrough() | vector_store_retriever
# print(chain.invoke({"query": "test"}))

# combine_docs_chain = create_stuff_documents_chain(
#     llm, custom_prompt
# )


# answer_chain = RunnablePassthrough.assign(input=lambda x: x["question"]) | create_retrieval_chain(
#     retriever,
#     create_stuff_documents_chain(llm, custom_prompt),
# )

def get_documents_chain(llm, query, optimize: bool) -> Runnable[Dict[str, Any], Any]:
    custom_prompt = PromptTemplate(template=PROMPT, input_variables=["context", "question"])
    if optimize:
        return create_stuff_refine_documents_chain(llm, custom_prompt, query)
    return create_stuff_documents_chain(llm, custom_prompt)

# TODO: Filters:
# - [] pre-processing
# - [] reranking
# - [] source summarization
# With all of these unselected, we (should) have the original unimproved behaviour from previous commits.
# Preprocessing requires re-ingestion, and so should simply toggle between a preprocessed and unprocessed vector set.
base_retriever = load_preprocess_retriever()
reranker_retriever = get_reranker_retriever(base_retriever)
answer_chain = RunnablePassthrough.assign(input=lambda x: x["question"]) | RunnableLambda(
    lambda x: create_retrieval_chain(
        # retriever,
        reranker_retriever,
        get_documents_chain(llm, x["input"], x.get("optimize", settings.optimize_by_default)),
        # create_stuff_documents_chain(llm, custom_prompt),
        # create_stuff_refine_documents_chain(llm, custom_prompt, x["input"])
    )
)

# answer_chain = RunnablePassthrough.assign(input=lambda x: x["question"]) | RunnableLambda(
#     lambda x: create_retrieval_chain(
#         get_reranker_retriever(
#             vector_store.as_retriever(search_kwargs=dict_subset(x, ["score_threshold", "k"]))
#         ),
#         # create_stuff_documents_chain(llm, custom_prompt, document_prompt=document_prompt),
#         create_stuff_refine_documents_chain(llm, custom_prompt, x["input"])
#     )
# )


# answer_chain = RunnablePassthrough.assign(input=lambda x: x["question"]) | RunnableLambda(
#     lambda x: create_retrieval_chain(
#         vector_store.as_retriever(search_kwargs=dict_subset(x, ["score_threshold", "k"])),
#         create_stuff_documents_chain(llm, custom_prompt),
#     )
# )


# answer_chain = RunnableLambda(
#     lambda x: create_retrieval_chain(
#         RunnablePassthrough.assign(question=lambda x: pre_embedding_process(x["question"])) | vector_store.as_retriever(
#             search_kwargs=dict_subset(x, ["score_threshold", "k"])
#         ),
#         combine_docs_chain,
#     ).invoke(x)
# )

# print(pre_embedding_process("What are the responsibilities of the First Captain during a fire?"))

# answer_chain = RunnableLambda(
#     lambda x: RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vector_store.as_retriever(
#             search_kwargs=dict_subset(x, ["score_threshold", "k"])
#         ),
#         callbacks=[handler],
#         return_source_documents=True,
#         chain_type_kwargs={
#             "prompt": custom_prompt,
#         }
#     )(x["query"])
# )


# TODO: Update direct search to optionally use preprocessing and reranking (source summarization is N/A)
def search_function(params: Dict[str, str]):
    print(f"params: {params}")
    if "score_threshold" in params:
        print("score_threshold in params")
        return vector_store.similarity_search_with_score_by_vector
    return vector_store.similarity_search_by_vector


search_chain = RunnableLambda(
    lambda x: reranker_retriever.get_relevant_documents(x["question"])
)

# search_chain = RunnableLambda(
#     lambda x: [
#         {**doc.dict(), "score": str(round(score, 3))}
#         for (doc, score) in vector_store.similarity_search_with_score_by_vector(
#             get_embeddings_model().embed_query(pre_embedding_process(x["query"])),
#             **dict_subset(x, ["score_threshold", "k"]),
#         )
#     ]
# )
