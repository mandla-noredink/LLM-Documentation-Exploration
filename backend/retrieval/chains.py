from typing import Any, Dict

from ingestion.ingest import load_vector_store
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import (Runnable, RunnableLambda,
                                      RunnablePassthrough)
from retrieval.retrievers import get_reranker_retriever
from retrieval.stuff_refine_documents_chain import \
    create_stuff_refine_documents_chain
from settings import settings

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


def _get_documents_chain(llm, query, optimize: bool) -> Runnable[Dict[str, Any], Any]:
    custom_prompt = PromptTemplate(
        template=PROMPT, input_variables=["context", "question"]
    )
    if optimize:
        return create_stuff_refine_documents_chain(llm, custom_prompt, query)
    return create_stuff_documents_chain(llm, custom_prompt)


_reranker_retriever = get_reranker_retriever(vector_store=load_vector_store())
llm = Ollama(model=settings.ollama_llm, temperature=0)

answer_chain = RunnablePassthrough.assign(
    input=lambda x: x["question"]
) | RunnableLambda(
    lambda x: create_retrieval_chain(
        _reranker_retriever,
        _get_documents_chain(
            llm, 
            x["input"], x.get("optimize", settings.optimize_by_default)
        ),
    )
)

search_chain = RunnableLambda(
    lambda x: _reranker_retriever.get_relevant_documents(x["question"])
)
