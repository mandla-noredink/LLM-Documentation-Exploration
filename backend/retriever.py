from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.llms import Ollama
from pydantic import BaseModel
from typing import Dict, List, Optional
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage

from ingest import load_vector_store, get_embeddings_model
from settings import settings


def dict_subset(di, keys):
    return {key: di[key] for key in keys if key in di}


llm = Ollama(model=settings.ollama_llm, temperature=0)
handler = StdOutCallbackHandler()
vector_store = load_vector_store()

answer_chain = RunnableLambda(
    lambda x: RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_kwargs=dict_subset(x, ["score_threshold", "k"])
        ),
        callbacks=[handler],
        return_source_documents=True,
    )(x["query"])
)

search_chain = RunnableLambda(
    lambda x: get_embeddings_model().embed_query(x["query"])
) | RunnableLambda(lambda x: vector_store.similarity_search_by_vector(x, k=4))
