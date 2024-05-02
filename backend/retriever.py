from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain_community.llms import Ollama
from pydantic import BaseModel
from typing import Dict, List, Optional

from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage, HumanMessage

from ingest import load_vector_store, get_embeddings_model
from settings import settings


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def format_docs(docs) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request):
    chat_history = request.get("chat_history") or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


llm = Ollama(model=settings.ollama_llm, temperature=0)
handler = StdOutCallbackHandler()
vector_store = load_vector_store()
retriever = vector_store.as_retriever()


answer_chain = RunnableLambda(
    lambda x: RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": x["threshold"]},
        ),
        callbacks=[handler],
        return_source_documents=True,
    )(x)
)

search_chain = RunnableLambda(
    lambda x: get_embeddings_model().embed_query(x["query"])
) | RunnableLambda(lambda x: vector_store.similarity_search_by_vector(x, k=4))
