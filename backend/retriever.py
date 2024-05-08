from typing import Dict

from ingest import get_embeddings_model, load_vector_store
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableLambda
from langsmith import Client
from settings import settings
from langchain.prompts import PromptTemplate

# https://github.com/langchain-ai/langchain/issues/14191
# https://stackoverflow.com/questions/77352474/langchain-how-to-get-complete-prompt-retrievalqa-from-chain-type

PROMPT = "Use the following pieces of context to answer the question at the end. Do not use phrases like 'according to the context' in your answer. If you don't know the answer, just say that you don't know, don't try to make up an answer.\n\n{context}\n\nQuestion: {question}\nHelpful Answer:"

# Create a PromptTemplate instance with your custom template
custom_prompt = PromptTemplate(
    template=PROMPT,
    input_variables=["context", "question"],
)

def dict_subset(di, keys):
    return {key: di[key] for key in keys if key in di}

client = Client()

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
        chain_type_kwargs={
            "prompt": custom_prompt,
        }
    )(x["query"])
)


def search_function(params: Dict[str, str]):
    print(f"params: {params}")
    if "score_threshold" in params:
        print("score_threshold in params")
        return vector_store.similarity_search_with_score_by_vector
    return vector_store.similarity_search_by_vector


search_chain = RunnableLambda(
    lambda x: [
        {**doc.dict(), "score": str(round(score, 3))}
        for (doc, score) in vector_store.similarity_search_with_score_by_vector(
            get_embeddings_model().embed_query(x["query"]),
            **dict_subset(x, ["score_threshold", "k"]),
        )
    ]
)
