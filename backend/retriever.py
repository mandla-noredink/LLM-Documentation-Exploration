from langchain.chains import RetrievalQA
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.llms import Ollama
from pydantic import BaseModel
from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.prompts import PromptTemplate, MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.messages import AIMessage, HumanMessage

from ingest import load_vector_store, get_embeddings_model
from settings import settings

RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""



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


qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[handler],
    return_source_documents=True,

)

# conversation_chain = condense_question_chain | retriever
# retriever_chain = RunnableBranch(
#     (
#         RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
#             run_name="HasChatHistoryCheck"
#         ),
#         conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
#     ),
#     (
#         RunnableLambda(itemgetter("question")) | retriever
#     ).with_config(run_name="RetrievalChainWithNoHistory"),
# ).with_config(run_name="RouteDependingOnChatHistory")


# context = (
#     RunnablePassthrough.assign(docs=retriever_chain)
#     .assign(context=lambda x: format_docs(x["docs"]))
#     .with_config(run_name="RetrieveDocs")
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", RESPONSE_TEMPLATE),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{question}"),
#     ]
# )

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', "Hello! I'm a language model trained on a large corpus of text. I can answer questions and provide information on a wide range of topics. What would you like to know?"),
        # MessagesPlaceholder("chat_history"),
        ("human", "{text}"),
    ]
)

answer_chain = qa_with_sources_chain

search_chain = RunnableLambda(lambda x: get_embeddings_model().embed_query(x["query"])) | RunnableLambda(lambda x: vector_store.similarity_search_by_vector(x, k=4))
# answer_chain = prompt | RunnableLambda(lambda x: qa_with_sources_chain({"query": x["text"]}))
# answer_chain = prompt | RunnableLambda(itemgetter("text")) | qa_with_sources_chain
# answer_chain = prompt | context | llm

# answer_chain = RunnablePassthrough.assign(chat_history=serialize_history) | context | default_response_synthesizer





# llm = Ollama(model=settings.ollama_llm, temperature=0)
# handler = StdOutCallbackHandler()
# vector_store = load_vector_store()
# retriever = vector_store.as_retriever()

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ('system', "Hello! I'm a language model trained on a large corpus of text. I can answer questions and provide information on a wide range of topics. What would you like to know?"),
#         # MessagesPlaceholder("chat_history"),
#         ("human", "{text}"),
#     ]
# )

# result = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=retriever,
#     callbacks=[handler],
#     return_source_documents=True,
# )

# answer_chain = prompt | result
