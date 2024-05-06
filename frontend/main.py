import streamlit as st
import requests
from typing import Optional, List
import json
import sseclient
import uuid
from dataclasses import dataclass

from settings import settings, Mode

import logging
logging.basicConfig(filename='info.log', level=logging.DEBUG)


@dataclass
class MessageID:
    mode: Mode
    i: int


def enum_from_str(cls, value: str | None) -> Mode | None:
    if not value:
        return None
    key = value.upper().replace(" ", "_")
    try:
        return cls[key]
    except ValueError:
        return None


def filter_none_values(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}

def message_index(active_mode: Mode) -> int:
    return len(st.session_state.messages[active_mode.value])

def get_payload(params: dict) -> dict:
    return {
        "metadata": {
            "app_id": settings.app_id,
            "environment": settings.environment,
        },
        "input": filter_none_values(params)
    }

def get_search_results(
    query: str,
    score_threshold: Optional[float] = None,
    num_sources: Optional[int] = None,
):
    payload = get_payload({
            "query": query,
            "score_threshold": score_threshold,
            "k": num_sources,
        }
    )
    
    response = requests.post(f"{settings.base_api_url}{settings.search_endpoint}", json=payload)
    return response.json()["output"]


def get_stream(
    query: str,
    score_threshold: Optional[float] = None,
    num_sources: Optional[int] = None,
):
    payload = get_payload({
            "query": query,
            "score_threshold": score_threshold,
            "k": num_sources,
        }
    )

    response = requests.post(f"{settings.base_api_url}{settings.query_endpoint}", json=payload, stream=True)
    client = sseclient.SSEClient(response)
    for event in client.events():
        output = json.loads(event.data)
        if output["event"] == "on_llm_stream":
            if chunk := output["data"].get("chunk"):
                yield chunk
        elif output["event"] == "on_chain_end":
            if source_documents := output["data"]["output"].get("source_documents"):
                st.session_state.sources = source_documents
                st.session_state.run_id = output["run_id"]

def send_feedback(run_id: str, question: str, sentiment: bool, comment: Optional[str] = None):
    payload = filter_none_values({
        "question": question,
        "run_id": run_id,
        "score": sentiment,
        "feedback_id": str(uuid.uuid4()),
        "comment": comment,
    })
    response = requests.post(f"{settings.base_api_url}{settings.feedback_endpoint}", json=payload)
    print(response.json())
    return response.json()

def display_documents(documents: List[dict], mode: Mode):
    if mode == Mode.CHATBOT:
        if documents:
            with st.expander("Result sources"):
                tabs = st.tabs([f"Source {i}" for i in range(1, len(documents) + 1)])
                for i, source in enumerate(documents):
                    with tabs[i]:
                        st.caption(f'From `{source["metadata"]["source"]}`')
                        st.markdown(source["page_content"])
    else:
        for document in documents:
            with st.expander(
                f'{document["score"]}: `{document["metadata"]["source"]}`'
            ):
                st.markdown(document["page_content"])

def display_feedback_interface(message_id: MessageID, run_id: str, question: str):
    def send_and_update_feedback(sentiment: bool):
        send_feedback(run_id, question, sentiment)
        st.session_state.messages[message_id.mode.value][message_id.i]["feedback"] = sentiment

    def feedback_button_params(sentiment: bool):
        feedback = st.session_state.messages[message_id.mode.value][message_id.i].get("feedback")
        return {
            "key": f"{message_id.mode.value}_{message_id.i}_{sentiment}",
            "disabled": feedback is not None,
            "type": "primary" if feedback == sentiment else "secondary",
            "on_click": lambda: send_and_update_feedback(sentiment),
        }

    col1, col2, _ = st.columns([.075, .075, .85])
    col1.button(":thumbsup:", **feedback_button_params(True))
    col2.button(":thumbsdown:", **feedback_button_params(False))

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = {"query": [], "search": []}

# Initialize sources
if "sources" not in st.session_state:
    st.session_state.sources = []

if "mode" not in st.session_state:
    st.session_state.mode = settings.default_mode.value

if "run_id" not in st.session_state:
    st.session_state.run_id = ""


st.title("AI Documentation Explorer")

# User Configuration Sidebar
num_sources = None
score_threshold = None
with st.sidebar:
    st.subheader("The Power of NRI Docs in the Palm of Your Hand")
    st.header("Settings")
    mode = st.radio("Search Mode", options=["Chatbot", "Doc Search"])
    active_mode = enum_from_str(Mode, mode) or settings.default_mode

    sources_filtering = st.radio(
        "Sources Filtering", options=["Num Docs", "Similarity Threshold"]
    )

    if sources_filtering == "Num Docs":
        num_sources = st.slider(
            label="Number of documents to reference",
            min_value=settings.min_num_sources,
            max_value=settings.max_num_sources,
            value=settings.default_num_sources,
        )
    elif sources_filtering == "Similarity Threshold":
        score_threshold = st.slider(
            label="Similarity matching threshold",
            min_value=settings.min_score_threshold,
            max_value=settings.max_score_threshold,
            value=settings.default_score_threshold,
            step=settings.score_threshold_step,
        )

st.divider()

# Display chat messages from history on app rerun
for message in st.session_state.messages[active_mode.value]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        display_documents(message["sources"], active_mode)
        if message["role"] == "assistant":
            display_feedback_interface(message["id"], message["run_id"], message["question"])


message_id = MessageID(active_mode, message_index(active_mode))
if active_mode == Mode.CHATBOT:
    # React to user input
    if prompt := st.chat_input("What would you like help with?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages[active_mode.value].append(
            {
                "id": message_id,
                "role": "user", 
                "content": prompt, 
                "sources": [],
                "feedback": None,
            }
        )

        with st.chat_message("assistant"):
            full_response = st.write_stream(
                get_stream(prompt, score_threshold, num_sources)
            )
            display_documents(st.session_state.sources, active_mode)
            display_feedback_interface(message_id, st.session_state.run_id, prompt)

        # Add assistant response to chat history
        st.session_state.messages[active_mode.value].append(
            {
                "id": message_id,
                "role": "assistant",
                "question": prompt,
                "content": full_response,
                "sources": st.session_state.sources[:],
                "run_id": st.session_state.run_id,
                "feedback": None,
            }
        )
else:
    if prompt := st.chat_input("Enter search query"):
        st.session_state.sources = []
        st.chat_message("user").markdown(prompt)
        st.session_state.messages[active_mode.value].append(
            {"role": "user", "content": prompt, "sources": []}
        )

        with st.chat_message("assistant"):
            results = get_search_results(prompt, score_threshold, num_sources)
            response = f"{len(results)} Results found:"
            st.write(response)
            display_documents(results, active_mode)
            display_feedback_interface(message_id, st.session_state.run_id, prompt)

        # Add assistant response to chat history
        st.session_state.messages[active_mode.value].append(
            {
                "id": message_id,
                "role": "assistant",
                "question": prompt,
                "content": response,
                "sources": results,
                "run_id": st.session_state.run_id,
                "feedback": None,
            }
        )
