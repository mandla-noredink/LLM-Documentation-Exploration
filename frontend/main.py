import streamlit as st
import requests
from enum import Enum
from typing import Optional, List
import json
import sseclient


class Mode(Enum):
    CHATBOT = "query"
    DOC_SEARCH = "search"


BASE_API_URL = "http://127.0.0.1:8000"
DEFAULT_MODE = Mode.CHATBOT

MIN_NUM_SOURCES = 2
MAX_NUM_SOURCES = 8
DEFAULT_NUM_SOURCES = 5

MAX_SCORE_THRESHOLD = 2.0
MIN_SCORE_THRESHOLD = 0.0
DEFAULT_SCORE_THRESHOLD = 1.0
SCORE_THRESHOLD_STEP = 0.1


def enum_from_str(cls, value: str | None):
    if not value:
        return None
    key = value.upper().replace(" ", "_")
    try:
        return cls[key]
    except ValueError:
        return None


def filter_none_values(d: dict):
    return {k: v for k, v in d.items() if v is not None}


def get_search_results(
    query: str,
    score_threshold: Optional[float] = None,
    num_sources: Optional[int] = None,
):
    endpoint = "/search/invoke/"
    payload = {
        "input": filter_none_values(
            {
                "query": query,
                "score_threshold": score_threshold,
                "k": num_sources,
            }
        )
    }
    response = requests.post(f"{BASE_API_URL}{endpoint}", json=payload)
    return response.json()["output"]


def get_stream(
    query: str,
    score_threshold: Optional[float] = None,
    num_sources: Optional[int] = None,
):
    endpoint = "/query/stream_events/"
    payload = {
        "input": filter_none_values(
            {
                "query": query,
                "score_threshold": score_threshold,
                "k": num_sources,
            }
        )
    }
    response = requests.post(f"{BASE_API_URL}{endpoint}", json=payload, stream=True)
    client = sseclient.SSEClient(response)
    for event in client.events():
        output = json.loads(event.data)
        if output["event"] == "on_llm_stream":
            if chunk := output["data"].get("chunk"):
                yield chunk
        elif output["event"] == "on_chain_end":
            if source_documents := output["data"]["output"].get("source_documents"):
                st.session_state.sources = source_documents


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


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = {"query": [], "search": []}

# Initialize sources
if "sources" not in st.session_state:
    st.session_state.sources = []

if "mode" not in st.session_state:
    st.session_state.mode = DEFAULT_MODE.value

st.title("AI Documentation Explorer")

# User Configuration Sidebar
num_sources = None
score_threshold = None
with st.sidebar:
    st.subheader("The Power of NRI Docs in the Palm of Your Hand")
    st.header("Settings")
    mode = st.radio("Search Mode", options=["Chatbot", "Doc Search"])
    active_mode = enum_from_str(Mode, mode) or DEFAULT_MODE

    sources_filtering = st.radio(
        "Sources Filtering", options=["Num Docs", "Similarity Threshold"]
    )

    if sources_filtering == "Num Docs":
        num_sources = st.slider(
            label="Number of documents to reference",
            min_value=MIN_NUM_SOURCES,
            max_value=MAX_NUM_SOURCES,
            value=DEFAULT_NUM_SOURCES,
        )
    elif sources_filtering == "Similarity Threshold":
        score_threshold = st.slider(
            label="Similarity matching threshold",
            min_value=MIN_SCORE_THRESHOLD,
            max_value=MAX_SCORE_THRESHOLD,
            value=DEFAULT_SCORE_THRESHOLD,
            step=SCORE_THRESHOLD_STEP,
        )

st.divider()

# Display chat messages from history on app rerun
for message in st.session_state.messages[active_mode.value]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        display_documents(message["sources"], active_mode)


if active_mode == Mode.CHATBOT:
    # React to user input
    if prompt := st.chat_input("What would you like help with?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages[active_mode.value].append(
            {"role": "user", "content": prompt, "sources": []}
        )

        with st.chat_message("assistant"):
            full_response = st.write_stream(
                get_stream(prompt, score_threshold, num_sources)
            )
            display_documents(st.session_state.sources, active_mode)

        # Add assistant response to chat history
        st.session_state.messages[active_mode.value].append(
            {
                "role": "assistant",
                "content": full_response,
                "sources": st.session_state.sources[:],
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

        # Add assistant response to chat history
        st.session_state.messages[active_mode.value].append(
            {
                "role": "assistant",
                "content": response,
                "sources": results,
            }
        )
