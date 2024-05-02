import streamlit as st
import requests
from enum import Enum
from typing import Optional, List
import json
import sseclient


class Domain(Enum):
    FIRES = "fires"


class Mode(Enum):
    CHATBOT = "query"
    DOC_SEARCH = "search"


DEFAULT_MODE = Mode.CHATBOT


def enum_from_str(cls, value: str | None):
    if not value:
        return None
    key = value.upper().replace(" ", "_")
    try:
        return cls(key)
    except ValueError:
        return None


def filter_none_values(d: dict):
    return {k: v for k, v in d.items() if v is not None}


def get_stream(
    query: str,
    mode: Mode,
    score_threshold: Optional[float] = None,
    num_sources: Optional[int] = None,
):
    url = f"http://127.0.0.1:8000/{mode.value}/stream_events/"
    payload = {
        "input": filter_none_values(
            {
                "query": query,
                "score_threshold": score_threshold,
                "k": num_sources,
            }
        )
    }
    response = requests.post(url, json=payload, stream=True)
    client = sseclient.SSEClient(response)
    for event in client.events():
        output = json.loads(event.data)
        if output["event"] == "on_llm_stream":
            if chunk := output["data"].get("chunk"):
                yield chunk
        elif output["event"] == "on_chain_end":
            if source_documents := output["data"]["output"].get("source_documents"):
                st.session_state.sources = source_documents


def display_sources(sources: List[dict]):
    if sources:
        with st.expander("Result sources"):
            tabs = st.tabs([f"Source {i}" for i in range(1, len(sources) + 1)])
            for i, source in enumerate(sources):
                with tabs[i]:
                    st.caption(f'From `{source["metadata"]["source"]}`')
                    st.markdown(source["page_content"])


st.title("AI Documentation Explorer")

# User Configuration Sidebar
num_sources = None
score_threshold = None
with st.sidebar:
    st.subheader("The Power of NRI Docs in the Palm of Your Hand")
    st.header("Settings")
    domain = st.radio("Topic Domain", options=["Fires"])
    mode = st.radio("Search Mode", options=["Chatbot", "Doc Search"])
    sources_filtering = st.radio(
        "Sources Filtering", options=["Num Docs", "Similarity Threshold"]
    )

    if sources_filtering == "Num Docs":
        num_sources = st.slider(
            label="Number of documents to reference",
            min_value=2,
            max_value=8,
            value=5,
        )
    elif sources_filtering == "Similarity Threshold":
        score_threshold = st.slider(
            label="Similarity matching threshold",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
        )

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize sources
if "sources" not in st.session_state:
    st.session_state.sources = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        display_sources(message["sources"])

active_mode = enum_from_str(Mode, mode) or DEFAULT_MODE

# React to user input
if prompt := st.chat_input("What would you like help with?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    with st.chat_message("assistant"):
        full_response = st.write_stream(
            get_stream(prompt, active_mode, score_threshold, num_sources)
        )
        display_sources(st.session_state.sources)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": st.session_state.sources[:],
        }
    )
