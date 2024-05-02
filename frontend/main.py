import streamlit as st
import requests
from enum import Enum
from typing import Optional
import json
import sseclient


# TODO: Use this to adjust backend settings
API_URL = "http://127.0.0.1:8000/{endpoint}/{mode}/"


class Domain(Enum):
    FIRES = "fires"


class Endpoint(Enum):
    CHATBOT = "query"
    DOC_SEARCH = "search"


class Mode(Enum):
    INVOKE = "invoke"
    STREAM = "stream_events"


def enum_from_str(cls, value: str | None):
    if not value:
        return None
    key = value.upper().replace(" ", "_")
    try:
        return cls(key)
    except ValueError:
        return None


def call_api(
    query: str,
    documents: int,  # TODO: Implement this
    endpoint: Optional[Endpoint] = None,
    domain: Optional[Domain] = None,
    mode: Optional[Mode] = None,
):
    endpoint = endpoint or Endpoint.CHATBOT
    domain = domain or Domain.FIRES
    mode = mode or Mode.STREAM

    return requests.post(
        API_URL.format(endpoint=endpoint.value, mode=mode.value),
        json={"input": {"query": query}},
        stream=True,
    )


def get_stream(app, query):
    url = "http://127.0.0.1:8000/query/stream_events/"
    response = requests.post(url, json={"input": {"query": query}}, stream=True)
    client = sseclient.SSEClient(response)
    for event in client.events():
        output = json.loads(event.data)
        if output["event"] == "on_llm_stream":
            if chunk := output["data"].get("chunk"):
                yield chunk
        elif output["event"] == "on_chain_end":
            if source_documents := output["data"]["output"].get("source_documents"):
                app.session_state.sources = source_documents


def display_sources(app, sources):
    if sources:
        with app.expander("Result sources"):
            tabs = app.tabs([f"Source {i}" for i in range(1, len(sources) + 1)])
            for i, source in enumerate(sources):
                with tabs[i]:
                    app.caption(f'From `{source["metadata"]["source"]}`')
                    app.markdown(source["page_content"])


st.title("AI Documentation Explorer")

# User Configuration Sidebar
with st.sidebar:
    st.subheader("The Power of NRI Docs in the Palm of Your Hand")
    st.header("Settings")
    domain = st.radio("Topic Domain", options=["Fires"])
    endpoint = st.radio("Search Mode", options=["Chatbot", "Doc Search"])
    documents = st.slider(
        label="Number of documents to reference",
        min_value=2,
        max_value=6,
        value=4,
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
        display_sources(st, message["sources"])

# React to user input
if prompt := st.chat_input("What would you like help with?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})

    with st.chat_message("assistant"):
        full_response = st.write_stream(get_stream(st, prompt))
        display_sources(st, st.session_state.sources)

    # Add assistant response to chat history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "sources": st.session_state.sources[:],
        }
    )
