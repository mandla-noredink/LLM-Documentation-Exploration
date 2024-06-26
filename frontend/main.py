import json
import uuid
from dataclasses import dataclass
from functools import partial
from typing import List, Optional

import requests
import sseclient
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from settings import Mode, settings


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
        "config": {
            "metadata": {
                "app_id": settings.app_id,
                "environment": settings.environment,
            }
        },
        "input": filter_none_values(params),
    }


def get_search_results(
    query: str,
    num_sources: Optional[int] = None,
):
    payload = get_payload(
        {
            "question": query,
            "k": num_sources,
        }
    )

    response = requests.post(
        f"{settings.base_api_url}{settings.search_endpoint}", json=payload
    )
    return response.json()["output"]


def get_stream(
    query: str,
    optimize: Optional[bool] = False,
    num_sources: Optional[int] = None,
):
    payload = get_payload(
        {
            "question": query,
            "optimize": optimize,
            "k": num_sources,
        }
    )

    response = requests.post(
        f"{settings.base_api_url}{settings.query_endpoint}", 
        json=payload, 
        stream=True,
    )
    client = sseclient.SSEClient(response)
    for event in client.events():
        output = json.loads(event.data)
        if output["event"] == "on_llm_stream" and "seq:step:3" in output["tags"]:
            if chunk := output["data"].get("chunk"):
                yield chunk
        elif output["event"] == "on_chain_end":
            if results := output["data"].get("output"):
                if isinstance(results, dict):
                    if source_documents := results.get("context"):
                        st.session_state.sources = source_documents
                        st.session_state.run_id = output["run_id"]


def send_feedback(
    run_id: str, question: str, sentiment: bool, comment: Optional[str] = None
):
    payload = filter_none_values(
        {
            "question": question,
            "run_id": run_id,
            "score": sentiment,
            "feedback_id": str(uuid.uuid4()),
            "comment": comment,
        }
    )
    response = requests.post(
        f"{settings.base_api_url}{settings.feedback_endpoint}", json=payload
    )
    print(response.json())
    return response.json()

def upload_content(content):
    files = {'file': content}
    response = requests.post(
        f"{settings.base_api_url}{settings.upload_endpoint}",
        files=files,
    )
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
                f'{document["metadata"]["relevance_score"]}: `{document["metadata"]["source"]}`'
            ):
                st.markdown(document["page_content"])


def display_feedback_interface(message_id: MessageID, run_id: str, question: str):
    def send_and_update_feedback(sentiment: bool):
        send_feedback(run_id, question, sentiment)
        st.session_state.messages[message_id.mode.value][message_id.i][
            "feedback"
        ] = sentiment

    # If feedback is not yet provided, display both thumbs up and thumbs down buttons
    # otherwise, display only the selected feedback as a disabled button
    options = [(True, "thumbsup"), (False, "thumbsdown")]
    messages = st.session_state.messages[message_id.mode.value]
    feedback = (
        messages[message_id.i].get("feedback") if message_id.i < len(messages) else None
    )
    active_options = [(k, v) for k, v in options if feedback is None or feedback == k]
    cols = st.columns(
        [0.075 for _ in active_options] + [1 - 0.075 * len(active_options)]
    )
    for i, (sentiment, icon) in enumerate(active_options):
        cols[i].button(
            f":{icon}:",
            key=f"{message_id.mode.value}_{message_id.i}_{sentiment}",
            disabled=feedback is not None,
            on_click=partial(send_and_update_feedback, sentiment),
        )


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
optimize = False
num_sources = None
with st.sidebar:
    st.subheader("The Power of NRI Docs in the Palm of Your Hand")
    st.header("Settings")
    mode = st.radio("Search Mode", options=["Chatbot", "Doc Search"])
    active_mode = enum_from_str(Mode, mode) or settings.default_mode

    if mode == "Chatbot":
        optimize = st.toggle("LLM Query Optimization")
        st.caption("Query optimization summarizes the source documents that get sent to the LLM. This generally improves performance by stripping out irrelevant information, but requires multiple LLM calls and thus takes more time. ")
    else:
        num_sources = st.slider(
            label="Number of documents to reference",
            min_value=settings.min_num_sources,
            max_value=settings.max_num_sources,
            value=settings.default_num_sources,
        )

    with st.expander("Documentation Upload"):
        st.caption("To keep the knowledge base up to date, a zip file containing the latest versions of the documents to be used can be uploaded and ingested through this interface.")
        uploaded_file = st.file_uploader("Choose a file (page refreshes after upload)")
        if uploaded_file is not None:
            with st.spinner('Uploading file...'):
                upload_content(uploaded_file)
            st.success("File successfully uploaded!")
            streamlit_js_eval(js_expressions="parent.window.location.reload()")

st.divider()

# Display chat messages from history on app rerun
for message in st.session_state.messages[active_mode.value]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        display_documents(message["sources"], active_mode)
        if message["role"] == "assistant":
            display_feedback_interface(
                message["id"], message["run_id"], message["question"]
            )


if active_mode == Mode.CHATBOT:
    # React to user input
    if prompt := st.chat_input("What would you like help with?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        user_message_id = MessageID(active_mode, message_index(active_mode))
        st.session_state.messages[active_mode.value].append(
            {
                "id": user_message_id,
                "role": "user",
                "content": prompt,
                "sources": [],
            }
        )

        assistant_message_id = MessageID(active_mode, message_index(active_mode))
        with st.chat_message("assistant"):
            full_response = st.write_stream(
                get_stream(prompt, optimize, num_sources)
            )
            display_documents(st.session_state.sources, active_mode)
            display_feedback_interface(
                assistant_message_id, st.session_state.run_id, prompt
            )

        # Add assistant response to chat history
        st.session_state.messages[active_mode.value].append(
            {
                "id": assistant_message_id,
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
        user_message_id = MessageID(active_mode, message_index(active_mode))
        st.session_state.messages[active_mode.value].append(
            {
                "id": user_message_id,
                "role": "user",
                "content": prompt,
                "sources": [],
            }
        )

        assistant_message_id = MessageID(active_mode, message_index(active_mode))
        with st.chat_message("assistant"):
            results = get_search_results(prompt, num_sources)
            response = f"{len(results)} Results found:"
            st.write(response)
            display_documents(results, active_mode)
            display_feedback_interface(
                assistant_message_id, st.session_state.run_id, prompt
            )

        # Add assistant response to chat history
        st.session_state.messages[active_mode.value].append(
            {
                "id": assistant_message_id,
                "role": "assistant",
                "question": prompt,
                "content": response,
                "sources": results,
                "run_id": st.session_state.run_id,
                "feedback": None,
            }
        )
