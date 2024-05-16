# LLM Documentation Explorer

## Overview

The LLM Documentation Explorer is a sophisticated application designed to browse and interact with NoRedInk's internal documentation. It leverages a locally hosted open-source Large Language Model (LLM) with Retrieval Augmented Generation (RAG) capabilities, ensuring efficient and intuitive access to extensive company knowledge. This tool integrates a core RAG pipeline, an API layer, and a user-friendly frontend interface.

## Technologies
- **[Langchain](https://python.langchain.com/v0.1/docs/get_started/introduction):** Manages the RAG pipeline.
- **[Langserve](https://python.langchain.com/v0.1/docs/langserve/):** API management built on [FastAPI](https://fastapi.tiangolo.com/).
- **[Langsmith](https://docs.smith.langchain.com/):** Used for monitoring and evaluation.
- **[Streamlit](https://streamlit.io):** Powers the frontend interface.
- **[Ollama](https://ollama.com/):** Hosts and runs the LLM.
- **PostgreSQL with [PGVector](https://github.com/pgvector/pgvector):** Manages the remote vector database.

## Architecture

![Lucidchart Blank Diagram](https://github.com/mandla-noredink/LLM-Documentation-Exploration/assets/99652839/36bc8f27-899f-41a7-bcbd-e45398edd393)

### Core RAG pipeline

The core RAG implementation allows for either local or remote operation. Local operation saves a FAISS vector store to a local file, running it in memory. Remote operation saves the vector database to a PostgreSQL database using the PGVector extension.

The implementation uses the Llama 3 8B model, run locally using Ollama.

Multiple optimization have been implemented to maximise performance:
- A parent document retrieval system is used, separating the data used for vector db retrieval (the child documents) from that sent to the LLM (the parent documents) into separate copies of the knowledge base
- The child document data embedded and stored in the vector db is preprocessed to optimize vector search, using the following steps:
  - Noise removal: Unhelpful symbols and whitespace removed
  - Case normalization: all text converted to lowercase
  - Lemmatization: words stemmed for better matching
- Retrieved documents are reranked to find the most appropriate matches
- Final documents are summarised to provide only useful and contextually relevant information to the LLM

### API

The API serves the following endpoints:

- **/query/*:** Processes queries using the RAG to generate responses.
- **/search/*:** Searches the vector database without LLM intervention.
- **/feedback:** Collects user feedback on query responses.
- **/ingest:** Initiates re-ingestion of knowledge base data.
- **/upload:** Manages the uploading and ingestion of new datasets.

_*Langserve automatically implements `invoke`, `batch`, `stream`, `stream_log`, and `stream_events` endpoint expansions for the `query` and `search` base endpoints (as well as their async versions)._

_NOTE: the `/feedback`, `/ingest`, and `/upload` endpoints have schemas that can be viewed and interacted with in the application playground at `/docs` url._

### Frontend

The Streamlit frontend provides a straightforward interface for querying the knowledge base.

It also allows for the latest version of the knowledge base documents to be uploaded directly.

## Running Locally

1. Install, setup, and run Ollama and Llama3
```sh
# https://github.com/ollama/ollama?tab=readme-ov-file
brew install ollama
brew services start ollama
ollama pull llama3
OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=2 ollama serve  # Run with concurrency
```

2. Download repo (if not in monorepo)

3. Install poetry and project dependencies
```sh
pip install poetry
poetry install
```

4. Setup `.env` file
```sh
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=llm-docs-explorer
LANGCHAIN_API_KEY=<key>

# Local values for these can be found by starting postgres: 
# $ aide setup-postgres
PG_HOST=127.0.0.1
PG_PORT=<port>
PG_USER=<user>
PG_PASSWORD=<password>
PG_DBNAME=llm_docs_explorer
```

5. Ensure PGVector is installed locally
- Apply the changes found in Micah's PR
- Run `CREATE EXTENSION IF NOT EXISTS vector;` in postgres to ensure verify


6. Run server and client
```sh
# Backend
cd backend
poetry run uvicorn main:app --reload

# Frontend
cd frontend
poetry run streamlit run main.py
```

7. Use the frontend to upload documents for the knowledge base
   - Go to `Engineering` folder in Dropbox Paper
   - Select `Fires` folder
   - Select "Export" from bottom menu
   - Select "Export as Markdown"
   - upload the `.zip` file using the Frontend


# Resources and references


## General RAG Resources:
- https://www.youtube.com/playlist?list=PLrSHiQgy4VjGQohoAmgX9VFH52psNOu71
- https://github.com/AI-Maker-Space/LLM-Ops-Cohort-1/tree/main


## Streaming Server Output:
- https://discuss.streamlit.io/t/listening-for-updates-from-an-api-server/48486


## Concurrency
- [Ollama concurrency support](https://github.com/ollama/ollama/issues/358#issuecomment-2082599253)
- [Lanchain LLM async integration table](https://python.langchain.com/docs/integrations/llms/)
- [Langchain Ollama LLM on remote URL](https://medium.com/@andrewnguonly/local-llm-in-the-browser-powered-by-ollama-236817f335da)


## RAG Performance Improvement Resources
- https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/parent_document_retriever/
- https://python.langchain.com/v0.1/docs/integrations/retrievers/flashrank-reranker/
- https://python.langchain.com/v0.1/docs/modules/model_io/prompts/partial/
- https://medium.com/@vinusebastianthomas/document-chains-in-langchain-d33c4bdbabd8
- https://python.langchain.com/v0.1/docs/integrations/retrievers/re_phrase/
- https://stepup.ai/rag_improving_retrieval/
