# LLM Documentation Explorer

## Running Locally

- install ollama
  - https://github.com/ollama/ollama?tab=readme-ov-file
  - `brew install ollama`
  - `brew services start ollama`
- download llama3
  - `ollama pull llama3`
- run ollama (with concurrency)
  - `OLLAMA_NUM_PARALLEL=2 OLLAMA_MAX_LOADED_MODELS=2 ollama serve`
- download repo
- download dropbox folder
  - Go to Engineering folder in dropbox paper
  - Select Fires folder
  - Select Export from bottom menu
  - Select export as markdown
  - Unzip folder once downloaded
  - Move to `./backend/.content`
    - Should look like `./backend/.content/Engineering/Fires/...`
- install poetry
- ingest documents
  - `poetry run python backend/ingest.py`
- run backend server
  - `cd backend`
  - `poetry run uvicorn main:app --reload`
- run frontend server
  - `cd frontend`
  - `poetry run streamlit run main.py`

## Resources:
- https://www.youtube.com/playlist?list=PLrSHiQgy4VjGQohoAmgX9VFH52psNOu71
- https://github.com/AI-Maker-Space/LLM-Ops-Cohort-1/tree/main


Streaming:
- https://discuss.streamlit.io/t/listening-for-updates-from-an-api-server/48486


## Concurrency

(Ollama concurrency support)[https://github.com/ollama/ollama/issues/358#issuecomment-2082599253]
(Lanchain LLM async integration table)[https://python.langchain.com/docs/integrations/llms/]
(Langchain Ollama LLM on remote URL)[https://medium.com/@andrewnguonly/local-llm-in-the-browser-powered-by-ollama-236817f335da]