from langchain_community.llms import Ollama
from settings import settings

llm = Ollama(model=settings.ollama_llm, temperature=0)
