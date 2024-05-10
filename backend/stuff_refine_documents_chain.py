"""Adds summary refinement to document combination chain"""
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import BasePromptTemplate, format_document
from langchain_core.pydantic_v1 import Extra, Field, root_validator
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.chains.summarize import load_summarize_chain

from langchain.chains.combine_documents.base import (
    DEFAULT_DOCUMENT_PROMPT,
    DEFAULT_DOCUMENT_SEPARATOR,
    DOCUMENTS_KEY,
    BaseCombineDocumentsChain,
    _validate_prompt,
)
from langchain.chains.llm import LLMChain


from langchain_core.prompts import PromptTemplate

# TODO: These should include document source as well as the text
REFINE_PROMPT_TMPL = """\
Your job is to produce a final summary.
Focus particularly on content that is relevant to the question "{query}".
Do not provide any preamble or mention that it is the summary. Simply provide the summary itself.
We have provided an existing summary up to a certain point: {existing_answer}
We have the opportunity to refine the existing summary (only if needed) with some more context below.
------------
{text}
------------
Given the new context, refine the original summary.
If the context isn't useful, return the original summary.\
"""  # noqa: E501
REFINE_PROMPT = PromptTemplate.from_template(REFINE_PROMPT_TMPL)


prompt_template = """Write a concise summary of the text below, delimited by backticks (```).
Do not provide any preamble or mention that it is the summary. Simply provide the summary itself.
Focus particularly on content that is relevant to the question "{query}".

Text:
```
"{text}"
```

CONCISE SUMMARY:"""
QUESTION_PROMPT = PromptTemplate.from_template(prompt_template)



def create_stuff_refine_documents_chain(
    llm: LanguageModelLike,
    prompt: BasePromptTemplate,
    query: str,
    *,
    output_parser: Optional[BaseOutputParser] = None,
    document_prompt: Optional[BasePromptTemplate] = None,
    document_separator: str = DEFAULT_DOCUMENT_SEPARATOR,
) -> Runnable[Dict[str, Any], Any]:
    """Create a chain for passing a list of Documents to a model.
    Builds off the `create_stuff_documents_chain` found in `langchain.chains.combine_documents`
    to add LLM summarization to document combination, producing a shorter and more relevant
    context for the final LLM to use to answer the question.

    Args:
        llm: Language model.
        prompt: Prompt template. Must contain input variable "context", which will be
            used for passing in the formatted documents.
        output_parser: Output parser. Defaults to StrOutputParser.
        document_prompt: Prompt used for formatting each document into a string. Input
            variables can be "page_content" or any metadata keys that are in all
            documents. "page_content" will automatically retrieve the
            `Document.page_content`, and all other inputs variables will be
            automatically retrieved from the `Document.metadata` dictionary. Default to
            a prompt that only contains `Document.page_content`.
        document_separator: String separator to use between formatted document strings.

    Returns:
        An LCEL Runnable. The input is a dictionary that must have a "context" key that
        maps to a List[Document], and any other input variables expected in the prompt.
        The Runnable return type depends on output_parser used.

    Example:
        .. code-block:: python

            # pip install -U langchain langchain-community

            from langchain_community.chat_models import ChatOpenAI
            from langchain_core.documents import Document
            from langchain_core.prompts import ChatPromptTemplate
            from langchain.chains.combine_documents import create_stuff_documents_chain

            prompt = ChatPromptTemplate.from_messages(
                [("system", "What are everyone's favorite colors:\\n\\n{context}")]
            )
            llm = ChatOpenAI(model="gpt-3.5-turbo")
            chain = create_stuff_documents_chain(llm, prompt)

            docs = [
                Document(page_content="Jesse loves red but not yellow"),
                Document(page_content = "Jamal loves green but not as much as he loves orange")
            ]

            chain.invoke({"context": docs})
    """  # noqa: E501

    _validate_prompt(prompt)
    _document_prompt = document_prompt or DEFAULT_DOCUMENT_PROMPT
    _output_parser = output_parser or StrOutputParser()

    def format_docs(inputs: dict) -> str:
        chain = load_summarize_chain(
            llm, 
            chain_type="refine", 
            verbose=False,
            question_prompt=QUESTION_PROMPT.partial(query=query),
            refine_prompt=REFINE_PROMPT.partial(query=query),

        )
        return chain.invoke({"input_documents": inputs[DOCUMENTS_KEY]}).get("output_text", "")
        # return document_separator.join(
        #     format_document(doc, _document_prompt) for doc in inputs[DOCUMENTS_KEY]
        # )

    return (
        RunnablePassthrough.assign(**{DOCUMENTS_KEY: format_docs}).with_config(
            run_name="format_inputs"
        )
        | prompt
        | llm
        | _output_parser
    ).with_config(run_name="stuff_documents_chain")