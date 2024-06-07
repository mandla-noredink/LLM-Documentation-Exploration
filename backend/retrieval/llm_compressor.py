from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.runnables import run_in_executor
from langchain_core.language_models.llms import BaseLLM


class LLMCompressor(BaseDocumentCompressor):
    """LMM document compressors."""

    llm: None | BaseLLM

    def __init__(self, llm=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = llm

    # def __init__(self, llm: BaseLLM):
    #     super().__init__()
    #     self.llm = llm

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""
        from langchain_core.prompts import ChatPromptTemplate

        rerank_system_prompt = '''You are an Assistant responsible for helping detect whether the retrieved context is relevant to the engineering question. For a given input, you need to output a single token: "Yes" or "No" indicating the retrieved content is relevant to the engineering question. Examples are below:

        Engineering Question: What caused the merge fire?
        Context: """Jenkins was unable to coordinate merges and was throwing out memory errors. This caused multiple days of merge queue outages."""
        Relevant: Yes

        Engineering Question: What caused the merge fire?
        Context: """Question ID 15232 was reported as taking more than 60 seconds to submit. I'm making a pull request to stop showing new topics."""
        Relevant: No

        Engineering Question: What can we do if the database maxes out CPU resources?
        Context: """The quiz engine is the largest consumer of database resources, so reducing its resources is the main way to reduce database load."""
        Relevant: Yes

        Engineering Question: What can we do if the database maxes out CPU resources?
        Context: """error: var "firecrackers" is not defined. Question ID 23342352354 unable to perform celebration animation. root/directory/jobs/haskell_in_ruby."""
        Relevant: No'''

        rerank_user_prompt = '''Engineering Question: {question}
        Context: """{context}"""
        Relevant:
        '''

        rerank_prompt = ChatPromptTemplate.from_messages([
            ("system", rerank_system_prompt),
            ("user", rerank_user_prompt)
        ])

        final_contexts = []
        if self.llm:
            for context in documents:
                reply = self.llm.invoke(rerank_prompt.format(question=query, context=context))
                print(f'=== REPLY: {reply} ===')
                if reply == 'Yes':
                    final_contexts.append(context)
        return final_contexts

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context."""
        return await run_in_executor(
            None, self.compress_documents, documents, query, callbacks
        )
