import uuid
from typing import Any, Coroutine, List, Optional, Sequence

from ingestion.preprocess import pre_embedding_process, preprocess_document
from langchain.retrievers import MultiVectorRetriever
from langchain_core.callbacks import (AsyncCallbackManagerForRetrieverRun,
                                      CallbackManagerForRetrieverRun)
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter


class ParentDocumentPreprocessRetriever(MultiVectorRetriever):
    """Updates `langchain.retrievers.ParentDocumentRetriever` to include a custom 
    document preprocessing pipeline.
    """

    child_splitter: TextSplitter
    """The text splitter to use to create child documents."""

    """The key to use to track the parent id. This will be stored in the
    metadata of child documents."""
    parent_splitter: Optional[TextSplitter] = None
    """The text splitter to use to create parent documents.
    If none, then the parent documents will be the raw documents passed in."""

    child_metadata_fields: Optional[Sequence[str]] = None
    """Metadata fields to leave in child documents. If None, leave all parent document 
        metadata.
    """

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        add_to_docstore: bool = True,
    ) -> None:
        """Adds documents to the docstore and vectorstores.

        Args:
            documents: List of documents to add
            ids: Optional list of ids for documents. If provided should be the same
                length as the list of documents. Can be provided if parent documents
                are already in the document store and you don't want to re-add
                to the docstore. If not provided, random UUIDs will be used as
                ids.
            add_to_docstore: Boolean of whether to add documents to docstore.
                This can be false if and only if `ids` are provided. You may want
                to set this to False if the documents are already in the docstore
                and you don't want to re-add them.
        """
        if self.parent_splitter is not None:
            documents = self.parent_splitter.split_documents(documents)
        if ids is None:
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            if not add_to_docstore:
                raise ValueError(
                    "If ids are not passed in, `add_to_docstore` MUST be True"
                )
        else:
            if len(documents) != len(ids):
                raise ValueError(
                    "Got uneven list of documents and ids. "
                    "If `ids` is provided, should be same length as `documents`."
                )
            doc_ids = ids

        docs = []
        full_docs = []
        for i, doc in enumerate(documents):
            _id = doc_ids[i]
            sub_docs = self.child_splitter.split_documents([doc])
            sub_docs = [preprocess_document(doc) for doc in sub_docs]
            if self.child_metadata_fields is not None:
                for _doc in sub_docs:
                    _doc.metadata = {
                        k: _doc.metadata[k] for k in self.child_metadata_fields
                    }
            for _doc in sub_docs:
                _doc.metadata[self.id_key] = _id
            docs.extend(sub_docs)
            full_docs.append((_id, doc))
        self.vectorstore.add_documents(docs)
        if add_to_docstore:
            self.docstore.mset(full_docs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        preprocessed_query = pre_embedding_process(query)
        # print(f"Processed query: {preprocessed_query}")
        return super(ParentDocumentPreprocessRetriever, self)._get_relevant_documents(preprocessed_query, run_manager=run_manager)

    def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> Coroutine[Any, Any, List[Document]]:
        preprocessed_query = pre_embedding_process(query)
        # print(f"Processed query: {preprocessed_query}")
        return super()._aget_relevant_documents(preprocessed_query, run_manager=run_manager)
