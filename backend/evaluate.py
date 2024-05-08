import datetime
import os
from typing import Optional

from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator, evaluate

from __version__ import __version__
from retriever import answer_chain, llm

client = Client()
MAXIMUM_CONCURRENCY = 5


def upload_dataset_from_csv(name: str, file_path: Optional[str] = None):
    file_path = file_path or "datasets"
    client.upload_csv(
        csv_file=os.path.join(file_path, f"{name}.csv"),
        input_keys=["query"],
        output_keys=["answer"],
        name=name,
    )


def get_context_from_documents(documents):
    document_separator = "\n\n"
    return document_separator.join(
        f"Document {i}:\n{doc.page_content}" for i, doc in enumerate(documents)
    )


def evaluate_pipeline(dataset_name):

    def prepare_data(run, example):
        return {
            "prediction": run.outputs["result"],
            "reference": example.outputs["answer"],
            "input": example.inputs["query"],
        }

    def context_prepare_data(run, example):
        return {
            "context": get_context_from_documents(run.outputs["source_documents"]),
            **prepare_data(run, example),
        }

    qa_evaluator = LangChainStringEvaluator(
        "qa", config={"llm": llm}, prepare_data=prepare_data
    )
    context_qa_evaluator = LangChainStringEvaluator(
        "context_qa", config={"llm": llm}, prepare_data=context_prepare_data
    )
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    evaluate(
        answer_chain.invoke,
        data=dataset_name,
        evaluators=[qa_evaluator, context_qa_evaluator],
        experiment_prefix=f"{__version__}/{timestamp}",
        metadata={"revision_id": __version__},
        max_concurrency=MAXIMUM_CONCURRENCY,
    )


if __name__ == "__main__":
    evaluate_pipeline(dataset_name="fire_docs_rag_qa_small")
