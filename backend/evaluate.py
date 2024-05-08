import os
import datetime
from typing import Optional

from langsmith import Client
from langsmith.evaluation import LangChainStringEvaluator, evaluate
from langchain.evaluation import ContextQAEvalChain, QAEvalChain
from langchain_core.runnables import RunnablePassthrough
from langchain import smith
from langsmith.schemas import Example, Run

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

# def dummy_chain(inputs):
#     print(inputs)
#     print(answer_chain.invoke(inputs))
#     return {"query": "Test query", "result": "Test result", "source_documents": []}



def evaluate_pipeline(dataset_name):
    # Required keys: {query, answer, context, result}

    def prepare_data(run, example):
        return {
            # "query": example.inputs['query'],
            # "answer": example.outputs['answer'],
            # "result": run.outputs['result'],
            "prediction": run.outputs['result'],
            "reference": example.outputs['answer'],
            "input": example.inputs['query'],
        }
    
    def context_prepare_data(run, example):
        return {
            # "query": example.inputs['query'],
            # "answer": example.outputs['answer'],
            "context": get_context_from_documents(run.outputs['source_documents']),
            **prepare_data(run, example)
        }
    
    
    qa_evaluator = LangChainStringEvaluator("qa", config={"llm": llm}, prepare_data=prepare_data)
    context_qa_evaluator = LangChainStringEvaluator("context_qa", config={"llm": llm}, prepare_data=context_prepare_data)

    # criteria_evaluator = LangChainStringEvaluator(
    #     "criteria",
    #     config={
    #         "criteria": {
    #             "concision": "Is this response concise and to the point? Concise responses will avoid unnecessary filler, avoid repetition of the question, and avoid phrases that explicitly refer to the context like 'according to the context'.",
    #         },
    #         "normalize_by": 10,
    #         "llm": llm,
    #     },
    #     prepare_data=prepare_data,
    # )

    # eval_config = smith.RunEvalConfig(
    #     evaluators=[qa_evaluator, context_qa_evaluator],
    #     custom_evaluators=[]
    #     if args.skip_component_evaluator
    #     else [rubric_evaluator, component_evaluator],
    # )

    # context_chain = RunnablePassthrough.assign(
    #     context=lambda x: get_context_from_documents(x["source_documents"])
    # )
    
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
