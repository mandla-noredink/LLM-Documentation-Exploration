from typing import Optional, Union
from pydantic import BaseModel
from uuid import UUID
import os

from fastapi import File, UploadFile, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langsmith import Client

from utils import unzip, create_folder, clear_folder
from ingestion.ingest import ingest_documents
from retrieval.chains import answer_chain, search_chain
from settings import settings, get_logger
from local_settings import LANGCHAIN_API_KEY

class Query(BaseModel):
    question: str


class SendFeedbackBody(BaseModel):
    run_id: UUID
    key: str = "user_score"
    score: Union[float, int, bool]

    feedback_id: Optional[UUID] = None
    comment: Optional[str] = None

logger = get_logger(__name__)
client = Client(api_key=LANGCHAIN_API_KEY)
app = FastAPI()

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

add_routes(
    app,
    answer_chain,
    path="/query",
    config_keys=["metadata"],
)

add_routes(
    app,
    search_chain,
    path="/search",
    config_keys=["metadata"],
)


@app.post("/feedback")
async def send_feedback(body: SendFeedbackBody):
    logger.info("Uploading feedback")
    client.create_feedback(
        body.run_id,
        body.key,
        score=body.score,
        comment=body.comment,
        feedback_id=body.feedback_id,
    )
    return {"result": "posted feedback successfully", "code": 200}


@app.get("/ingest")
async def run_ingestion():
    logger.info("Ingesting documents")
    ingest_documents()
    logger.info("Document ingestion complete")
    return {"result": "content ingested successfully", "code": 200}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    create_folder(settings.temp_folder)
    create_folder(settings.download_folder)

    try:
        logger.info(f"Uploading file: {file.filename}")
        contents = file.file.read()
        assert file.filename
        file_path = os.path.join(settings.temp_folder, file.filename)
        logger.debug("Opening file to write")
        with open(file_path, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    try:
        logger.debug("Finished writing file")
        clear_folder(settings.download_folder)
        logger.debug("Unzipping folder")
        unzip(file_path, settings.download_folder)
        logger.debug("Folder unzipped")
        clear_folder(settings.temp_folder, delete_folder=True)
        logger.debug("Ingesting documents")
        ingest_documents()    
        logger.info("Upload and ingestion complete")
    except Exception:
        return {"message": "There was an error processing the uploaded file"}
    
    return {"message": f"Successfully uploaded {file.filename}"}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
