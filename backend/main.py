from typing import Optional, Union
from uuid import UUID

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from langsmith import Client
from pydantic import BaseModel
from retriever import answer_chain, search_chain

# import logging
# logging.basicConfig(filename='info.log', level=logging.DEBUG)


# def log_info(req_body, res_body):
#     logging.info(req_body)
#     logging.info(res_body)


class Query(BaseModel):
    question: str

class SendFeedbackBody(BaseModel):
    run_id: UUID
    key: str = "user_score"

    score: Union[float, int, bool, None] = None
    feedback_id: Optional[UUID] = None
    comment: Optional[str] = None

client = Client()

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
    client.create_feedback(
        body.run_id,
        body.key,
        score=body.score,
        comment=body.comment,
        feedback_id=body.feedback_id,
    )
    return {"result": "posted feedback successfully", "code": 200}

# @app.middleware('http')
# async def some_middleware(request: Request, call_next):
#     print("Middleware")
#     req_body = await request.body()
#     print(req_body)
#     response = await call_next(request)

#     res_body = b''
#     async for chunk in response.body_iterator:
#         res_body += chunk

#     task = BackgroundTask(log_info, req_body, res_body)
#     return Response(content=res_body, status_code=response.status_code,
#         headers=dict(response.headers), media_type=response.media_type, background=task)


@app.get("/test")
async def search():
    return {"message": "Beep boop doing a search.."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
