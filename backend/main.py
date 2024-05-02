from fastapi import FastAPI, Request, Response
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from pydantic import BaseModel
import logging

from retriever import ChatRequest, answer_chain, search_chain

# logging.basicConfig(filename='info.log', level=logging.DEBUG)


# def log_info(req_body, res_body):
#     logging.info(req_body)
#     logging.info(res_body)

class Query(BaseModel):
    question: str

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
)

add_routes(
    app,
    search_chain,
    path="/search",
)

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