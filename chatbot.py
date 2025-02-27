from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chatbot_model

app = FastAPI()

cmodel = chatbot_model.ChatbotModel()

class ChatRequest(BaseModel):
    text: str
    user: str

@app.post("/chat")
def chat(request: ChatRequest):
    response = cmodel.get_response(request.user, request.text)
    return {"content": response}
