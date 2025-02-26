from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import app_model
import chatbot_model
app = FastAPI()

model = app_model.AppModel()
cmodel = chatbot_model.ChatbotModel()

@app.get("/chat")
def chat(text: str = Query(), user: str = Query()):
    response = cmodel.get_response(user, text)
    return {"content" :response}

