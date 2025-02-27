from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

import chatbot_model
app = FastAPI()

cmodel = chatbot_model.ChatbotModel()

@app.get("/chat")
def chat(text: str = Query(), user: str = Query()):
    response = cmodel.get_response(user, text)
    return {"content" :response}

