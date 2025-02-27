import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List


from app import chatbot_model

cmodel = chatbot_model.ChatbotModel()

app = FastAPI(title="Chatbot API")

class ChatRequest(BaseModel):
    message:str
    userId:str

@app.post("/chat")
def chat(request: ChatRequest):
    response = cmodel.get_response(request.userId, request.message)
    return {"content" :response}

class UserData(BaseModel):
    id: str
    age: int
    like: List[str]

@app.post("/user")
def setUserData(request: UserData):
    cmodel.get_user_info(request.id, request.age, request.like)
    return {"message": "User data received", "data": request}


# 음성으로 전달하는거 만들예정 : Nick
# @app.websocket("/voiceChat")
# def chat():
#     response = cmodel.get_response(request.userId, request.message)
#     return {"content" :response}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)