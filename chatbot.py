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


# 사용자 정보 입력용 
class UserInfo(BaseModel):
    age: int
    like: str

@app.post("/set_user_info")
def set_user_info(user_info: UserInfo):
    # 올바르게 user_info의 값을 전달
    cmodel.get_user_info(user_info.age, user_info.like)
    return {"message": "User information updated successfully"}