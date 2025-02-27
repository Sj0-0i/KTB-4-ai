import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket
from typing import List
from fastapi.middleware.cors import CORSMiddleware


from app import chatbot_model

cmodel = chatbot_model.ChatbotModel()

app = FastAPI(title="Chatbot API")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (보안 상 특정 도메인만 허용하는 것이 좋음)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

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

@app.websocket("/chat-stream")
async def chat_stream(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    await websocket.send_text(f"Welcome client : {websocket.client}")

    user_id = None  # 첫 번째 메시지에서 user_id를 저장

    try:
        while True:
            data = await websocket.receive_text()

            if not data:
                break

            if user_id is None:
                # 첫 번째 메시지는 user_id로 처리
                user_id = data
                print(f"User ID received: {user_id}")
                await websocket.send_text(f"User ID {user_id} registered.")
            else:
                print(f"Received message: {data}")

                async for chunk in cmodel.get_stream_response(user_id, data):
                    await websocket.send_bytes(chunk)

    except Exception as e:
        print(f"WebSocket error: {e}")


    finally:
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)