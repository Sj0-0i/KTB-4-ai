import uvicorn
import pymysql
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# 챗봇 모델 임포트
from app import chatbot_model

cmodel = chatbot_model.ChatbotModel()

# FastAPI 애플리케이션 생성
app = FastAPI(title="Chatbot API")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인 허용 (보안 상 특정 도메인만 허용하는 것이 좋음)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

# .env 파일 로드
load_dotenv()

# RDS MySQL 연결 정보 (환경 변수에서 가져오기)
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# MySQL 연결 함수
def get_db_connection():
    try:
        conn = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            cursorclass=pymysql.cursors.DictCursor
        )
        print("DB 연결 성공")
        return conn
    except Exception as e:
        print(f"DB 연결 실패: {e}")
        return None

# FastAPI 실행 시 DB 연결 테스트
db_test_conn = get_db_connection()
if db_test_conn:
    db_test_conn.close()
else:
    print("FastAPI 실행 중, DB 연결 불가")

# 챗봇 응답 API
class ChatRequest(BaseModel):
    message: str
    userId: str

# 사용자 정보 저장 API
class UserData(BaseModel):
    id: str
    age: int
    like: List[str]

@app.options("/chat")
async def options_chat(request: Request):
    return JSONResponse(content=None, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
    })


@app.post("/chat")
def chat(request: ChatRequest):
    response = cmodel.get_response(request.userId, request.message)
    return JSONResponse(
        content={"content": response},
        headers={"Access-Control-Allow-Origin": "*"}  # 명시적 허용
    )

@app.post("/user")
def setUserData(request: UserData):
    session_id = request.id  # id 값을 session_id로 저장
    cmodel.get_user_info(session_id, request.age, request.like)

    return {"message": "User data received", "session_id": session_id, "data": request}

# DB 상태 확인 API
@app.get("/chat/db-status")
def check_db_status():
    conn = get_db_connection()
    if conn:
        conn.close()
        return JSONResponse(content={"status": "DB 연결 성공"})
    else:
        return JSONResponse(content={"status": "DB 연결 실패"}, status_code=500)

# 특정 세션의 대화 메시지 조회 API
@app.get("/chat/messages")
def get_chat_messages(session_id: str = Query(..., description="Session ID for chat history")):
    conn = get_db_connection()
    if not conn:
        return JSONResponse(content={"error": "DB 연결 실패"}, status_code=500)

    cursor = conn.cursor()
    query = "SELECT id, session_id, message FROM message_store WHERE session_id = %s ORDER BY id ASC"
    cursor.execute(query, (session_id,))
    messages = cursor.fetchall()

    # JSON 필드를 파싱하여 반환
    parsed_messages = []
    for msg in messages:
        try:
            msg_data = json.loads(msg["message"])  # JSON 파싱
            parsed_messages.append({
                "id": msg["id"],
                "session_id": msg["session_id"],
                "type": msg_data.get("type"),
                "content": msg_data["data"].get("content")
            })
        except json.JSONDecodeError:
            parsed_messages.append({
                "id": msg["id"],
                "session_id": msg["session_id"],
                "type": "unknown",
                "content": "Invalid JSON format"
            })

    cursor.close()
    conn.close()

    return JSONResponse(content={"messages": parsed_messages})

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
# 메시지 저장 API (JSON 형식으로 데이터 삽입)
class ChatMessage(BaseModel):
    session_id: str  # 기존대로 유지
    type: str  # "human" 또는 "ai"
    content: str

@app.post("/chat/save")
def save_chat_message(request: ChatMessage):
    conn = get_db_connection()
    if not conn:
        return JSONResponse(content={"error": "DB 연결 실패"}, status_code=500)

    cursor = conn.cursor()

    # MySQL에 저장할 JSON 데이터 생성
    message_json = json.dumps({
        "type": request.type,
        "data": {
            "content": request.content,
            "additional_kwargs": {},
            "response_metadata": {},
            "type": request.type,
            "name": None,
            "id": None,
            "example": False
        }
    })

    query = "INSERT INTO message_store (session_id, message) VALUES (%s, %s)"
    cursor.execute(query, (request.session_id, message_json))

    conn.commit()
    cursor.close()
    conn.close()

    return {"status": "success", "message": "Chat message saved"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)