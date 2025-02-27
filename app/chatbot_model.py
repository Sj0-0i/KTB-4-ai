import json
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
import pymysql
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_openai import ChatOpenAI
import asyncio
import edge_tts
import io

# 환경 변수 로딩
load_dotenv()

# RDS MySQL 연결 정보
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")


# MySQL 연결 함수
def get_db_connection():
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        cursorclass=pymysql.cursors.DictCursor
    )
    return conn


class ChatbotModel:
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        language: str

    def __init__(self, model_name="gpt-4o-mini", model_provider="openai"):
        self.model = init_chat_model(model_name, model_provider=model_provider)
        self.model_name = model_name
        self.session_id = None
        self.age = None
        self.like = None

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", """
                당신은 친절한 어시스턴트입니다. 
                시니어가 이해하기 쉽게 짧고 간결하며 친절하게 답변해주세요.
                나의 나이는 {age}
                나의 관심사는 {like}야
                나의 나이에 맞는 답변을 해줘
                """),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )

        self.memory = MemorySaver()
        self.chat_message_history = None

        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)

    def _build_workflow(self):
        workflow = StateGraph(state_schema=self.State)
        workflow.add_node("model", self._call_model)
        workflow.add_edge(START, "model")
        return workflow

    def _call_model(self, state: State):
        if self.chat_message_history is None:
            return {"messages": ["Session ID가 설정되지 않았습니다."]}

        # 사용자 정보 조회
        if self.session_id and (self.age is None or self.like is None):
            self._load_user_info_from_db(self.session_id)

        # 메시지 처리
        past_messages = []
        try:
            past_messages = self.chat_message_history.messages
        except:
            pass

        all_messages = past_messages + state["messages"]

        last_human_message = next((msg.content for msg in reversed(all_messages) if isinstance(msg, HumanMessage)),
                                  None)
        if last_human_message is None:
            last_human_message = "Hello, how can I help you?"

        # 사용자 정보 기본값 설정
        age_value = self.age if self.age is not None else "알 수 없음"
        like_value = self.like if self.like is not None else "알 수 없음"

        # 응답 생성
        prompt = self.prompt_template.invoke({
            "history": all_messages,
            "question": last_human_message,
            "age": age_value,
            "like": like_value,
        })
        response = self.model.invoke(prompt)

        return {"messages": all_messages + [response]}

    def get_response(self, session_id, message):
        # 세션 설정
        self.session_id = session_id
        self.chat_message_history = SQLChatMessageHistory(
            session_id=self.session_id,
            connection_string=f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

        # 사용자 정보 로드
        self._load_user_info_from_db(session_id)

        # 메시지 처리
        config = {"configurable": {"thread_id": session_id}}
        input_messages = [HumanMessage(content=message)]

        try:
            self.chat_message_history.add_user_message(message)
        except:
            pass

        # LLM 호출 및 응답 처리
        output = self.app.invoke({"messages": input_messages}, config)

        if not isinstance(output, dict) or "messages" not in output:
            return "응답을 생성할 수 없습니다."

        ai_response = output["messages"]

        if not ai_response:
            return "응답을 생성할 수 없습니다."

        # 응답이 리스트이고 마지막 항목이 AIMessage인 경우
        if isinstance(ai_response, list) and ai_response and isinstance(ai_response[-1], AIMessage):
            response_content = ai_response[-1].content
            try:
                self.chat_message_history.add_ai_message(response_content)
            except:
                pass
            return response_content

        # 응답이 문자열인 경우
        elif isinstance(ai_response, str):
            try:
                self.chat_message_history.add_ai_message(ai_response)
            except:
                pass
            return ai_response

        # 기타 경우 - 문자열로 변환
        return str(ai_response)

    def _load_user_info_from_db(self, session_id):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            query = "SELECT age, likes FROM user_info WHERE session_id = %s"
            cursor.execute(query, (session_id,))
            user_info = cursor.fetchone()

            if user_info:
                self.age = user_info['age']
                if user_info['likes']:
                    try:
                        self.like = json.loads(user_info['likes'])
                    except:
                        self.like = user_info['likes']
                else:
                    self.like = None
            else:
                self.age = None
                self.like = None

            cursor.close()
            conn.close()
        except:
            self.age = None
            self.like = None

    def get_user_info(self, session_id, age, like):
        self.session_id = session_id
        self.age = age
        self.like = like

        # DB에 사용자 정보 저장
        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # 기존 사용자 정보가 있는지 확인
            check_query = "SELECT * FROM user_info WHERE session_id = %s"
            cursor.execute(check_query, (session_id,))
            existing_user = cursor.fetchone()

            if existing_user:
                # 기존 사용자 정보 업데이트
                update_query = "UPDATE user_info SET age = %s, likes = %s WHERE session_id = %s"
                cursor.execute(update_query, (age, json.dumps(like), session_id))
            else:
                # 새 사용자 정보 추가
                insert_query = "INSERT INTO user_info (session_id, age, likes) VALUES (%s, %s, %s)"
                cursor.execute(insert_query, (session_id, age, json.dumps(like)))

            conn.commit()
            cursor.close()
            conn.close()
        except:
            pass

    async def get_stream_response(self, user_id, message):
        response = self.get_response(user_id, message)
        yield self.text_to_speech(response)
        await asyncio.sleep(0.1)
        async for chunk in response:
            if "choices" in chunk and chunk["choices"]:
                yield chunk["choices"][0]["delta"]["content"]

    async def text_to_speech(self, text):
        tts = edge_tts.Communicate(text, "ko-KR-SunHiNeural")
        mp3_data = io.BytesIO()

        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                mp3_data.write(chunk["data"])

        return mp3_data.getvalue()