import json
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver  
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
import pymysql
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
import asyncio  # 비동기 처리를 위한 모듈
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

class ChatbotModel:
    class State(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        language: str

    def __init__(self, model_name="gpt-4o-mini", model_provider="openai"):
        self.model = init_chat_model(model_name, model_provider=model_provider)
        self.llm = ChatOpenAI(model_name=model_name)
        self.model_name = model_name
        self.session_id = None  # 기존 userId → session_id로 변경
        self.age = None
        self.like = None

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system",  """
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
        self.chat_message_history = None  # `session_id` 설정 후 초기화하도록 변경

        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.memory)

        self.chain = self.prompt_template | self.llm  
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            lambda session_id: SQLChatMessageHistory(
                session_id=session_id, 
                connection_string=f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            ),
            input_messages_key="question",
            history_messages_key="history",
        )

    def _build_workflow(self):
        workflow = StateGraph(state_schema=self.State)
        workflow.add_node("model", self._call_model)
        workflow.add_edge(START, "model")
        return workflow

    def _call_model(self, state: State):
        """LLM 모델 호출 및 응답 생성 함수"""
        if self.chat_message_history is None:
            return {"messages": ["Session ID가 설정되지 않았습니다."]}

        # 현재 세션 ID에 대한 사용자 정보가 없는 경우 DB에서 로드 시도
        if self.session_id and (self.age is None or self.like is None):
            self._load_user_info_from_db(self.session_id)

        past_messages = self.chat_message_history.messages  # 대화 기록 리스트
        all_messages = past_messages + state["messages"]

        last_human_message = next((msg.content for msg in reversed(all_messages) if isinstance(msg, HumanMessage)),
                                  None)
        if last_human_message is None:
            last_human_message = "Hello, how can I help you?"  # 기본 질문 설정

        # 사용자 정보 기본값 설정
        age_value = self.age if self.age is not None else "알 수 없음"
        like_value = self.like if self.like is not None else "알 수 없음"

        # 디버깅을 위한 로그
        print(f"프롬프트에 포함되는 사용자 정보 - 나이: {age_value}, 관심사: {like_value}")

        prompt = self.prompt_template.invoke({
            "history": all_messages,
            "question": last_human_message,
            "age": age_value,
            "like": like_value,
        })
        response = self.model.invoke(prompt)

        return {"messages": all_messages + [response]}

    def get_response(self, session_id, message):
        """ 세션 ID를 기반으로 사용자 입력에 대한 챗봇 응답을 반환 """
        if self.session_id != session_id:
            self.session_id = session_id
            self.chat_message_history = SQLChatMessageHistory(
                session_id=self.session_id,
                connection_string=f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            )

            # 세션 ID가 변경되면 DB에서 사용자 정보 로드
            self._load_user_info_from_db(session_id)

        # 이미 로드된 사용자 정보가 없으면 다시 로드 시도
        if self.age is None or self.like is None:
            self._load_user_info_from_db(session_id)

        config = {"configurable": {"thread_id": session_id}}
        input_messages = [HumanMessage(content=message)]

        # 사용자 메시지 저장 시 예외 처리
        try:
            self.chat_message_history.add_user_message(message)
        except Exception as user_msg_err:
            print(f"사용자 메시지 저장 실패: {user_msg_err}")
            # 오류가 발생해도 계속 진행

        try:
            # LLM을 통해 응답 생성
            output = self.app.invoke({"messages": input_messages}, config)

            # output에서 응답 추출
            if not isinstance(output, dict) or "messages" not in output:
                return "응답 형식이 올바르지 않습니다."

            ai_response = output["messages"]

            # 응답이 없는 경우
            if not ai_response:
                return "응답을 생성하는 중 오류가 발생했습니다."

            # 응답이 리스트이고 마지막 항목이 AIMessage인 경우
            if isinstance(ai_response, list) and ai_response and isinstance(ai_response[-1], AIMessage):
                final_message = ai_response[-1]
                response_content = final_message.content

                # 응답 저장 시도
                try:
                    self.chat_message_history.add_ai_message(response_content)
                except Exception as save_err:
                    print(f"AI 응답 저장 실패: {save_err}")

                return response_content

            # 응답이 문자열인 경우
            elif isinstance(ai_response, str):
                try:
                    self.chat_message_history.add_ai_message(ai_response)
                except Exception as save_err:
                    print(f"AI 응답 저장 실패: {save_err}")

                return ai_response

            # 그 외의 경우: 응답 전체를 문자열로 변환
            else:
                # 마지막 메시지가 있으면 그 내용을 추출
                if isinstance(ai_response, list) and ai_response:
                    last_item = ai_response[-1]

                    # 마지막 항목에서 'content' 속성 추출 시도
                    if hasattr(last_item, 'content'):
                        response_content = last_item.content
                        try:
                            self.chat_message_history.add_ai_message(response_content)
                        except Exception as save_err:
                            print(f"AI 응답 저장 실패: {save_err}")

                        return response_content

                # 마지막 수단: 전체 응답을 문자열로 변환
                response_text = str(ai_response)
                try:
                    # 응답이 너무 길면 잘라서 저장
                    if len(response_text) > 1000:
                        truncated_text = response_text[:997] + "..."
                        self.chat_message_history.add_ai_message(truncated_text)
                    else:
                        self.chat_message_history.add_ai_message(response_text)
                except Exception as save_err:
                    print(f"AI 응답 저장 실패: {save_err}")

                return response_text

        except Exception as e:
            error_msg = f"응답 생성 중 오류 발생: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg

    def _load_user_info_from_db(self, session_id):
        """DB에서 사용자 정보를 로드하는 헬퍼 함수"""
        try:
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                query = "SELECT age, likes FROM user_info WHERE session_id = %s"
                cursor.execute(query, (session_id,))
                user_info = cursor.fetchone()

                if user_info:
                    self.age = user_info['age']
                    # likes 필드가 JSON 문자열이라면 파싱
                    if user_info['likes']:
                        try:
                            self.like = json.loads(user_info['likes'])
                        except json.JSONDecodeError:
                            # JSON 파싱 실패 시 그대로 사용
                            self.like = user_info['likes']
                    else:
                        self.like = None

                    print(f"사용자 정보 로드 성공 - 나이: {self.age}, 관심사: {self.like}")
                else:
                    print(f"세션 ID {session_id}에 대한 사용자 정보가 없습니다.")
                    self.age = None
                    self.like = None

                cursor.close()
                conn.close()
        except Exception as e:
            print(f"사용자 정보 로드 중 오류 발생: {e}")
            self.age = None
            self.like = None

    # get_user_info 함수 수정 - DB에 사용자 정보 저장
    def get_user_info(self, session_id, age, like):
        """ 사용자 정보를 DB에 저장하고 이후 대화에서 활용 """
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
        except Exception as e:
            print(f"사용자 정보 저장 중 오류 발생: {e}")
        finally:
            if conn:
                conn.close()

    async def get_stream_response(self, user_id, message):
        response = self.get_response(user_id, message)
        yield self.text_to_speech(response)
        await asyncio.sleep(0.1)
        async for chunk in response:
            if "choices" in chunk and chunk["choices"]:
                yield chunk["choices"][0]["delta"]["content"]
        # 채팅 기록 초기화 (MySQL 연결)
        self.chat_message_history = SQLChatMessageHistory(
            session_id=self.session_id,
            connection_string=f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

    async def text_to_speech(self, text):
        """ TTS를 사용하여 텍스트를 음성으로 변환 (Edge TTS) """
        tts = edge_tts.Communicate(text, "ko-KR-SunHiNeural")  # 한국어 음성 선택
        mp3_data = io.BytesIO()

        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                mp3_data.write(chunk["data"])

        return mp3_data.getvalue()  # MP3 바이너리 데이터 반환