import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver  
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
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
                connection=f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
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
        if self.chat_message_history is None:
            return {"messages": ["Session ID가 설정되지 않았습니다."]}

        past_messages = self.chat_message_history.messages  # 대화 기록 리스트
        all_messages = past_messages + state["messages"]

        last_human_message = next((msg.content for msg in reversed(all_messages) if isinstance(msg, HumanMessage)), None)
        if last_human_message is None:
            last_human_message = "Hello, how can I help you?"  # 기본 질문 설정

        prompt = self.prompt_template.invoke({
            "history": all_messages,
            "question": last_human_message,
            "age": self.age,
            "like": self.like,  
        })
        response = self.model.invoke(prompt)

        return {"messages": all_messages + [response]}

    def get_response(self, session_id, message):
        """ 세션 ID를 기반으로 사용자 입력에 대한 챗봇 응답을 반환 """
        if self.session_id != session_id:
            self.session_id = session_id
            self.chat_message_history = SQLChatMessageHistory(
                session_id=self.session_id, 
                connection=f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            )

        config = {"configurable": {"thread_id": session_id}}
        input_messages = [HumanMessage(message)]
        self.chat_message_history.add_user_message(message)
        output = self.app.invoke({"messages": input_messages}, config)
        ai_response = output["messages"][-1].content
        self.chat_message_history.add_ai_message(ai_response)

        return ai_response
    
    def get_user_info(self, session_id, age, like):
        """ 사용자 정보를 설정하고 이후 대화에서 활용 """
        self.session_id = session_id  # 기존 userId → session_id로 변경
        self.age = age
        self.like = like

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
            connection=f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )

    async def text_to_speech(self, text):
        """ TTS를 사용하여 텍스트를 음성으로 변환 (Edge TTS) """
        tts = edge_tts.Communicate(text, "ko-KR-SunHiNeural")  # 한국어 음성 선택
        mp3_data = io.BytesIO()

        async for chunk in tts.stream():
            if chunk["type"] == "audio":
                mp3_data.write(chunk["data"])

        return mp3_data.getvalue()  # MP3 바이너리 데이터 반환