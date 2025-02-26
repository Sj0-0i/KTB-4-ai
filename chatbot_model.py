from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain_core.messages import trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict



class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

class ChatbotModel:
  def __init__(self):
    load_dotenv() 
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                당신은 친절한 어시스턴트입니다. 
                시니어가 이해하기 쉽게 짧고 간결하게 답변해주세요.
                나의 성별은 남성 이고
                나의 생년월일은 1951년 8월 21일 이야
                나의 거주지는 대한민국 판교 유스페이스1 A동이고
                나의 취미는 운동이야
                나의 나이에 맞는 답변을 해줘
                """,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    trimmer = trim_messages(
        max_tokens=65,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )
    workflow = StateGraph(state_schema=State)

    def call_model(state: State):
        trimmed_messages = trimmer.invoke(state["messages"])
        prompt = prompt_template.invoke(
            {"messages": trimmed_messages, "language": state["language"]}
        )
        response = model.invoke(prompt)
        return {"messages": response}
    
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    memory = MemorySaver()
    self.app = workflow.compile(checkpointer=memory)

    self.messages = []


  def get_response(self, thread_id, language, message):
    config = {"configurable": {"thread_id": thread_id}}
    input_messages = self.messages + [HumanMessage(message)]
    output = self.app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )

    # print(self.messages)
    return output["messages"][-1]
  

