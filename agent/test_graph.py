from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
class State(TypedDict):
    message: str


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)


def llm_node(state: State):
    response = llm.invoke(state["message"])

    return {
        "message": response.content
    }


builder = StateGraph(State)

builder.add_node("llm", llm_node)

builder.add_edge(START, "llm")
builder.add_edge("llm", END)

graph = builder.compile()


result = graph.invoke({
    "message": "What is a LangGraph Supervisor?"
})

print(result["message"])