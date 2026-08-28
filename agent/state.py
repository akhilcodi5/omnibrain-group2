from typing import TypedDict


class AgentState(TypedDict):
    user_query: str
    messages: list
    next_agent: str
    retrieved_context: list
    vision_result: str
    sql_result: str
    final_answer: str