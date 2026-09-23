import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict):
    user_query: str

    # `operator.add` reducer so every node that returns "messages" appends
    # to the running conversation instead of overwriting it.
    messages: Annotated[list, operator.add]

    next_agent: str
    retrieved_context: list
    vision_result: str
    sql_result: str
    final_answer: str

    # Safety counter incremented by the supervisor on every routing decision,
    # used to force termination and avoid infinite supervisor <-> agent loops.
    iteration_count: int