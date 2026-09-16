from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# -----------------------------
# 1. LangGraph State
# -----------------------------
class State(TypedDict):
    user_input: str
    route: str
    response: str


# -----------------------------
# 2. Supervisor Node
# -----------------------------
def supervisor(state: State):
    user_input = state["user_input"].lower()

    # Simple routing logic for Role 1
    if any(word in user_input for word in [
        "chart", "image", "graph", "visual", "figure"
    ]):
        route = "vision"

    elif any(word in user_input for word in [
        "stock", "revenue", "profit", "sql", "database",
        "financial year", "percentage", "calculate"
    ]):
        route = "sql"

    else:
        route = "search"

    return {
        "route": route
    }


# -----------------------------
# 3. Search Agent
# -----------------------------
def search_agent(state: State):
    return {
        "response": (
            "Search Agent selected. "
            "This agent will search the Qdrant/vector database "
            "for relevant document content."
        )
    }


# -----------------------------
# 4. SQL Agent
# -----------------------------
def sql_agent(state: State):
    return {
        "response": (
            "SQL Agent selected. "
            "This agent will handle structured financial and "
            "database queries."
        )
    }


# -----------------------------
# 5. Vision Agent
# -----------------------------
def vision_agent(state: State):
    return {
        "response": (
            "Vision Agent selected. "
            "This agent will analyze charts, images, and "
            "visual information from documents."
        )
    }


# -----------------------------
# 6. Routing Function
# -----------------------------
def route_to_agent(
    state: State,
) -> Literal["search", "sql", "vision"]:

    return state["route"]


# -----------------------------
# 7. Build LangGraph
# -----------------------------
builder = StateGraph(State)

builder.add_node("supervisor", supervisor)
builder.add_node("search", search_agent)
builder.add_node("sql", sql_agent)
builder.add_node("vision", vision_agent)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "search": "search",
        "sql": "sql",
        "vision": "vision",
    },
)

builder.add_edge("search", END)
builder.add_edge("sql", END)
builder.add_edge("vision", END)


# -----------------------------
# 8. Compile the Graph
# -----------------------------
app = builder.compile()


# -----------------------------
# 9. Test the Role 1 Graph
# -----------------------------
if __name__ == "__main__":

    test_queries = [
        "Find information about the company in the financial report.",
        "What was the company's revenue?",
        "Explain the chart in the report."
    ]

    for query in test_queries:
        result = app.invoke({
            "user_input": query,
            "route": "",
            "response": ""
        })

        print("\nUser:", query)
        print("Route:", result["route"])
        print("Response:", result["response"]) 