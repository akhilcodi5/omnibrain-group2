import logging
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from .state import AgentState


load_dotenv()

logger = logging.getLogger(__name__)


# --------------------------------------------------
# 1. Possible routing decisions
# --------------------------------------------------

class SupervisorDecision(BaseModel):
    next_agent: Literal["rag", "vision", "sql", "FINISH"] = Field(
        description=(
            "The next agent that should handle the task. "
            "Choose rag for document retrieval/search, "
            "vision for images/charts/visual content, "
            "sql for database queries, "
            "or FINISH when the task is complete."
        )
    )


# --------------------------------------------------
# 2. Gemini model
# --------------------------------------------------

# Model name is configurable via env var so it can be changed without a
# code change / redeploy. Falls back to a valid Gemini model.
SUPERVISOR_MODEL = os.getenv("SUPERVISOR_MODEL", "gemini-3.6-flash")

# Safety limit on supervisor <-> agent hops to guarantee termination even
# if the LLM never returns FINISH.
MAX_SUPERVISOR_ITERATIONS = int(os.getenv("MAX_SUPERVISOR_ITERATIONS", "6"))

llm = ChatGoogleGenerativeAI(
    model=SUPERVISOR_MODEL,
    temperature=0,
)


# Force Gemini to return the routing decision
structured_llm = llm.with_structured_output(
    SupervisorDecision,
    method="json_schema"
)


# --------------------------------------------------
# 3. Supervisor system prompt (persona / instructions only)
# --------------------------------------------------

SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor of a multi-agent AI system.

Your job is ONLY to decide which specialized agent should work next.

Available agents:

1. rag
   - Searches/retrieves information from documents.
   - Handles PDF text retrieval, citations and document-based questions.

2. vision
   - Handles images, charts, figures and visual information.
   - Extracts information from visual content.

3. sql
   - Handles structured data and database queries.

4. FINISH
   - Choose this when the available information is sufficient
     and the task can be completed.

Routing rules:

- If the question requires information from documents/PDF text,
  choose rag.
- If the question requires understanding an image, chart, graph,
  table or other visual content, choose vision.
- If the question requires querying structured database information,
  choose sql.
- If the required information has already been obtained and no
  additional agent is necessary, choose FINISH.

Return ONLY the routing decision."""


def _build_state_prompt(
    user_query: str,
    retrieved_context: list,
    vision_result: str,
    sql_result: str,
    final_answer: str,
) -> str:
    return f"""USER QUERY:
{user_query}

CURRENT STATE:

Retrieved context:
{retrieved_context}

Vision result:
{vision_result}

SQL result:
{sql_result}

Current final answer:
{final_answer}
"""


# --------------------------------------------------
# 4. Supervisor node
# --------------------------------------------------

def supervisor_node(state: AgentState):

    iteration_count = state.get("iteration_count", 0) + 1

    # Hard safety cutoff: force FINISH instead of looping forever if the
    # LLM keeps delegating to agents without resolving the task.
    if iteration_count > MAX_SUPERVISOR_ITERATIONS:
        logger.warning(
            "Supervisor reached MAX_SUPERVISOR_ITERATIONS=%s. Forcing FINISH.",
            MAX_SUPERVISOR_ITERATIONS,
        )
        return {
            "next_agent": "FINISH",
            "iteration_count": iteration_count,
        }

    user_query = state.get("user_query", "")

    retrieved_context = state.get("retrieved_context", [])
    vision_result = state.get("vision_result", "")
    sql_result = state.get("sql_result", "")
    final_answer = state.get("final_answer", "")

    state_prompt = _build_state_prompt(
        user_query, retrieved_context, vision_result, sql_result, final_answer
    )

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=state_prompt),
    ]

    try:
        decision = structured_llm.invoke(messages)
        next_agent = decision.next_agent
    except Exception:
        # If the LLM call/parsing fails (rate limit, network error,
        # malformed structured output, etc.) fail safe instead of
        # crashing the whole graph run.
        logger.exception(
            "Supervisor routing decision failed; falling back to FINISH."
        )
        next_agent = "FINISH"

    return {
        "next_agent": next_agent,
        "iteration_count": iteration_count,
    }


# --------------------------------------------------
# 5. Routing logic (conditional edge function)
# --------------------------------------------------

def route_after_supervisor(state: AgentState) -> Literal["rag", "vision", "sql", "__end__"]:
    """Maps the supervisor's routing decision to the next LangGraph node.

    Intended to be wired up by whoever assembles the full graph via:

        builder.add_conditional_edges(
            "supervisor",
            route_after_supervisor,
            {"rag": "rag", "vision": "vision", "sql": "sql", "__end__": END},
        )
    """

    next_agent = state.get("next_agent", "FINISH")

    if next_agent == "FINISH":
        return END

    return next_agent