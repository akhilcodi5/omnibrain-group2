"""Text-to-SQL Agent for querying structured historical stock and financial databases."""

import logging
import re
import time
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agents.state import AgentState
from storage.sql_db import FinancialDatabase, get_financial_db
from app.core.telemetry import get_telemetry_manager

import os
import time
from app.core.telemetry import get_telemetry_manager

logger = logging.getLogger(__name__)


class SQLAgent:
    """Specialist agent responsible for inspecting financial database schemas and executing Text-to-SQL queries."""

    def __init__(self, db: Optional[FinancialDatabase] = None):
        self.db = db or get_financial_db()

    def generate_sql(self, query: str) -> str:
        """Map user query intent to SQL query using Gemini LLM."""
        import google.generativeai as genai
        import os
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found. Falling back to simple rule engine.")
            return "SELECT * FROM stocks LIMIT 5;"

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash")
            
            schema = '''
            Table: stocks
            Columns: id (INTEGER), ticker (TEXT), company_name (TEXT), current_price (REAL), pe_ratio (REAL), market_cap_billions (REAL), fifty_two_week_high (REAL), fifty_two_week_low (REAL)
            
            Table: quarterly_financials
            Columns: id (INTEGER), ticker (TEXT), fiscal_quarter (TEXT), revenue_millions (REAL), operating_margin_pct (REAL), net_income_millions (REAL), eps (REAL)
            '''

            prompt = f'''
            You are a SQLite expert. Generate a single SQLite query to answer the user's question.
            Only output the raw SQL query, without markdown backticks or any other text.
            
            Database Schema:
            {schema}
            
            Question: {query}
            '''

            response = model.generate_content(prompt)
            sql = response.text.strip()
            # Clean up potential markdown formatting
            if sql.startswith("```sql"):
                sql = sql[6:]
            if sql.startswith("```"):
                sql = sql[3:]
            if sql.endswith("```"):
                sql = sql[:-3]
            
            return sql.strip()
        except Exception as e:
            logger.error(f"Gemini SQL generation error: {e}")
            return "SELECT * FROM stocks LIMIT 5;"

    def execute(self, query: str) -> Dict[str, Any]:
        """Execute Text-to-SQL resolution and query execution."""
        sql_statement = self.generate_sql(query)
        logger.info(f"Generated SQL query: {sql_statement}")
        
        try:
            results = self.db.execute_query(sql_statement)
            summary_lines = [f"SQL Query: `{sql_statement}`", f"Returned {len(results)} structured rows:"]
            for r in results:
                summary_lines.append(f"- {r}")

            return {
                "sql_query": sql_statement,
                "sql_results": results,
                "summary": "\n".join(summary_lines),
            }
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return {
                "sql_query": sql_statement,
                "sql_results": [],
                "summary": f"SQL execution error: {str(e)}",
            }


@tool
def execute_sql_query_tool(query: str) -> str:
    """Executes a natural language Text-to-SQL query against structured financial tables.
    
    Args:
        query: Natural language question regarding stock prices, market caps, or quarterly metrics.
        
    Returns:
        Stringified JSON results.
    """
    agent = SQLAgent()
    res = agent.execute(query)
    return res.get("summary", "")


def sql_agent_node(state: AgentState) -> Dict[str, Any]:
    """LangGraph node handler for the Text-to-SQL Agent."""
    query = state.get("query", "")
    trace_id = state.get("trace_id")
    logger.info(f"Executing sql_agent_node for query: '{query}'")

    start_time = time.time()
    agent = SQLAgent()
    result = agent.execute(query)
    elapsed = time.time() - start_time

    if trace_id:
        get_telemetry_manager().log_agent_step(
            trace_id=trace_id,
            agent_name="SQLAgent",
            action="GenerateSQL",
            model="gemini-2.5-flash",
            input_data=query,
            output_data=result["summary"],
            prompt_tokens=400, # approximate
            completion_tokens=50,
            latency_seconds=elapsed
        )

    ai_message = AIMessage(
        content=f"**[SQL Agent Results]**\n\n{result['summary']}",
        name="SQLAgent",
    )

    return {
        "messages": [ai_message],
        "sql_query": result["sql_query"],
        "sql_results": result["sql_results"],
        "next_agent": "Supervisor",
    }

