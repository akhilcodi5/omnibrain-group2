"""Text-to-SQL Agent for querying structured historical stock and financial databases."""

import logging
import re
from typing import Any, Dict, List, Optional
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from agents.state import AgentState
from storage.sql_db import FinancialDatabase, get_financial_db

logger = logging.getLogger(__name__)


class SQLAgent:
    """Specialist agent responsible for inspecting financial database schemas and executing Text-to-SQL queries."""

    def __init__(self, db: Optional[FinancialDatabase] = None):
        self.db = db or get_financial_db()

    def generate_sql(self, query: str) -> str:
        """Map user query intent to SQL query using rule-based and keyword extraction."""
        q = query.upper()

        # Check for target ticker
        ticker = "APEX"
        for t in ["NVDA", "AAPL", "MSFT", "APEX"]:
            if t in q:
                ticker = t
                break

        if "PRICE" in q or "P/E" in q or "MARKET CAP" in q or "52-WEEK" in q or "HIGH" in q or "LOW" in q:
            return f"SELECT ticker, company_name, current_price, pe_ratio, market_cap_billions, fifty_two_week_high, fifty_two_week_low FROM stocks WHERE ticker = '{ticker}';"
        
        elif "QUARTER" in q or "REVENUE" in q or "MARGIN" in q or "EPS" in q or "NET INCOME" in q:
            return f"SELECT fiscal_quarter, revenue_millions, operating_margin_pct, net_income_millions, eps FROM quarterly_financials WHERE ticker = '{ticker}' ORDER BY id ASC;"

        return f"SELECT * FROM stocks WHERE ticker = '{ticker}';"

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
    logger.info(f"Executing sql_agent_node for query: '{query}'")

    agent = SQLAgent()
    result = agent.execute(query)

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
