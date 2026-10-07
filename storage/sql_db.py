"""Relational SQL Database connection, schema creation, and financial data management."""

import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FinancialDatabase:
    """Manages SQLite database connections and structured financial queries for the SQL Agent."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("DATABASE_URL", "sqlite:///./storage/financial_data.db").replace("sqlite:///", "")
        self._conn: Optional[sqlite3.Connection] = None
        
        # Ensure parent directory exists
        if self.db_path != ":memory:":
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
        else:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._conn.row_factory = sqlite3.Row

        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create or return a sqlite3 connection with dict cursor factory."""
        if self.db_path == ":memory:":
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._conn.row_factory = sqlite3.Row
            return self._conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create sample financial tables and populate mock benchmark data."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Stocks table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS stocks (
                ticker TEXT PRIMARY KEY,
                company_name TEXT NOT NULL,
                current_price REAL NOT NULL,
                pe_ratio REAL,
                market_cap_billions REAL,
                fifty_two_week_high REAL,
                fifty_two_week_low REAL
            )
            """)

            # 2. Quarterly financials table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS quarterly_financials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                fiscal_quarter TEXT NOT NULL,
                revenue_millions REAL NOT NULL,
                operating_margin_pct REAL,
                net_income_millions REAL,
                eps REAL,
                FOREIGN KEY (ticker) REFERENCES stocks (ticker)
            )
            """)

            # Insert sample rows if empty
            cursor.execute("SELECT COUNT(*) FROM stocks")
            if cursor.fetchone()[0] == 0:
                cursor.executemany("""
                INSERT INTO stocks (ticker, company_name, current_price, pe_ratio, market_cap_billions, fifty_two_week_high, fifty_two_week_low)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [
                    ("APEX", "Apex Holdings Corp", 184.50, 24.6, 42.5, 192.00, 128.40),
                    ("NVDA", "NVIDIA Corporation", 125.80, 48.2, 3100.0, 140.76, 45.11),
                    ("AAPL", "Apple Inc", 228.40, 33.5, 3450.0, 237.23, 164.08),
                    ("MSFT", "Microsoft Corp", 418.20, 35.1, 3120.0, 468.35, 309.45),
                ])

                cursor.executemany("""
                INSERT INTO quarterly_financials (ticker, fiscal_quarter, revenue_millions, operating_margin_pct, net_income_millions, eps)
                VALUES (?, ?, ?, ?, ?, ?)
                """, [
                    ("APEX", "Q1 FY24", 120.5, 22.5, 27.1, 0.85),
                    ("APEX", "Q2 FY24", 135.2, 23.8, 32.2, 0.95),
                    ("APEX", "Q3 FY24", 148.8, 25.0, 37.2, 1.10),
                    ("APEX", "Q4 FY24", 162.0, 27.2, 44.1, 1.30),
                ])
            conn.commit()

    def get_schema_summary(self) -> str:
        """Return the schema definition for Text-to-SQL prompting."""
        summary = ""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table_row in tables:
                table_name = table_row[0]
                if table_name == "sqlite_sequence":
                    continue
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = cursor.fetchall()
                cols_str = ", ".join([f"{c['name']} ({c['type']})" for c in columns])
                summary += f"Table: {table_name}\nColumns: {cols_str}\n\n"
        return summary.strip()

    def ingest_pdf_tables(self, pdf_name: str, tables_by_page: List[List[List[Optional[str]]]]):
        """Dynamically ingest extracted 2D PDF tables into SQLite."""
        import re
        clean_pdf_name = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.splitext(os.path.basename(pdf_name))[0]).lower()
        table_metadata = []
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for page_idx, tables in enumerate(tables_by_page):
                for tbl_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                        
                    # Clean headers
                    raw_headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
                    headers = []
                    for h in raw_headers:
                        clean_h = re.sub(r'[^a-zA-Z0-9_]', '_', h).lower()
                        clean_h = re.sub(r'^_+|_+$', '', clean_h)
                        if not clean_h or clean_h[0].isdigit():
                            clean_h = f"col_{clean_h}"
                        headers.append(clean_h)
                        
                    # Ensure unique headers
                    seen = {}
                    unique_headers = []
                    for h in headers:
                        if h in seen:
                            seen[h] += 1
                            unique_headers.append(f"{h}_{seen[h]}")
                        else:
                            seen[h] = 0
                            unique_headers.append(h)
                    
                    table_name = f"tbl_{clean_pdf_name}_p{page_idx+1}_{tbl_idx+1}"
                    
                    # Create table
                    cols_def = ", ".join([f"{h} TEXT" for h in unique_headers])
                    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({cols_def})")
                    
                    # Insert rows
                    insert_sql = f"INSERT INTO {table_name} VALUES ({','.join(['?']*len(unique_headers))})"
                    for row in table[1:]:
                        padded_row = [str(cell) if cell is not None else "" for cell in row]
                        while len(padded_row) < len(unique_headers):
                            padded_row.append("")
                        padded_row = padded_row[:len(unique_headers)]
                        cursor.execute(insert_sql, padded_row)
                        
                    table_metadata.append({
                        "page_number": page_idx + 1,
                        "table_name": table_name,
                        "columns": unique_headers,
                        "rows_inserted": len(table[1:])
                    })
            conn.commit()
        return table_metadata


    def execute_query(self, sql_query: str) -> List[Dict[str, Any]]:
        """Execute a read-only SQL query and return results as dictionaries."""
        # Enforce read-only constraint
        clean_query = sql_query.strip().upper()
        if not clean_query.startswith("SELECT") and not clean_query.startswith("WITH"):
            raise ValueError("Only SELECT and WITH queries are permitted.")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


_default_db = None


def get_financial_db() -> FinancialDatabase:
    """Singleton getter for FinancialDatabase."""
    global _default_db
    if _default_db is None:
        _default_db = FinancialDatabase()
    return _default_db
