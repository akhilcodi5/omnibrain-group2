"""SQLAlchemy ORM database entity models for structured financial and document records."""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class StockEntity(Base):
    """Stock entity representing corporate equities and market benchmarks."""
    __tablename__ = "stocks"

    ticker = Column(String(10), primary_key=True, index=True)
    company_name = Column(String(255), nullable=False)
    current_price = Column(Float, nullable=False)
    pe_ratio = Column(Float, nullable=True)
    market_cap_billions = Column(Float, nullable=True)
    fifty_two_week_high = Column(Float, nullable=True)
    fifty_two_week_low = Column(Float, nullable=True)

    quarterly_financials = relationship("QuarterlyFinancialEntity", back_populates="stock")


class QuarterlyFinancialEntity(Base):
    """Quarterly financial line items and margin metrics."""
    __tablename__ = "quarterly_financials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), ForeignKey("stocks.ticker"), nullable=False, index=True)
    fiscal_quarter = Column(String(20), nullable=False)
    revenue_millions = Column(Float, nullable=False)
    operating_margin_pct = Column(Float, nullable=True)
    net_income_millions = Column(Float, nullable=True)
    eps = Column(Float, nullable=True)

    stock = relationship("StockEntity", back_populates="quarterly_financials")


class DocumentEntity(Base):
    """Ingested PDF document metadata entity."""
    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    total_pages = Column(Integer, nullable=False)
    total_chunks = Column(Integer, default=0)
    total_images_extracted = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    visual_assets = relationship("VisualAssetEntity", back_populates="document")


class VisualAssetEntity(Base):
    """Extracted visual chart or table asset entity."""
    __tablename__ = "visual_assets"

    id = Column(String(64), primary_key=True, index=True)
    document_id = Column(String(64), ForeignKey("documents.id"), nullable=False)
    page_number = Column(Integer, nullable=False)
    figure_type = Column(String(50), default="unknown")
    image_path = Column(String(512), nullable=False)
    summary = Column(Text, nullable=True)

    document = relationship("DocumentEntity", back_populates="visual_assets")
