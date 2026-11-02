"""The shared SQLAlchemy engine. All database I/O in the product (API routes and Temporal activities) goes through this one engine."""
from sqlalchemy import create_engine

from services.app.settings import settings

_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url.replace("postgresql://", "postgresql+psycopg://"), pool_pre_ping=True)
    return _engine
