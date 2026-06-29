import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import JSON, Column, DateTime, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./retinascan.db")

connect_args = {}
if DATABASE_URL.startswith("sqlite:///"):
    database_path = Path(DATABASE_URL.replace("sqlite:///", "", 1))
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class AnalysisRecord(Base):
    __tablename__ = "analyses"

    id = Column(String, primary_key=True)  # uuid
    filename = Column(String, nullable=False)  # nom original
    image_path = Column(String, nullable=False)  # chemin sur disque
    results = Column(JSON, nullable=False)  # liste d'exsudats
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
