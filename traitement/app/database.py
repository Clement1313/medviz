from sqlalchemy import create_engine, Column, String, JSON, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./retinascan.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class AnalysisRecord(Base):
    __tablename__ = "analyses"
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    results = Column(JSON, nullable=False)
    diagnosis = Column(JSON, nullable=True)  # nouveau champ
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
