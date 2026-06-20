import os
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database import get_db, AnalysisRecord


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()

    record_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    save_path = UPLOAD_DIR / f"{record_id}{ext}"

    with open(save_path, "wb") as f:
        f.write(contents)

    # TODO: appliquer le traitement sur content

    results = [
        {
            "id": 1,
            "x": 38,
            "y": 42,
            "radius": 3.2,
            "confidence": 94,
            "severity": "modéré",
            "type": "Mou",
            "size": "Grand",
        },
        {
            "id": 2,
            "x": 52,
            "y": 58,
            "radius": 2.1,
            "confidence": 87,
            "severity": "modéré",
            "type": "Dur",
            "size": "Grand",
        },
    ]

    record = AnalysisRecord(
        id=record_id,
        filename=file.filename,
        image_path=str(save_path),
        results=results,
    )
    db.add(record)
    db.commit()
    return {"id": record_id, "results": results}


@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    records = db.query(AnalysisRecord).order_by(AnalysisRecord.created_at.desc()).all()
    return [
        {
            "id": r.id,
            "name": r.filename,
            "date": int(r.created_at.timestamp() * 1000),
            "results": r.results,
        }
        for r in records
    ]


@app.get("/history/{record_id}/image")
def get_history_image(record_id: str, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record:
        return {"error": "not found"}
    return FileResponse(record.image_path)
