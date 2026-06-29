import os
import uuid
from pathlib import Path
import numpy as np
from skimage import io
from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database import get_db, AnalysisRecord
import joblib

from segmentation.segmentation import segment

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(exist_ok=True)
CLF_PATH = Path("app/clf.joblib")
clf = joblib.load(CLF_PATH)


@app.get("/")
async def root():
    return {"message": "Hello World"}


def compute_diagnosis(masks, image_shape):
    height, width = image_shape[:2]

    crop_size = min(width, height)
    retina_area = np.pi * (crop_size / 2) ** 2

    exudates_area = sum(mask.sum() for _, mask in masks)

    n_exudates = len(masks)
    surface_ratio = exudates_area / retina_area if retina_area > 0 else 0

    if surface_ratio >= 0.05:
        stage = "Sévère / proche PDR"
        interpretation = "Forte accumulation d'exsudats, atteinte rétinienne importante"
    elif surface_ratio >= 0.01:
        stage = "Modéré / NPDR avancé"
        interpretation = "Progression visible des dépôts lipidiques"
    else:
        stage = "Léger / NPDR précoce"
        interpretation = "Peu d'exsudats, petites lésions localisées"

    return {
        "n_exudates": n_exudates,
        "surface_ratio": round(surface_ratio, 4),
        "stage": stage,
        "interpretation": interpretation,
    }


def mask_to_result(label, mask, image_shape, idx):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    height, width = image_shape[:2]

    crop_size = min(width, height)
    x_offset = (width - crop_size) / 2
    y_offset = (height - crop_size) / 2

    x_in_crop = xs.mean() - x_offset
    y_in_crop = ys.mean() - y_offset

    x_pct = round(float(x_in_crop) / crop_size * 100, 1)
    y_pct = round(float(y_in_crop) / crop_size * 100, 1)

    area = len(xs)
    radius = round(float(np.sqrt(area / np.pi)) / crop_size * 100, 2)

    severity, exudate_type, size, confidence = classify_exudate(mask, label, area)

    return {
        "id": idx,
        "x": x_pct,
        "y": y_pct,
        "radius": radius,
        "confidence": confidence,
        "severity": severity,
        "type": exudate_type,
        "size": size,
    }


def classify_exudate(mask, label, area, proba=None):
    exudate_type = "Dur"

    SIZE_THRESHOLD = 300  # pixels
    size = "Grand" if area > SIZE_THRESHOLD else "Petit"

    if area < 250:
        severity = "faible"
    elif area < 10000:
        severity = "modéré"
    else:
        severity = "élevé"

    if proba is not None:
        confidence = round(float(proba) * 100)
    else:
        confidence = 90

    return severity, exudate_type, size, confidence


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await file.read()

    record_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    save_path = UPLOAD_DIR / f"{record_id}{ext}"

    with open(save_path, "wb") as f:
        f.write(contents)

    masks = segment(str(save_path), clf=clf, threshold=0.95)
    # print(f">>> Nombre de masques retournés par segment(): {len(masks)}")
    # for label, mask in masks:
    #     print(f"   label={label}, aire={mask.sum()}, bbox=({np.where(mask)[1].min()}-{np.where(mask)[1].max()}, {np.where(mask)[0].min()}-{np.where(mask)[0].max()})")

    image = io.imread(save_path)
    image_shape = image.shape

    results = []
    exudate_masks = []
    for idx, (label, mask) in enumerate(masks, start=1):
        if label == 0:
            continue
        exudate_masks.append((label, mask))
        r = mask_to_result(label, mask, image_shape, idx)
        if r is not None:
            results.append(r)

    diagnosis = compute_diagnosis(exudate_masks, image_shape)

    record = AnalysisRecord(
        id=record_id,
        filename=file.filename,
        image_path=str(save_path),
        results=results,
        diagnosis=diagnosis,
    )
    db.add(record)
    db.commit()
    return {"id": record_id, "results": results, "diagnosis": diagnosis}


@app.get("/history")
def get_history(db: Session = Depends(get_db)):
    records = db.query(AnalysisRecord).order_by(AnalysisRecord.created_at.desc()).all()
    return [
        {
            "id": r.id,
            "name": r.filename,
            "date": int(r.created_at.timestamp() * 1000),
            "results": r.results,
            "diagnosis": r.diagnosis,
        }
        for r in records
    ]


@app.get("/history/{record_id}/image")
def get_history_image(record_id: str, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record:
        return {"error": "not found"}
    return FileResponse(record.image_path)
